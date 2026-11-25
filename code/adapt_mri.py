"""mRI 어댑터 (변환기 v0.3): 공개 정답 + 독립 2D(ViTPose, 공개 blurred 영상) → 피험자별 팩 (dev/packs/mri_v03/{S}.npz).
v0.3: (1) 바닥 부호 오류 수정 — v0.2는 카메라 높이를 c − 0.08로 잡아 바닥이 16 cm 위에 있었다 (정답 발목이 z = −0.08).
      (2) 바닥을 정답 없이: ViTPose 두 시점 삼각측량 발목(모든 피험자, 입력만)의 RANSAC 평면 − 발목 높이 0.08 m.
      (3) cam2 규약을 정답 없이: 두 시점 ViTPose 키포인트의 에피폴라 거리로 고른다.
- 입력: ViTPose COCO-17 키포인트. 공개 영상은 좌우 반전돼 있어 u' = W−1−u, 좌우 관절 이름을 교환한다
  (교환 후 공개 pose_2d와 중앙 5.3–5.7 px: color0 ↔ cam1(원점), color1 ↔ cam2). 공개 pose_2d는 정답을 만든 2D라 쓰지 않는다.
- 정답: refined_gt_kps (COCO-17, m). 정답 좌표 = cam1 좌표에서 x, y를 뒤집은 것 (공개 cam1 K로 투영하면 pose_2d_r과 약 4 px).
- 깊이 없음 (Kinect 깊이 비공개) → 깊이 토큰 없음.
- 사양(학습 피험자만): cam1·cam2 K와 cam2 외부 파라미터는 공개값. 2D 오차는 Student-t(ν) + 속도 항 MLE.
  바닥: 깊이가 없어 학습 피험자 정답 발목 평면(RANSAC) − 발목 높이 사전값 0.08 m.
사용: python adapt_mri.py [--spec-only]"""
import io, json, pickle, sys, zlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import f as fdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_common as cc

CODE = Path(__file__).resolve().parent
LAB = Path("/home/interns/unisense/dev/data/mri/extracted/dataset_release/aligned_data/pose_labels")
DET = Path("/home/interns/unisense/dev/data/mri/derived/vitpose_b")
OUT = Path("/home/interns/unisense/dev/packs/mri_v03")
SPEC_PATH = CODE / "specs" / "mri_v03.json"
SPL = json.loads((CODE / "splits" / "mri.json").read_text())
W_IMG, N_PTS, ANKLE_H = 1920, 256, 0.08
SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
FLIP_GT = np.array([-1.0, -1.0, 1.0])                         # 정답 좌표 → cam1 좌표
ALLOW = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}


class _Safe(pickle.Unpickler):
    def find_class(self, m, n):
        if (m, n) in ALLOW:
            return super().find_class(m, n)
        raise pickle.UnpicklingError(f"blocked {m}.{n}")


def labels(s):
    return _Safe(io.BytesIO((LAB / f"subject{s}_all_labels.cpl").read_bytes())).load()


def detections(s, cam):
    d = np.load(DET / f"subject{s}_color{cam}.npz")
    kp = d["kp_px"][:, SWAP].astype(float); kp[..., 0] = W_IMG - 1 - kp[..., 0]
    return kp, d["kp_score"][:, SWAP].astype(float)


def project(X, K):
    uv = X @ np.asarray(K).T
    return uv[..., :2] / uv[..., 2:3]


def subject_arrays(s):
    """정답 구간 프레임만: cam1 좌표 정답 (T,17,3), cam1·cam2 키포인트, 점수, 동작 구간 번호."""
    L = labels(s); g0, g1 = L["gt_avail_frames"]
    G = L["refined_gt_kps"].transpose(0, 2, 1).astype(float) * FLIP_GT
    k1, s1 = detections(s, 0); k2, s2 = detections(s, 1)
    T = min(g1, len(k1), len(k2), len(G)); sl = slice(g0, T)
    act = np.full(len(G), -1, np.int16)
    for i, (name, (a, b)) in enumerate(L["video_label"].items()):
        act[a:b] = i
    return dict(G=G[sl], k1=k1[sl], s1=s1[sl], k2=k2[sl], s2=s2[sl], act=act[sl], frame0=g0,
                actions=list(L["video_label"].keys()), cm=L["camera_matrix"])


def fit_noise(arrs, K, fps, step=2):
    """학습 피험자 cam1: 잔차 = ViTPose − 정답 투영. 정의 차이 중앙값을 빼고 2변량 Student-t, scale² = (g_j s)² + (β v)² MLE."""
    R, Sc, V, Jc = [], [], [], []
    for a in arrs:
        kp, G = a["k1"], a["G"]
        res = kp - project(G, K)
        s = cc.person_scale(kp); v = cc.keypoint_speed(kp, fps)
        frozen = np.zeros(G.shape[:2], bool); frozen[1:] = np.all(G[1:] == G[:-1], -1)
        ok = np.isfinite(res).all(-1) & ~frozen
        idx = np.arange(0, len(kp), step)
        for j in range(17):
            m = ok[idx, j]
            R.append(res[idx, j][m]); Sc.append(s[idx][m]); V.append(v[idx, j][m]); Jc.append(np.full(m.sum(), j))
    R, Sc, V, Jc = map(np.concatenate, (R, Sc, V, Jc))
    off = np.stack([np.median(R[Jc == j] / Sc[Jc == j, None], 0) for j in range(17)])
    r2 = np.sum((R - off[Jc] * Sc[:, None]) ** 2, -1)

    def nll(th):
        g, beta, nu = np.exp(th[:17]), np.exp(th[17]), np.exp(th[18])
        sig2 = (g[Jc] * Sc) ** 2 + (beta * V) ** 2
        return -np.mean(gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(nu * np.pi) - np.log(sig2)
                        - (nu + 2) / 2 * np.log1p(r2 / (nu * sig2)))
    opt = minimize(nll, np.r_[np.log(0.3 * cc.COCO_OKS_SIGMA), np.log(0.03), np.log(3.0)], method="L-BFGS-B")
    g, beta, nu = np.exp(opt.x[:17]), float(np.exp(opt.x[17])), float(np.exp(opt.x[18]))
    pit = fdist.cdf(r2 / ((g[Jc] * Sc) ** 2 + (beta * V) ** 2) / 2, 2, nu)
    return dict(g_per_joint=g.tolist(), gamma_per_joint=(g / cc.COCO_OKS_SIGMA).round(4).tolist(), beta_s=beta, nu=nu,
                offset_s_units=off.round(4).tolist(), n_obs=int(len(r2)),
                coverage_fit_set={f"{int(p * 100)}": round(float(np.mean(pit <= p)), 3) for p in (0.5, 0.68, 0.9, 0.95, 0.99)},
                resid_px_median=round(float(np.median(np.sqrt(r2))), 2))


def norm_rays(k, K):
    a = np.concatenate([k, np.ones(k.shape[:-1] + (1,))], -1) @ np.linalg.inv(np.asarray(K, float)).T
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def triangulate_cam1(a, K1, K2, R12, t12):
    """두 시점 ViTPose 광선의 신뢰도 가중 최소제곱 교점 (cam1 좌표). 신뢰도 0.3 미만은 NaN. (T,17,3)"""
    d1 = norm_rays(a["k1"], K1); d2 = norm_rays(a["k2"], K2) @ R12; o2 = -R12.T @ t12      # X1 = R12ᵀ (X2 − t12)
    w1, w2 = a["s1"] ** 2, a["s2"] ** 2
    A1 = (np.eye(3) - d1[..., :, None] * d1[..., None, :]) * w1[..., None, None]
    A2 = (np.eye(3) - d2[..., :, None] * d2[..., None, :]) * w2[..., None, None]
    X = np.linalg.solve(A1 + A2, (A2 @ o2)[..., None])[..., 0]
    X[(a["s1"] < 0.3) | (a["s2"] < 0.3)] = np.nan
    return X


def fit_floor(arrs, K1, K2, R12, t12, seed=0):
    """정답 없는 바닥: 프레임마다 더 낮은 삼각측량 발목(cam1 좌표, 모든 피험자 입력)에 RANSAC 평면 → 발목 높이 사전값만큼 내린다.
    fit_floor_plane의 c = 카메라에서 발목 평면까지 높이 → 카메라 높이 = c + ANKLE_H (v0.2는 부호가 반대였다)."""
    A = []
    for a in arrs:
        X = triangulate_cam1(a, K1, K2, R12, t12)[:, 15:17]
        h = np.where(np.isfinite(X[..., 1]), X[..., 1], -np.inf); k = np.argmax(h, 1)          # cam1에서 아래 = +y → 더 낮은 발목
        low = X[np.arange(len(X)), k]; A.append(low[np.isfinite(low).all(-1)])
    A = np.concatenate(A)
    n, c, k = cc.fit_floor_plane(A, (0, -1, 0), 0.8, tol=0.03, min_height=0.2, seed=seed)
    return n, c + ANKLE_H, k, len(A)


def cam2_extrinsic(cm, arrs):
    """cam1 → cam2 변환 규약을 정답 없이 고른다: 두 시점 ViTPose 키포인트(신뢰도 ≥ 0.5)의 에피폴라 각도 거리 중앙값이 가장 작은 것."""
    K2 = np.asarray(cm["cam2_intrinsic"], float)
    E = np.asarray(cm["cam2_extrinsic"], float); R = np.asarray(cm["cam2_R"], float); T = np.asarray(cm["cam2_T"], float)
    cands = {"extrinsic[R|t]": (E[:, :3], E[:, 3] / 1000), "R,T": (R, T / 1000), "R,-RT": (R, -R @ T / 1000),
             "Rt,-RtT": (R.T, -R.T @ T / 1000)}
    K1 = np.asarray(cm["cam1_intrinsic"], float); errs = {}
    for name, (Rc, tc) in cands.items():
        Ex = np.array([[0, -tc[2], tc[1]], [tc[2], 0, -tc[0]], [-tc[1], tc[0], 0]]) @ Rc        # 본질 행렬 E = [t]× R
        e = []
        for a in arrs[:3]:
            ok = (a["s1"] >= 0.5) & (a["s2"] >= 0.5)
            x1, x2 = norm_rays(a["k1"][ok], K1), norm_rays(a["k2"][ok], K2)
            l = x1 @ Ex.T                                                                      # cam2의 에피폴라 평면 법선
            e.append(np.median(np.abs(np.sum(x2 * l, -1)) / np.linalg.norm(l, axis=-1)) * 1000)  # mrad
        errs[name] = float(np.median(e))
    best = min(errs, key=errs.get)
    return cands[best][0], cands[best][1], best, errs


def main():
    train = [int(x) for x in SPL["train"]]
    tr = [subject_arrays(s) for s in train]
    cm = tr[0]["cm"]
    K1 = np.asarray(cm["cam1_intrinsic"], float); K2 = np.asarray(cm["cam2_intrinsic"], float)
    fps = 1.0 / 0.115                                                   # 정렬 프레임 간격 중앙값 약 115 ms
    noise = fit_noise(tr, K1, fps)
    R12, t12, conv, errs = cam2_extrinsic(cm, tr)
    n, c, k, nA = fit_floor([subject_arrays(s) for s in range(1, 21)], K1, K2, R12, t12)     # 입력만 쓰므로 모든 피험자
    R_c2r, t_c2r = cc.room_frame(n, c)
    # cam2 → 방: X_room = R_c2r (R12ᵀ (X2 − t12)) + t_c2r
    R_c2r_2, t_c2r_2 = cc.compose(R12.T, -R12.T @ t12, R_c2r, t_c2r)
    tilt = float(np.degrees(np.arcsin(np.clip(n @ np.array([0, 0, 1.0]), -1, 1))))
    S_ = dict(dataset="mri", version="v0.3", spec_path=str(SPEC_PATH), fps=fps, label_family="coco17",
              label_source="mri_hrnet_stereo_triangulated_refined",
              rgb=dict(type="rgb_keypoint", sensor="Kinect v2 colour (2 views)", image_wh=[1920, 1080], kp_family="coco17",
                       detector="ViTPose-B (usyd-community/vitpose-base-simple) + RT-DETR boxes, on mirrored-back blurred videos",
                       K=K1.tolist(), K_cam2=K2.tolist(), R_cam1_to_cam2=R12.tolist(), t_cam1_to_cam2=t12.tolist(),
                       cam2_convention=conv, cam2_epipolar_mrad=errs, sigma_model=dict(type="student_t", **noise),
                       provenance=f"K: released camera_matrix. 2D model: Student-t MLE on train subjects {train} (cam1, frozen GT excluded)"),
              depth=None, room=dict(R_c2r=R_c2r.tolist(), t_c2r=t_c2r.tolist(), R_c2r_cam2=R_c2r_2.tolist(), t_c2r_cam2=t_c2r_2.tolist(),
                                    camera_height_m=float(c), optical_axis_tilt_deg=tilt, floor_inliers=int(k), n_ankle_pts=nA,
                                    provenance=f"floor = RANSAC plane of lower triangulated ViTPose ankles (2 views, all subjects, GT-free) − {ANKLE_H} m; cam2 convention by epipolar distance (GT-free)"))
    SPEC_PATH.write_text(json.dumps(S_, indent=1))
    print(json.dumps({"noise": {k_: noise[k_] for k_ in ("nu", "beta_s", "coverage_fit_set", "resid_px_median")},
                      "gamma": noise["gamma_per_joint"], "camera_height": round(float(c), 3), "tilt_deg": round(tilt, 2),
                      "cam2": conv, "cam2_epipolar_mrad": {k_: round(v, 2) for k_, v in errs.items()}}, indent=1), flush=True)
    if "--spec-only" in sys.argv:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for s in range(1, 21):
        a = subject_arrays(s); T = len(a["G"])
        G13 = cc.to_common13(a["G"], "coco17")
        q = np.zeros((T, 13), np.uint8)
        fr = np.zeros((T, 17), bool); fr[1:] = np.all(a["G"][1:] == a["G"][:-1], -1)
        frc = np.stack([fr[:, [3, 4]].any(1)] + [fr[:, j] for j in range(5, 17)], 1)
        q[frc] |= 1
        meta = dict(dataset="mri", subject=f"S{s:02d}", T=int(T), fps=fps, frame0=int(a["frame0"]), label_family="coco17",
                    label_source=S_["label_source"], kp_family="coco17", detector="vitpose_b", spec=str(SPEC_PATH),
                    version="v0.3", actions=a["actions"], seed=int(zlib.crc32(f"mri/S{s:02d}".encode())))
        np.savez_compressed(OUT / f"S{s:02d}.npz", kp_px=a["k1"].astype(np.float32), kp_score=a["s1"].astype(np.float32),
                            kp_px_cam2=a["k2"].astype(np.float32), kp_score_cam2=a["s2"].astype(np.float32),
                            pts_uvz=np.zeros((T, N_PTS, 3), np.float32), pts_edge=np.zeros((T, N_PTS), bool),
                            pts_valid=np.zeros((T, N_PTS), bool),
                            labels=(G13 @ R_c2r.T + t_c2r).astype(np.float32), labels_cam=G13.astype(np.float32),
                            label_mask=np.isfinite(G13).all(-1), label_quality=q, action=a["act"],
                            R_c2r=R_c2r, t_c2r=t_c2r, R_d2r=np.eye(3), t_d2r=np.zeros(3), meta=json.dumps(meta))
        print("pack", s, T, flush=True)


if __name__ == "__main__":
    main()
