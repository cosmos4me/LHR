"""MM-Fi 어댑터 (변환기 v0.2; v0.2 = 정답 X, Y를 정답 없는 RGB 초점거리로 구한 배율 s로 보정, 깊이 보정 재적합): 원본 → 시퀀스별 팩 (dev/packs/mmfi_v02_spec{A,B}/{E}/{S}_{A}.npz).
- 입력으로 쓰는 것: RGB 2D 키포인트(rgb/, COCO-17, 640×480 px)와 깊이(depth/, D435, mm). infra1/infra2는 정답을 만든 입력이라 쓰지 않는다.
- 정답: ground_truth.npy (H36M-17, RGB 카메라 좌표, m) → 공통 13관절, 방 좌표. 정답 품질 비트(정지·뼈 길이 이상·움직임 감쇠)를 함께 저장.
- 사양은 분할마다 학습 피험자로만 맞춘다 (specs/mmfi_v02_spec{A,B}.json):
    A = E01 S01–S06 (E1, E2-시험 E02/E03용),  B = E02 S11–S16 + E03 S21–S26 (E2-시험 E01용).
  RGB: K(최소제곱), 2D 오차 = 2변량 Student-t(ν), scale² = (g_j·s)² + (β·|v_kp|)², 정의 차이 중앙값을 뺀 뒤 MLE.
       정지 정답·움직임 감쇠 구간은 맞춤에서 제외.
  깊이: calib_depth.py 결과(학습 피험자, t_z = 0, 표면 오프셋 자유)와 정답 없이 구한 C = f·b = 19.175 px·m.
  바닥 평면: 환경별로 깊이만으로 (정답 없음).
- 사람 영역 깊이 점: RGB 키포인트로 영역을 잡고(정답 없음), 섞인 가장자리 화소를 빼고, 바닥 근처 점을 뺀 뒤 256점 표본.
  가장자리 근처 점에는 edge 표시.
사용: python adapt_mmfi.py --spec A [--spec-only]"""
import json, sys, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_common as cc

CODE = Path(__file__).resolve().parent
ROOT = Path("/home/interns/unisense/dev/data/mmfi/extracted")
NOTES = Path("/home/interns/unisense/notes")
ENVS = ["E01", "E02", "E03"]
S = lambda a, b: [f"S{i:02d}" for i in range(a, b + 1)]
SPEC_FIT = {"A": [("E01", S(1, 6))], "B": [("E02", S(11, 16)), ("E03", S(21, 26))]}
GT_SCALE = json.loads(Path("/home/interns/unisense/notes/gt_scale_mmfi.json").read_text())["s"]   # 정답 X, Y 축소 보정 (사양별)
DEPTH_VARIANT = "C0"                   # calib_depth.py 변형 중 검증 피험자 적중률로 고른 것 (notes/calib_depth_spec*.json)
N_PTS, FPS, INVALID = 256, 10, (0, 65535)
UP_HINT, MIN_COS = (0.0, -1.0, 0.0), 0.8   # MM-Fi 카메라는 거의 수평 (바닥 평면 추정 시 위쪽 방향)
PAIRS = {0: 9, 1: 10, 2: 10, 3: 10, 4: 10, 5: 11, 6: 14, 7: 12, 8: 15, 9: 13, 10: 16, 11: 4, 12: 1, 13: 5,
         14: 2, 15: 6, 16: 3}            # COCO → H36M (얼굴은 코 → 9, 눈·귀 → 10 대용)
BODY = list(range(5, 17))
C13_H36M = cc.JOINT_MAP["h36m17"]
BONES = [(11, 12), (12, 13), (14, 15), (15, 16), (4, 5), (5, 6), (1, 2), (2, 3), (1, 4), (11, 14), (8, 10)]


def frames(seq, kind):
    return sorted((seq / kind).glob("frame*"), key=lambda p: int(p.stem[5:]))


def sequences(env, subjects=None):
    seqs = sorted(p.parent for p in (ROOT / env).glob("S*/A*/ground_truth.npy"))
    return [s for s in seqs if subjects is None or s.parent.name in subjects]


def read_depth(path):
    d = np.array(Image.open(path)).astype(np.float32)
    d[np.isin(d, INVALID)] = np.nan
    return d * 1e-3


def load_gt(path, spec_id):
    """정답 (RGB 카메라 좌표). X, Y를 사양별 배율 s로 나눠 물리 크기로 보정 (Z는 깊이 표면과 일치해 그대로)."""
    gt = np.load(path).astype(float)
    gt[..., :2] /= GT_SCALE[spec_id]
    return gt


def load_seq(s, spec_id):
    gt = load_gt(s / "ground_truth.npy", spec_id); rf = frames(s, "rgb")
    T = min(len(gt), len(rf))
    return np.stack([np.load(rf[t]) for t in range(T)]).astype(float), gt[:T]


def project(G, K):
    fx, fy, cx, cy = K
    return np.stack([fx * G[..., 0] / G[..., 2] + cx, fy * G[..., 1] / G[..., 2] + cy], -1)


def gt_flags(kp, gt, K):
    """정답 품질 (관절별 COCO 쌍 기준): 정지(직전 프레임과 정확히 같음), 움직임 감쇠(정답/키포인트 이동 범위 < 0.6)."""
    h = np.array([PAIRS[c] for c in range(17)])
    G = gt[:, h]
    frozen = np.zeros(G.shape[:2], bool); frozen[1:] = np.all(G[1:] == G[:-1], axis=-1)
    exc = lambda X: np.linalg.norm(np.percentile(X, 95, axis=0) - np.percentile(X, 5, axis=0), axis=-1)
    ek, eg = exc(kp), exc(project(G, K))
    damped = (ek > 40) & (eg / np.maximum(ek, 1e-6) < 0.6)
    return frozen, np.broadcast_to(damped[None], frozen.shape)


def fit_rgb(groups, spec_id, step=2):
    """RGB K (몸통 관절 최소제곱) → 2D 오차 모형 (Student-t, scale² = (g_j s)² + (β v)²)을 깨끗한 관측으로 MLE."""
    seqs = [s for env, subs in groups for s in sequences(env, subs)]
    data = [load_seq(s, spec_id) for s in seqs]
    KP = np.concatenate([k[::5] for k, _ in data]); G = np.concatenate([g[::5] for _, g in data])
    c, h = np.array(BODY), np.array([PAIRS[j] for j in BODY])
    X, Y, Z = G[:, h, 0], G[:, h, 1], G[:, h, 2]
    (fx, cx), *_ = np.linalg.lstsq(np.c_[(X / Z).ravel(), np.ones(X.size)], KP[:, c, 0].ravel(), rcond=None)
    (fy, cy), *_ = np.linalg.lstsq(np.c_[(Y / Z).ravel(), np.ones(Y.size)], KP[:, c, 1].ravel(), rcond=None)
    K = [float(fx), float(fy), float(cx), float(cy)]
    R, Sc, V, Jc, n_all = [], [], [], [], 0
    hh = np.array([PAIRS[j] for j in range(17)])
    for kp, gt in data:
        res = kp - project(gt[:, hh], K)                                   # (T,17,2)
        s = cc.person_scale(kp); v = cc.keypoint_speed(kp, FPS)
        frozen, damped = gt_flags(kp, gt, K)
        ok = ~(frozen | damped)
        idx = np.arange(0, len(kp), step)
        for j in range(17):
            m = ok[idx, j]; n_all += len(idx)
            R.append(res[idx, j][m]); Sc.append(s[idx][m]); V.append(v[idx, j][m]); Jc.append(np.full(m.sum(), j))
    R, Sc, V, Jc = map(np.concatenate, (R, Sc, V, Jc))
    off = np.stack([np.median(R[Jc == j] / Sc[Jc == j, None], axis=0) for j in range(17)])   # 정의 차이 (s 단위)
    Rc = R - off[Jc] * Sc[:, None]
    r2 = np.sum(Rc ** 2, -1)

    def nll(th):
        g, beta, nu = np.exp(th[:17]), np.exp(th[17]), np.exp(th[18])
        sig2 = (g[Jc] * Sc) ** 2 + (beta * V) ** 2
        return -np.mean(gammaln((nu + 2) / 2) - gammaln(nu / 2) - np.log(nu * np.pi) - np.log(sig2)
                        - (nu + 2) / 2 * np.log1p(r2 / (nu * sig2)))
    th0 = np.r_[np.log(0.3 * cc.COCO_OKS_SIGMA), np.log(0.03), np.log(3.0)]
    opt = minimize(nll, th0, method="L-BFGS-B")
    g, beta, nu = np.exp(opt.x[:17]), float(np.exp(opt.x[17])), float(np.exp(opt.x[18]))
    # 맞춤 집합에서 포함률 (2변량 t: m²/2 ~ F(2, ν))
    from scipy.stats import f as fdist
    sig2 = (g[Jc] * Sc) ** 2 + (beta * V) ** 2
    pit = fdist.cdf(r2 / sig2 / 2.0, 2, nu)
    cov = {f"{int(p * 100)}": round(float(np.mean(pit <= p)), 3) for p in (0.5, 0.68, 0.9, 0.95, 0.99)}
    return dict(K=K, g=g.tolist(), gamma=(g / cc.COCO_OKS_SIGMA).round(4).tolist(), beta_s=beta, nu=nu,
                offset_s_units=off.round(4).tolist(), n_obs=int(len(r2)), frac_excluded=round(1 - len(r2) / n_all, 3),
                coverage_fit_set=cov, converged=bool(opt.success))


def depth_calib(spec_id):
    C = json.loads((NOTES / f"calib_depth_spec{spec_id}_v02.json").read_text())["variants"][DEPTH_VARIANT]
    R = Rotation.from_rotvec(np.radians(C["rotvec_deg"])).as_matrix(); t = np.array(C["t_m"])   # X_d = R X_rgb + t
    return C, R, t


def floor_for_env(env, K_d, R_rgb2d, t_rgb2d, n_seq=12, seed=0):
    """환경별 바닥 평면 (RGB 카메라 좌표): 시퀀스별 시간 중앙값 깊이 → 역투영 → RANSAC. 깊이만 사용."""
    rng = np.random.default_rng(seed)
    seqs = sequences(env); pick = rng.choice(len(seqs), min(n_seq, len(seqs)), replace=False)
    fx, fy, cx, cy = K_d
    vv, uu = np.mgrid[0:480:4, 0:640:4]
    P = []
    for i in pick:
        D = np.stack([read_depth(f)[::4, ::4] for f in frames(seqs[i], "depth")[::3]])
        with np.errstate(all="ignore"):
            z = np.nanmedian(D, 0)
        m = (np.mean(np.isfinite(D), 0) > 0.5) & np.isfinite(z) & (z < 8)
        Pd = np.c_[(uu[m] - cx) / fx * z[m], (vv[m] - cy) / fy * z[m], z[m]]
        P.append((Pd - t_rgb2d) @ R_rgb2d)
    n, c, cnt = cc.fit_floor_plane(np.concatenate(P), UP_HINT, MIN_COS, seed=seed)
    return dict(n=n.tolist(), c=c, inliers=cnt, up_hint=list(UP_HINT), min_cos=MIN_COS)


def segment(d, kp, S_):
    """RGB 키포인트로 사람 영역 깊이 점 (u, v, z, edge)를 고른다. 정답은 쓰지 않는다.
    섞인 가장자리 화소(7×7 창의 범위 > 6σ_z이고 창의 최소·최대 양쪽에서 3σ_z 넘게 떨어진 화소)는 뺀다."""
    K_rgb, K_d = S_["rgb"]["K"], S_["depth"]["K"]
    R_rgb2d, t_rgb2d = np.array(S_["depth"]["R_rgb2d"]), np.array(S_["depth"]["t_rgb2d"])
    fxr, fyr, cxr, cyr = K_rgb; fx, fy, cx, cy = K_d
    dm = minimum_filter(np.nan_to_num(d, nan=99.0), size=5)
    zg = 3.0
    for _ in range(2):
        X = np.c_[(kp[:, 0] - cxr) / fxr * zg, (kp[:, 1] - cyr) / fyr * zg, np.full(len(kp), zg)] @ R_rgb2d.T + t_rgb2d
        kd = np.c_[fx * X[:, 0] / X[:, 2] + cx, fy * X[:, 1] / X[:, 2] + cy]
        ui = np.clip(np.rint(kd[:, 0]).astype(int), 0, 639); vi = np.clip(np.rint(kd[:, 1]).astype(int), 0, 479)
        zk = dm[vi, ui]; zk = zk[zk < 60]
        if zk.size < 5:
            return np.zeros((0, 4), np.float32), 0
        zg = float(np.median(zk))
    w, h = kd.max(0) - kd.min(0)
    lo = kd.min(0) - 0.15 * np.r_[w, h]; hi = kd.max(0) + 0.15 * np.r_[w, h]
    u0, v0 = np.maximum(np.floor(lo).astype(int) - 4, 0); u1, v1 = np.minimum(np.ceil(hi).astype(int) + 4, [639, 479])
    sub = d[v0:v1 + 1, u0:u1 + 1]
    inv = ~np.isfinite(sub)
    smax = maximum_filter(np.where(inv, -np.inf, sub), size=7); smin = minimum_filter(np.where(inv, np.inf, sub), size=7)
    sz = cc.axial_sigma(np.nan_to_num(sub, nan=3.0), S_["depth"])
    spread = smax - smin
    with np.errstate(invalid="ignore"):
        mixed = (spread > 6 * sz) & (sub - smin > 3 * sz) & (smax - sub > 3 * sz)
        band = np.abs(sub - zg) < 0.5
    edge = maximum_filter(inv, size=7) | (spread > 0.2)
    keep = band & ~mixed
    vv, uu = np.nonzero(keep)
    z = sub[vv, uu]; u = (uu + u0).astype(np.float32); v = (vv + v0).astype(np.float32)
    R_d2r, t_d2r = np.array(S_["room_cur"]["R_d2r"]), np.array(S_["room_cur"]["t_d2r"])
    P = np.c_[(u - cx) / fx * z, (v - cy) / fy * z, z] @ R_d2r.T + t_d2r
    ok = P[:, 2] > 0.05                                                     # 바닥 근처 점 제거
    n_mixed = int((band & mixed).sum())
    return np.c_[u, v, z, edge[vv, uu].astype(np.float32)][ok].astype(np.float32), n_mixed


def label_quality(gt, kp, K):
    """공통 13관절별 품질 비트 (T,13): 1 정지, 2 뼈 길이 이상(구간 중앙값 대비 ±25%), 4 움직임 감쇠."""
    T = len(gt); q = np.zeros((T, 13), np.uint8)
    h2c = {h: i for i, h in enumerate(C13_H36M)}
    fr = np.zeros((T, 17), bool); fr[1:] = np.all(gt[1:] == gt[:-1], axis=-1)
    for h, i in h2c.items():
        q[fr[:, h], i] |= 1
    for a, b in BONES:
        L = np.linalg.norm(gt[:, a] - gt[:, b], axis=-1)
        bad = np.abs(L / np.median(L) - 1) > 0.25
        for x in (a, b):
            if x in h2c:
                q[bad, h2c[x]] |= 2
    _, damped = gt_flags(kp, gt, K)
    coco_of_c13 = {i: c for c, i in zip(BODY, range(1, 13))}
    for i, c in coco_of_c13.items():
        q[damped[:, c], i] |= 4
    return q


def process(job):
    seq, env, S_, out_root = job
    S_ = dict(S_, room_cur=S_["room"][env])
    out = out_root / env / f"{seq.parent.name}_{seq.name}.npz"
    if out.exists():
        return str(out), "skip"
    gt = load_gt(seq / "ground_truth.npy", S_["spec_id"]); rf, df = frames(seq, "rgb"), frames(seq, "depth")
    T = min(len(gt), len(rf), len(df))
    kp = np.stack([np.load(rf[t]) for t in range(T)]).astype(np.float32)
    seed = zlib.crc32(f"mmfi/{env}/{seq.parent.name}/{seq.name}".encode())
    rng = np.random.default_rng(seed)
    pts = np.zeros((T, N_PTS, 4), np.float32); valid = np.zeros((T, N_PTS), bool)
    n_raw = np.zeros(T, np.int32); n_mixed = np.zeros(T, np.int32)
    for t in range(T):
        p, nm = segment(read_depth(df[t]), kp[t].astype(float), S_)
        n_raw[t], n_mixed[t] = len(p), nm
        if len(p):
            pick = rng.choice(len(p), min(N_PTS, len(p)), replace=False)
            pts[t, :len(pick)] = p[pick]; valid[t, :len(pick)] = True
    R_c2r, t_c2r = np.array(S_["room_cur"]["R_c2r"]), np.array(S_["room_cur"]["t_c2r"])
    lab_cam = cc.to_common13(gt[:T], "h36m17")
    meta = dict(dataset="mmfi", env=env, subject=seq.parent.name, action=seq.name, T=int(T), fps=FPS, seed=int(seed),
                label_family="h36m17", label_source="mmfi_ir_stereo_hrnet_triangulated", kp_family="coco17",
                spec=S_["spec_path"], spec_id=S_["spec_id"], version="v0.2")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, kp_px=kp, pts_uvz=pts[..., :3], pts_edge=pts[..., 3] > 0.5, pts_valid=valid,
                        pts_n_raw=n_raw, pts_n_mixed=n_mixed,
                        labels=(lab_cam @ R_c2r.T + t_c2r).astype(np.float32), labels_cam=lab_cam.astype(np.float32),
                        label_mask=np.ones((T, 13), bool), label_quality=label_quality(gt[:T], kp.astype(float), S_["rgb"]["K"]),
                        R_c2r=R_c2r, t_c2r=t_c2r, R_d2r=np.array(S_["room_cur"]["R_d2r"]), t_d2r=np.array(S_["room_cur"]["t_d2r"]),
                        meta=json.dumps(meta))
    return str(out), "ok"


def build_spec(spec_id):
    groups = SPEC_FIT[spec_id]
    rgb = fit_rgb(groups, spec_id)
    C, R_rgb2d, t_rgb2d = depth_calib(spec_id)
    fit_desc = ", ".join(f"{e} {s[0]}–{s[-1]}" for e, s in groups)
    path = CODE / "specs" / f"mmfi_v02_spec{spec_id}.json"
    S_ = dict(dataset="mmfi", spec_id=spec_id, spec_path=str(path), version="v0.2",
              rgb=dict(type="rgb_keypoint", sensor="RealSense D435 colour", image_wh=[640, 480], kp_family="coco17",
                       detector="HRNet (MM-Fi release keypoints)", K=rgb["K"],
                       sigma_model=dict(type="student_t", nu=rgb["nu"], g_per_joint=rgb["g"], gamma_per_joint=rgb["gamma"],
                                        beta_s=rgb["beta_s"], offset_s_units=rgb["offset_s_units"],
                                        coverage_fit_set=rgb["coverage_fit_set"]),
                       provenance=f"K: lstsq on {fit_desc} GT (body joints). 2D model: bivariate Student-t MLE on clean obs "
                                  f"(frozen/damped GT excluded, {rgb['frac_excluded']:.1%}), n={rgb['n_obs']}, "
                                  f"definitional median offset removed per keypoint; face keypoints use H36M 9/10 proxies"),
              depth=dict(type="stereo", sensor="RealSense D435", image_wh=[640, 480], K=C["K"], baseline_m=0.050,
                         stereo_constant_px_m=19.175, subpixel_steps=32, sigma_disp_px=0.08, f_px=float(C["K"][0]),
                         sigma_pix_px=float(1 / np.sqrt(12)), corr_len_px=8.0, n_eff_per_256=31,
                         surface_offset_r_m=C["surface_offset_r_m"],
                         R_rgb2d=R_rgb2d.tolist(), t_rgb2d=t_rgb2d.tolist(),
                         provenance=f"calib_depth.py variant {DEPTH_VARIANT} on {fit_desc} GT (t_z fixed 0, surface offset free); "
                                    "C = f·b = 19.175 px·m from disparity quantization levels (GT-free, 96% of mm levels in E01–E03); "
                                    "sigma_disp 0.08 px Intel spec; D0 temporal noise eff. 0.064–0.119 px (lower bound); "
                                    "spatial noise correlation ~8 px → N_eff ≈ 31 of 256 (converter verification)"),
              fps=FPS, label_family="h36m17", label_source="mmfi_ir_stereo_hrnet_triangulated",
              gt_scale_xy=GT_SCALE[spec_id], gt_scale_provenance="est_gt_scale.py: GT-free RGB fy (stature from LiDAR-validated depth) / GT-fit fy", room={})
    for env in ENVS:
        fl = floor_for_env(env, C["K"], R_rgb2d, t_rgb2d)
        R_c2r, t_c2r = cc.room_frame(np.array(fl["n"]), fl["c"])
        R_d2r, t_d2r = cc.compose(R_rgb2d.T, -R_rgb2d.T @ t_rgb2d, R_c2r, t_c2r)
        tilt = float(np.degrees(np.arcsin(np.clip(np.array(fl["n"]) @ np.array([0, 0, 1.0]), -1, 1))))
        S_["room"][env] = dict(floor=fl, camera_height_m=fl["c"], optical_axis_tilt_deg=tilt, R_c2r=R_c2r.tolist(),
                               t_c2r=t_c2r.tolist(), R_d2r=R_d2r.tolist(), t_d2r=t_d2r.tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(S_, indent=1))
    return S_


if __name__ == "__main__":
    sid = sys.argv[sys.argv.index("--spec") + 1]
    S_ = build_spec(sid)
    sm = S_["rgb"]["sigma_model"]
    print(json.dumps({"K": S_["rgb"]["K"], "nu": sm["nu"], "beta_s": sm["beta_s"], "gamma": sm["gamma_per_joint"],
                      "coverage_fit_set": sm["coverage_fit_set"], "depth_K": S_["depth"]["K"],
                      "room": {e: (round(S_["room"][e]["camera_height_m"], 4), round(S_["room"][e]["optical_axis_tilt_deg"], 3))
                               for e in ENVS}}, indent=1), flush=True)
    if "--spec-only" in sys.argv:
        sys.exit(0)
    out_root = Path(f"/home/interns/unisense/dev/packs/mmfi_v02_spec{sid}")
    jobs = [(s, e, S_, out_root) for e in ENVS for s in sequences(e)]
    with Pool(14) as pool:
        for i, _ in enumerate(pool.imap_unordered(process, jobs), 1):
            if i % 100 == 0:
                print(i, "/", len(jobs), flush=True)
    print("packs done", len(jobs))
