"""ITOP 변환기 (평가 전용 개발 데이터셋, 2026-11-09). 학습에 쓰지 않는다. 원본: dev/data/itop/raw (Zenodo 3932973, CC-BY-4.0, 옆 시점 test 분할).
- 입력은 깊이만 (Asus Xtion PRO, 구조광, 320×240). 라벨에 딸린 segmentation (정답 기반 사람 마스크) 은 쓰지 않는다.
- 사람 점 (정답 없음): 사람 (시퀀스) 마다 화소별 깊이 90% 분위수를 배경으로 두고, 배경보다 0.12 m 넘게 가깝고 바닥 위 5 cm 이상인 화소의 가장 큰 연결 성분.
  섞인 가장자리 화소 제거·가장자리 표시는 다른 변환기와 같은 규칙 (7×7 창), 256점 표본.
- 방 좌표 (정답 없음): 사람마다 영상 아래쪽 1/3 깊이 점으로 바닥 평면 RANSAC (위쪽 힌트 = 카메라 −y, 기울기 18° 이내) → conv_common.room_frame. (2026-11-09 첫 판은 0.8 허용으로 35° 기운 면을 잡았음)
- 정답: real_world_coordinates (m, 카메라 x 오른쪽 · y 위 · z 앞) 15관절 → 공통 13관절 (손 → 손목, 발 → 발목 자리; 관례 차이는 영샷 평가에서 그대로 오차로 남는다). is_valid 프레임만.
- 사양 (사전, 정답 없음): f = 1/0.0035 px (ITOP 코드), 구조광 축 잡음 1.42e-3·z² (Khoshelham & Oude Elberink 2012) = stereo 모형 C = 21.43 px·m, σ_d = 0.0304 px.
- 프레임 속도는 원본 문서에 없어 10 fps 로 둔다 (시간 창 간격에만 쓰임).
사용: python -I adapt_itop.py   (dev/packs/itop_v01/U0x.npz, specs/itop_v01.json)"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, binary_opening, label as cc_label
CODE = Path("/home/interns/unisense/code"); sys.path.insert(0, str(CODE))
import conv_common as cc
RAW = Path("/home/interns/unisense/dev/data/itop/raw"); OUT = Path("/home/interns/unisense/dev/packs/itop_v01"); SPEC = CODE / "specs" / "itop_v01.json"
F_PX, CX, CY, W, H, N_PTS, FPS = 1 / 0.0035, 160.0, 120.0, 320, 240, 256, 10.0
J13 = [0, 3, 2, 5, 4, 7, 6, 10, 9, 12, 11, 14, 13]                 # ITOP: 0 머리 1 목 2 R어깨 3 L어깨 4 R팔꿈치 5 L팔꿈치 6 R손 7 L손 8 몸통 9 R엉덩이 10 L엉덩이 11 R무릎 12 L무릎 13 R발 14 L발
DM = dict(type="stereo", sensor="Asus Xtion PRO (ITOP side)", image_wh=[W, H], K=[[F_PX, 0, CX], [0, F_PX, CY], [0, 0, 1]], f_px=F_PX, stereo_constant_px_m=21.43, sigma_disp_px=0.0304, subpixel_steps=8,
          sigma_pix_px=float(1 / np.sqrt(12)), corr_len_px=4.0, n_eff_per_256=31, quant_step_m=0.0,
          provenance="사전: ITOP 코드 f = 1/0.0035 px, 주점 (160, 120); 축 잡음 1.42e-3 z^2 (Kinect v1 계열 구조광, Khoshelham 2012) 을 stereo 식 z^2 sigma_d / C 로 (C = f·b, b = 0.075 m)")


def cam_std(P_itop):
    """ITOP 카메라 (x 오른쪽, y 위) → 표준 핀홀 (x 오른쪽, y 아래)."""
    P = np.array(P_itop, float); P[..., 1] *= -1; return P


def unproject(z):
    v, u = np.mgrid[0:H, 0:W]; return np.stack([(u - CX) / F_PX * z, (v - CY) / F_PX * z, z], -1)


def person_points(z, bg, R, t, rng):
    inv = ~(np.isfinite(z) & (z > 0.3))
    zz = np.where(inv, np.nan, z); Pr = unproject(np.nan_to_num(zz)) @ R.T + t; hgt = Pr[..., 2]
    with np.errstate(invalid="ignore"):
        fg = ~inv & (zz < bg - 0.12) & (hgt > 0.05) & (zz < 8.0)
    fg = binary_opening(fg, iterations=1); lab, n = cc_label(fg)
    if n == 0:
        return None, 0
    k = 1 + np.argmax(np.bincount(lab.ravel())[1:]); comp = lab == k
    smax = maximum_filter(np.where(inv, -np.inf, zz), size=7); smin = minimum_filter(np.where(inv, np.inf, zz), size=7); sz = cc.axial_sigma(np.nan_to_num(zz, nan=3.0), DM); spread = smax - smin
    with np.errstate(invalid="ignore"):
        mixed = (spread > 6 * sz) & (zz - smin > 3 * sz) & (smax - zz > 3 * sz)
    edge = maximum_filter(inv, size=7) | (spread > 0.2); keep = comp & ~mixed; vv, uu = np.nonzero(keep)
    if len(vv) == 0:
        return None, int((comp & mixed).sum())
    pick = rng.choice(len(vv), min(N_PTS, len(vv)), replace=False)
    return np.c_[uu[pick], vv[pick], zz[vv[pick], uu[pick]], edge[vv[pick], uu[pick]]].astype(np.float32), int((comp & mixed).sum())


def main():
    import h5py
    OUT.mkdir(parents=True, exist_ok=True); SPEC.write_text(json.dumps(dict(dataset="itop", version="v0.1", fps=FPS, depth=DM, note="평가 전용 (학습 금지)"), indent=1, ensure_ascii=False))
    fl = h5py.File(RAW / "ITOP_side_test_labels.h5", "r"); fd = h5py.File(RAW / "ITOP_side_test_depth_map.h5", "r")
    ids = np.array([x.decode() for x in fl["id"][:]]); assert (np.array([x.decode() for x in fd["id"][:]]) == ids).all()
    valid = fl["is_valid"][:].astype(bool); rw = fl["real_world_coordinates"][:].astype(np.float64); depth = fd["data"]; rng = np.random.default_rng(0); summary = {}
    for pid in sorted(set(i.split("_")[0] for i in ids)):
        sel = np.where(np.char.startswith(ids, pid + "_"))[0]; Z = np.asarray(depth[sel[0]:sel[-1] + 1], np.float32); T = len(sel)
        Zn = np.where(Z > 0.3, Z, np.nan); bg = np.nanpercentile(Zn[::5], 90, axis=0)                 # 배경 = 화소별 90% 분위수 (사람이 지나간 화소도 대부분 배경)
        Pf = []
        for k in range(0, T, max(1, T // 60)):                                                         # 바닥 RANSAC 표본 (정답 없음)
            p = unproject(np.nan_to_num(Zn[k]))[160:].reshape(-1, 3); p = p[np.isfinite(p).all(1) & (p[:, 2] > 0.3)]; Pf.append(p[rng.choice(len(p), min(3000, len(p)), replace=False)])   # 영상 아래쪽 1/3 (바닥 후보)
        n, c, cnt = cc.fit_floor_plane(np.concatenate(Pf), (0.0, -1.0, 0.0), 0.95, seed=0, min_height=0.6); R, t = cc.room_frame(np.asarray(n), c)                       # 카메라 거의 수평: 기울기 18° 이내
        pts = np.zeros((T, N_PTS, 4), np.float32); val = np.zeros((T, N_PTS), bool); n_mix = np.zeros(T, np.int32)
        for k in range(T):
            p, nm = person_points(Z[k], bg, R, t, rng); n_mix[k] = nm
            if p is not None:
                pts[k, :len(p)] = p; val[k, :len(p)] = True
        y = cam_std(rw[sel])[:, J13] @ R.T + t; ok = valid[sel]; y[~ok] = np.nan; lm = np.repeat(ok[:, None], 13, 1)
        meta = dict(dataset="itop", subject=f"U{int(pid):02d}", env="itop_side", action="unknown", T=int(T), fps=FPS, label_family="itop15", label_source="itop real_world_coordinates (manual+refined, 15 joints)",
                    kp_family=None, has_rgb=False, spec=str(SPEC), version="v0.1", eval_only=True, floor=dict(n=np.asarray(n).tolist(), c=float(c), inliers=int(cnt)), camera_height_m=float(c), license="ITOP CC-BY-4.0 (Zenodo 3932973)")
        np.savez_compressed(OUT / f"U{int(pid):02d}.npz", meta=np.array(json.dumps(meta)), kp_px=np.zeros((T, 17, 2), np.float32), kp_valid=np.zeros((T, 17), bool), pts_uvz=pts[..., :3], pts_edge=pts[..., 3] > 0.5, pts_valid=val,
                            pts_n_raw=val.sum(1).astype(np.int32), pts_n_mixed=n_mix, comp_size=val.sum(1).astype(np.int32), labels=np.nan_to_num(y).astype(np.float32), label_mask=lm, label_quality=np.zeros((T, 13), np.uint8),
                            R_c2r=R, t_c2r=t, R_d2r=R, t_d2r=t)
        g = 0.5 * (y[ok, 7] + y[ok, 8]); summary[pid] = dict(tilt_deg=round(float(np.degrees(np.arccos(abs(np.asarray(n)[1])))), 2), pts_z=np.percentile((unproject(np.nan_to_num(np.where(Z[len(Z) // 2] > 0.3, Z[len(Z) // 2], 0)))[val[len(Z) // 2].any() and 0 or 0:]).reshape(-1, 3)[:1] @ R.T + t, [50])[0].round(2).tolist() if False else None, T=T, valid=int(ok.sum()), cam_h=round(float(c), 3), floor_inliers=int(cnt), pts_frac=float((val.sum(1) >= 20).mean()), ankle_z_med=float(np.nanmedian(np.minimum(y[ok, 11, 2], y[ok, 12, 2]))), pelvis_dist_med=float(np.median(np.linalg.norm(g[:, :2], axis=-1))))
        print(pid, summary[pid], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
