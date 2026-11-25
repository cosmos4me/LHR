"""MM-Fi 정답 배율 s 추정. 정답 없이 RGB 초점거리 fy 추정: 깊이(물리, f=383.5)로 잰 키와 같은 프레임의 RGB 키포인트 세로 길이(눈→발목)를 비교.
해부학 사전값: 정수리→눈 0.115 m, 발목→발바닥 0.075 m. s = fy_gtfree / fy_fit(사양별, 정답 기반). 키는 바닥 기준 높이라 깊이↔RGB 외부 파라미터와 무관.
출력: notes/gt_scale_mmfi.json. 사용: python est_gt_scale.py"""
import sys, json
sys.path.insert(0, "/home/interns/unisense/code")
from pathlib import Path
import numpy as np
import conv_common as cc
PK = Path("/home/interns/unisense/dev/packs/mmfi_v01_specA")
spec = json.loads(Path("/home/interns/unisense/code/specs/mmfi_v01_specA.json").read_text())
TRAIN = set(json.loads(Path("/home/interns/unisense/code/splits/mmfi.json").read_text())["E1_cross_subject"]["train"])
HT, AS = 0.115, 0.075
est, est_gtcheck = [], []
for f in sorted(PK.glob("E*/*.npz")):
    if f.name.split("_")[0] not in TRAIN: continue
    P = np.load(f); kp = P["kp_px"].astype(float); v = P["pts_valid"]
    pts, _ = cc.depth_point_tokens(P["pts_uvz"].astype(float), spec["depth"]["K"], P["R_d2r"], P["t_d2r"], spec["depth"])
    T = len(kp); stat = np.array([np.percentile(pts[t][v[t]][:, 2], 99) for t in range(T)])
    zc = np.array([np.median(P["pts_uvz"][t][v[t]][:, 2]) for t in range(T)]) + 0.05     # 관절 평면 ≈ 표면 + 5 cm
    dv = 0.5 * (kp[:, 15, 1] + kp[:, 16, 1]) - 0.5 * (kp[:, 1, 1] + kp[:, 2, 1])
    up = stat > np.percentile(stat, 80) - 0.02                                             # 서 있는 프레임
    if up.sum() < 20: continue
    fy = dv[up] * zc[up] / (stat[up] - HT - AS)
    est.append(np.median(fy))
    lab = P["labels_cam"].astype(float)
    est_gtcheck.append(np.median((lab[up, 11:13, 1].mean(1) - lab[up, 0, 1])))            # 정답 머리→발목 세로 [m]
e = np.array(est)
fit = {k: json.loads(Path(f"/home/interns/unisense/code/specs/mmfi_v01_spec{k}.json").read_text())["rgb"]["K"][1] for k in "AB"}
res = {"fy_gtfree": round(float(np.median(e)), 2), "fy_gtfree_iqr": np.percentile(e, [25, 75]).round(1).tolist(), "n_seq": len(e),
       "fy_fit": fit, "s": {k: round(float(np.median(e) / v), 4) for k, v in fit.items()},
       "assumptions": {"head_top_to_eye_m": HT, "ankle_to_sole_m": AS, "joint_plane_behind_surface_m": 0.05, "isotropic_xy": True}}
Path("/home/interns/unisense/notes/gt_scale_mmfi.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res))
print(json.dumps({"n_seq": len(e), "fy_rgb_gtfree_median": round(float(np.median(e)), 1), "iqr": np.percentile(e, [25, 75]).round(1).tolist(),
                  "fy_gt_fit": round(spec["rgb"]["K"][1], 1), "ratio_gtfree_over_fit (=GT vertical scale s_y)": round(float(np.median(e) / spec["rgb"]["K"][1]), 3),
                  "fx_gt_fit": round(spec["rgb"]["K"][0], 1)}, indent=1))
