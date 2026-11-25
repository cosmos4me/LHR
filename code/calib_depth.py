"""MM-Fi 깊이 카메라 보정 (일반화판): GT(RGB 카메라 좌표) → 깊이 카메라 좌표. 학습 피험자만 사용.
변형 비교:
  C0    : C + t_y = 0 (색 카메라는 x축으로만 떨어져 있음)
  free  : fx, fy 따로 자유 (v0 방식)
  iso   : fx = fy 자유
  C     : fx = fy = C / b (정답 없이 양자화 격자로 구한 C = 19.175 px·m, D435 기선 b = 50 mm)
공통: 회전 3, t_x, t_y 자유. t_z = 0 고정 (D435 영상 센서가 같은 평면에 있음). 몸 표면 깊이 오프셋 r은 자유 방해 변수
(v0에서는 r = 8 cm를 가정해 t_z가 그 값을 흡수했다).
목적: 가는 관절 투영 화소(7×7 최솟값)의 깊이 − 관절 깊이 = −r 근처가 되도록 하는 부드러운 점수.
사용: python calib_depth.py --train E01:S01-S06 [--train E02:S11-S16 ...] --val E01:S07 --val E02:S17 --val E03:S27 --out NAME"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = Path("/home/interns/unisense/dev/data/mmfi/extracted")
J = [10, 12, 13, 15, 16, 2, 3, 5, 6, 0, 8]
C_STEREO, BASE = 19.175, 0.050
GT_SCALE = 1.0                 # 정답 X, Y 배율 보정 (notes/gt_scale_mmfi.json); --gt-scale로 지정
rng = np.random.default_rng(3)


def parse(spec):
    env, rg = spec.split(":"); a, b = rg.split("-") if "-" in rg else (rg, rg)
    return env, [f"S{i:02d}" for i in range(int(a[1:]), int(b[1:]) + 1)]


def load(groups, per_seq=3):
    D, G = [], []
    for env, subs in groups:
        for s in sorted(p.parent for p in (ROOT / env).glob("S*/A*/ground_truth.npy")):
            if s.parent.name not in subs:
                continue
            gt = np.load(s / "ground_truth.npy").astype(float); gt[..., :2] /= GT_SCALE
            df = sorted((s / "depth").glob("frame*"), key=lambda p: int(p.stem[5:]))
            for t in rng.choice(min(len(df), len(gt)), per_seq, replace=False):
                d = np.array(Image.open(df[t])).astype(np.float32) / 1000.0
                d[(d <= 0) | (d > 60)] = 99.0
                D.append(minimum_filter(d, size=7)); G.append(gt[t, J])
    return np.stack(D), np.stack(G)


def unpack(q, mode):
    if mode == "free":
        fx, fy, cx, cy = q[:4]; rest = q[4:]
    elif mode == "iso":
        fx = fy = q[0]; cx, cy = q[1:3]; rest = q[3:]
    elif mode == "C":
        fx = fy = C_STEREO / BASE; cx, cy = q[:2]; rest = q[2:]
    else:                                                    # C0: C에 더해 t_y = 0 (D435 색 카메라는 x축으로만 떨어져 있음)
        fx = fy = C_STEREO / BASE; cx, cy = q[:2]; rv, tx, r = q[2:5], q[5], q[6]
        return dict(K=[fx, fy, cx, cy], R=Rotation.from_rotvec(rv).as_matrix(), t=np.r_[tx, 0.0, 0.0], r=r, rotvec=rv)
    rv, txy, r = rest[:3], rest[3:5], rest[5]
    return dict(K=[fx, fy, cx, cy], R=Rotation.from_rotvec(rv).as_matrix(), t=np.r_[txy, 0.0], r=r, rotvec=rv)


def dz_of(P, D, G):
    fx, fy, cx, cy = P["K"]
    Xd = G @ P["R"].T + P["t"]
    u = np.rint(fx * Xd[..., 0] / Xd[..., 2] + cx).astype(int); v = np.rint(fy * Xd[..., 1] / Xd[..., 2] + cy).astype(int)
    ok = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    n = np.broadcast_to(np.arange(len(D))[:, None], u.shape)
    dz = np.full(u.shape, np.nan); dz[ok] = D[n[ok], v[ok], u[ok]] - Xd[..., 2][ok]
    return dz


def loss(q, mode, D, G):
    P = unpack(q, mode); dz = dz_of(P, D, G)
    s = np.exp(-((dz + P["r"]) / 0.10) ** 2)
    return -np.nanmean(np.where(np.isfinite(dz), s, 0.0))


def hit(P, D, G):
    dz = dz_of(P, D, G); f = np.isfinite(dz)
    return float(np.mean(np.abs(dz[f] + P["r"]) < 0.15))


def fit(mode, D, G):
    best = None
    f0s = {"free": [383.5, 424.0], "iso": [383.5, 424.0], "C": [None], "C0": [None]}[mode]
    for f0 in f0s:
        for tx0 in (0.0, 0.015, -0.015):
            for cy0 in (240.0, 254.0):
                base = [320.0, cy0, 0, 0, 0, tx0, 0.0, 0.08]
                q0 = {"free": [f0, f0] + base, "iso": [f0] + base, "C": base,
                      "C0": [320.0, cy0, 0, 0, 0, tx0, 0.08]}[mode]
                r = minimize(loss, np.array(q0, float), args=(mode, D, G), method="Powell",
                             options=dict(maxiter=8000, xtol=1e-3, ftol=1e-6))
                if best is None or r.fun < best.fun:
                    best = r
    return unpack(best.x, mode)


def main():
    global GT_SCALE
    args = sys.argv[1:]
    if "--gt-scale" in args:
        GT_SCALE = float(args[args.index("--gt-scale") + 1])
    tr = [parse(args[i + 1]) for i, a in enumerate(args) if a == "--train"]
    va = [parse(args[i + 1]) for i, a in enumerate(args) if a == "--val"]
    name = args[args.index("--out") + 1]
    Dtr, Gtr = load(tr); Dva, Gva = load(va)
    res = {"train": [f"{e}:{s[0]}-{s[-1]}" for e, s in tr], "val": [f"{e}:{s[0]}-{s[-1]}" for e, s in va],
           "gt_scale": GT_SCALE, "n_train_frames": int(len(Dtr)), "n_val_frames": int(len(Dva)), "variants": {}}
    modes = args[args.index("--modes") + 1].split(",") if "--modes" in args else ["free", "iso", "C", "C0"]
    prev = Path(f"/home/interns/unisense/notes/calib_depth_{name}.json")
    if prev.exists():
        res["variants"] = json.loads(prev.read_text())["variants"]
    for mode in modes:
        P = fit(mode, Dtr, Gtr)
        res["variants"][mode] = dict(K=[round(float(x), 3) for x in P["K"]], rotvec_deg=np.degrees(P["rotvec"]).round(4).tolist(),
                                     t_m=np.round(P["t"], 4).tolist(), surface_offset_r_m=round(float(P["r"]), 4),
                                     hit_train=round(hit(P, Dtr, Gtr), 4), hit_val=round(hit(P, Dva, Gva), 4),
                                     dz_val_median_m=round(float(np.nanmedian(dz_of(P, Dva, Gva))), 4))
        print(mode, res["variants"][mode], flush=True)
    Path(f"/home/interns/unisense/notes/calib_depth_{name}.json").write_text(json.dumps(res, indent=1))


main()
