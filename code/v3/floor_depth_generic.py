"""v3 바닥 (OctoNet, ITOP; 깊이 RANSAC). 정답 안 읽음.
입력: floor_extract_raw.py 의 배경 (dev/packs/v3/floor/raw_bg/*.npz) + 팩의 R_d2r, t_d2r (+ OctoNet 사양의 K, 깊이 배율 a).
- OctoNet: 녹화별 배경(40프레임 90 백분위, 8비트 → 양자화 약 3.5 cm) × a → 역투영(2화소 간격), 깊이 0.5–6 m.
  허용 오차 max(2.5 cm, 1%·깊이) (양자화 ± q/2 의 바닥 수직 성분 ≤ 1 cm + D455 잡음).
- ITOP: 사람별 배경(90 백분위) + 전체 6장, 깊이 0.5–6 m, 허용 오차 max(2 cm, 1%·깊이).
- 위쪽 사전 = 옛 방 z (20° 안), 카메라보다 0.3 m 아래, 가장 낮은 큰 면 (아래 점 ≤ 3%). 세 점 표본은 옛 바닥 ±0.4 m 점에서만 (3000회).
출력: dev/packs/v3/floor/{octonet,itop}_depth.json. 실행: python -I floor_depth_generic.py octonet|itop"""
import sys, os, glob, json
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
import floor_common as fc

U = "/home/interns/unisense"; BG = f"{U}/dev/packs/v3/floor/raw_bg"
OSPEC = json.load(open(f"{U}/code/specs/octonet_v07.json"))
ITOP_K = [1 / 0.0035, 1 / 0.0035, 160.0, 120.0]
BGKEY = "med"              # 시간 중앙값 배경 (90 백분위는 잡음 큰 바닥에서 먼 쪽 치우침 → 바닥이 낮게 보임, OctoNet env 2 에서 확인)


def backproject(D, K, step=2, zmin=0.5, zmax=6.0, valid=None):
    fx, fy, cx, cy = K
    d = D[::step, ::step].astype(float)
    m = np.isfinite(d) & (d > zmin) & (d < zmax)
    if valid is not None:
        m &= valid[::step, ::step].astype(float) > 0.5
    ii, jj = np.nonzero(m); z = d[ii, jj]; u = step * jj; v = step * ii
    return np.c_[(u - cx) / fx * z, (v - cy) / fy * z, z]


def fit(X, Z, t, tol0, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > 60000:
        k = rng.choice(len(X), 60000, replace=False); X, Z = X[k], Z[k]
    res = fc.ransac_plane(X, sensor=t, tol_fn=lambda Q: np.maximum(tol0, 0.01 * Z), max_tilt_deg=20.0, seed=seed, iters=3000, sample_mask=np.abs(X[:, 2]) < 0.4)
    if res is None:
        return None
    Q = X[res["inl"]]; bs = fc.block_bootstrap(Q, B=60, cell=0.5, seed=1)
    rest = ~res["inl"]; alt = fc.ransac_plane(X[rest], sensor=t, tol=0.03, max_tilt_deg=10.0, seed=2, refine=1) if rest.sum() > 500 else None
    alt_info = None
    if alt is not None:
        a, h = fc.deviation(alt["n"], alt["d"], res["n"], res["d"]); alt_info = dict(n_inl=alt["n_inl"], angle_deg=a, dh_m=h)
    return dict(ok=True, n=res["n"], d=res["d"], n_points=res["n_inl"], n_cand=int(len(X)), tilt_deg=res["tilt_deg"],
                cam_height_new=float(res["n"] @ t + res["d"]), xy_extent=np.percentile(Q[:, :2], 95, 0) - np.percentile(Q[:, :2], 5, 0),
                resid_mad_m=float(1.4826 * np.median(np.abs(Q @ res["n"] + res["d"]))), second_plane=alt_info, **bs)


def octo_one(pk):
    st = os.path.basename(pk)[:-4]; f = f"{BG}/octonet_{st}.npz"
    if not os.path.exists(f):
        return pk, dict(ok=False, why="no bg")
    P = fc.safe_npz(pk); m = P.meta(); R, t = P["R_d2r"].astype(float), P["t_d2r"].astype(float)
    S = OSPEC["depth"][f"{m['env']}_{m['day']}"]; a = float(S["depth_scale_a"])
    B = np.load(f, allow_pickle=False)
    Xc = backproject(B[BGKEY].astype(np.float32) * a, S["K"], valid=B["valid"])
    r = fit(Xc @ R.T + t, Xc[:, 2], t, 0.025)
    if r is None:
        return pk, dict(ok=False, why="no plane")
    r.update(env=m["env"], day=m["day"], subject=m.get("subject"), action=m.get("action"), n_bg_frames=int(B["n_frames"]))
    return pk, r


def itop_one(pk):
    u = os.path.basename(pk)[:-4]; B = np.load(f"{BG}/itop_{u}.npz", allow_pickle=False)
    P = fc.safe_npz(pk); R, t = P["R_d2r"].astype(float), P["t_d2r"].astype(float)
    X = [backproject(B[BGKEY].astype(np.float32), ITOP_K, valid=B["valid"])] + [backproject(f.astype(np.float32), ITOP_K, step=3) for f in B["full"]]
    Xc = np.concatenate(X)
    r = fit(Xc @ R.T + t, Xc[:, 2], t, 0.02)
    if r is None:
        return pk, dict(ok=False, why="no plane")
    r.update(subject=u)
    return pk, r


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "octonet":
        pks = sorted(glob.glob(f"{U}/dev/packs/octonet_v07/*.npz")); fn = octo_one
    else:
        pks = sorted(glob.glob(f"{U}/dev/packs/itop_v01/U*.npz")); fn = itop_one
    res = {}
    with Pool(24) as pool:
        for pk, r in pool.imap_unordered(fn, pks):
            res[pk] = r
    open(f"{U}/dev/packs/v3/floor/{what}_depth.json", "w").write(fc.jdump(res))
    ok = [r for r in res.values() if r.get("ok")]
    print(what, "done", len(res), "ok", len(ok), "tilt med/p95", np.round(np.median([r["tilt_deg"] for r in ok]), 3), np.round(np.percentile([r["tilt_deg"] for r in ok], 95), 3),
          "d med/min/max", np.round(np.median([r["d"] for r in ok]), 4), np.round(min(r["d"] for r in ok), 4), np.round(max(r["d"] for r in ok), 4), flush=True)
