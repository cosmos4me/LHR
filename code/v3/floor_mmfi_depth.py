"""v3 바닥 (MM-Fi, 깊이): 녹화마다 깊이 배경(시간 중앙값 ¼ 해상도 bg_med) + 전체 깊이 3장 → 팩의 현재 방 좌표에서 RANSAC 바닥.
입력 (정답 안 읽음): dev/data/mmfi/cache_v1/{E}/{S}_{A}.npz 의 bg_med, bg_valid, full (원시 mm), 팩의 R_d2r, t_d2r, 사양 깊이 K.
- 점: 유효 비율 > 0.5 인 bg_med 화소 + 전체 깊이 3장(2화소 간격), 깊이 0.5–5.0 m.
- 허용 오차: max(2 cm, 1%·깊이) (D435 축 잡음 0.0042·Z² 의 바닥 수직 성분 ≈ 0.004·Z, 고정 패턴 여유).
- 위쪽 사전 = 옛 방 z (환경별 깊이 바닥으로 이미 거의 수평), 20° 안, 카메라보다 0.3 m 아래, 가장 낮은 큰 면 (아래 점 ≤ 3%).
  세 점 표본은 옛 바닥 ±0.4 m 점에서만 (3000회), 내점 평가는 모든 점.
출력: dev/packs/v3/floor/mmfi_depth.json (녹화별 평면, 내점 수, 부트스트랩 σ).
실행: python -I floor_mmfi_depth.py [--limit N]"""
import sys, os, glob, json
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
import floor_common as fc

U = "/home/interns/unisense"
OUT = f"{U}/dev/packs/v3/floor/mmfi_depth.json"
ZMIN, ZMAX = 0.5, 5.0


def depth_K(spec_path):
    return json.load(open(spec_path))["depth"]["K"]


def points(C, K, R, t):
    fx, fy, cx, cy = K
    bg = C["bg_med"].astype(float); bv = C["bg_valid"].astype(float)
    ii, jj = np.nonzero(np.isfinite(bg) & (bv > 0.5) & (bg > ZMIN) & (bg < ZMAX))
    z = bg[ii, jj]; u = 4.0 * jj; v = 4.0 * ii
    P = [np.c_[(u - cx) / fx * z, (v - cy) / fy * z, z]]
    full = C["full"]
    for f in full:
        d = f[::2, ::2].astype(float) * 1e-3; d[(f[::2, ::2] == 0) | (f[::2, ::2] == 65535)] = np.nan
        ii, jj = np.nonzero(np.isfinite(d) & (d > ZMIN) & (d < ZMAX))
        z = d[ii, jj]; u = 2.0 * jj; v = 2.0 * ii
        P.append(np.c_[(u - cx) / fx * z, (v - cy) / fy * z, z])
    Pd = np.concatenate(P)
    return Pd @ R.T + t, Pd[:, 2]


def one(pk):
    env = pk.split("/")[-2]; nm = os.path.basename(pk)[:-4]
    P = fc.safe_npz(pk); m = P.meta(); R = P["R_d2r"].astype(float); t = P["t_d2r"].astype(float)
    C = fc.safe_npz(f"{U}/dev/data/mmfi/cache_v1/{env}/{nm}.npz")
    K = depth_K(m["spec"] if env != "E04" else f"{U}/code/specs/mmfi_v02_specA_e04.json")
    X, Z = points(C, K, R, t)
    rng = np.random.default_rng(0)
    if len(X) > 60000:
        k = rng.choice(len(X), 60000, replace=False); X, Z = X[k], Z[k]
    res = fc.ransac_plane(X, sensor=t, tol_fn=lambda Q: np.maximum(0.02, 0.01 * Z), max_tilt_deg=20.0, seed=0, iters=3000, sample_mask=np.abs(X[:, 2]) < 0.4)
    if res is None:
        return pk, dict(ok=False, why="no plane")
    Q = X[res["inl"]]
    bs = fc.block_bootstrap(Q, B=60, cell=0.5, seed=1)
    # 두 번째로 큰 수평면 (탁자 등) — 바닥보다 아래에 큰 면이 있으면 표시
    rest = ~res["inl"]; alt = fc.ransac_plane(X[rest], sensor=t, tol_fn=None, tol=0.03, max_tilt_deg=10.0, seed=2, refine=1) if rest.sum() > 500 else None
    alt_info = None
    if alt is not None:
        a, h = fc.deviation(alt["n"], alt["d"], res["n"], res["d"])
        alt_info = dict(n_inl=alt["n_inl"], angle_deg=a, dh_m=h)
    out = dict(ok=True, env=env, subject=m.get("subject"), action=m.get("action"), n=res["n"], d=res["d"], n_points=res["n_inl"],
               n_cand=int(len(X)), tilt_deg=res["tilt_deg"], cam_height_new=float(res["n"] @ t + res["d"]),
               xy_extent=(np.percentile(Q[:, :2], 95, 0) - np.percentile(Q[:, :2], 5, 0)), resid_mad_m=float(1.4826 * np.median(np.abs(Q @ res["n"] + res["d"]))),
               second_plane=alt_info, **bs)
    return pk, out


def main():
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    pks = sorted(glob.glob(f"{U}/dev/packs/mmfi_v02_specA/E0[1-4]/S*_A*.npz"))
    if lim:
        pks = pks[::max(1, len(pks) // lim)][:lim]
    res = {}
    with Pool(24) as pool:
        for i, (pk, r) in enumerate(pool.imap_unordered(one, pks, chunksize=2), 1):
            res[pk] = r
            if i % 100 == 0:
                print(i, "/", len(pks), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = OUT if not lim else OUT.replace(".json", f"_test{lim}.json")
    open(out, "w").write(fc.jdump(res))
    T = np.array([r["tilt_deg"] for r in res.values() if r.get("ok")]); D = np.array([r["d"] for r in res.values() if r.get("ok")])
    print("done", len(res), "ok", len(T), "tilt med/p95", np.round(np.median(T), 3), np.round(np.percentile(T, 95), 3),
          "d med/p5/p95", np.round(np.median(D), 4), np.round(np.percentile(D, 5), 4), np.round(np.percentile(D, 95), 4), flush=True)


if __name__ == "__main__":
    main()
