"""v3 바닥 교차 확인 (MM-Fi, LiDAR vs 깊이). 정답 안 읽음.
A. 원시 LiDAR(Ouster OS1-32, xyz, LiDAR 좌표 z 위) 녹화마다 30프레임 → RANSAC 바닥 (위쪽 사전 +z, 20° 안) → LiDAR 높이 h_L, 법선 기울기.
   주의: 공개 LiDAR 점군은 상자로 잘려 z ≥ −1.2 m 만 남아 있다 (바닥이 약 −1.2) → 바닥 점의 아래 절반이 잘릴 수 있음 → clip_frac 로 보고.
B. 같은 녹화의 사람 높이 분포 비교 (바닥 오프셋 일치 검사, 외부 파라미터 불필요):
   - LiDAR: 자신의 바닥으로 수평 맞춘 높이, 사람 점 = 바닥 위 0.05–2.3 m 이고 (0.3–2.0 m 점들의 수평 중앙값)에서 0.8 m 안.
   - 깊이: 팩 사람 점(pts_uvz, RGB 키포인트 영역) → 옛 방 → (옛 z) 와 (v3 깊이 바닥 보정 z).
   - 프레임별 p97 (머리 꼭대기 쪽) 과 p50, 녹화별 중앙값. Δ = 깊이 − LiDAR. LiDAR 링 간격(3 m 에서 약 7 cm) 때문에 꼭대기는 LiDAR 가 낮게 나오는 치우침이 있다.
C. rig 일관성: h_cam(깊이 새 바닥) − h_L 이 같은 환경·피험자 안에서 일정해야 한다 (강체 플랫폼).
입력: dev/data/mmfi/extracted_lidar/{E}/{S}/{A}/lidar/frame*.bin (신뢰하지 않는 데이터: np.fromfile 로만 읽음), dev/packs/v3/floor/mmfi_depth.json.
출력: dev/packs/v3/floor/mmfi_lidar.json. 실행: python -I floor_mmfi_lidar.py"""
import sys, os, glob, json, re
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
import floor_common as fc

U = "/home/interns/unisense"; LID = f"{U}/dev/data/mmfi/extracted_lidar"
OUT = f"{U}/dev/packs/v3/floor/mmfi_lidar.json"
DEPTH = json.load(open(f"{U}/dev/packs/v3/floor/mmfi_depth.json"))
NAME = re.compile(r"^frame(\d+)\.bin$")


def lidar_frames(env, s, a, n=30):
    d = os.path.join(LID, env, s, a, "lidar")
    if not os.path.isdir(d):
        return []
    fs = sorted([f for f in os.listdir(d) if NAME.match(f)], key=lambda f: int(NAME.match(f).group(1)))
    out = []
    for f in [fs[i] for i in np.linspace(0, len(fs) - 1, min(n, len(fs))).astype(int)]:
        p = os.path.realpath(os.path.join(d, f))
        if not p.startswith(os.path.realpath(LID) + os.sep):
            continue
        a_ = np.fromfile(p, dtype=np.float64)
        if a_.size % 3:
            continue
        out.append(a_.reshape(-1, 3))
    return out


def person_heights(f, n, d):
    h = f @ n + d; Q = f - np.outer(f @ n, n)                                  # 수평 성분
    m = (h > 0.3) & (h < 2.0)
    if m.sum() < 15:
        return None
    med = np.median(Q[m], 0); k = (h > 0.05) & (h < 2.3) & (np.linalg.norm(Q - med, axis=1) < 0.8)
    return h[k] if k.sum() >= 15 else None


def one(pk):
    env = pk.split("/")[-2]; nm = os.path.basename(pk)[:-4]; s, a = nm.split("_")
    fr = lidar_frames(env, s, a)
    if not fr:
        return pk, dict(ok=False, why="no lidar")
    X = np.concatenate(fr)
    res = fc.ransac_plane(X, tol=0.02, sensor=np.zeros(3), max_tilt_deg=20.0, seed=0)
    if res is None:
        return pk, dict(ok=False, why="no plane")
    n, d = res["n"], res["d"]; Q = X[res["inl"]]
    clip = float(np.mean(Q[:, 2] < -1.195))
    bs = fc.block_bootstrap(Q, B=40, cell=0.5, seed=1)
    # B. 사람 높이
    allfr = lidar_frames(env, s, a, n=100)
    lt, lm = [], []
    for f in allfr:
        h = person_heights(f, n, d)
        if h is not None:
            lt.append(np.percentile(h, 97)); lm.append(np.percentile(h, 50))
    P = fc.safe_npz(pk); m = P.meta(); R, t = P["R_d2r"].astype(float), P["t_d2r"].astype(float)
    K = json.load(open(m["spec"] if env != "E04" else f"{U}/code/specs/mmfi_v02_specA_e04.json"))["depth"]["K"]
    uvz, val = P["pts_uvz"], P["pts_valid"]; fx, fy, cx, cy = K
    D = DEPTH[pk]; nD, dD = np.array(D["n"]), D["d"]
    dt_old, dm_old, dt_new, dm_new = [], [], [], []
    for k in range(0, len(uvz), 3):
        v = val[k]
        if v.sum() < 100:
            continue
        u_, w_, z = uvz[k, v].astype(float).T
        Xr = np.c_[(u_ - cx) / fx * z, (w_ - cy) / fy * z, z] @ R.T + t
        zo = Xr[:, 2]; zn = Xr @ nD + dD
        dt_old.append(np.percentile(zo, 97)); dm_old.append(np.percentile(zo, 50)); dt_new.append(np.percentile(zn, 97)); dm_new.append(np.percentile(zn, 50))
    med = lambda x: float(np.median(x)) if len(x) else float("nan")
    out = dict(ok=True, env=env, subject=s, action=a, lidar_n=n, lidar_d=d, lidar_height_m=float(d), lidar_tilt_deg=res["tilt_deg"], n_points=res["n_inl"],
               clip_frac=clip, resid_mad_m=float(1.4826 * np.median(np.abs(Q @ n + d))), **{"lidar_" + k: v for k, v in bs.items()},
               lidar_top=med(lt), lidar_mid=med(lm), n_lidar_frames=len(lt),
               depth_top_old=med(dt_old), depth_mid_old=med(dm_old), depth_top_new=med(dt_new), depth_mid_new=med(dm_new), n_depth_frames=len(dt_new),
               cam_height_new=D["cam_height_new"])
    return pk, out


def main():
    pks = sorted(glob.glob(f"{U}/dev/packs/mmfi_v02_specA/E0[1-4]/S*_A*.npz"))
    res = {}
    with Pool(24) as pool:
        for i, (pk, r) in enumerate(pool.imap_unordered(one, pks, chunksize=2), 1):
            res[pk] = r
            if i % 100 == 0:
                print(i, "/", len(pks), flush=True)
    open(OUT, "w").write(fc.jdump(res))
    print("done", len(res), "ok", sum(r.get("ok", False) for r in res.values()), flush=True)


if __name__ == "__main__":
    main()
