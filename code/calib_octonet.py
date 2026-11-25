"""OctoNet 노드 1 깊이 카메라(D455, 8비트 0–10 m) 보정: 내부 파라미터(f, cx, cy)와 모캡 세계 좌표(z 위, m)에서의 자세.
학습 사용자 정답만 사용 (splits/octonet.json). 환경별로 따로 맞춘다.
1) 깊이 정적 배경(여러 녹화의 시간 중앙값)에서 바닥 평면 → 카메라 높이·기울기 (정답 없음).
2) 사람 평균 위치를 중심으로 카메라 방위 θ·거리 ρ 격자 탐색 (카메라는 사람을 향함).
3) 전체 파라미터 Powell: 관절 투영 화소(7×7 최솟값) 깊이 − 관절 깊이 ≈ −r (몸 표면 오프셋 r 자유).
검증: 검증 사용자 적중률, 깊이 바닥 평면을 세계로 옮겼을 때 z ≈ 0인지(정답 없는 점검).
사용: python calib_octonet.py --env 1"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_common as cc
import octonet_io as oi

CACHE = Path("/home/interns/unisense/dev/data/octonet/derived/mocap13")
SPL = json.loads((Path(__file__).resolve().parent / "splits" / "octonet.json").read_text())
THIN = [0, 3, 4, 5, 6, 9, 10, 11, 12, 1, 2]          # 머리, 팔꿈치, 손목, 무릎, 발목, 어깨
W, H = 640, 480
rng = np.random.default_rng(0)


def sample(env, users, per_rec=8):
    D, G, Dfull = [], [], []
    for r in oi.recordings():
        if r["env"] != env or r["user"] not in users:
            continue
        M = np.load(CACHE / f"{r['stamp']}.npz"); t, J = M["t"], M["joints13"].astype(float)
        Z = oi.DepthZip(r["depth"])
        ok = np.where((Z.t > t[0] + 0.5) & (Z.t < t[-1] - 0.5))[0]
        for i in ok[np.linspace(0, len(ok) - 1, per_rec).astype(int)]:
            z = Z.depth(i); j, _ = oi.nearest(t, Z.t[i:i + 1])
            D.append(minimum_filter(np.nan_to_num(z, nan=99.0), size=7)); G.append(J[j[0]][THIN]); Dfull.append(z)
    return np.stack(D), np.stack(G), Dfull


def static_floor(env, f0=385.0, n=20):
    """정답 없이: 여러 녹화 깊이의 시간·녹화 중앙값(사람이 지워진 배경) → 바닥 평면 (카메라 좌표)."""
    Z = []
    recs = [r for r in oi.recordings() if r["env"] == env]
    for r in [recs[i] for i in rng.choice(len(recs), min(n, len(recs)), replace=False)]:
        dz = oi.DepthZip(r["depth"])
        idx = np.linspace(0, len(dz) - 1, 12).astype(int)
        with np.errstate(all="ignore"):
            Z.append(np.nanmedian(np.stack([dz.depth(i) for i in idx]), 0))
    with np.errstate(all="ignore"):
        z = np.nanmedian(np.stack(Z), 0)
    vv, uu = np.mgrid[0:H:2, 0:W:2]; zz = z[::2, ::2]; m = np.isfinite(zz) & (zz < 9.5) & (zz > 0.3)
    P = np.c_[(uu[m] - W / 2) / f0 * zz[m], (vv[m] - H / 2) / f0 * zz[m], zz[m]]
    n_, c, k = cc.fit_floor_plane(P, (0, -1, 0), 0.7, iters=3000)
    return n_, c, k, z


def look_at(C, target, up=np.array([0, 0, 1.0])):
    f = target - C; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])                                    # 세계 → 카메라 (x 오른쪽, y 아래, z 앞)


def unpack(q, R0):
    f, cx, cy = q[0], q[1], q[2]
    R = Rotation.from_rotvec(q[3:6]).as_matrix() @ R0
    return dict(K=[f, f, cx, cy], R=R, C=q[6:9], r=q[9])


def dz_of(P, D, G):
    fx, fy, cx, cy = P["K"]
    Xc = (G - P["C"]) @ P["R"].T
    with np.errstate(all="ignore"):
        u = np.rint(fx * Xc[..., 0] / Xc[..., 2] + cx); v = np.rint(fy * Xc[..., 1] / Xc[..., 2] + cy)
    ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Xc[..., 2] > 0.3)
    n = np.broadcast_to(np.arange(len(D))[:, None], u.shape)
    dz = np.full(u.shape, np.nan)
    dz[ok] = D[n[ok], v[ok].astype(int), u[ok].astype(int)] - Xc[..., 2][ok]
    return dz


def loss(q, R0, D, G):
    P = unpack(q, R0); dz = dz_of(P, D, G)
    return -np.nanmean(np.where(np.isfinite(dz), np.exp(-((dz + P["r"]) / 0.10) ** 2), 0.0))


def hit(P, D, G):
    dz = dz_of(P, D, G); f = np.isfinite(dz)
    return float(np.mean(np.abs(dz[f] + P["r"]) < 0.15)) if f.any() else 0.0


def main():
    env = int(sys.argv[sys.argv.index("--env") + 1])
    Dtr, Gtr, _ = sample(env, set(SPL["train"])); Dva, Gva, _ = sample(env, set(SPL["val"]), per_rec=6)
    n_cam, c_cam, k_floor, _ = static_floor(env)
    subj = np.nanmean(Gtr.reshape(-1, 3), 0)
    best = None
    for th in np.radians(np.arange(0, 360, 10)):
        for rho in (2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
            C = np.r_[subj[:2] + rho * np.array([np.cos(th), np.sin(th)]), c_cam]
            R0 = look_at(C, np.r_[subj[:2], c_cam])
            q = np.r_[385.0, W / 2, H / 2, 0, 0, 0, C, 0.05]
            l = loss(q, R0, Dtr, Gtr)
            if best is None or l < best[0]:
                best = (l, th, rho, q, R0)
    l0, th, rho, q, R0 = best
    starts = [(q, R0)]
    if env == 1:                                                      # 모캡에 기록된 Node1 강체 위치에서도 시작 (카메라는 그 근처)
        nd = [np.load(CACHE / f"{r['stamp']}.npz") for r in oi.recordings() if r["env"] == 1][:5]
        node1 = np.nanmean(np.concatenate([x["rigid"][:, list(x["rigid_names"]).index("Node1"), :3] for x in nd]), 0)
        R1 = look_at(node1, np.r_[subj[:2], node1[2]])
        starts.append((np.r_[385.0, W / 2, H / 2, 0, 0, 0, node1, 0.05], R1))
    res, R0 = None, None
    for qs, Rs in starts:
      for f0 in (370.0, 385.0, 400.0, 420.0):
        q0 = qs.copy(); q0[0] = f0
        r = minimize(loss, q0, args=(Rs, Dtr, Gtr), method="Powell", options=dict(maxiter=20000, xtol=1e-4, ftol=1e-7))
        if res is None or r.fun < res.fun:
            res, R0 = r, Rs
    res_unused = None
    P = unpack(res.x, R0)
    # 정답 없는 점검: 깊이 바닥 평면을 세계로 옮기면 z = 0 평면이어야 한다 (카메라 좌표 평면은 f0로 근사 추정)
    n_w = P["R"].T @ n_cam; cam_h_world = P["C"][2]
    out = dict(env=env, K=[round(float(x), 3) for x in P["K"]], R_w2c=P["R"].round(6).tolist(), C_world=np.round(P["C"], 4).tolist(),
               surface_offset_r_m=round(float(P["r"]), 4), hit_train=round(hit(P, Dtr, Gtr), 4), hit_val=round(hit(P, Dva, Gva), 4),
               grid_best=dict(theta_deg=round(float(np.degrees(th)), 1), rho_m=rho, score=round(-l0, 4)),
               floor_check=dict(camera_height_from_depth_floor_m=round(float(c_cam), 3), camera_height_fit_m=round(float(cam_h_world), 3),
                                floor_normal_world=np.round(n_w, 4).tolist(), floor_inliers=int(k_floor)),
               n_train_frames=int(len(Dtr)), n_val_frames=int(len(Dva)))
    if env == 1:
        nodes = [np.load(CACHE / f"{r['stamp']}.npz") for r in oi.recordings() if r["env"] == 1][:5]
        node1 = np.nanmean(np.concatenate([nd["rigid"][:, list(nd["rigid_names"]).index("Node1"), :3] for nd in nodes]), 0)
        out["node1_rigid_body_world"] = np.round(node1, 4).tolist()
        out["camera_minus_node1_m"] = np.round(P["C"] - node1, 4).tolist()
    Path(f"/home/interns/unisense/notes/calib_octonet_env{env}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


main()
