"""v3 OctoNet 레이더: 정답 없는 정리(clean) + 정합(register) 핵심 함수. 정답(관절·라벨)은 읽지 않는다.
좌표 약속
- 레이더 좌표 (TI): x 오른쪽, y 앞(조준축), z 위.  깊이 카메라: x 오른쪽, y 아래, z 앞.  방: z 위, 바닥 z = 0 (팩 R_d2r/t_d2r).
- 레이더→방 = 카메라 기울기 공유 가정 (노드 1 에 함께 달림): R = Rz(δψ) · R_c2r · R_T2C,  원점 = 카메라 중심 + (δx, δy, 0).
  → z (높이), roll, pitch 는 깊이 바닥(팩 카메라 자세, 정답 없는 바닥)에서 오고, 레이더 고도로 추정하지 않는다. x, y, yaw 만 추정.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter1d, gaussian_filter

import octo_io as io_

U = Path("/home/interns/unisense")
OFF = {x["stamp"]: x for x in json.loads((U / "notes/octonet_align_v2.json").read_text())["offsets"]}
BG = U / "dev/data/octonet/derived/backgrounds_p90.npz"
R_T2C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])      # TI → 카메라 축

# ---- 파라미터 [제안] ----
P = dict(
    vmax_valid=10.0,           # |도플러| > 10 m/s 는 깨진 값
    static_vox=0.2, static_occ=0.2, static_v=0.14,   # 녹화 안 0.2 m 칸이 레이더 프레임 20 % 넘게 차 있고 |v| ≤ 1 도플러 칸 (0.137 m/s)
    person_rmin=1.2, person_rmax=5.5,                # 사람 후보 거리 (깊이 사람 수평 거리 1–99 % = 1.9–4.0 m 에 여유)
    gate_r=0.8,                # 레이더 자체 사람 추적 중심에서 수평 0.8 m (깊이 추적기 R_GATE 와 같은 값)
    gate_static_r=0.45,        # 정지 표시 점도 추적 중심 0.45 m 안이고 장면 정지 지도에 없으면 몸 점으로 둠 (서 있는 사람)
    track_win=5,               # 추적 중심 이동 중앙값 창 (±5 프레임 ≈ ±0.55 s)
    z_lo=-0.3, z_hi=2.4,       # 방 높이 밖 (바닥 아래 = 바닥 다중경로 거울상, 천장 쪽) 은 유령점
    wall_margin=0.30,          # 깊이 배경 첫 벽보다 0.3 m 넘게 뒤 = 벽 너머 유령점
    tol_s=0.06,                # 레이더 프레임 ↔ 팩 프레임 짝 허용 시차
    huber=0.10, trunc=0.50,    # BEV 정합 강건 손실 (m)
    shifts=np.round(np.arange(-3.0, 1.0001, 0.02), 3),  # 시차 탐색 격자 (s)
)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# ------------------------------------------------------------------ 읽기
def load_recording(r, read_pack=True):
    """레이더 (레이더 좌표) + 팩 프레임 시각 + 깊이 사람 점 (방 좌표). 팩의 labels 는 열지 않는다."""
    ts, pts = io_.read_radar(r["radar"])
    pts = [p[np.abs(p[:, 3]) < P["vmax_valid"]] for p in pts]
    a, b, fps, n = io_.mocap_span(r["mocap"]); tz = io_.depth_times(r["depth"]); dt = OFF[r["stamp"]]["dt"]
    idx = np.where((tz + dt > a) & (tz + dt < b))[0][::3]; tk = tz[idx]
    out = dict(rec=r, ts=ts, pts=pts, tk=tk)
    if read_pack:
        f = U / "dev/packs/octonet_v07" / f"{r['stamp']}.npz"
        with np.load(f, allow_pickle=False) as Z:
            meta = json.loads(str(Z["meta"])); uvz = Z["pts_uvz"].astype(float); val = Z["pts_valid"].copy()
            R_d2r, t_d2r = Z["R_d2r"].astype(float), Z["t_d2r"].astype(float)
        assert len(tk) == uvz.shape[0], (r["stamp"], len(tk), uvz.shape)
        g = io_.room_geometry(r["env"], r["day"]); fx, fy, cx, cy = g["K"]
        assert np.allclose(g["R_c2r"], R_d2r, atol=1e-6) and np.allclose(g["t_c2r"], t_d2r, atol=1e-6)
        D = []
        for k in range(len(tk)):
            if val[k].sum() < 20:
                D.append(None); continue
            u, v, z = uvz[k, val[k]].T
            D.append(np.c_[(u - cx) / fx * z, (v - cy) / fy * z, z] @ R_d2r.T + t_d2r)
        out.update(meta=meta, D=D, R_d2r=R_d2r, t_d2r=t_d2r, K=g["K"], a=g["a"])
    return out


def depth_scene(env, day, g=None):
    """날짜별 깊이 정적 배경 (90 백분위, 정답 없음) → 방 좌표 점 + 카메라 기준 방위별 첫 벽 거리 지도."""
    g = g or io_.room_geometry(env, day); fx, fy, cx, cy = g["K"]
    with np.load(BG) as Z:
        z = Z[f"e{env}_{day}"].astype(float) * g["a"]
    vv, uu = np.mgrid[0:480, 0:640]; ok = np.isfinite(z) & (z > 0.3) & (z < 9.5)
    Pc = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], -1)[ok]; Pr = Pc @ g["R_c2r"].T + g["t_c2r"]
    vert = Pr[(Pr[:, 2] > 0.2) & (Pr[:, 2] < 2.0)]                       # 벽·가구 (바닥·천장 제외)
    c = g["t_c2r"][:2]; d = vert[:, :2] - c; az = np.degrees(np.arctan2(d[:, 1], d[:, 0])); rh = np.linalg.norm(d, axis=1)
    bins = np.arange(-60, 60.01, 1.0); wall = np.full(len(bins) - 1, np.nan)
    ib = np.digitize(az, bins) - 1
    for i in range(len(wall)):
        m = ib == i
        if m.sum() >= 20:
            wall[i] = np.percentile(rh[m], 5)                            # 그 방위의 첫 표면 (5 백분위, 잡음 화소 무시)
    return dict(vert=vert, wall_bins=bins, wall=wall, cam=g["t_c2r"].copy())


# ------------------------------------------------------------------ 정리
def static_mask(pts):
    """녹화 안 정지 클러터 (레이더 좌표, 외부 파라미터 불필요)."""
    if not len(pts) or sum(len(p) for p in pts) == 0:
        return [np.zeros(len(p), bool) for p in pts], np.zeros(0, np.int64)
    allp = np.concatenate(pts); fid = np.concatenate([np.full(len(p), i) for i, p in enumerate(pts)])
    v = np.floor(allp[:, :3] / P["static_vox"]).astype(np.int64); code = (v[:, 0] * 100003 + v[:, 1]) * 100003 + v[:, 2]
    u, inv = np.unique(code, return_inverse=True); fr = np.unique(np.c_[inv, fid], axis=0)
    occ = np.bincount(fr[:, 0], minlength=len(u)) / len(pts)
    m = (occ[inv] > P["static_occ"]) & (np.abs(allp[:, 3]) <= P["static_v"])
    return np.split(m, np.cumsum([len(p) for p in pts])[:-1]), u[occ > P["static_occ"]]


def scene_static_codes(static_lists):
    """같은 (환경, 날) 녹화들의 정지 칸 중 절반 넘는 녹화에서 정지인 칸 = 장면 정지 (가구·벽). 사람이 한자리에 오래 선 칸은 녹화마다 달라 빠짐."""
    from collections import Counter
    c = Counter()
    for codes in static_lists:
        c.update(set(codes.tolist()))
    n = len(static_lists)
    return np.array([k for k, v in c.items() if v > 0.5 * n], np.int64)


def vox_code(p):
    v = np.floor(p[:, :3] / P["static_vox"]).astype(np.int64)
    return (v[:, 0] * 100003 + v[:, 1]) * 100003 + v[:, 2]


def radar_track(ts, pts, stat):
    """레이더만으로 사람 추적 (레이더 좌표 BEV x, y; 외부 파라미터·깊이 불필요).
    후보 = 정지 아님 + 거리 [rmin, rmax]. 활동 중심 = 후보 BEV 밀도 최빈값. 중심 = 반경을 줄여 가며 (2.0 → 1.2 → 0.8 m) 이전 중심 근처 후보의 중앙값, 시간 이동 중앙값으로 다듬음."""
    F = len(pts)
    cand = []
    for p, s in zip(pts, stat):
        rg = np.linalg.norm(p[:, :2], axis=1)
        cand.append(p[(~s) & (rg >= P["person_rmin"]) & (rg <= P["person_rmax"])])
    allc = np.concatenate([c[:, :2] for c in cand]) if any(len(c) for c in cand) else np.zeros((0, 2))
    if len(allc) < 20:
        return np.full((F, 2), np.nan), np.zeros(F, bool), None
    H, xe, ye = np.histogram2d(allc[:, 0], allc[:, 1], bins=[np.arange(-4, 4.01, 0.1), np.arange(0, 6.01, 0.1)])
    H = gaussian_filter(H, 3); i, j = np.unravel_index(np.argmax(H), H.shape); centre = np.array([xe[i] + 0.05, ye[j] + 0.05])
    C = np.tile(centre, (F, 1))
    for R in (2.0, 1.2, P["gate_r"]):
        M = np.full((F, 2), np.nan)
        for f in range(F):
            lo, hi = max(0, f - 1), min(F, f + 2)                          # ±1 프레임 점을 합쳐 희소 프레임 안정화
            q = np.concatenate([cand[j][:, :2] for j in range(lo, hi)])
            if len(q):
                q = q[np.linalg.norm(q - C[f], axis=1) < R]
            if len(q) >= 2:
                M[f] = np.median(q, 0)
        ok = np.isfinite(M).all(1)
        if ok.sum() < 5:
            break
        Mi = np.c_[np.interp(np.arange(F), np.flatnonzero(ok), M[ok, 0]), np.interp(np.arange(F), np.flatnonzero(ok), M[ok, 1])]
        w = P["track_win"]
        C = np.array([np.median(Mi[max(0, f - w):f + w + 1], 0) for f in range(F)])
    seen = np.zeros(F, bool)
    for f in range(F):
        seen[f] = len(cand[f]) > 0 and (np.linalg.norm(cand[f][:, :2] - C[f], axis=1) < P["gate_r"]).any()
    return C, seen, centre


def classify(pts, stat, C, scene_codes):
    """점 분류 (레이더 좌표, 외부 파라미터 불필요): 0 몸, 1 정지 클러터, 2 추적 밖 (다중경로·다른 물체·잡음), 3 거리 밖."""
    cls = []
    for f, (p, s) in enumerate(zip(pts, stat)):
        c = np.full(len(p), 2, np.int8)
        rg = np.linalg.norm(p[:, :2], axis=1); dc = np.linalg.norm(p[:, :2] - C[f], axis=1) if np.isfinite(C[f]).all() else np.full(len(p), 99.0)
        inr = (rg >= P["person_rmin"]) & (rg <= P["person_rmax"])
        sc = np.isin(vox_code(p), scene_codes) if len(scene_codes) else np.zeros(len(p), bool)
        body = inr & (((~s) & (dc < P["gate_r"])) | (s & ~sc & (dc < P["gate_static_r"])))
        c[s] = 1; c[~inr & ~s] = 3; c[body] = 0
        cls.append(c)
    return cls


def ghost_room(q_room, scene):
    """방 좌표 점에서 유령점: 바닥 아래·천장 쪽, 깊이 첫 벽 너머 (깊이 시야 방위 안에서만)."""
    bad = (q_room[:, 2] < P["z_lo"]) | (q_room[:, 2] > P["z_hi"])
    d = q_room[:, :2] - scene["cam"][:2]; az = np.degrees(np.arctan2(d[:, 1], d[:, 0])); rh = np.linalg.norm(d, axis=1)
    ib = np.clip(np.digitize(az, scene["wall_bins"]) - 1, 0, len(scene["wall"]) - 1)
    inside = (az > scene["wall_bins"][0]) & (az < scene["wall_bins"][-1])
    w = scene["wall"][ib]; behind = inside & np.isfinite(w) & (rh > w + P["wall_margin"])
    return bad, behind


# ------------------------------------------------------------------ 변환
def base_pose(R_d2r, t_d2r):
    """레이더→방 기준 자세 (δ = 0): 카메라 기울기 공유, 원점 = 카메라 중심."""
    return R_d2r @ R_T2C, t_d2r.copy()


def apply(x, R0, c, b):
    """x = (δx, δy, δψ): 방 점 = Rz(δψ) R0 b + c + (δx, δy, 0)."""
    return b @ (Rz(x[2]) @ R0).T + c + np.array([x[0], x[1], 0.0])


def full_RT(x, R0, c):
    return Rz(x[2]) @ R0, c + np.array([x[0], x[1], 0.0])


# ------------------------------------------------------------------ 시간
def depth_motion(D):
    """깊이 사람 모양 변화 (연속 프레임 점 집합 사이 양방향 최근접 거리 중앙값) — 외부 파라미터 불필요."""
    T = len(D); E = np.full(T, np.nan)
    for k in range(1, T):
        if D[k] is not None and D[k - 1] is not None:
            E[k] = 0.5 * (np.median(cKDTree(D[k - 1]).query(D[k])[0]) + np.median(cKDTree(D[k]).query(D[k - 1])[0]))
    return E


def radar_motion(pts, cls):
    """레이더 움직임 에너지 = 몸 후보 점 중 |v| > 1 칸인 점 수 (외부 파라미터 불필요)."""
    return np.array([float(((c == 0) & (np.abs(p[:, 3]) > P["static_v"])).sum()) for p, c in zip(pts, cls)])


def xcorr_curve(ts, er, tk, ed, shifts=None):
    shifts = P["shifts"] if shifts is None else shifts
    ok = np.isfinite(ed); eds = uniform_filter1d(np.where(ok, ed, np.nanmedian(ed) if ok.any() else 0.0), 3); ers = uniform_filter1d(er, 3)
    C = np.full(len(shifts), np.nan)
    for i, sh in enumerate(shifts):
        ri = np.interp(tk - sh, ts, ers, left=np.nan, right=np.nan); m = ok & np.isfinite(ri)
        if m.sum() >= 50 and ri[m].std() > 0 and eds[m].std() > 0:
            C[i] = np.corrcoef(ri[m], eds[m])[0, 1]
    return C


def pair_frames(ts, tk, sh):
    """레이더 프레임 f ↔ 팩 프레임 k (|ts + sh − tk| ≤ tol). 반환 (f 배열, k 배열)."""
    if len(ts) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    j, e = io_.nearest(ts + sh, tk); ok = np.abs(e) <= P["tol_s"]
    return j[ok], np.flatnonzero(ok)


# ------------------------------------------------------------------ BEV 정합 (사람 보이는 표면)
class _FrameTree:
    """한 프레임 BEV 점 (query 만 필요할 때)."""
    def __init__(self, a):
        self.data = a

    def query(self, q):
        return cKDTree(self.data).query(q)


class PersonBEV:
    """팩 프레임별 깊이 사람 점 BEV (카메라에서 보이는 표면 = 같은 노드의 레이더가 보는 표면).
    빠른 최근접: 모든 프레임을 (x, y, 프레임·1000 m) 3차원 점 하나의 KD-tree 로 → 같은 프레임 안에서만 최근접이 잡힘."""
    SEP = 1000.0

    def __init__(self, D):
        self.trees = [None if d is None else _FrameTree(np.asarray(d[:, :2], float)) for d in D]
        self.cen = np.array([[np.nan] * 2 if d is None else np.median(d[:, :2], 0) for d in D])
        ks = [k for k, d in enumerate(D) if d is not None]
        if ks:
            self.all = np.concatenate([np.c_[np.asarray(D[k][:, :2], float), np.full(len(D[k]), k * self.SEP)] for k in ks])
            self.tree = cKDTree(self.all)
        else:
            self.all = np.zeros((0, 3)); self.tree = None

    def nn(self, q2, K):
        d, i = self.tree.query(np.c_[q2, K * self.SEP])
        return d, self.all[i, :2]


def gather(ts, tk, pts, cls, bev, sh, keep_frames=None):
    """짝지은 (레이더 몸 점 b (레이더 좌표), 팩 프레임 k) 목록."""
    fs, ks = pair_frames(ts, tk, sh); B, K = [], []
    for f, k in zip(fs, ks):
        if bev.trees[k] is None or (keep_frames is not None and not keep_frames[k]):
            continue
        b = pts[f][np.isin(cls[f], KEEP)]
        if len(b):
            B.append(b[:, :3]); K.append(np.full(len(b), k))
    if not B:
        return np.zeros((0, 3)), np.zeros(0, int)
    return np.concatenate(B), np.concatenate(K)


def residuals(x, R0, c, B, K, bev):
    q = apply(x, R0, c, B)[:, :2]
    if len(q) == 0:
        return q, np.zeros((0, 2)), np.zeros(0)
    d, nn = bev.nn(q, K)
    return q, nn, d


def weights(d):
    w = np.where(d <= P["huber"], 1.0, P["huber"] / np.maximum(d, 1e-9)); w[d > P["trunc"]] = 0.0
    return w


def solve_bev(R0, c, B, K, bev, x0=(0.0, 0.0, 0.0), iters=30, prior=None):
    """3자유도 (δx, δy, δψ) 강건 ICP (점–점, 2D). 회전은 레이더 원점 둘레. prior = (μ (3,), σ (3,)) 가우시안 사전 (선택)."""
    x = np.array(x0, float)
    for it in range(iters):
        q, nn, d = residuals(x, R0, c, B, K, bev); w = weights(d)
        if w.sum() < 10:
            return x, dict(ok=False, n=int((w > 0).sum()))
        # 선형화: q(x+Δ) ≈ q + [1,0,−(q_y − o_y)];[0,1,(q_x − o_x)] Δ, o = 레이더 원점
        o = c[:2] + x[:2]; rx, ry = q[:, 0] - o[0], q[:, 1] - o[1]
        J = np.zeros((2 * len(q), 3)); J[0::2, 0] = 1; J[1::2, 1] = 1; J[0::2, 2] = -ry; J[1::2, 2] = rx
        r = (nn - q).reshape(-1); W = np.repeat(w, 2)
        A = J.T @ (J * W[:, None]); bb = J.T @ (W * r)
        if prior is not None:
            mu, sg = prior; Pi = np.diag(1 / np.asarray(sg) ** 2); A = A + Pi; bb = bb + Pi @ (np.asarray(mu) - x)
        dx = np.linalg.solve(A + 1e-9 * np.eye(3), bb); x = x + dx
        if np.abs(dx[:2]).max() < 5e-4 and abs(dx[2]) < 1e-4:
            break
    q, nn, d = residuals(x, R0, c, B, K, bev); w = weights(d)
    return x, dict(ok=True, n=int(len(d)), inlier=float((d <= P["trunc"]).mean()), med=float(np.median(d)), mean_w=float(w.mean()), iters=it + 1)


def cost(x, R0, c, B, K, bev):
    _, _, d = residuals(x, R0, c, B, K, bev)
    rho = np.where(d <= P["huber"], 0.5 * d ** 2, P["huber"] * (np.minimum(d, P["trunc"]) - 0.5 * P["huber"]))
    return float(rho.mean()), float(np.median(d))


# ------------------------------------------------------------------ 정적 클러터 ↔ 깊이 배경 (BEV, 몸과 무관)
def static_pre(scene):
    """깊이 배경 수직 구조 BEV 점, KD-tree, 법선 (이웃 12점 PCA), 평평함 표시 — 한 번만 계산."""
    V = scene["vert"][:, :2]; tree = cKDTree(V)
    _, nb = tree.query(V, k=12); Q = V[nb] - V[nb].mean(1, keepdims=True); cov = np.einsum("nki,nkj->nij", Q, Q)
    ev, evec = np.linalg.eigh(cov)
    return dict(V=V, tree=tree, N=evec[:, :, 0], flat=ev[:, 0] / np.maximum(ev[:, 1], 1e-12) < 0.1)


def solve_static(R0, c, S, scene, x0=(0.0, 0.0, 0.0), iters=30, theta_max=np.radians(5), t_max=0.5, eps=0.05, pre=None):
    """레이더 정지 클러터 (레이더 좌표 점) ↔ 깊이 배경 수직 구조 BEV, 점–평면(수직 평면 = BEV 선) ICP, Shin 식 거리 비례 허용폭.
    초기값에서 (t_max, θ_max) 넘게 벗어나면 발산으로 보고 ok=False."""
    pre = pre or static_pre(scene); V, tree, N, flat = pre["V"], pre["tree"], pre["N"], pre["flat"]
    x = np.array(x0, float); info = dict(ok=False)
    for it in range(iters):
        q = apply(x, R0, c, S)[:, :2]; o = c[:2] + x[:2]; rng_ = np.linalg.norm(q - o, axis=1)
        d, ii = tree.query(q); gate = d < theta_max * rng_ + t_max + eps
        m = gate & flat[ii]
        if m.sum() < 15:
            info = dict(ok=False, n=int(m.sum()), overlap=float(gate.mean())); return x, info
        n = N[ii[m]]; r = ((V[ii[m]] - q[m]) * n).sum(1)
        w = np.where(np.abs(r) <= P["huber"], 1.0, P["huber"] / np.abs(r))
        rx, ry = q[m, 0] - o[0], q[m, 1] - o[1]
        J = np.c_[n[:, 0], n[:, 1], -ry * n[:, 0] + rx * n[:, 1]]
        A = J.T @ (J * w[:, None]); bb = J.T @ (w * r)
        dx = np.linalg.lstsq(A, bb, rcond=None)[0]; x = x + dx
        info = dict(ok=True, n=int(m.sum()), overlap=float(gate.mean()), med_abs=float(np.median(np.abs(r))), cond=float(np.linalg.cond(A)))
        dev = x - np.asarray(x0, float)
        if np.abs(dev[:2]).max() > t_max or abs(dev[2]) > theta_max:              # Shin 초기 오차 한계 (0.5 m, 5°) 밖 = 발산
            return x, dict(info, ok=False, why="diverged beyond initial-error bound (0.5 m / 5°)")
        if np.abs(dx[:2]).max() < 1e-4 and abs(dx[2]) < 1e-5:
            break
    return x, info


# ------------------------------------------------------------------ 녹화 단위 도우미 (prep 결과 + 팩 깊이 사람)
def depth_person(stamp):
    """팩의 깊이 사람 점 (방 좌표) 프레임 목록 + 카메라→방. labels 는 열지 않는다."""
    f = U / "dev/packs/octonet_v07" / f"{stamp}.npz"
    with np.load(f, allow_pickle=False) as Z:
        meta = json.loads(str(Z["meta"])); uvz = Z["pts_uvz"].astype(float); val = Z["pts_valid"].copy()
        R_d2r, t_d2r = Z["R_d2r"].astype(float), Z["t_d2r"].astype(float)
    g = io_.room_geometry(meta["env"], meta["day"]); fx, fy, cx, cy = g["K"]
    D = []
    for k in range(uvz.shape[0]):
        if val[k].sum() < 20:
            D.append(None); continue
        u, v, z = uvz[k, val[k]].T
        D.append((np.c_[(u - cx) / fx * z, (v - cy) / fy * z, z] @ R_d2r.T + t_d2r).astype(np.float32))
    return D, R_d2r, t_d2r, meta


def split_pts(pts, off):
    return [pts[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def running_median(X, w):
    return np.array([np.nanmedian(X[max(0, i - w):i + w + 1], 0) for i in range(len(X))])


def velocity_xcorr(ts, track, R0, tk, D, centre_shift, half=0.4, step=0.01):
    """레이더 추적 속도 (BEV, 방 축) ↔ 깊이 사람 중심 속도: 위치 치우침과 무관한 시차 다듬기 (Peršić 원리, 벡터 내적 상관).
    반환 (최적 시차, 상관 최대값, 깊이 속력 표준편차, 곡선)."""
    F = len(ts)
    cen = np.array([[np.nan, np.nan] if d is None else np.median(d[:, :2], 0) for d in D])
    ok = np.isfinite(cen).all(1)
    if ok.sum() < 50 or F < 50:
        return np.nan, np.nan, 0.0, None
    k_ = np.arange(len(D)); cen = np.c_[np.interp(k_, k_[ok], cen[ok, 0]), np.interp(k_, k_[ok], cen[ok, 1])]
    cen = running_median(cen, 3)
    vd = np.gradient(cen, tk, axis=0)
    tr = np.c_[track, np.zeros(F)] @ R0.T; tr = tr[:, :2]
    vr = np.gradient(tr, ts, axis=0)
    sp = float(np.linalg.norm(vd - vd.mean(0), axis=1).std())
    shifts = np.round(np.arange(centre_shift - half, centre_shift + half + 1e-9, step), 3); C = np.full(len(shifts), np.nan)
    for i, sh in enumerate(shifts):
        rx = np.interp(tk - sh, ts, vr[:, 0], left=np.nan, right=np.nan); ry = np.interp(tk - sh, ts, vr[:, 1], left=np.nan, right=np.nan)
        m = np.isfinite(rx) & ok
        if m.sum() < 50:
            continue
        a = np.c_[rx[m], ry[m]]; b = vd[m]; a = a - a.mean(0); b = b - b.mean(0)
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            C[i] = (a * b).sum() / den
    if not np.isfinite(C).any():
        return np.nan, np.nan, sp, None
    i = int(np.nanargmax(C))
    return float(shifts[i]), float(C[i]), sp, (shifts, C)


# ------------------------------------------------------------------ 최종 분류 (방 좌표 검사 포함)
CLS = {0: "body (정지 아님, 추적 0.8 m 안)", 6: "body_static (정지 표시지만 추적 0.45 m 안·깊이 배경 구조 아님 = 서 있는 사람)",
       1: "static clutter (녹화 안 0.2 m 칸 점유 > 20 %, |v| ≤ 1 칸)", 2: "off-track (추적 밖: 다중경로·다른 물체·잡음)",
       3: "out of person range (거리 < 1.2 또는 > 5.5 m)", 4: "z out of room (바닥 0.5 m 아래 = 바닥 반사 거울상, 2.5 m 위)",
       5: "behind wall (깊이 첫 표면 0.3 m 너머 = 벽 너머 유령점)"}
KEEP = (0, 6)


class Scene:
    def __init__(self, env, day):
        s = depth_scene(env, day); self.__dict__.update(s); self.tree = cKDTree(self.vert[:, :2])

    def wall_at(self, q):
        d = q[:, :2] - self.cam[:2]; az = np.degrees(np.arctan2(d[:, 1], d[:, 0])); rh = np.linalg.norm(d, axis=1)
        inside = (az > self.wall_bins[0]) & (az < self.wall_bins[-1])
        ib = np.clip(np.digitize(az, self.wall_bins) - 1, 0, len(self.wall) - 1)
        w = np.where(inside, self.wall[ib], np.nan)
        return w, rh


def classify_full(pts, stat, track, x, R0, c, scene):
    """정리 규칙 (레이더 좌표 규칙 + 방 좌표 유령 검사). 반환 프레임별 클래스 배열 (CLS)."""
    out = []
    for f, (p, s) in enumerate(zip(pts, stat)):
        cl = np.full(len(p), 2, np.int8)
        if len(p) == 0:
            out.append(cl); continue
        rg = np.linalg.norm(p[:, :2], axis=1)
        dc = np.linalg.norm(p[:, :2] - track[f], axis=1) if np.isfinite(track[f]).all() else np.full(len(p), 99.0)
        inr = (rg >= P["person_rmin"]) & (rg <= P["person_rmax"])
        q = apply(x, R0, c, p[:, :3]); w, rh = scene.wall_at(q)
        dstruct = scene.tree.query(q[:, :2])[0]
        free = np.isfinite(w) & (rh < w - 0.3) & (dstruct > 0.3) & (q[:, 2] > -0.3) & (q[:, 2] < 2.2)
        cl[s] = 1; cl[~s & ~inr] = 3
        cl[~s & inr & (dc < P["gate_r"])] = 0
        cl[s & inr & (dc < P["gate_static_r"]) & free] = 6
        body = np.isin(cl, KEEP)
        zbad = body & ((q[:, 2] < -0.5) | (q[:, 2] > 2.5)); cl[zbad] = 4
        behind = np.isin(cl, KEEP) & np.isfinite(w) & (rh > w + P["wall_margin"]); cl[behind] = 5
        out.append(cl)
    return out
