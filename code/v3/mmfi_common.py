"""v3 MM-Fi 정답 없는 정합 공통 모듈 (2026-11-24).
- 정답 차단: 빌드 코드는 SafeNpz 로만 npz 를 연다. 허용 목록 밖의 키, 라벨·관절·GT·IR 키는 읽으면 예외.
- 기준(정답 없음): 깊이 팩 (mmfi_v02_specA) 의 사람 점 pts_uvz (RGB 검출기 키포인트로 고른 깊이 사람 점) → R_d2r/t_d2r 로 방 좌표.
  방 좌표 = 환경별 깊이 바닥 (z = 0) + RGB 광축 수평 투영 (x) (adapt_mmfi.room_frame). 깊이 원본 crop / 배경 중앙값은 cache_v1.
- 원시 센서: LiDAR extracted_lidar/.../frameNNN.bin (float64 xyz, LiDAR 좌표), 레이더 extracted_rf/.../mmwave/frameNNN.bin (float64 x y z snr doppler).
내려받은 데이터를 읽으므로 python -I 로 실행하고 sys.path 에 이 폴더만 수동으로 넣는다."""
import json, re
from pathlib import Path
import numpy as np

U = Path("/home/interns/unisense")
DATA = U / "dev/data/mmfi"
LID = DATA / "extracted_lidar"
RF = DATA / "extracted_rf"
CACHE = DATA / "cache_v1"
DPK = U / "dev/packs/mmfi_v02_specA"
OUT = U / "dev/packs/v3"
OUT_L = OUT / "mmfi_lidar"
OUT_R = OUT / "mmfi_radar"
WORK_L = OUT_L / "_reg"            # 녹화별 중간 결과 (JSON/npz)
WORK_R = OUT_R / "_reg"
ENVS = ("E01", "E02", "E03", "E04")
FPS = 10.0

# ---------------- 정답 차단 ----------------
FORBID = re.compile(r"^(gt|labels.*|label_.*|joint.*|kp3d.*|ir.*|infra.*|.*ground_truth.*|y|ym)$")
ALLOW_DEPTH_PACK = {"pts_uvz", "pts_valid", "pts_edge", "pts_n_raw", "R_d2r", "t_d2r", "R_c2r", "t_c2r", "meta"}
ALLOW_CACHE = {"boxes", "crop_data", "crop_off", "bg_med", "bg_valid", "n_depth", "meta"}
ACCESS_LOG = set()


class SafeNpz:
    """허용된 키만 읽는 npz 래퍼. 라벨·GT 키는 허용 목록에 있어도 거부한다."""
    def __init__(self, path, allow):
        self.path = str(path); self._z = np.load(path, allow_pickle=False); self.allow = set(allow)
        self.files = [k for k in self._z.files if k in self.allow and not FORBID.match(k)]

    def __getitem__(self, k):
        if k not in self.allow or FORBID.match(k):
            raise KeyError(f"GT guard: key '{k}' is not allowed in build code ({self.path})")
        ACCESS_LOG.add(k)
        return self._z[k]

    def __contains__(self, k):
        return k in self.files


def assert_no_gt_access():
    bad = [k for k in ACCESS_LOG if FORBID.match(k)]
    assert not bad, f"GT keys accessed: {bad}"
    return sorted(ACCESS_LOG)


# ---------------- 목록 ----------------
def recordings(envs=ENVS):
    """ms_cache2 가 읽는 녹화 = 깊이 팩 mmfi_v02_specA/E0x/Syy_Azz.npz 전부."""
    out = []
    for e in envs:
        for f in sorted((DPK / e).glob("S*_A*.npz")):
            out.append((e, f.stem))
    return out


def spec_depth_K(env):
    sp = U / "code/specs" / ("mmfi_v02_specA_e04.json" if env == "E04" else "mmfi_v02_specA.json")
    return np.array(json.loads(sp.read_text())["depth"]["K"], float)


# ---------------- 깊이 기준 ----------------
def depth_person(env, nm, drop_edge=False):
    """프레임별 깊이 사람 점 (방 좌표) 목록과 R_d2r, t_d2r, 프레임별 카메라 z 중앙값."""
    P = SafeNpz(DPK / env / f"{nm}.npz", ALLOW_DEPTH_PACK)
    uvz = P["pts_uvz"].astype(float); v = P["pts_valid"]; e = P["pts_edge"]; R = np.asarray(P["R_d2r"], float); t = np.asarray(P["t_d2r"], float)
    fx, fy, cx, cy = spec_depth_K(env); pts, zc = [], np.full(len(uvz), np.nan)
    for k in range(len(uvz)):
        m = v[k] & (~e[k] if drop_edge else True)
        u, w, z = uvz[k, m].T
        pts.append(np.c_[(u - cx) / fx * z, (w - cy) / fy * z, z] @ R.T + t)
        if m.sum():
            zc[k] = np.median(z)
    return pts, R, t, zc


def depth_floor_rec(env, nm, R_d2r, t_d2r):
    """녹화별 깊이 바닥 (cache_v1 bg_med, ¼ 해상도 시간 중앙값) → 방 좌표 평면 n·X + c = 0 (n 위쪽).
    방 좌표의 바닥은 정의상 z = 0 (환경 단위 적합); 녹화별 바닥이 그와 얼마나 다른지 = 플랫폼 이동 진단."""
    Z = SafeNpz(CACHE / env / f"{nm}.npz", ALLOW_CACHE)
    z = Z["bg_med"].astype(float); val = Z["bg_valid"]
    fx, fy, cx, cy = spec_depth_K(env); vv, uu = np.mgrid[0:480:4, 0:640:4]
    m = (val > 0.5) & np.isfinite(z) & (z < 8) & (z > 0.3)
    X = np.c_[(uu[m] - cx) / fx * z[m], (vv[m] - cy) / fy * z[m], z[m]] @ R_d2r.T + t_d2r
    X = X[np.abs(X[:, 2]) < 0.25]
    if len(X) < 200:
        return None
    n, c, inl, sig = ransac_plane(X, up=np.array([0, 0, 1.0]), min_cos=np.cos(np.radians(10)), tol=0.02, iters=400, seed=0)
    if n is None:
        return None
    return dict(n=n.tolist(), c=float(c), inliers=int(inl), sigma=float(sig), tilt_deg=float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))),
                z_at_person=float(-(c + n[0] * 3.1) / n[2]))


def depth_motion_energy(env, nm, zc):
    """깊이 원본 crop (cache_v1) 의 사람 근처 프레임 차분 에너지 e(t) = 평균 |D_t − D_{t−1}| (m), 사람 깊이 ±0.6 m 안 화소만.
    zc = 프레임별 사람 카메라 z 중앙값 (깊이 팩). 외부 파라미터와 무관한 움직임 신호 (시간 정합용)."""
    Z = SafeNpz(CACHE / env / f"{nm}.npz", ALLOW_CACHE)
    bx = Z["boxes"].astype(int); data = Z["crop_data"]; off = Z["crop_off"]; T = len(bx)
    def crop(t):
        u0, v0, u1, v1 = bx[t]; d = data[off[t]:off[t + 1]].reshape(v1 - v0 + 1, u1 - u0 + 1).astype(float) * 1e-3
        d[(d <= 0) | (d > 60)] = np.nan
        return d
    e = np.full(T, np.nan); prev = crop(0) if T else None
    for t in range(1, T):
        cur = crop(t); a, b = bx[t - 1], bx[t]
        u0, v0 = max(a[0], b[0]), max(a[1], b[1]); u1, v1 = min(a[2], b[2]), min(a[3], b[3])
        if u1 > u0 + 4 and v1 > v0 + 4:
            A = prev[v0 - a[1]:v1 - a[1] + 1, u0 - a[0]:u1 - a[0] + 1]; B = cur[v0 - b[1]:v1 - b[1] + 1, u0 - b[0]:u1 - b[0] + 1]
            zp = zc[t] if np.isfinite(zc[t]) else (zc[t - 1] if np.isfinite(zc[t - 1]) else np.nan)
            if np.isfinite(zp):
                with np.errstate(invalid="ignore"):
                    m = np.isfinite(A) & np.isfinite(B) & ((np.abs(A - zp) < 0.6) | (np.abs(B - zp) < 0.6))
                if m.sum() > 50:
                    e[t] = float(np.mean(np.minimum(np.abs(A[m] - B[m]), 0.3)))
        prev = cur
    return e


# ---------------- 원시 센서 ----------------
def lidar_frames(env, nm):
    s, a = nm.split("_"); d = LID / env / s / a / "lidar"
    fs = sorted(d.glob("frame*.bin"), key=lambda f: int(f.stem[5:])) if d.exists() else []
    return [np.fromfile(f, dtype=np.float64).reshape(-1, 3) for f in fs]


def radar_frames(env, nm, T):
    """열: x (앞) y z snr doppler — 원시 레이더 좌표. 파일 없음/깨짐 = 빈 배열."""
    s, a = nm.split("_"); out = []
    for k in range(T):
        f = RF / env / s / a / "mmwave" / f"frame{k + 1:03d}.bin"
        r = np.zeros((0, 5))
        if f.exists():
            x = np.fromfile(f, dtype=np.float64)
            if x.size % 5 == 0:
                r = x.reshape(-1, 5)
        out.append(r)
    return out


# ---------------- 기하 ----------------
def ransac_plane(P, up, min_cos, tol=0.02, iters=500, seed=0):
    """평면 n·X + c = 0 (n 은 up 쪽). 반환 n, c, 내점 수, 내점 잔차 σ (MAD)."""
    rng = np.random.default_rng(seed); best, bi = 0, None
    if len(P) < 3:
        return None, None, 0, np.nan
    for _ in range(iters):
        a, b, c3 = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c3 - a); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if n @ up < 0:
            n = -n
        if n @ up < min_cos:
            continue
        inl = np.abs(P @ n - n @ a) < tol; cnt = int(inl.sum())
        if cnt > best:
            best, bi = cnt, inl
    if bi is None:
        return None, None, 0, np.nan
    for _ in range(3):                                                    # 내점 최소제곱 재적합 (SVD) + 재선택
        Q = P[bi]; m = Q.mean(0); n = np.linalg.svd(Q - m, full_matrices=False)[2][-1]
        if n @ up < 0:
            n = -n
        c = -n @ m; r = P @ n + c; bi = np.abs(r) < tol
    r = P[bi] @ n + c
    return n, float(c), int(bi.sum()), float(1.4826 * np.median(np.abs(r - np.median(r))))


def rot_a_to_b(a, b):
    """a 를 b 로 보내는 최소 회전 (Rodrigues)."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); s = np.linalg.norm(v); c = a @ b
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s; th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def procrustes2d(A, B, w):
    """가중 2D 강체 (회전 + 이동): R A + t ≈ B. 반환 ψ, t (2,)."""
    w = w / w.sum(); ma = (w[:, None] * A).sum(0); mb = (w[:, None] * B).sum(0); A0, B0 = A - ma, B - mb
    H = (w[:, None] * A0).T @ B0; psi = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = np.cos(psi), np.sin(psi); R = np.array([[c, -s], [s, c]])
    return float(psi), mb - R @ ma


def huber_w(r, d):
    a = np.abs(r)
    return np.where(a <= d, 1.0, d / np.maximum(a, 1e-12))


def box_smooth(x, w=5, centred=True):
    """NaN 무시 상자 평활. centred=False 면 뒤쪽 창 [t−w+1, t]."""
    x = np.asarray(x, float); T = len(x); out = np.full(T, np.nan)
    for t in range(T):
        lo, hi = (t - w // 2, t + w // 2 + 1) if centred else (t - w + 1, t + 1)
        s = x[max(0, lo):min(T, hi)]
        if np.isfinite(s).sum() >= max(1, w // 2):
            out[t] = np.nanmean(s)
    return out


def xcorr_lags(a, b, lags):
    """정규화 상호상관 ρ(L) = corr(a[t], b[t − L]) (NaN 쌍 제외). a 가 b 보다 L 프레임 늦으면 최대."""
    out = []
    for L in lags:
        if L >= 0:
            x, y = a[L:], b[:len(b) - L]
        else:
            x, y = a[:L], b[-L:]
        m = np.isfinite(x) & np.isfinite(y)
        out.append(np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 20 and np.std(x[m]) > 0 and np.std(y[m]) > 0 else np.nan)
    return np.array(out)
