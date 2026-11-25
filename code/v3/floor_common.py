"""v3 바닥 공통 도구 (정답 없음). 모든 데이터셋 바닥 스크립트가 같이 쓴다.
- 평면 표기: 옛 방 좌표 X에서 n·X + d = 0, n = 위쪽 단위 법선 (n_z > 0). 새 높이 z' = n·X + d.
- 보정 변환 (옛 방 → 새 방, X' = R X + t):
    z' 축 = n (바닥 법선), x' = 옛 x를 바닥 평면에 투영해 정규화, y' = z' × x',
    원점 = 옛 원점을 바닥에 투영한 점 o' = −d n  →  R = [x'; y'; z'],  t = −R o' = (0, 0, d).
- RANSAC: 위쪽 사전(옛 z)과 max_tilt_deg 안의 법선만, 센서보다 min_below m 아래 평면만. 내점은 SVD(전체 최소제곱)로 다시 맞춘다.
- 불확도: 공간 블록 부트스트랩 (0.5 m 칸 재표본, 깊이 잡음이 공간 상관이므로 점 단위가 아니라 칸 단위).
- 정답 차단: safe_npz 는 정답 키(labels, y, gt, joints, mesh, refined_gt_kps, pose_2d 등)를 열려고 하면 예외를 낸다."""
import json
import numpy as np

FORBIDDEN = ("labels", "labels_cam", "labels_world", "label", "y", "gt", "joints", "mesh", "refined_gt_kps", "naive_gt_kps",
             "pose_2d", "pose_2d_l", "pose_2d_r", "pose", "pose_hand", "shape", "bbox_master", "bbox_sub", "label_mask",
             "label_quality", "joint_speed")
UP = np.array([0.0, 0.0, 1.0])


class SafeNpz:
    """np.load(allow_pickle=False) 래퍼: 정답 키 접근을 막는다 (지연 로딩이라 열지 않은 키는 읽지 않음)."""
    def __init__(self, path):
        self._z = np.load(path, allow_pickle=False)
        self.files = [k for k in self._z.files if k not in FORBIDDEN]

    def __getitem__(self, k):
        if k in FORBIDDEN or k.startswith("label") or "gt" in k.split("_"):
            raise KeyError(f"GT key blocked in build code: {k}")
        return self._z[k]

    def meta(self):
        return json.loads(str(self._z["meta"])) if "meta" in self._z.files else {}


def safe_npz(path):
    return SafeNpz(path)


# ---------------- 평면 ----------------
def plane_from_pts(Q, w=None):
    """가중 SVD 평면 (n 위쪽, d). Q (N,3)."""
    if w is None:
        w = np.ones(len(Q))
    w = w / w.sum(); m = (Q * w[:, None]).sum(0)
    C = ((Q - m) * w[:, None]).T @ (Q - m)
    n = np.linalg.eigh(C)[1][:, 0]
    if n[2] < 0:
        n = -n
    return n, -float(n @ m)


def tilt_deg(n):
    return float(np.degrees(np.arccos(np.clip(n[2] / np.linalg.norm(n), -1, 1))))


def ransac_plane(P, tol=0.02, iters=1500, max_tilt_deg=20.0, sensor=None, min_below=0.3, seed=0, tol_fn=None, refine=3, w=None, max_below_frac=0.03, sample_mask=None):
    """옛 방 좌표 점 P (N,3) → 바닥 평면. tol_fn(P) 가 있으면 점별 허용 오차 (깊이에 따라 커짐).
    sensor: 센서 위치 (옛 방 좌표); 평면이 센서보다 min_below 이상 아래여야 한다.
    sample_mask: 세 점 표본을 뽑을 후보 (예: 옛 바닥 ±0.4 m; 바닥 점이 5% 뿐인 ITOP 에서 RANSAC 성공 확률을 올림). 내점 평가는 모든 점.
    바닥 = 가장 낮은 큰 면: 평면보다 max(5 cm, 3·허용 오차) 넘게 아래에 있는 점이 max_below_frac 을 넘으면 후보에서 뺀다 (탁자·경사면 거부).
    반환 dict(n, d, inl (bool N), n_inl, tilt_deg) 또는 None."""
    rng = np.random.default_rng(seed); N = len(P)
    if N < 50:
        return None
    tl = tol_fn(P) if tol_fn is not None else np.full(N, tol)
    cmin = np.cos(np.radians(max_tilt_deg)); best, best_inl = -1, None
    pool = np.flatnonzero(sample_mask) if sample_mask is not None else np.arange(N)
    if len(pool) < 3:
        return None
    idx = pool[rng.integers(0, len(pool), size=(iters, 3))]
    for a, b, c in idx:
        nn = np.cross(P[b] - P[a], P[c] - P[a]); s = np.linalg.norm(nn)
        if s < 1e-9:
            continue
        nn /= s
        if nn[2] < 0:
            nn = -nn
        if nn[2] < cmin:
            continue
        d = -nn @ P[a]
        if sensor is not None and nn @ sensor + d < min_below:
            continue
        r = P @ nn + d
        inl = np.abs(r) < tl
        k = int(inl.sum()) if w is None else float(w[inl].sum())
        if k > best and max_below_frac is not None and np.mean(r < -np.maximum(0.05, 3 * tl)) > max_below_frac:
            continue
        if k > best:
            best, best_inl = k, inl
    if best_inl is None or best_inl.sum() < 30:
        return None
    inl = best_inl
    for _ in range(refine):
        n, d = plane_from_pts(P[inl], None if w is None else w[inl])
        inl2 = np.abs(P @ n + d) < tl
        if inl2.sum() < 30:
            break
        inl = inl2
    n, d = plane_from_pts(P[inl], None if w is None else w[inl])
    return dict(n=n, d=d, inl=inl, n_inl=int(inl.sum()), tilt_deg=tilt_deg(n))


def block_bootstrap(Q, B=100, cell=0.5, seed=0, w=None, ref_pts=None):
    """평면 불확도: xy 칸(cell m) 단위 재표본. 반환 σ_tilt [deg] (법선 각 흔들림 RMS), σ_h0 [m] (옛 원점에서 높이),
    ref_pts 가 있으면 그 점들에서 높이 σ의 최대값."""
    rng = np.random.default_rng(seed)
    key = np.floor(Q[:, :2] / cell).astype(np.int64); _, cid = np.unique(key, axis=0, return_inverse=True); cid = cid.ravel()
    nc = cid.max() + 1
    members = [np.flatnonzero(cid == c) for c in range(nc)]
    n0, d0 = plane_from_pts(Q, w)
    ang, h0, hr = [], [], []
    for _ in range(B):
        pick = rng.integers(0, nc, nc)
        ii = np.concatenate([members[c] for c in pick])
        if len(ii) < 10:
            continue
        n, d = plane_from_pts(Q[ii], None if w is None else w[ii])
        ang.append(np.degrees(np.arccos(np.clip(n @ n0, -1, 1)))); h0.append(d - d0)
        if ref_pts is not None:
            hr.append((ref_pts @ n + d) - (ref_pts @ n0 + d0))
    out = dict(sigma_tilt_deg=float(np.sqrt(np.mean(np.square(ang)))) if ang else float("nan"),
               sigma_h0_m=float(np.std(h0)) if h0 else float("nan"), n_cells=int(nc))
    if hr:
        out["sigma_h_ref_m"] = float(np.max(np.std(np.array(hr), 0)))
    return out


# ---------------- 보정 변환 ----------------
def correction(n, d):
    """평면 (n, d) (옛 방 좌표) → 옛 방 → 새 방 강체 변환 R (3,3), t (3,)."""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    if n[2] < 0:
        n, d = -n, -d
    ex = np.array([1.0, 0, 0]); x = ex - (ex @ n) * n; x /= np.linalg.norm(x)
    y = np.cross(n, x); R = np.stack([x, y, n])
    t = np.array([0.0, 0.0, float(d)])
    return R, t


def apply(R, t, X):
    return X @ np.asarray(R).T + np.asarray(t)


def plane_summary(n, d):
    """보고용: 법선 각 변화 [deg], 높이 변화 [m] (옛 원점이 새 바닥 위로 얼마나 있는지 = d),
    기울기 방향 (옛 x 쪽 pitch, 옛 y 쪽 roll) [deg]."""
    n = np.asarray(n, float)
    return dict(angle_change_deg=tilt_deg(n), height_change_m=float(d),
                tilt_x_deg=float(np.degrees(np.arctan2(n[0], n[2]))), tilt_y_deg=float(np.degrees(np.arctan2(n[1], n[2]))))


def consensus(planes, weights=None):
    """같은 장면(같은 옛 방 좌표) 녹화들의 평면 합의: 법선 성분별 중앙값 → 정규화, d 중앙값. planes = [(n, d)]."""
    N = np.array([p[0] for p in planes]); D = np.array([p[1] for p in planes])
    n = np.median(N, 0); n /= np.linalg.norm(n)
    return n, float(np.median(D))


def deviation(n1, d1, n2, d2):
    """두 평면 차이: 각 [deg], 옛 원점 높이 차 [m]."""
    return float(np.degrees(np.arccos(np.clip(np.dot(n1, n2) / np.linalg.norm(n1) / np.linalg.norm(n2), -1, 1)))), float(d1 - d2)


# 장면 합의 규칙 (모든 데이터셋 공통, mRI 실행 전에 고정; mRI 는 floor_feet.py 의 사전 등록 규칙을 따로 씀):
#  허용 = (각 max(ANG_TOL, 3σ_녹화, 3·장면 흩어짐), 높이 max(H_TOL, 3σ_녹화, 3·장면 흩어짐)); 장면 흩어짐 = 녹화 편차의 1.4826·MAD.
#  허용 안 → 장면 합의값; 실패 한계(FAIL_ANG, FAIL_H) 밖 → 녹화 맞춤 실패로 보고 장면 합의값 (표시);
#  그 사이 → 녹화 자신의 값 (rig 이동 의심 표시).
ANG_TOL, H_TOL = 1.0, 0.02
FAIL_ANG, FAIL_H = 3.0, 0.10


def choose_final(rec, cons, spread_ang=0.0, spread_h=0.0):
    """rec, cons: dict(n, d, sigma_tilt_deg, sigma_h0_m). 반환 (n, d, 선택 이름, 각 차, 높이 차)."""
    a, h = deviation(rec["n"], rec["d"], cons["n"], cons["d"])
    st = rec.get("sigma_tilt_deg", 0.0); sh = rec.get("sigma_h0_m", 0.0)
    st = 0.0 if not np.isfinite(st) else st; sh = 0.0 if not np.isfinite(sh) else sh
    if a <= max(ANG_TOL, 3 * st, 3 * spread_ang) and abs(h) <= max(H_TOL, 3 * sh, 3 * spread_h):
        return np.asarray(cons["n"]), float(cons["d"]), "scene_consensus", a, h
    if a > max(FAIL_ANG, 3 * spread_ang) or abs(h) > max(FAIL_H, 3 * spread_h):
        return np.asarray(cons["n"]), float(cons["d"]), "scene_consensus(own_fit_rejected_outlier)", a, h
    return np.asarray(rec["n"]), float(rec["d"]), "own_recording(deviates_from_scene)", a, h


def scene_spread(recs, cons):
    """장면 안 녹화 편차의 강건 흩어짐 (각 [deg], 높이 [m])."""
    dv = np.array([deviation(r["n"], r["d"], cons["n"], cons["d"]) for r in recs])
    if len(dv) < 3:
        return 0.0, 0.0
    return float(1.4826 * np.median(np.abs(dv[:, 0]))), float(1.4826 * np.median(np.abs(dv[:, 1] - np.median(dv[:, 1]))))


def jdump(x):
    def cv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    return json.dumps(x, default=cv, indent=1, ensure_ascii=False)
