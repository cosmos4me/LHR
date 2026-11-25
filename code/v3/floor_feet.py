"""v3 발 접촉 바닥 (사람 기반, 정답 없음). MM-Fi 에서 절차·상수를 고정한 뒤 mRI 에 한 번 적용한다.
절차 (고정, 2026-11-24, mRI 실행 전):
 1. 발목 3D (방 좌표): mRI = 두 시점 ViTPose 발목 삼각측량 (두 시점 점수 ≥ SCORE_MIN), MM-Fi 검증 = RGB 발목 + 깊이.
 2. 접촉 프레임: 발목 2D 속도 → 미터 속도 v = |Δuv|/2 · Z/f · fps (중앙 차분, 시점이 둘이면 큰 쪽) 이고 3프레임 중앙값 < V_CONTACT,
    그리고 그 발목이 두 발목 중 낮은 쪽 (+ LOW_SLACK 안). 공중·걷는 발은 빠진다.
 3. 접촉 구간: 같은 발목의 연속 접촉 프레임 MIN_SEG 개 이상 → 구간 3D 중앙값 (좁은 기선의 깊이 잡음을 구간 안에서 평균).
 4. 평면: 구간 점에 RANSAC (허용 TOL, 옛 z 와 20° 안, 아래 점 ≤ 5%) → 내점에 수직 잔차 Huber IRLS (z = a·x + b·y + c, 구간 길이 가중).
    구간 점의 수평 짧은 축 표준편차 < MINOR_MIN 이면 그 축 기울기는 관측 불가 → 0 으로 고정 (옛 방 기울기 유지).
 5. 바닥 = 발목 평면을 법선 방향으로 ANKLE_TO_SOLE 만큼 내림.
    ANKLE_TO_SOLE = 0.039·H (Drillis & Contini 1966 / Winter 2009 체절 비율: 바닥 → 발목 관절) + 신발 밑창 0.025 m, H = 1.70 m → 0.091 → 0.09 m, 사전 σ 0.02 m.
    사전 등록 규칙: MM-Fi 검증(깊이 바닥 위 RGB 발목 높이, 접촉 프레임 중앙값)이 사전값 ±0.02 m 안이면 사전값을 쓰고, 벗어나면 MM-Fi 측정값으로 바꾼다.
 6. 불확도: 구간 부트스트랩 (B = 200) → σ_기울기, σ_높이(원점·사람 중심) ⊕ 발목 사전 σ.
MM-Fi 검증 결과 (floor_feet_validate_mmfi.py, 2026-11-24, mRI 실행 전):
 - 접촉 발목의 깊이 바닥 위 높이 중앙값 0.073 m (E01 0.099 / E02 0.070 / E03 0.059 / E04 0.068) → |0.073 − 0.09| ≤ 0.02 → 사전값 0.09 유지 (규칙대로).
 - 환경 묶음 발 평면 vs 깊이 바닥: 기울기 1.55 / 0.53 / 1.27 / 3.20°, 사람 위치 높이 +1.4 / −1.8 / −2.3 / −1.7 cm.
   → 높이는 3 cm 안, 기울기는 방법 오차 RMS 약 1.9° (부트스트랩 σ 0.35–0.83° 보다 큼 = 계통 오차). 피험자 단위: 기울기 중앙 1.15°, 최대 5.7°, 높이 |·| 최대 5.1 cm.
 - mRI 비슷한 스테레오 잡음 모의 (f 1030 px, 기선 0.25 m): σ 2 px 에서 기울기 0.23° (p90 0.63°), 높이 < 0.3 cm → 좁은 기선 잡음은 주 오차원이 아님.
mRI 적용 규칙 (사전 등록): 법선 = 모든 피험자 접촉 구간을 묶은 평면 (한 책상 rig), 높이 = 묶은 평면;
 단 피험자 자신의 평면이 그 피험자 사람 위치에서 5 cm 넘게 다르면 그 피험자만 자기 높이 (법선은 묶은 것, rig 높이 이동 표시).
 기울기 불확도 = max(부트스트랩, 방법 오차 1.9°), 높이 불확도 = √(맞춤² + 0.02² (발목 사전) + 0.02² (방법)).
"""
METHOD_TILT_SIGMA = 1.9   # deg, MM-Fi 환경 묶음 검증 RMS
METHOD_H_SIGMA = 0.02     # m
SUBJECT_H_SWITCH = 0.05   # m
import numpy as np
import floor_common as fc

SCORE_MIN = 0.5
V_CONTACT = 0.25          # m/s
LOW_SLACK = 0.05          # m
MIN_SEG = 3               # 프레임
TOL = 0.03                # m
MINOR_MIN = 0.3           # m
ANKLE_TO_SOLE = 0.09      # m (사전; 위 규칙으로 MM-Fi 검증 후 확정)
ANKLE_PRIOR_SIGMA = 0.02  # m
ANK = (15, 16)            # COCO 왼·오른 발목


def metric_speed(kp_views, Z, f_views, fps):
    """kp_views: [(T,2) 발목 2D] 시점별, Z: (T,) 카메라 거리, f_views: 초점거리. 반환 (T,) m/s (시점 중 큰 값), 3프레임 중앙값."""
    vs = []
    for kp, f in zip(kp_views, f_views):
        d = np.full(len(kp), np.nan)
        d[1:-1] = np.linalg.norm(kp[2:] - kp[:-2], axis=-1) / 2.0
        vs.append(d * Z / f * fps)
    v = np.nanmax(np.stack(vs), 0) if len(vs) > 1 else vs[0]
    vm = v.copy()
    for t in range(1, len(v) - 1):
        vm[t] = np.nanmedian(v[t - 1:t + 2])
    return vm


def contact_segments(X, speed, valid):
    """X: (T,2,3) 두 발목 3D (방 좌표), speed: (T,2) m/s, valid: (T,2). 반환 [(점 (3,), 길이, 발목, 시작)]."""
    T = len(X); h = np.where(valid, X[..., 2], np.inf); low = np.min(h, 1)
    c = valid & (speed < V_CONTACT) & (h <= low[:, None] + LOW_SLACK) & np.isfinite(speed)
    segs = []
    for j in range(2):
        t = 0
        while t < T:
            if not c[t, j]:
                t += 1; continue
            s = t
            while t < T and c[t, j]:
                t += 1
            if t - s >= MIN_SEG:
                segs.append((np.median(X[s:t, j], 0), t - s, j, s))
    return segs


def fit_contact_plane(P, w, sensor=None, seed=0):
    """P (N,3) 구간 점, w (N,) 가중. 반환 dict(n, d (발목 평면), inl, minor, constrained) 또는 None."""
    if len(P) < 8:
        return None
    res = fc.ransac_plane(P, tol=TOL, iters=3000, max_tilt_deg=20.0, sensor=sensor, min_below=0.2, seed=seed, max_below_frac=0.05, refine=1)
    if res is None:
        return None
    inl = res["inl"]; Q, wq = P[inl], w[inl]
    n, d, minor = huber_plane(Q, wq)
    return dict(n=n, d=d, inl=inl, minor=minor, constrained=bool(minor < MINOR_MIN), n_inl=int(inl.sum()))


def huber_plane(Q, wq):
    xy0 = np.average(Q[:, :2], axis=0, weights=wq); xy = Q[:, :2] - xy0
    C = np.cov(xy.T, aweights=wq); ev, V = np.linalg.eigh(C); minor = float(np.sqrt(max(ev[0], 0)))
    A = np.c_[xy, np.ones(len(Q))] if minor >= MINOR_MIN else np.c_[xy @ V[:, 1], np.ones(len(Q))]
    w = np.ones(len(Q))
    for _ in range(15):
        ww = np.sqrt(w * wq)
        coef = np.linalg.lstsq(A * ww[:, None], Q[:, 2] * ww, rcond=None)[0]; r = Q[:, 2] - A @ coef
        s_ = 1.4826 * np.median(np.abs(r)) + 1e-6; w = np.minimum(1.0, 1.345 * s_ / np.maximum(np.abs(r), 1e-9))
    g = coef[:2] if minor >= MINOR_MIN else coef[0] * V[:, 1]
    n = np.array([-g[0], -g[1], 1.0]); n /= np.linalg.norm(n)
    c0 = coef[-1] - g @ xy0                                                  # z = g·xy + c0  (옛 좌표)
    return n, float(-c0 * n[2]), minor


def floor_from_ankles(segs, sensor=None, B=200, seed=0, ankle_to_sole=None):
    """접촉 구간 → 바닥 평면 + 불확도. 반환 dict 또는 None."""
    A_ = ANKLE_TO_SOLE if ankle_to_sole is None else ankle_to_sole
    if len(segs) < 8:
        return None
    P = np.array([s[0] for s in segs]); L = np.array([s[1] for s in segs], float); w = np.sqrt(np.minimum(L, 30))
    r = fit_contact_plane(P, w, sensor=sensor, seed=seed)
    if r is None:
        return None
    n, d_ank = r["n"], r["d"]; d_floor = d_ank + A_
    Q, wq = P[r["inl"]], w[r["inl"]]; pc = np.average(Q, axis=0, weights=wq)
    rng = np.random.default_rng(seed + 1); ang, h0, hp = [], [], []
    for _ in range(B):
        k = rng.integers(0, len(Q), len(Q))
        n2, d2, _ = huber_plane(Q[k], wq[k])
        ang.append(np.degrees(np.arccos(np.clip(n2 @ n, -1, 1)))); h0.append(d2 - d_ank); hp.append((pc @ n2 + d2) - (pc @ n + d_ank))
    s_h0 = float(np.std(h0)); s_hp = float(np.std(hp))
    resid = Q @ n + d_ank
    return dict(n=n, d=float(d_floor), d_ankle_plane=float(d_ank), ankle_to_sole=A_, n_points=int(r["n_inl"]), n_segments=len(segs),
                n_contact_frames=int(L.sum()), xy_minor_sd_m=r["minor"], slope_constrained=r["constrained"], person_center=pc,
                xy_extent=np.percentile(Q[:, :2], 95, 0) - np.percentile(Q[:, :2], 5, 0), resid_mad_m=float(1.4826 * np.median(np.abs(resid))),
                sigma_tilt_deg=float(np.sqrt(np.mean(np.square(ang)))), sigma_h0_fit_m=s_h0, sigma_h_person_fit_m=s_hp,
                sigma_h0_m=float(np.hypot(s_h0, ANKLE_PRIOR_SIGMA)), sigma_h_person_m=float(np.hypot(s_hp, ANKLE_PRIOR_SIGMA)))


# ---------------- 스테레오 삼각측량 ----------------
def norm_rays(k, K):
    a = np.concatenate([k, np.ones(k.shape[:-1] + (1,))], -1) @ np.linalg.inv(np.asarray(K, float)).T
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def triangulate(k1, k2, s1, s2, K1, K2, R12, t12, smin=SCORE_MIN):
    """cam1 좌표 신뢰도 가중 최소제곱 광선 교점. X2 = R12 X1 + t12. 점수 < smin 이면 NaN. (…,3)"""
    d1 = norm_rays(k1, K1); d2 = norm_rays(k2, K2) @ R12; o2 = -R12.T @ t12
    w1, w2 = s1 ** 2, s2 ** 2
    A1 = (np.eye(3) - d1[..., :, None] * d1[..., None, :]) * w1[..., None, None]
    A2 = (np.eye(3) - d2[..., :, None] * d2[..., None, :]) * w2[..., None, None]
    X = np.linalg.solve(A1 + A2 + 1e-12 * np.eye(3), (A2 @ o2)[..., None])[..., 0]
    X[(s1 < smin) | (s2 < smin)] = np.nan
    return X
