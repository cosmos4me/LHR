"""변환기 공통 코드 (v0.1): 방 좌표계, 관절 대응, 토큰화(광선·점), 사양 기반 입력 Σ.
팩(데이터셋 어댑터 출력)에는 원시 측정(2D 픽셀, 깊이 픽셀·깊이)과 좌표 변환만 저장하고, 토큰과 Σ는 이 함수들로
불러올 때 만든다. 그래야 가상 센서 실험(E3)에서 K·해상도·잡음을 바꿔 다시 토큰화할 수 있다.
v0.1 변경 (변환기 검증 반영):
- 2D 오차: Student-t(ν), 관절별 크기 g_j·s, 속도 항 β·|v_kp|.
- 광선 σ: 투영 야코비안 반영 (방사 cos²θ, 접선 cosθ).
- 스테레오: 양자화 격자로 구한 C = f·b, 시차 양자화 항.
- 바닥 평면: up_hint를 데이터셋마다 명시."""
import numpy as np

COMMON13 = ["head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
            "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
# 라벨 계열 → 공통 13 (정수 = 원래 관절 번호, 목록 = 그 관절들의 평균)
JOINT_MAP = {
    "h36m17": [10, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3],
    "coco17": [[3, 4], 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],   # 머리 대용 = 두 귀의 중점
}
# COCO 평가 코드의 관절별 OKS σ. 2D 검출 오차 사전값의 "모양"으로 쓰고 크기는 γ_j로 맞춘다
COCO_OKS_SIGMA = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062,
                           .107, .107, .087, .087, .089, .089])
# 깊이 PNG가 0/65535로 표시하는 무효 값 등 상수는 어댑터가 사양으로 넘긴다


def to_common13(J, family):
    """(..., J_family, 3) → (..., 13, 3)"""
    cols = [J[..., m, :].mean(-2) if isinstance(m, list) else J[..., m, :] for m in JOINT_MAP[family]]
    return np.stack(cols, axis=-2)


# ---------------- 방 좌표계 (정답 없이, 깊이에서 바닥 평면) ----------------
def fit_floor_plane(P, up_hint, min_cos, iters=3000, tol=0.02, min_height=0.2, seed=0):
    """카메라 좌표 점 P (N,3)에서 바닥 평면 n·X + c = 0 을 RANSAC으로 찾는다.
    up_hint: 카메라 좌표에서 대략의 위쪽 방향 (데이터셋마다 명시; 카메라가 크게 숙여져 있으면 그에 맞게).
    n: 위쪽 단위 벡터, c: 카메라 높이 [m] (> min_height). 반환 (n, c, 내점 수)."""
    rng = np.random.default_rng(seed)
    up = np.asarray(up_hint, float); up /= np.linalg.norm(up)
    best_cnt, best_inl = 0, None
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
        c = -n @ a
        if c < min_height:
            continue
        inl = np.abs(P @ n + c) < tol
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
    if best_inl is None:
        raise RuntimeError("floor plane not found: check up_hint/min_cos")
    Q = P[best_inl]; m = Q.mean(0)
    n = np.linalg.svd(Q - m, full_matrices=False)[2][-1]
    if n @ up < 0:
        n = -n
    return n, float(-n @ m), best_cnt


def room_frame(n, c):
    """카메라 좌표 → 방 좌표 변환 (X_room = R @ X_cam + t).
    z = 위쪽(바닥 법선), 원점 = 카메라 바로 아래 바닥점, x = 광축을 바닥에 투영한 방향. 카메라 중심은 (0, 0, c)."""
    z = n / np.linalg.norm(n)
    f = np.array([0.0, 0.0, 1.0]); x = f - (f @ z) * z; x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ (-c * z)
    return R, t


def compose(R1, t1, R2, t2):
    """X ↦ R2 (R1 X + t1) + t2"""
    return R2 @ R1, R2 @ t1 + t2


# ---------------- RGB: 2D 키포인트 → 광선 ----------------
def person_scale(kp_px):
    """사람 크기 s [px] = 키포인트 상자 넓이의 제곱근. (...,J,2) → (...)"""
    wh = kp_px.max(-2) - kp_px.min(-2)
    return np.sqrt(np.prod(np.maximum(wh, 1.0), -1))


def keypoint_speed(kp_seq, fps):
    """입력 키포인트 속도 |v| [px/s] (중앙 차분, 끝은 한쪽 차분). (T,J,2) → (T,J)"""
    v = np.gradient(kp_seq, axis=0) * fps
    return np.linalg.norm(v, axis=-1)


def keypoint_sigma_px(kp_seq, g, beta_s, fps):
    """관절별 2D 오차 크기 σ [px] (Student-t의 scale). (T,J,2) → (T,J)
    σ² = (g_j · s)² + (β · |v_kp|)², g_j = γ_j κ_j (사양), β [s] (동기·흔들림 항, 프레임률에 무관한 단위)."""
    s = person_scale(kp_seq)[..., None]
    v = keypoint_speed(kp_seq, fps)
    return np.sqrt((np.asarray(g)[None] * s) ** 2 + (beta_s * v) ** 2)


def ray_tokens(kp_px, K, R_c2r, t_c2r, sigma_px):
    """2D 키포인트 (...,J,2) → 방 좌표 광선.
    반환: 원점 (3,), 방향 d (...,J,3), 방사 각도 σ_rad, 접선 각도 σ_tan [rad] (...,J), 방사 단위벡터 e_rad (...,J,3).
    핀홀에서 주점으로부터 거리 ρ인 화소의 각도 민감도는 방사 cos²θ/f, 접선 cosθ/f (θ = 광축과의 각).
    광선 방향의 거리 정보는 0 (정보 행렬 계수 2)."""
    fx, fy, cx, cy = K
    f = 0.5 * (fx + fy)
    x, y = (kp_px[..., 0] - cx) / fx, (kp_px[..., 1] - cy) / fy
    a = np.stack([x, y, np.ones_like(x)], -1)
    nrm = np.linalg.norm(a, axis=-1, keepdims=True)
    d = a / nrm
    cos_t = 1.0 / nrm[..., 0]
    rho = np.sqrt(x ** 2 + y ** 2)
    rad = np.where(rho[..., None] > 1e-9, np.stack([x, y, np.zeros_like(x)], -1) / np.maximum(rho, 1e-9)[..., None],
                   np.array([1.0, 0.0, 0.0]))
    e = rad - np.sum(rad * d, -1, keepdims=True) * d
    e /= np.linalg.norm(e, axis=-1, keepdims=True)
    R = np.asarray(R_c2r)
    return (np.asarray(t_c2r, float).copy(), d @ R.T, sigma_px * cos_t ** 2 / f, sigma_px * cos_t / f, e @ R.T)


# ---------------- 깊이: 픽셀 (u, v, z) → 3D 점 + 입력 Σ ----------------
def axial_sigma(z, model):
    """광선 방향(깊이) 잡음 σ [m].
    stereo: z²·σ_d / C (C = f·b [px·m], 양자화 격자로 추정한 값이 있으면 그것) + 시차 양자화 z²/(n_sub·C·√12).
    tof: σ0 + k·z. 공통: 저장 양자화 Δ²/12 (예: 8비트 0–10 m → Δ = 39.2 mm)."""
    if model["type"] == "stereo":
        C = model.get("stereo_constant_px_m") or model["f_px"] * model["baseline_m"]
        var = (z ** 2 * model["sigma_disp_px"] / C) ** 2
        if model.get("subpixel_steps"):
            var = var + (z ** 2 / (model["subpixel_steps"] * C)) ** 2 / 12.0
    elif model["type"] == "tof":
        var = (model["sigma0_m"] + model["k_rel"] * z) ** 2
    else:
        raise ValueError(model["type"])
    return np.sqrt(var + model.get("quant_step_m", 0.0) ** 2 / 12.0)


def depth_point_tokens(uvz, K_d, R_d2r, t_d2r, model):
    """uvz (...,3): 깊이 영상 픽셀 좌표와 깊이 z [m]. 반환: 방 좌표 점 (...,3), 공분산 (...,3,3).
    P = z·a, a = ((u−cx)/fx, (v−cy)/fy, 1) 이므로 dP = a·dz + z·(du/fx, dv/fy, 0):
    Σ_cam = σ_z²·a aᵀ + z²·diag(σ_u²/fx², σ_v²/fy², 0). 이후 Σ_room = R Σ_cam Rᵀ.
    주의: 이 Σ는 점별 주변분포다. 실측 깊이 잡음은 약 8 px 상관 길이로 이웃 점끼리 공유된다(사양 corr_len_px)."""
    fx, fy, cx, cy = K_d
    u, v, z = uvz[..., 0], uvz[..., 1], uvz[..., 2]
    a = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(z)], -1)
    P = a * z[..., None]
    sz = axial_sigma(z, model)
    sp = model.get("sigma_pix_px", 1.0 / np.sqrt(12.0))
    S = (sz ** 2)[..., None, None] * a[..., :, None] * a[..., None, :]
    S[..., 0, 0] += (z * sp / fx) ** 2
    S[..., 1, 1] += (z * sp / fy) ** 2
    R = np.asarray(R_d2r)
    return P @ R.T + np.asarray(t_d2r), R @ S @ R.T


def cov6(S):
    """(...,3,3) → (...,6) = xx, xy, xz, yy, yz, zz"""
    return np.stack([S[..., 0, 0], S[..., 0, 1], S[..., 0, 2], S[..., 1, 1], S[..., 1, 2], S[..., 2, 2]], -1)
