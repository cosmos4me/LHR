"""v3 mRI 레이더 → 방 정합 (정답 없음) 공용 함수.

봉인 규칙: mRI 팩에서는 kp_px, kp_score, kp_px_cam2, kp_score_cam2, meta 만 읽는다.
labels / labels_cam / label_mask / label_quality / action 및 mRI 원본의 pose_labels·GT 는 열지 않는다.
사양(specs/mri_v03.json)에서는 K, K_cam2, room(R_c2r, t_c2r, R_c2r_cam2, t_c2r_cam2, camera_height_m) 만 쓴다.
sigma_model (학습 피험자 정답으로 맞춘 2D 잡음 모형) 은 쓰지 않는다 — 잡음은 에피폴라 잔차와 시간 고주파 잔차로 정답 없이 잰다.

좌표: 방(mri_v03) x = 카메라 앞, y = 왼쪽, z = 위 (바닥 = 정답 없는 ViTPose 발목 평면).
"""
import json, itertools
import numpy as np

U = "/home/interns/unisense"
CSV = f"{U}/dev/data/mri/extracted/dataset_release/aligned_data/radar/singleframe"
PK = f"{U}/dev/packs/mri_v03"
SPEC = f"{U}/code/specs/mri_v03.json"
OUT = f"{U}/dev/packs/v3/mri_radar"
WORK = f"{OUT}/work"
OLD = f"{U}/dev/packs/mri_v03_radar"
SUBJ = list(range(1, 21))
KP_KEYS = ("kp_px", "kp_score", "kp_px_cam2", "kp_score_cam2")      # 팩에서 읽는 유일한 배열
SCORE_MIN = 0.3
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


# ---------------------------------------------------------------- 입력
def load_spec():
    S = json.load(open(SPEC))
    rgb, rm = S["rgb"], S["room"]
    cams = [dict(K=np.array(rgb["K"], float), R=np.array(rm["R_c2r"], float), t=np.array(rm["t_c2r"], float)),
            dict(K=np.array(rgb["K_cam2"], float), R=np.array(rm["R_c2r_cam2"], float), t=np.array(rm["t_c2r_cam2"], float))]
    return cams, float(rm["camera_height_m"]), float(S["fps"])


def load_kp(s):
    """팩에서 2D 키포인트와 meta 만 읽음 (라벨 배열은 접근하지 않음)."""
    P = np.load(f"{PK}/S{s:02d}.npz", allow_pickle=False)
    out = {k: P[k].astype(float) for k in KP_KEYS}
    m = json.loads(str(P["meta"]))
    return out, dict(T=int(m["T"]), fps=float(m["fps"]), frame0=int(m["frame0"]))


def read_radar_csv(s):
    """원본 CSV (데이터만). 열: Frame #, # Obj, X, Y, Z, Doppler, Intensity, Abs Time, Camera Frame.
    반환: dict 카메라 프레임 → (레이더 프레임 번호, (n,5) 점 x y z 도플러 세기 [센서 좌표])."""
    A = np.loadtxt(f"{CSV}/subject{s}.csv", delimiter=",", skiprows=1)
    fr, cf = A[:, 0].astype(np.int64), A[:, 8].astype(np.int64)
    order = np.lexsort((fr, cf)); A, fr, cf = A[order], fr[order], cf[order]
    out = {}
    cuts = np.flatnonzero(np.diff(cf)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(cf)]):
        c = int(cf[a]); f = fr[a:b]; last = f.max()                       # 같은 정렬 프레임에 레이더 프레임 여럿이면 마지막 (v0.3 과 같음)
        sel = f == last
        out[c] = (int(last), A[a:b][sel][:, [2, 3, 4, 5, 6]].astype(float))
    return out


# ---------------------------------------------------------------- 스테레오 삼각측량
def ray_dirs(kp, cam):
    """(…,2) 화소 → 방 좌표 단위 광선 (K 의 skew 무시, ms_cache2.rays 와 같은 규약)."""
    K = cam["K"]; x = (kp[..., 0] - K[0, 2]) / K[0, 0]; y = (kp[..., 1] - K[1, 2]) / K[1, 1]
    a = np.stack([x, y, np.ones_like(x)], -1); d = a / np.linalg.norm(a, axis=-1, keepdims=True)
    return d @ cam["R"].T


def triangulate(u1, u2, cams, sig_theta):
    """두 광선 가중 최소제곱 교점 + 이방성 공분산.
    u1,u2 (N,2) 화소. sig_theta (2,) 각 카메라의 광선 각 잡음 [rad].
    점 X 는 Σ_i w_i (I − d_i d_iᵀ)(X − o_i) = 0 의 해, w_i = 1/(r_i σ_i)² (r_i = 카메라–점 거리).
    공분산 = (Σ_i w_i (I − d_i d_iᵀ))⁻¹ — 좁은 기선에서는 시선(깊이) 방향으로 길쭉함.
    에피폴라 잔차: 두 광선 최단 거리의 부호 있는 각 (기선 ⟂ 성분) — 잡음 추정용."""
    d1, d2 = ray_dirs(u1, cams[0]), ray_dirs(u2, cams[1]); o1, o2 = cams[0]["t"], cams[1]["t"]
    I = np.eye(3)
    P1 = I[None] - d1[:, :, None] * d1[:, None, :]; P2 = I[None] - d2[:, :, None] * d2[:, None, :]
    X = None; r1 = r2 = np.full(len(u1), 2.5)
    for _ in range(2):                                                    # 거리 가중을 한 번 갱신
        w1 = 1.0 / (r1 * sig_theta[0]) ** 2; w2 = 1.0 / (r2 * sig_theta[1]) ** 2
        A = w1[:, None, None] * P1 + w2[:, None, None] * P2
        b = w1[:, None] * (P1 @ o1) + w2[:, None] * (P2 @ o2)
        X = np.linalg.solve(A, b[..., None])[..., 0]
        r1 = np.linalg.norm(X - o1, axis=1); r2 = np.linalg.norm(X - o2, axis=1)
    cov = np.linalg.inv(A)
    n = np.cross(d1, d2); nn = np.linalg.norm(n, axis=1, keepdims=True); n = n / np.maximum(nn, 1e-12)
    n = n * np.sign(n[:, 2:3] + 1e-12)                                     # 부호 고정 (위쪽 성분 +)
    epi = ((o2 - o1)[None] * n).sum(1) / np.maximum(0.5 * (r1 + r2), 1e-6)   # rad, 부호 있음
    depth1 = ((X - o1) @ cams[0]["R"])[:, 2]                               # 카메라1 광축 방향 깊이
    return X, cov, epi, depth1


def mid2d(kp, sc, a, b):
    ok = (sc[:, a] > SCORE_MIN) & (sc[:, b] > SCORE_MIN)
    return 0.5 * (kp[:, a] + kp[:, b]), ok


def nan_median_filter(x, w):
    """(T,…) 시간 중앙값 필터, NaN 무시. 창 w (홀수)."""
    T = len(x); h = w // 2; pad = np.full((h,) + x.shape[1:], np.nan)
    xp = np.concatenate([pad, x, pad]); st = np.stack([xp[i:i + T] for i in range(w)], 0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(st, 0), np.isfinite(st).all(-1).sum(0) if st.ndim > 2 else np.isfinite(st).sum(0)


# ---------------------------------------------------------------- 축 배치 (행렬식 +1, 24 가지)
def perms24():
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, perm[i]] = sg[i]
            if np.linalg.det(R) > 0:
                out.append((R, list(perm), list(sg)))
    return out


def perms_reflect24():
    """진단 전용: 행렬식 −1 배치 24 가지 (채택하지 않음)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[i, perm[i]] = sg[i]
            if np.linalg.det(R) < 0:
                out.append((R, list(perm), list(sg)))
    return out


def rotz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot2(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------- BEV 3-DoF 가중 강건 정합
def bev_fit(a, b, Sb, Sa_fn, psi0=0.0, tau0=None, huber_k=2.0, iters=30, w_ext=None):
    """a: (N,2) 레이더 중심 (R0 적용 뒤 방 xy, 회전 전). b: (N,2) 스테레오 사람 xy. Sb: (N,2,2) 스테레오 공분산.
    Sa_fn(psi, a) → (N,2,2) 레이더 중심 공분산 (방 좌표). 최소화: Σ ρ_Huber( rᵀ (Sb+Sa)⁻¹ r ), r = Rot(ψ) a + τ − b.
    반환 dict(psi, tau, cov(3×3, Gauss–Newton), w, m (마할라노비스 제곱), r)."""
    psi = float(psi0)
    tau = np.median(b - a @ rot2(psi).T, 0) if tau0 is None else np.asarray(tau0, float).copy()
    N = len(a); w = np.ones(N) if w_ext is None else w_ext.copy()
    for it in range(iters):
        Rm = rot2(psi); Ra = a @ Rm.T; r = Ra + tau - b
        W = np.linalg.inv(Sb + Sa_fn(psi, a))
        m = np.einsum("ni,nij,nj->n", r, W, r); sm = np.sqrt(np.maximum(m, 1e-12))
        wh = np.where(sm <= huber_k, 1.0, huber_k / sm)
        wt = wh * (1.0 if w_ext is None else w_ext)
        dR = np.stack([-Ra[:, 1], Ra[:, 0]], 1)                          # ∂(Rot a)/∂ψ
        J = np.zeros((N, 2, 3)); J[:, :, 0] = dR; J[:, 0, 1] = 1; J[:, 1, 2] = 1
        H = np.einsum("n,nia,nij,njb->ab", wt, J, W, J); g = np.einsum("n,nia,nij,nj->a", wt, J, W, r)
        step = -np.linalg.solve(H + 1e-9 * np.eye(3), g)
        psi += step[0]; tau = tau + step[1:]
        if np.abs(step).max() < 1e-7:
            break
    Rm = rot2(psi); r = a @ Rm.T + tau - b; W = np.linalg.inv(Sb + Sa_fn(psi, a)); m = np.einsum("ni,nij,nj->n", r, W, r)
    return dict(psi=psi, tau=tau, cov=np.linalg.inv(H), w=wt, m=m, r=r, n=N, it=it + 1)


def lag_xcorr(x, y, maxlag):
    """정규화 상호상관. x(t) 와 y(t+L) 의 상관을 L ∈ [−maxlag, maxlag] 에서 계산 (NaN 무시).
    L > 0: y 가 x 보다 늦게 같은 변화를 보임 (y 지연)."""
    out = []
    for L in range(-maxlag, maxlag + 1):
        if L >= 0:
            xa, ya = x[:len(x) - L], y[L:]
        else:
            xa, ya = x[-L:], y[:len(y) + L]
        ok = np.isfinite(xa) & np.isfinite(ya)
        if ok.sum() < 50:
            out.append(np.nan); continue
        u, v = xa[ok] - xa[ok].mean(), ya[ok] - ya[ok].mean()
        out.append(float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum() + 1e-12)))
    c = np.array(out); L = np.arange(-maxlag, maxlag + 1); k = int(np.nanargmax(c))
    sub = 0.0
    if 0 < k < len(c) - 1 and np.isfinite(c[k - 1]) and np.isfinite(c[k + 1]):
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        sub = float(0.5 * (c[k - 1] - c[k + 1]) / den) if den < 0 else 0.0
    return dict(lags=L.tolist(), corr=[None if not np.isfinite(v) else round(v, 4) for v in c], best=int(L[k]), best_sub=float(L[k] + sub), peak=float(c[k]))


def jdump(obj, path):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=conv)


# ---------------------------------------------------------------- 정합 데이터 묶음 + 적합
def smooth3(x, w=3):
    k = np.ones(w) / w; y = np.convolve(np.nan_to_num(x), k, "same"); n = np.convolve(np.isfinite(x).astype(float), k, "same")
    y = y / np.maximum(n, 1e-9); y[n < 0.5] = np.nan
    return y


def load_subject(s, variant="dyn"):
    st = np.load(f"{WORK}/stereo_S{s:02d}.npz", allow_pickle=False); rt = np.load(f"{WORK}/radar_track_S{s:02d}.npz", allow_pickle=False)
    a = rt[variant].astype(float); a_raw = rt[f"{variant}_raw"].astype(float)
    a[~rt["uniq"]] = np.nan
    return dict(s=s, T=int(st["T"]), fps=float(st["fps"]), b3=st["torso"].astype(float), Sb3=st["torso_cov"].astype(float), hip=st["hip"].astype(float),
                a3=a, a3_raw=a_raw, img=st["img_speed"].astype(float), npts=np.where(rt["uniq"], rt["n_gate"], np.nan).astype(float),
                dopE=np.where(rt["uniq"], rt["dop_energy"], np.nan).astype(float), uniq=rt["uniq"], dup=rt["dup"])


def radar_hf_cov(D):
    """레이더 궤적 고주파 잔차 공분산 (센서 좌표, 축 배치 무관): 원 중심 − 창 5 중앙값 평활, 로버스트 (MCD 대신 MAD 상관)."""
    R = []
    for d in D.values():
        x, nv = nan_median_filter(d["a3_raw"], 5); r = d["a3_raw"] - x; ok = np.isfinite(r).all(1) & d["uniq"]; R.append(r[ok])
    R = np.concatenate(R); med = np.median(R, 0); Z = R - med
    s = 1.4826 * np.median(np.abs(Z), 0); keep = (np.abs(Z) < 4 * s).all(1)
    C = np.cov(Z[keep].T) / 1.16
    return C


def pairs(D, lags, R0, subj=None, frames=None):
    """스테레오 프레임 t ↔ 레이더 프레임 t + L_s. 반환: a (N,3) R0 적용 방 좌표 (회전 ψ 전), b (N,2), Sb (N,2,2), sid, t."""
    A, B, S, I, TT, B3 = [], [], [], [], [], []
    for s, d in D.items():
        if subj is not None and s not in subj:
            continue
        L = int(lags.get(s, 0)); T = d["T"]; t = np.arange(T); tr = t + L; ok = (tr >= 0) & (tr < T)
        t, tr = t[ok], tr[ok]
        a = d["a3"][tr]; b = d["b3"][t]; Sb = d["Sb3"][t][:, :2, :2]
        v = np.isfinite(a).all(1) & np.isfinite(b).all(1) & np.isfinite(Sb).all((1, 2))
        if frames is not None:
            v &= frames(s, t, T)
        A.append(a[v] @ R0.T); B.append(b[v, :2]); S.append(Sb[v]); I.append(np.full(v.sum(), s)); TT.append(t[v]); B3.append(b[v])
    return dict(a=np.concatenate(A), b=np.concatenate(B), Sb=np.concatenate(S), sid=np.concatenate(I), t=np.concatenate(TT), b3=np.concatenate(B3))


def make_Sa(Ca_room_xy):
    def f(psi, a):
        Rm = rot2(psi); C = Rm @ Ca_room_xy @ Rm.T
        return np.broadcast_to(C, (len(a), 2, 2))
    return f


def fit_extra(P, Ca_sensor, R0, psi0=0.0, tau0=None, n_outer=4, huber_k=2.0, extra0=None):
    """BEV 3-DoF 적합 + 모형 밖 잔차 분산 Σ_extra = diag(e_x², e_y²) (방 x=카메라 깊이, y=옆) 을 잔차에서 추정 (정답 없음).
    총 공분산 = Σ_stereo(프레임별, 이방성) + R Σ_radar R ᵀ + Σ_extra."""
    Ca_room = (R0 @ Ca_sensor @ R0.T)[:2, :2]
    ex = np.array([0.15, 0.10]) if extra0 is None else np.asarray(extra0, float)
    psi, tau = psi0, tau0
    for k in range(n_outer):
        Cx = np.diag(ex ** 2)
        Sa = make_Sa(Ca_room)
        fn = lambda p, a, Sa=Sa, Cx=Cx: Sa(p, a) + Cx
        res = bev_fit(P["a"][:, :2], P["b"], P["Sb"], fn, psi0=psi, tau0=tau, huber_k=huber_k)
        psi, tau = res["psi"], res["tau"]
        r = res["r"]; med = np.median(r, 0); mad2 = (1.4826 * np.median(np.abs(r - med), 0)) ** 2
        base = np.median(np.einsum("nii->ni", P["Sb"]), 0) + np.diag(Rm := rot2(psi) @ Ca_room @ rot2(psi).T)
        ex = np.sqrt(np.maximum(mad2 - base, 1e-6))
    res["extra_sigma"] = ex; res["Ca_room_xy"] = Ca_room
    return res


# ---------------------------------------------------------------- 레이더 사람 궤적 (센서 좌표)
TRACK_PAR = dict(STATIC_OCC=0.30, VOX=0.10, GATE0=1.5, GATE1=1.0, MIN_PTS=3, DOP_STEP=0.356, TQ=0.25)


def tmean(p, q=0.25):
    lo, hi = np.quantile(p, q, axis=0), np.quantile(p, 1 - q, axis=0)
    m = (p >= lo) & (p <= hi)
    return np.array([p[m[:, k], k].mean() if m[:, k].any() else np.median(p[:, k]) for k in range(p.shape[1])])


def attach_frames(R, T, f0):
    """CSV dict → 팩 프레임별 점 목록 (v0.3 과 같은 대응 t = Camera Frame − frame0) + 레이더 프레임 번호 + 중복 표시."""
    pts = [None] * T; rf = np.full(T, -1, np.int64)
    for c, (fnum, p) in R.items():
        t = c - f0
        if 0 <= t < T:
            pts[t] = p; rf[t] = fnum
    has = rf >= 0; dup = np.zeros(T, bool); prev = -2
    for t in range(T):
        if has[t]:
            dup[t] = rf[t] == prev; prev = rf[t]
    return pts, rf, has, dup


def build_track(pts, uniq, par=TRACK_PAR):
    """레이더 사람 중심 궤적 (센서 좌표). 회전과 무관한 3D 게이트 + 좌표별 절사 평균 (TI 방위 양자화 대응)."""
    T = len(pts); VOX = par["VOX"]
    idx = np.flatnonzero(uniq)
    A = np.concatenate([pts[t] for t in idx])
    occ = {}
    for t in idx:
        for key in set(map(tuple, np.floor(pts[t][:, :3] / VOX).astype(np.int64))):
            occ[key] = occ.get(key, 0) + 1
    nu = len(idx); stat_keys = {k for k, c in occ.items() if c / nu >= par["STATIC_OCC"]}
    max_occ = max(occ.values()) / nu
    c0 = np.median(A[:, :3], 0)
    coarse = np.full((T, 3), np.nan); keep = [None] * T
    for t in idx:
        p = pts[t]
        st_m = np.array([tuple(x) in stat_keys for x in np.floor(p[:, :3] / VOX).astype(np.int64)], bool) if stat_keys else np.zeros(len(p), bool)
        k = (np.linalg.norm(p[:, :3] - c0, axis=1) <= par["GATE0"]) & ~st_m; keep[t] = k
        if k.sum() >= par["MIN_PTS"]:
            coarse[t] = tmean(p[k, :3], par["TQ"])
    coarse_s, _ = nan_median_filter(coarse, 9)
    raw = {m: np.full((T, 3), np.nan) for m in ["dyn", "all"]}; npt = {m: np.zeros(T, int) for m in ["dyn", "all"]}
    dop_e = np.full(T, np.nan); n_g = np.zeros(T, int)
    for t in idx:
        if not np.isfinite(coarse_s[t]).all():
            continue
        p = pts[t]; k = keep[t] & (np.linalg.norm(p[:, :3] - coarse_s[t], axis=1) <= par["GATE1"]); n_g[t] = int(k.sum())
        if k.sum():
            dop_e[t] = float(np.abs(p[k, 3]).mean())
        for m, sel in [("all", k), ("dyn", k & (np.abs(p[:, 3]) >= 0.5 * par["DOP_STEP"]))]:
            npt[m][t] = int(sel.sum())
            if sel.sum() >= par["MIN_PTS"]:
                raw[m][t] = tmean(p[sel, :3], par["TQ"])
    sm = {}
    for m in raw:
        x, nv = nan_median_filter(raw[m], 5); x[nv < 3] = np.nan; sm[m] = x
    return dict(dyn=sm["dyn"], all=sm["all"], dyn_raw=raw["dyn"], all_raw=raw["all"], n_dyn=npt["dyn"], n_all=npt["all"], dop_energy=dop_e, n_gate=n_g,
                c0=c0, max_occ=max_occ, n_static_vox=len(stat_keys), n_pts=len(A), frac_far=float((np.linalg.norm(A[:, :3] - c0, axis=1) > par["GATE0"]).mean()))


def inv2(S):
    a, b, c, d = S[:, 0, 0], S[:, 0, 1], S[:, 1, 0], S[:, 1, 1]; det = a * d - b * c
    return np.stack([np.stack([d, -b], -1), np.stack([-c, a], -1)], -2) / det[:, None, None]


def bev_fit_fast(a, b, Sb, Cconst, psi0=0.0, tau0=None, huber_k=2.0, iters=30, Ca_room=None):
    """bev_fit 과 같은 목적함수 (2×2 해석적 역행렬). 총 공분산 = Sb(프레임별) + Rot(ψ) Ca_room Rot(ψ)ᵀ + Cconst."""
    psi = float(psi0)
    tau = np.median(b - a @ rot2(psi).T, 0) if tau0 is None else np.asarray(tau0, float).copy()
    Ca_room = np.zeros((2, 2)) if Ca_room is None else Ca_room
    for it in range(iters):
        Rm = rot2(psi); Ra = a @ Rm.T; r = Ra + tau - b
        W = inv2(Sb + (Rm @ Ca_room @ Rm.T + Cconst)[None])
        Wr = np.einsum("nij,nj->ni", W, r); m = (r * Wr).sum(1); sm = np.sqrt(np.maximum(m, 1e-12))
        wt = np.where(sm <= huber_k, 1.0, huber_k / sm)
        dR = np.stack([-Ra[:, 1], Ra[:, 0]], 1)
        WdR = np.einsum("nij,nj->ni", W, dR)
        H = np.zeros((3, 3)); g = np.zeros(3)
        H[0, 0] = (wt * (dR * WdR).sum(1)).sum(); H[0, 1:] = (wt[:, None] * WdR).sum(0); H[1:, 0] = H[0, 1:]
        H[1:, 1:] = (wt[:, None, None] * W).sum(0)
        g[0] = (wt * (dR * Wr).sum(1)).sum(); g[1:] = (wt[:, None] * Wr).sum(0)
        step = -np.linalg.solve(H + 1e-9 * np.eye(3), g)
        psi += step[0]; tau = tau + step[1:]
        if np.abs(step).max() < 1e-7:
            break
    Rm = rot2(psi); r = a @ Rm.T + tau - b; W = inv2(Sb + (Rm @ Ca_room @ Rm.T + Cconst)[None]); m = np.einsum("ni,nij,nj->n", r, W, r)
    return dict(psi=psi, tau=tau, cov=np.linalg.inv(H), w=wt, m=m, r=r, n=len(a), it=it + 1)


def fit_extra_fast(P, Ca_sensor, R0, psi0=0.0, tau0=None, n_outer=4, huber_k=2.0, extra0=None, sub=None):
    """fit_extra 와 같음 (빠른 판). sub: 프레임 부분 표본 인덱스."""
    a, b, Sb = P["a"][:, :2], P["b"], P["Sb"]
    if sub is not None:
        a, b, Sb = a[sub], b[sub], Sb[sub]
    Ca_room = (R0 @ Ca_sensor @ R0.T)[:2, :2]
    ex = np.array([0.15, 0.10]) if extra0 is None else np.asarray(extra0, float)
    psi, tau = psi0, tau0
    for k in range(n_outer):
        res = bev_fit_fast(a, b, Sb, np.diag(ex ** 2), psi0=psi, tau0=tau, huber_k=huber_k, Ca_room=Ca_room)
        psi, tau = res["psi"], res["tau"]
        if k < n_outer - 1 or extra0 is None:
            r = res["r"]; med = np.median(r, 0); mad2 = (1.4826 * np.median(np.abs(r - med), 0)) ** 2
            base = np.median(np.einsum("nii->ni", Sb), 0) + np.diag(rot2(psi) @ Ca_room @ rot2(psi).T)
            ex_new = np.sqrt(np.maximum(mad2 - base, 1e-6))
            if k < n_outer - 1:
                ex = ex_new
    res["extra_sigma"] = ex; res["Ca_room_xy"] = Ca_room
    return res
