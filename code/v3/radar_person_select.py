"""캐시 v3 — 정답 없는 레이더 사람 점 고르기 (센서 무관 규칙, 파라미터는 물리 단위).

입력은 방 좌표(바닥 z = 0, z 위) 점, 도플러(m/s), 세기, 프레임 오프셋, 레이더 수평 위치, 도플러 간격, 프레임 속도뿐이다.
정답 관절·포즈·라벨은 이 모듈이 받지도 않는다 (빌드 스크립트가 허용 키만 읽어 넘김).

절차 (녹화 하나 단위):
  0. 유효·관심 영역: |v| ≤ V_MAX (사람 팔다리 ≤ 5 m/s 보다 큰 값은 접힘·잡음), Z_LO ≤ z ≤ Z_HI (사람 높이대),
     수평 거리 R_MIN ≤ r_xy ≤ R_MAX (실내 사람 센싱 작업 거리).
  1. 정적 배경 (녹화별 복셀 점유): VOX 복셀마다 점유율 occ = 점이 있던 프레임 비율,
     동적 비율 dyn = 점유 프레임 중 움직이는 점(|v| > V_TH)이 있던 비율. occ ≥ OCC_MIN 이고 dyn ≤ DYN_MAX 인 복셀 = 배경.
     (제자리 동작하는 사람 복셀은 점유율이 높아도 동적 비율이 높아 배경이 되지 않음; 단순 점유율 모형은 제자리 사람을 지움)
  2. 이동 씨앗: |v| > V_TH, V_TH = max(V_MIN, 0.5·Δv) (도플러 간격 Δv 가 V_MIN 보다 잘면 영 근처 칸들은 정적 취급:
     Arbe Δv 0.071 → |v| ≥ 2칸 = 0.142 m/s, TI Δv 0.36–0.6 → 0 아닌 칸 전부).
  3. 군집: 프레임마다 씨앗을 바닥면(xy)에서 DBSCAN(EPS, MIN_PTS).
  4. 다중 트랙 (감쇠 등속 칼만, 상태 x y vx vy): 예측 → 게이트(χ²₂ 99 %, 반경 GATE_MIN–GATE_MAX) 안 군집을 거리순 탐욕 연결 →
     측정 = 군집 중심 R_SEL 안 배경 아닌 모든 점(도플러 0 포함)의 xy 중앙값 (2회 재중심).
     연결 못 한 트랙은 예측 R_SEL 안 배경 아닌 점이 N_STAT 이상이면 정지 측정(큰 σ), 아니면 예측만(coast); coast > T_LOST 초면 종료.
     연결 안 된 군집(씨앗 ≥ N_INIT, 기존 트랙에서 R_SPAWN 밖)은 새 트랙. 두 트랙이 R_MERGE 안이면 누적 지지가 작은 쪽 종료.
     오프라인: 트랙마다 RTS 평활, 트랙 시작 전은 정지 측정으로 뒤로 연장 (사람이 처음에 가만히 있던 구간).
  5. 사람 트랙 고르기: 프레임마다 덮는 트랙들의 지지 = ±SUP_WIN 초 동안 R_SEL 안 배경 아닌 점 수의 평균.
     가장 큰 트랙을 고르되, 현재 트랙이 최대의 1/SWITCH_RATIO 이상이면 유지 (이력 현상).
     (사람은 몸 전체가 점을 내므로 지지가 크고, 선풍기·흔들리는 물체·벽 반사 유령은 작다)
  6. 고르기: 고른 트랙 위치에서 수평 R_SEL 안, Z_LO–Z_HI 안의 배경 아닌 점 (도플러 0 포함). 세기 상위 CAP 개로 자름.
"""
import numpy as np
from scipy.spatial import cKDTree

PARAMS = dict(
    V_MAX=8.0,        # m/s. 사람 팔다리 micro-Doppler 최대 약 5 m/s (Deep 2020) + 여유. 그보다 큰 |v| 점은 버림
    Z_LO=-0.10,       # m, 바닥 기준. 바닥 아래(바닥 거울 다중경로) 제외
    Z_HI=2.30,        # m, 바닥 기준. 사람 키 + 팔 올림 여유
    R_MIN=0.5,        # m, 레이더 수평 거리 하한 (근거리 누설)
    R_MAX=10.0,       # m, 실내 사람 센싱 작업 거리 상한
    VOX=0.20,         # m, 배경 복셀 (Arbe 거리 해상 0.4 m 의 절반, 5 m 에서 2° 각 셀 ≈ 0.17 m)
    OCC_MIN=0.05,     # 녹화 프레임의 5 % 이상 점유
    DYN_MAX=0.005,    # 점유 프레임 중 움직이는 점이 있던 비율 ≤ 0.5 % (정적 반사체는 움직임 0; 0.02 는 거의 가만한 사람 복셀을 지움 —
                      #  11-24 정답 없는 진단: 트랙에서 1.5 m 밖 클러터 복셀의 92–99.8 % 가 dyn = 0, 트랙 근처 사람 복셀은 0.005–0.3 연속)
    V_MIN=0.10,       # m/s, 움직임 하한 (몸통 흔들림 수준)
    EPS=0.50,         # m, 씨앗 DBSCAN 반경 (Durmaz 2025 의 0.5 m)
    MIN_PTS=3,        # DBSCAN 최소 점 (Durmaz 2025)
    N_INIT=5,         # 트랙 시작 군집 최소 씨앗 수
    SIGMA_A=1.0,      # m/s², 사람 수평 가속 σ (걷기 시작·정지)
    TAU_V=1.0,        # s, 속도 감쇠 시간 상수 (멈춘 사람으로 흘러가지 않게)
    SIGMA_MOV=0.15,   # m, 이동 군집 측정 σ (몸통 반폭 수준)
    SIGMA_STAT=0.30,  # m, 정지 측정 σ
    GATE_CHI2=9.21,   # χ²₂ 99 %
    GATE_MIN=0.50,    # m, 게이트 최소 반경 (사람 반폭 + 거리 해상)
    GATE_MAX=1.50,    # m, 게이트 최대 반경 (2 m/s × 0.75 s)
    N_STAT=10,        # 정지 측정에 필요한 R_SEL 안 배경 아닌 점 수
    T_LOST=2.0,       # s, coast 허용 시간
    R_SPAWN=1.0,      # m, 기존 트랙에서 이 거리 안 군집은 새 트랙을 만들지 않음 (한 사람의 팔·다리 군집)
    R_MERGE=0.5,      # m, 두 트랙이 이 거리 안이면 하나로
    SUP_WIN=1.0,      # s, 트랙 지지 평균 창 (±)
    SWITCH_RATIO=1.5, # 다른 트랙 지지가 현재의 1.5 배를 넘을 때만 바꿈
    R_SEL=0.70,       # m, 사람 수평 반경 (어깨 반폭 0.25 + 팔 0.45; 정답 없이 정한 고정값)
    CAP=512,          # 프레임당 최대 점 수 (v2 신탁과 같은 상한), 세기 상위
)


def dbscan_xy(P, eps, min_pts):
    """작은 점 집합용 DBSCAN (xy). 반환 라벨, −1 = 잡음."""
    n = len(P)
    lab = -np.ones(n, int)
    if n == 0:
        return lab
    tree = cKDTree(P); nb = tree.query_ball_point(P, eps); core = np.array([len(x) >= min_pts for x in nb])
    c = 0
    for i in np.where(core)[0]:
        if lab[i] >= 0:
            continue
        lab[i] = c; st = [i]
        while st:
            j = st.pop()
            if not core[j]:
                continue
            for k in nb[j]:
                if lab[k] < 0:
                    lab[k] = c; st.append(k)
        c += 1
    return lab


def background_mask(q, v, fi, T, roi, mov, p):
    """정적 배경 점 표시 (녹화 단위 복셀 점유 + 동적 비율). 반환 (점별 bool, 복셀 통계 dict). v 는 쓰지 않음 (mov 로 받음)."""
    vox = np.floor(q[roi] / p["VOX"]).astype(np.int64) + 2048
    key = (vox[:, 0] << 24) | (vox[:, 1] << 12) | vox[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    f = fi[roi].astype(np.int64); m = mov[roi]
    occ = np.zeros(len(uk)); dyn = np.zeros(len(uk))
    kf = np.unique(inv.astype(np.int64) * (T + 1) + f); np.add.at(occ, kf // (T + 1), 1)
    km = np.unique(inv[m].astype(np.int64) * (T + 1) + f[m]); np.add.at(dyn, km // (T + 1), 1)
    dr = dyn / np.maximum(occ, 1); occ = occ / T
    bgv = (occ >= p["OCC_MIN"]) & (dr <= p["DYN_MAX"])
    bg = np.zeros(len(q), bool); bg[np.where(roi)[0]] = bgv[inv]
    cent = (np.stack([(uk >> 24) & 4095, (uk >> 12) & 4095, uk & 4095], 1) - 2048 + 0.5) * p["VOX"]
    return bg, dict(n_vox=int(len(uk)), n_bg_vox=int(bgv.sum()), bg_centres=cent[bgv].astype(np.float32),
                    roi_pts_bg_frac=float(bgv[inv].mean()) if len(inv) else float("nan"))


def _center(Q, c, r, it=2):
    """c 둘레 수평 r 안 점들의 xy 중앙값으로 재중심 (it 회). 반환 (중심, 점 수)."""
    n = 0
    for _ in range(it):
        d = np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1]); s = d <= r; n = int(s.sum())
        if n == 0:
            break
        c = np.median(Q[s, :2], 0)
    return c, n


def _mats(dt, p):
    a = np.exp(-dt / p["TAU_V"])
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, a, 0], [0, 0, 0, a]], float)
    q1 = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * p["SIGMA_A"] ** 2
    Q = np.zeros((4, 4)); Q[np.ix_([0, 2], [0, 2])] = q1; Q[np.ix_([1, 3], [1, 3])] = q1
    H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1
    return F, Q, H


def _update(x, P, z, R, H):
    S = H @ P @ H.T + np.eye(2) * R; K = P @ H.T @ np.linalg.inv(S)
    return x + K @ (z - x[:2]), (np.eye(4) - K @ H) @ P


def _gated(x, P, c, H, p):
    S = H @ P @ H.T + np.eye(2) * p["SIGMA_MOV"] ** 2; e = c - x[:2]; d2 = float(e @ np.linalg.solve(S, e)); dist = float(np.hypot(*e))
    return ((d2 < p["GATE_CHI2"]) or dist < p["GATE_MIN"]) and dist < p["GATE_MAX"], dist


def _support(Q, c, r):
    return int((np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1]) <= r).sum())


def run_tracks(frames, dt, p):
    """다중 트랙 앞 방향 통과. frames[t] = (clusters [(c, n_seed)], cand_xyz). 반환 트랙 목록
    (dict: t0, t1, xf, Pf, xp, Pp (프레임별), st (2 이동, 1 정지, 0 coast), sup (R_SEL 안 배경 아닌 점 수))."""
    F, Q, H = _mats(dt, p); T = len(frames); done = []; act = []
    lost_n = int(np.ceil(p["T_LOST"] / dt))
    for t in range(T):
        clusters, cand = frames[t]
        for tr in act:
            x = F @ tr["x"]; P = F @ tr["P"] @ F.T + Q; tr["x"], tr["P"] = x, P; tr["xp"].append(x.copy()); tr["Pp"].append(P.copy())
        pairs = []
        for j, tr in enumerate(act):
            for k, (c, n) in enumerate(clusters):
                ok, dist = _gated(tr["x"], tr["P"], c, H, p)
                if ok:
                    pairs.append((dist, j, k))
        pairs.sort(); aj = {}; used = set()
        for dist, j, k in pairs:
            if j in aj or k in used:
                continue
            aj[j] = k; used.add(k)
        for j, tr in enumerate(act):
            z = None
            if j in aj:
                z, _ = _center(cand, clusters[aj[j]][0], p["R_SEL"]); R = p["SIGMA_MOV"] ** 2; s_ = 2
            else:
                c_, n_ = _center(cand, tr["x"][:2], p["R_SEL"], it=1)
                if n_ >= p["N_STAT"]:
                    z = c_; R = p["SIGMA_STAT"] ** 2; s_ = 1
            if z is not None:
                tr["x"], tr["P"] = _update(tr["x"], tr["P"], z, R, H); tr["last"] = t
            else:
                s_ = 0
            tr["xf"].append(tr["x"].copy()); tr["Pf"].append(tr["P"].copy()); tr["st"].append(s_); tr["sup"].append(_support(cand, tr["x"][:2], p["R_SEL"]))
        # 종료: coast 초과 (뒤쪽 coast 프레임은 마지막 측정까지 잘라냄)
        keep = []
        for tr in act:
            if t - tr["last"] > lost_n:
                cut = tr["last"] - tr["t0"] + 1
                for k_ in ("xf", "Pf", "xp", "Pp", "st", "sup"):
                    tr[k_] = tr[k_][:cut]
                done.append(tr)
            else:
                keep.append(tr)
        act = keep
        # 합치기: 가까운 두 트랙 중 누적 지지가 작은 쪽 종료 (그 프레임까지)
        act.sort(key=lambda tr: -sum(tr["sup"]))
        keep = []
        for tr in act:
            if any(np.hypot(*(tr["x"][:2] - o["x"][:2])) < p["R_MERGE"] for o in keep):
                tr["merged"] = True; done.append(tr)
            else:
                keep.append(tr)
        act = keep
        # 새 트랙
        for k, (c, n) in enumerate(clusters):
            if k in used or n < p["N_INIT"]:
                continue
            if any(np.hypot(*(c - tr["x"][:2])) < p["R_SPAWN"] for tr in act):
                continue
            c0, _ = _center(cand, c, p["R_SEL"]); x = np.r_[c0, 0.0, 0.0]; P = np.diag([p["SIGMA_MOV"] ** 2] * 2 + [1.0, 1.0])
            act.append(dict(t0=t, x=x, P=P, last=t, xf=[x.copy()], Pf=[P.copy()], xp=[x.copy()], Pp=[P.copy()], st=[2], sup=[_support(cand, c0, p["R_SEL"])]))
    done += act
    for tr in done:
        tr["t1"] = tr["t0"] + len(tr["st"])          # 끝 (배타)
    return [tr for tr in done if tr["t1"] > tr["t0"]]


def rts(tr, dt, p):
    """트랙 하나 RTS 평활. 반환 xy (n,2), σ (n,)."""
    F, _, _ = _mats(dt, p); n = len(tr["xf"]); xs = [None] * n; Ps = [None] * n
    xs[-1] = tr["xf"][-1]; Ps[-1] = tr["Pf"][-1]
    for i in range(n - 2, -1, -1):
        C = tr["Pf"][i] @ F.T @ np.linalg.inv(tr["Pp"][i + 1])
        xs[i] = tr["xf"][i] + C @ (xs[i + 1] - tr["xp"][i + 1]); Ps[i] = tr["Pf"][i] + C @ (Ps[i + 1] - tr["Pp"][i + 1]) @ C.T
    return np.array([x[:2] for x in xs]), np.array([np.sqrt(np.trace(P[:2, :2]) / 2) for P in Ps])


def extend_back(tr, xy0, frames, dt, p):
    """트랙 시작 전 프레임으로 뒤로 연장 (정지 측정 또는 게이트 안 군집). 반환 (프레임 목록, xy, 상태)."""
    F, Q, H = _mats(dt, p); x = np.r_[xy0, 0.0, 0.0]; P = np.diag([p["SIGMA_MOV"] ** 2] * 2 + [0.25, 0.25]); last = tr["t0"]
    ts, xy, st = [], [], []; lost_n = int(np.ceil(p["T_LOST"] / dt))
    for t in range(tr["t0"] - 1, -1, -1):
        clusters, cand = frames[t]; x = F @ x; P = F @ P @ F.T + Q; z = None
        best = None
        for c, n in clusters:
            ok, dist = _gated(x, P, c, H, p)
            if ok and (best is None or dist < best[1]):
                best = (c, dist)
        if best is not None:
            z, _ = _center(cand, best[0], p["R_SEL"]); R = p["SIGMA_MOV"] ** 2; s_ = 2
        else:
            c_, n_ = _center(cand, x[:2], p["R_SEL"], it=1)
            if n_ >= p["N_STAT"]:
                z = c_; R = p["SIGMA_STAT"] ** 2; s_ = 1
        if z is None:
            if last - t > lost_n:
                break
            s_ = 0
        else:
            x, P = _update(x, P, z, R, H); last = t
        ts.append(t); xy.append(x[:2].copy()); st.append(s_)
    k = sum(1 for t_ in ts if t_ >= last)                                   # 마지막 측정보다 앞쪽(더 이른) coast 는 잘라냄
    return ts[:k], np.array(xy[:k]).reshape(-1, 2), st[:k]


def choose_person(tracks, frames, T, dt, p):
    """프레임별 사람 트랙 선택 (지지 최대 + 이력 현상). 반환 xy (T,2), σ (T,), status (T,), track id (T,)."""
    W = int(round(p["SUP_WIN"] / dt)); cov = []
    for i, tr in enumerate(tracks):
        xy, sig = rts(tr, dt, p); ts = list(range(tr["t0"], tr["t1"])); st = list(tr["st"])
        bt, bxy, bst = ([], None, []) if tr.get("merged") else extend_back(tr, xy[0], frames, dt, p)   # 합쳐진 중복 트랙은 연장 안 함
        if bt:
            ts = bt[::-1] + ts; xy = np.r_[bxy[::-1], xy]; sig = np.r_[np.full(len(bt), np.nan), sig]; st = bst[::-1] + st
        sup = np.array([_support(frames[t][1], xy[j], p["R_SEL"]) for j, t in enumerate(ts)], float)
        cs = np.r_[0, np.cumsum(sup)]; n = len(sup); lo = np.clip(np.arange(n) - W, 0, n); hi = np.clip(np.arange(n) + W + 1, 0, n)
        cov.append(dict(t0=ts[0], xy=xy, sig=sig, st=np.array(st), msup=(cs[hi] - cs[lo]) / (hi - lo), sup=sup))
    out = np.full((T, 2), np.nan); sig = np.full(T, np.nan); stat = -np.ones(T, int); tid = -np.ones(T, int); cur = -1
    for t in range(T):
        cand = [(c["msup"][t - c["t0"]], i) for i, c in enumerate(cov) if c["t0"] <= t < c["t0"] + len(c["msup"])]
        if not cand:
            cur = -1; continue
        best = max(cand)
        cs = [s for s, i in cand if i == cur]
        if not (cs and cs[0] * p["SWITCH_RATIO"] >= best[0]):
            cur = best[1]
        c = cov[cur]; j = t - c["t0"]; out[t] = c["xy"][j]; sig[t] = c["sig"][j]; stat[t] = c["st"][j]; tid[t] = cur
    return out, sig, stat, tid, cov


def select_sequence(q, v, inten, off, s_xy, doppler_step, fps, p=PARAMS):
    """한 녹화의 사람 점 고르기. q (N,3) 방 좌표, v (N,) m/s, inten (N,), off (T+1,), s_xy 레이더 수평 위치.
    반환 dict: sel (프레임별 원 점 인덱스 배열 목록), trk_xy (T,2), trk_sig (T,), status (T,), trk_id (T,), 진단."""
    T = len(off) - 1; dt = 1.0 / fps
    fi = np.repeat(np.arange(T), np.diff(off))
    ok = np.isfinite(q).all(1) & np.isfinite(v) & (np.abs(v) <= p["V_MAX"])
    rxy = np.hypot(q[:, 0] - s_xy[0], q[:, 1] - s_xy[1])
    roi = ok & (q[:, 2] >= p["Z_LO"]) & (q[:, 2] <= p["Z_HI"]) & (rxy >= p["R_MIN"]) & (rxy <= p["R_MAX"])
    v_th = max(p["V_MIN"], 0.5 * doppler_step)
    mov = roi & (np.abs(v) > v_th)
    bg, bgs = background_mask(q, v, fi, T, roi, mov, p)
    cand = roi & ~bg; seed = cand & mov
    frames = []; ncl = np.zeros(T, int)
    for t in range(T):
        a, b = off[t], off[t + 1]
        Q = q[a:b][cand[a:b]]
        S = q[a:b][seed[a:b], :2]; lab = dbscan_xy(S, p["EPS"], p["MIN_PTS"])
        cl = [(np.median(S[lab == k], 0), int((lab == k).sum())) for k in range(lab.max() + 1)] if len(lab) else []
        frames.append((cl, Q)); ncl[t] = len(cl)
    tracks = run_tracks(frames, dt, p)
    xy, sig, st, tid, cov = choose_person(tracks, frames, T, dt, p)
    sel = []
    for t in range(T):
        a, b = off[t], off[t + 1]
        if not np.isfinite(xy[t, 0]):
            sel.append(np.zeros(0, np.int64)); continue
        idx = np.arange(a, b)[cand[a:b]]
        d = np.hypot(q[idx, 0] - xy[t, 0], q[idx, 1] - xy[t, 1]); idx = idx[d <= p["R_SEL"]]
        if len(idx) > p["CAP"]:
            idx = idx[np.argsort(-inten[idx], kind="stable")[:p["CAP"]]]
        sel.append(np.sort(idx))
    # 트랙 요약 (지지 큰 순)
    tsum = sorted([dict(t0=int(c["t0"]), n=int(len(c["sup"])), sup_mean=round(float(c["sup"].mean()), 1),
                        xy_median=[round(float(x), 2) for x in np.median(c["xy"], 0)], xy_std=[round(float(x), 3) for x in c["xy"].std(0)],
                        chosen_frames=int((tid == i).sum())) for i, c in enumerate(cov)], key=lambda d: -d["sup_mean"] * d["n"])
    diag = dict(v_th=v_th, roi_pts_per_frame=float(np.median(np.bincount(fi[roi], minlength=T))),
                cand_pts_per_frame=float(np.median(np.bincount(fi[cand], minlength=T))),
                seed_pts_per_frame=float(np.median(np.bincount(fi[seed], minlength=T))),
                n_clusters_median=float(np.median(ncl)), bg=bgs, n_tracks=len(tracks), tracks=tsum[:8],
                n_switch=int(((tid[1:] != tid[:-1]) & (tid[1:] >= 0) & (tid[:-1] >= 0)).sum()))
    return dict(sel=sel, trk_xy=xy, trk_sig=sig, status=st, trk_id=tid, diag=diag, bg_pt=bg, cand=cand, seed=seed, roi=roi, mov=mov, fi=fi)
