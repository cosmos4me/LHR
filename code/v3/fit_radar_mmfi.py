"""v3 MM-Fi 레이더 방 단위 적합 + 정답 없는 검증 (2026-11-24). 입력 = reg_radar_mmfi.py stageA 결과 (_reg/A_*.npz). 정답을 읽지 않는다.
절차 (방마다):
 1. 축 부호: R0 = diag(1, −1, −1) (x 축 둘레 180°, det +1). 근거 = 녹화 안 레이더 옆 좌표 ↔ 깊이 사람 옆 좌표 상관 (+ 이어야 함), 높이 상관.
 2. 시간: 레이더 |도플러| 합 ↔ 깊이 프레임 차분 에너지 (중심 맞춤: (e_t + e_{t+1})/2, 5 프레임 중심 평활) 상호상관, 방 평균 곡선의 포물선 봉우리 → τ (프레임).
    정수 프레임 이동보다 작으면 팩은 옮기지 않고 τ 와 σ 만 기록.
 3. 수평 (ψ, tx, ty): 프레임별 레이더 사람 점 중앙 (게이트: 거리 1–6 m, |고도| < 30°, 앵커 1.0 m → 0.7 m 원통, 높이 −0.2–2.3 m)
    ↔ 깊이 보이는 표면 중앙 + LOS_PRIOR·u (u = 레이더 → 사람 수평 단위 벡터). Huber 0.10 m IRLS, 게이트 3 회 갱신.
 4. 수직 H (3 m 유효 높이): V2 압축 모형 (주), V1 중앙값 일치 (대안) — 차이를 z 불확도로 기록.
 5. 검증: 녹화 군집 부트스트랩 σ, 세션 (피험자) 별 적합 퍼짐, LOSO, 홀수/짝수 피험자 hold-out 잔차, 주입 복원.
출력: dev/packs/v3/mmfi_radar/_reg/fit.json (방별 변환·검증), rec_diag.json (녹화별 진단). 사용: python -I fit_radar_mmfi.py"""
import sys, os, json
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
import mmfi_common as mc
from reg_radar_mmfi import R0, M, QP

LOS_PRIOR = 0.10          # 레이더 중심은 깊이 보이는 표면보다 시선 방향으로 0.10 m 뒤 (번역기 HALF_BODY 와 같은 센서 종류 사전값; 데이터셋 키 아님)
HUB = 0.10
LAGS = np.arange(-6, 7)
STEP = 2                  # 프레임 간격 (레이더 0.5 s 누적이라 이웃 프레임은 거의 같음)
H_RULE = "V1"             # 주: V1 (이동 등가: 주입 복원 통과, 방 사이 반복성 더 좋음). V2 는 대안으로 기록 (첫 실행 결과: V2 는 주입 δH 를 δH/s 로 복원 — 압축 모형 가정 때문)
X3 = 3.1


def load_room(env):
    out = {}
    for e, nm in mc.recordings((env,)):
        A = dict(np.load(mc.WORK_R / f"A_{e}_{nm}.npz", allow_pickle=False)); out[nm] = A
    return out


def nominal(A):
    """원시 점 → 카메라 평행 수평 좌표 (R_d2r M R0 x). 반환 점 (n,3), 도플러, 게이트 (거리·고도), 프레임 오프셋."""
    P = A["pts"].astype(float); x = P[:, :3] @ R0.T; rg = np.linalg.norm(x, axis=1)
    el = np.degrees(np.arcsin(np.clip(x[:, 2] / np.maximum(rg, 1e-6), -1, 1)))
    g = (rg > 1.0) & (rg < 6.0) & (np.abs(el) < 30)
    L = x @ (np.asarray(A["R_d2r"], float) @ M).T
    return L, P[:, 4], g, A["off"]


def time_curve(A, L, dop, g):
    T = int(A["T"]); off = A["off"]; sr = np.full(T, np.nan)
    for k in range(T):
        m = np.arange(off[k], off[k + 1]); m = m[g[m]]
        if len(m):
            sr[k] = np.abs(dop[m]).sum()
    e = np.asarray(A["ed"], float); ec = np.full(T, np.nan); ec[:-1] = 0.5 * (e[:-1] + e[1:])     # 프레임 t 중심 움직임
    return mc.xcorr_lags(sr, mc.box_smooth(ec, 5), LAGS)


def parab_peak(c, lags=LAGS):
    i = int(np.nanargmax(c))
    if 0 < i < len(c) - 1 and np.isfinite(c[i - 1:i + 2]).all():
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]; den = y0 - 2 * y1 + y2
        return float(lags[i] + (0.5 * (y0 - y2) / den if den < 0 else 0.0))
    return float(lags[i])


def frame_pairs(A, L, g, x, H, gate_r):
    """변환 x = (ψ, tx, ty) 와 높이 H 로 게이트한 프레임별 (레이더 수평 중앙 [공칭], 깊이 목표 수평, 레이더 높이 분위수 [H=0], 깊이 분위수, 점 수)."""
    psi, tx, ty = x; c, s = np.cos(psi), np.sin(psi); R2 = np.array([[c, -s], [s, c]]); t = np.array([tx, ty])
    T = int(A["T"]); off = A["off"]; cd = A["cd"]; dq = A["dq"]
    cds = np.c_[mc.box_smooth(cd[:, 0], 5), mc.box_smooth(cd[:, 1], 5)]
    rows = []
    for k in range(0, T, STEP):
        if not np.isfinite(cds[k]).all() or not np.isfinite(dq[k]).all():
            continue
        m = np.arange(off[k], off[k + 1]); m = m[g[m]]
        if len(m) < 3:
            continue
        Q = L[m]; Qr = Q[:, :2] @ R2.T + t; zr = Q[:, 2] + H
        u = cds[k] - t; u = u / max(np.linalg.norm(u), 1e-6); tgt = cds[k] + LOS_PRIOR * u
        keep = (np.linalg.norm(Qr - tgt, axis=1) < gate_r) & (zr > -0.2) & (zr < 2.3)
        if keep.sum() < 3:
            continue
        rows.append((k, *np.median(Q[keep, :2], 0), *tgt, *np.quantile(Q[keep, 2], QP), *dq[k], keep.sum()))
    return np.array(rows) if rows else np.zeros((0, 6 + 2 * len(QP)))


def fit_h(rows, x0):
    """IRLS Huber: 공칭 수평 r → 목표 tgt, 2D 강체 (ψ, t). 레이더→사람 u 가 t 에 의존하지만 rows 가 이미 목표를 담음."""
    r = rows[:, 1:3]; tgt = rows[:, 3:5]; w = np.ones(len(r)); psi, t = x0[0], np.array(x0[1:3])
    for _ in range(30):
        p2, t2 = mc.procrustes2d(r, tgt, w); c, s = np.cos(p2), np.sin(p2)
        res = np.linalg.norm(r @ np.array([[c, s], [-s, c]]) + t2 - tgt, axis=1); w = mc.huber_w(res, HUB)
        done = abs(p2 - psi) < 1e-6 and np.linalg.norm(t2 - t) < 1e-5; psi, t = p2, t2
        if done:
            break
    return np.array([psi, *t]), res


def fit_v(rows):
    """V2: q_r0 = s q_d + a  (q_r0 = H=0 일 때 레이더 높이 분위수) → H = −a/s. V1: H = 중앙(깊이 중앙값) − 중앙(레이더 중앙값). Huber IRLS."""
    nq = len(QP); qr = rows[:, 5:5 + nq]; qd = rows[:, 5 + nq:5 + 2 * nq]
    X = qd.ravel(); Y = qr.ravel(); w = np.ones(len(X)); b = np.array([0.5, -0.5])
    for _ in range(30):
        A_ = np.c_[X, np.ones(len(X))] * np.sqrt(w)[:, None]; b = np.linalg.lstsq(A_, Y * np.sqrt(w), rcond=None)[0]
        res = Y - (b[0] * X + b[1]); w = mc.huber_w(res, 0.10)
    s, a = b; H2 = -a / s if s > 0.05 else np.nan
    H1 = float(np.median(qd[:, nq // 2]) - np.median(qr[:, nq // 2]))
    return dict(s=float(s), a=float(a), H_v2=float(H2), H_v1=H1)


def room_fit(R, recs, H0, x0=(0.0, 0.0, 0.0), it=3):
    x = np.array(x0, float); H = H0
    for i in range(it):
        gate = 1.0 if i == 0 else 0.7
        rows = [frame_pairs(R[nm][0], R[nm][1], R[nm][2], x, H, gate) for nm in recs]
        allr = np.concatenate([r for r in rows if len(r)])
        x, res = fit_h(allr, x); v = fit_v(allr); H = v["H_v1"] if H_RULE == "V1" else v["H_v2"]
    return x, H, v, allr, rows


def residuals(R, recs, x, H):
    """녹화별 잔차 (방 좌표): 수평 (레이더 중앙 → 목표) 의 시선·옆 성분 중앙값, 높이 중앙값 차 (H 적용, V1 기준)."""
    out = {}
    for nm in recs:
        rows = frame_pairs(R[nm][0], R[nm][1], R[nm][2], x, H, 0.7)
        if len(rows) < 10:
            out[nm] = None; continue
        c, s = np.cos(x[0]), np.sin(x[0]); r = rows[:, 1:3] @ np.array([[c, s], [-s, c]]) + x[1:3]; e = r - rows[:, 3:5]
        u = rows[:, 3:5] - x[1:3]; u /= np.linalg.norm(u, axis=1, keepdims=True); lat = np.c_[-u[:, 1], u[:, 0]]
        nq = len(QP); dz = (rows[:, 5 + nq // 2] + H) - rows[:, 5 + nq + nq // 2]
        out[nm] = dict(n=int(len(rows)), los=float(np.median((e * u).sum(1))), lat=float(np.median((e * lat).sum(1))), dz=float(np.median(dz)),
                       los_std=float(np.std((e * u).sum(1))), lat_std=float(np.std((e * lat).sum(1))), pts=float(np.median(rows[:, -1])))
    return out


def inject(R, recs, dpsi, dt, dH, x, H):
    """원시 점을 알려진 변환만큼 흔든 사본으로 다시 적합 → 복원값 − 주입값."""
    c, s = np.cos(dpsi), np.sin(dpsi); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]); R2 = {}
    for nm in recs:
        A, L, g, off = R[nm]; L2 = L @ Rz.T + np.array([dt[0], dt[1], dH]); R2[nm] = (A, L2, g, off)
    x2, H2, v2, _, _ = room_fit(R2, recs, H, x0=x)
    # 주입: L' = Rz L + d  → 원래 변환을 L 에 적용하려면 새 변환 x2 는 x ∘ (Rz, d)^{-1}
    psi_rec = x[0] - x2[0]; Rb = np.array([[np.cos(x2[0]), -np.sin(x2[0])], [np.sin(x2[0]), np.cos(x2[0])]])
    # x2 가 L' 에 적용되어 방으로: R(x2)(Rz L + d) + t2 = R(x) L + t → R(x2)Rz = R(x), R(x2) d + t2 = t
    d_rec = np.linalg.solve(Rb, np.array(x[1:3]) - x2[1:3])
    return dict(inj=[float(np.degrees(dpsi)), *map(float, dt), float(dH)], rec=[float(np.degrees(psi_rec)), *map(float, d_rec), float(H - H2)])


def proc_room(env):
    Rraw = load_room(env); R = {}
    for nm, A in Rraw.items():
        L, dop, g, off = nominal(A); R[nm] = (A, L, g, off); A["_dop"] = dop
    recs = sorted(R); subj = sorted({nm[:3] for nm in recs})
    out = dict(env=env, n_rec=len(recs))
    # 1. 축 부호 증거 (공칭 좌표, 변환 전): 녹화 안 상관
    lat, ver = [], []
    for nm in recs:
        A, L, g, off = R[nm]; T = int(A["T"]); cy = np.full(T, np.nan); cz = np.full(T, np.nan)
        for k in range(T):
            m = np.arange(off[k], off[k + 1]); m = m[g[m]]
            if len(m) >= 3:
                cy[k] = np.median(L[m, 1]); cz[k] = np.median(L[m, 2])
        cd = A["cd"]; ok = np.isfinite(cy) & np.isfinite(cd[:, 1])
        if ok.sum() > 50 and np.std(cd[ok, 1]) > 0.05:
            lat.append(np.corrcoef(cy[ok], cd[ok, 1])[0, 1])
        dz = A["dq"][:, len(QP) // 2]; ok = np.isfinite(cz) & np.isfinite(dz)
        if ok.sum() > 50 and np.std(dz[ok]) > 0.08:
            ver.append(np.corrcoef(cz[ok], dz[ok])[0, 1])
    out["axis_check"] = dict(R0=R0.tolist(), det=float(np.linalg.det(R0)), lateral_corr_med=float(np.median(lat)), lateral_frac_pos=float(np.mean(np.array(lat) > 0)), n_lat=len(lat),
                             vertical_corr_med=float(np.median(ver)), vertical_frac_pos=float(np.mean(np.array(ver) > 0)), n_ver=len(ver),
                             note="옆 상관 > 0 이 강하게 성립 (y 뒤집기 확정). x 는 앞 (거리 약 3 m). det = +1 을 지키려면 z 도 뒤집어야 하며, 높이 상관도 약하게 + (TI 고도 약함)")
    # 2. 시간
    C = np.array([time_curve(R[nm][0], R[nm][1], Rraw[nm]["_dop"], R[nm][2]) for nm in recs]); mean_c = np.nanmean(C, 0)
    rng = np.random.default_rng(int(env[1:]))
    bs = [parab_peak(np.nanmean(C[rng.integers(0, len(C), len(C))], 0)) for _ in range(200)]
    halves = []
    for half in (0, 1):
        Ch = []
        for nm in recs:
            A, L, g, off = R[nm]; A2 = dict(A); T = int(A["T"]); h0, h1 = (0, T // 2) if half == 0 else (T // 2, T)
            ed = np.asarray(A["ed"], float).copy(); mask = np.ones(T, bool); mask[h0:h1] = False; ed[mask] = np.nan; A2["ed"] = ed
            Ch.append(time_curve(A2, L, Rraw[nm]["_dop"], g))
        halves.append(parab_peak(np.nanmean(np.array(Ch), 0)))
    tau = parab_peak(mean_c)
    out["time"] = dict(signal="radar sum|doppler| vs depth frame-diff energy (centred, 5-box)", lags=LAGS.tolist(), mean_xcorr=np.round(mean_c, 4).tolist(),
                       tau_frames=tau, tau_s=tau / mc.FPS, sigma_frames=float(np.std(bs)), first_half=halves[0], second_half=halves[1],
                       rec_peak_hist={int(k): int(v) for k, v in zip(*np.unique(LAGS[np.nanargmax(np.where(np.isfinite(C), C, -9), 1)], return_counts=True))},
                       applied_shift_frames=0, note="음수 = 레이더가 깊이보다 앞섬 (레이더 프레임 k 의 움직임이 깊이 k+|τ| 와 맞음). |τ| < 1 프레임이라 팩은 옮기지 않음")
    # 3–4. 방 적합
    H0 = float(np.asarray(Rraw[recs[0]]["t_d2r"])[2])
    x, H, v, allr, rows = room_fit(R, recs, H0)
    x_pooled = x.copy()
    out["fit"] = dict(psi_deg=float(np.degrees(x[0])), tx=float(x[1]), ty=float(x[2]), H=float(H), H_v2=v["H_v2"], H_v1=v["H_v1"], elev_slope_s=v["s"], n_pairs=int(len(allr)),
                      H_choice=H_RULE, los_prior_m=LOS_PRIOR, tx_pooled=float(x[1]))
    # 5a. 녹화 군집 부트스트랩 (빠르게: 게이트 고정, 행 재사용)
    rows_d = {nm: r for nm, r in zip(recs, rows) if len(r)}; keys = sorted(rows_d); bsx = []
    for b in range(200):
        pick = rng.integers(0, len(keys), len(keys)); rr = np.concatenate([rows_d[keys[i]] for i in pick])
        xb, _ = fit_h(rr, x); vb = fit_v(rr); bsx.append([np.degrees(xb[0]), xb[1], xb[2], vb["H_v2"], vb["H_v1"], vb["s"]])
    bsx = np.array(bsx)
    out["boot_rec"] = dict(sigma_psi_deg=float(np.std(bsx[:, 0])), sigma_tx=float(np.std(bsx[:, 1])), sigma_ty=float(np.std(bsx[:, 2])), sigma_H_v2=float(np.nanstd(bsx[:, 3])),
                           sigma_H_v1=float(np.std(bsx[:, 4])), sigma_s=float(np.std(bsx[:, 5])),
                           sigma_lat_at_3m=float(np.std(bsx[:, 2] + 3.1 * np.sin(np.radians(bsx[:, 0])))))
    # 5b. 피험자 (세션) 별 적합과 LOSO
    per_s, loso = {}, {}
    for sj in subj:
        rs = [rows_d[k] for k in keys if k.startswith(sj)]
        if rs:
            rr = np.concatenate(rs); xs, _ = fit_h(rr, x); vs = fit_v(rr)
            per_s[sj] = dict(psi_deg=float(np.degrees(xs[0])), tx=float(xs[1]), ty=float(xs[2]), lat3=float(xs[2] + 3.1 * np.sin(xs[0])), H_v2=vs["H_v2"], H_v1=vs["H_v1"], s=vs["s"], n=int(len(rr)))
        ro = [rows_d[k] for k in keys if not k.startswith(sj)]; rr = np.concatenate(ro); xo, _ = fit_h(rr, x); vo = fit_v(rr)
        loso[sj] = dict(psi_deg=float(np.degrees(xo[0])), tx=float(xo[1]), ty=float(xo[2]), lat3=float(xo[2] + 3.1 * np.sin(xo[0])), H_v2=vo["H_v2"], H_v1=vo["H_v1"], s=vo["s"])
    def spread(d, k):
        a = np.array([v[k] for v in d.values()], float); return dict(std=float(np.nanstd(a)), range=float(np.nanmax(a) - np.nanmin(a)), median=float(np.nanmedian(a)))
    out["per_subject"] = per_s; out["loso"] = loso
    # 앞 (x3 = 3.1 m 기준점의 방 x) 은 세션 중앙값들의 중앙값 (세션별 깊이 시차 오프셋에 강건; LiDAR 와 같은 규칙)
    x3s = np.array([v["tx"] + X3 * np.cos(np.radians(v["psi_deg"])) for v in per_s.values()]); x3f = float(np.median(x3s))
    x = np.array([x[0], x3f - X3 * np.cos(x[0]), x[2]]); out["fit"]["tx"] = float(x[1]); out["fit"]["x3"] = x3f; out["fit"]["y3"] = float(x[2] + X3 * np.sin(x[0]))
    out["boot_rec"]["sigma_x3_session_median"] = float(np.std([np.median(rng.choice(x3s, len(x3s))) for _ in range(500)]))
    out["repeat"] = {k: dict(per_subject=spread(per_s, k), loso=spread(loso, k)) for k in ("psi_deg", "tx", "ty", "lat3", "H_v2", "H_v1", "s")}
    # 5c. hold-out: 홀수 피험자로 적합 → 짝수 피험자 잔차 (반대로도)
    ho = {}
    for name, fit_set in (("odd_fit", [s for s in subj if int(s[1:]) % 2 == 1]), ("even_fit", [s for s in subj if int(s[1:]) % 2 == 0])):
        tr = [k for k in keys if k[:3] in fit_set]; te = [k for k in keys if k[:3] not in fit_set]
        xf, _ = fit_h(np.concatenate([rows_d[k] for k in tr]), x); vf = fit_v(np.concatenate([rows_d[k] for k in tr]))
        Hf = vf["H_v2"] if H_RULE == "V2" else vf["H_v1"]
        x3f_ = float(np.median([per_s[s]["tx"] + X3 * np.cos(np.radians(per_s[s]["psi_deg"])) for s in fit_set if s in per_s])); xf = np.array([xf[0], x3f_ - X3 * np.cos(xf[0]), xf[2]])
        r_te = residuals(R, te, xf, Hf); r_in = residuals(R, te, x, H)
        f = lambda d, k: float(np.median([abs(v[k]) for v in d.values() if v]))
        ho[name] = dict(test_abs_los=f(r_te, "los"), test_abs_lat=f(r_te, "lat"), test_abs_dz=f(r_te, "dz"), full_abs_los=f(r_in, "los"), full_abs_lat=f(r_in, "lat"), full_abs_dz=f(r_in, "dz"))
    out["holdout"] = ho
    # 5d. 주입 복원 (방 전체)
    out["injection"] = [inject(R, recs, np.radians(3.0), (0.10, -0.10), 0.15, x_pooled, H), inject(R, recs, np.radians(-5.0), (-0.20, 0.15), -0.20, x_pooled, H)]
    # 녹화별 진단
    out["rec"] = residuals(R, recs, x, H)
    print(env, json.dumps({k: out[k] for k in ("axis_check", "fit", "boot_rec")}, ensure_ascii=False)[:1500], flush=True)
    print(env, "time", {k: out["time"][k] for k in ("tau_frames", "sigma_frames", "first_half", "second_half")}, "holdout", ho, "inject", out["injection"], flush=True)
    return env, out


if __name__ == "__main__":
    envs = tuple(sys.argv[1:]) or mc.ENVS
    with Pool(len(envs)) as p:
        res = dict(p.map(proc_room, envs))
    f = mc.WORK_R / "fit.json"; old = json.load(open(f)) if f.exists() else {}
    old.update(res); json.dump(old, open(f, "w"), indent=1, ensure_ascii=False)
    print("RADAR FIT DONE", flush=True)
