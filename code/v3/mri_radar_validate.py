"""v3 mRI 레이더 정합의 정답 없는 검증.

A. 보류 잔차: 앞 절반(모든 피험자)으로 맞추고 뒤 절반에서 잼 (반대 방향도). 잔차 중앙값·축별 치우침·정규화 잔차 적중률.
B. 주입 복원: 원 레이더 점(센서 좌표)에 알려진 yaw / 이동 / 시간 지연을 넣고 궤적부터 다시 만들어 전체 적합을 다시 돌림.
   기대값: ψ' = ψ + δ,  τ' = τ + (Rz(ψ) R0 d)_xy,  지연' = 지연 + k.  pitch 5° 주입은 수평 가정에 대한 민감도(복원 대상 아님).
C. 잔차 구조: 최종 공통 변환의 잔차 (방 x = 깊이, y = 옆) 를 공변량(거리, 옆 위치, 몸통 속도, 2D 움직임, 레이더 점 수)에 회귀.
   피험자 블록 부트스트랩 표준오차. 움직임 구간별·걷는 방향별 중앙값. 높이 진단 (레이더 z − 스테레오 골반·몸통 z).
D. 이전 정답-ICP 변환과 파라미터만 비교 (회전 차, 이동 차, 사람 위치에서의 사상 차). 라벨은 읽지 않음 (이전 팩의 meta 만).
출력: work/validation.json
"""
import sys, json, time
sys.path.insert(0, "/home/interns/unisense/code/v3")
import numpy as np
from multiprocessing import Pool
from mri_radar_common import (load_subject, radar_hf_cov, pairs, fit_extra_fast, bev_fit_fast, rotz, rot2, smooth3, lag_xcorr, read_radar_csv,
                              attach_frames, build_track, load_spec, WORK, SUBJ, OLD, jdump, nan_median_filter)

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
REG = json.load(open(f"{WORK}/registration_dyn.json"))
R0 = np.array(REG["perm_best"]["R0"]); psi = np.radians(REG["final"]["psi_deg"]); tau = np.array(REG["final"]["tau"])
EX = np.array(REG["final"]["extra_sigma"]); Rf = np.array(REG["final"]["R"]); tf = np.array(REG["final"]["t"])
LAG = REG["time"]["applied_integer_lag"]; LAGS = {s: LAG for s in SUBJ}
cams, cam_h, _ = load_spec()
D = {s: load_subject(s, "dyn") for s in SUBJ}
Ca = radar_hf_cov(D); CaR = (R0 @ Ca @ R0.T)[:2, :2]
V = {}
CHI2_68, CHI2_95 = 2.279, 5.991


def metrics(P_, psi_, tau_, ex_):
    r = P_["a"][:, :2] @ rot2(psi_).T + tau_ - P_["b"]
    Rm = rot2(psi_); S = P_["Sb"] + (Rm @ CaR @ Rm.T + np.diag(ex_ ** 2))[None]
    m = np.einsum("ni,nij,nj->n", r, np.linalg.inv(S), r)
    med = np.median(r, 0)
    return dict(med_abs_r=float(np.median(np.linalg.norm(r, axis=1))), median_xy=med.tolist(), robust_sd_xy=(1.4826 * np.median(np.abs(r - med), 0)).tolist(),
                cover68=float((m <= CHI2_68).mean()), cover95=float((m <= CHI2_95).mean()), n=int(len(r)))


# ---------------------------------------------------------------- A. 보류 잔차
first = lambda s, t_, T: t_ < T // 2
second = lambda s, t_, T: t_ >= T // 2
A = {}
for tr_nm, te_nm, ftr, fte in [("first", "second", first, second), ("second", "first", second, first)]:
    Ptr = pairs(D, LAGS, R0, frames=ftr); Pte = pairs(D, LAGS, R0, frames=fte)
    f = fit_extra_fast(Ptr, Ca, R0, psi0=psi, tau0=tau, n_outer=4)
    A[f"fit_{tr_nm}"] = dict(psi_deg=float(np.degrees(f["psi"])), tau=f["tau"].tolist(), extra_sigma=f["extra_sigma"].tolist(),
                             in_sample=metrics(Ptr, f["psi"], f["tau"], f["extra_sigma"]), held_out=metrics(Pte, f["psi"], f["tau"], f["extra_sigma"]))
    per = {}
    for s in SUBJ:
        m = Pte["sid"] == s; r = Pte["a"][m, :2] @ rot2(f["psi"]).T + f["tau"] - Pte["b"][m]; per[f"S{s:02d}"] = np.median(r, 0).round(4).tolist()
    A[f"fit_{tr_nm}"]["held_out_per_subject_median_xy"] = per
    log("보류", tr_nm, "→", te_nm, json.dumps(A[f"fit_{tr_nm}"]["in_sample"]), json.dumps(A[f"fit_{tr_nm}"]["held_out"]))
Pall = pairs(D, LAGS, R0)
A["final_in_sample"] = metrics(Pall, psi, tau, EX)
V["held_out"] = A

# ---------------------------------------------------------------- C. 잔차 구조
r = Pall["a"][:, :2] @ rot2(psi).T + tau - Pall["b"]
sid, tt = Pall["sid"], Pall["t"]
vel = np.zeros((len(r), 2)); img = np.zeros(len(r)); npt = np.zeros(len(r)); zr = np.zeros(len(r)); zh = np.zeros(len(r)); zt = np.zeros(len(r))
for s in SUBJ:
    m = sid == s; d = D[s]; b3 = d["b3"]
    bs, _ = nan_median_filter(b3, 5)
    v = np.vstack([np.full((1, 3), np.nan), np.diff(bs, axis=0)]) * d["fps"]
    v3 = 0.5 * (v + np.vstack([v[1:], np.full((1, 3), np.nan)]))          # 중앙 차분
    vel[m] = v3[tt[m], :2]; img[m] = smooth3(d["img"])[tt[m]]; npt[m] = smooth3(d["npts"])[np.clip(tt[m] + LAG, 0, d["T"] - 1)]
    a3 = d["a3"][np.clip(tt[m] + LAG, 0, d["T"] - 1)]; zr[m] = (a3 @ Rf.T + tf)[:, 2]; zh[m] = d["hip"][tt[m], 2]; zt[m] = b3[tt[m], 2]
X = np.c_[np.ones(len(r)), Pall["b"][:, 0] - 2.4, Pall["b"][:, 1], vel, img, npt / 30.0]
names = ["const", "range_x_minus_2.4 (m/m)", "lateral_y (m/m)", "vel_x (m per m/s = −Δt s)", "vel_y (m per m/s = −Δt s)", "img_speed (m per m/s)", "radar_pts/30"]
ok = np.isfinite(X).all(1) & np.isfinite(r).all(1)


def huber_ls(Xm, y, k=1.345, it=20):
    beta = np.linalg.lstsq(Xm, y, rcond=None)[0]
    for _ in range(it):
        e = y - Xm @ beta; s_ = 1.4826 * np.median(np.abs(e - np.median(e))) + 1e-9; u = np.abs(e) / s_; w = np.where(u <= k, 1, k / u)
        beta = np.linalg.lstsq(Xm * w[:, None] ** 0.5, y * w ** 0.5, rcond=None)[0]
    return beta


C = {}
rng = np.random.default_rng(7)
for ax, nm in [(0, "res_x_depth"), (1, "res_y_lateral")]:
    beta = huber_ls(X[ok], r[ok, ax]); bs = []
    idx_s = {s: np.flatnonzero(ok & (sid == s)) for s in SUBJ}
    for b_ in range(100):
        pick = rng.choice(SUBJ, len(SUBJ)); ii = np.concatenate([idx_s[s] for s in pick]); bs.append(huber_ls(X[ii], r[ii, ax], it=8))
    se = np.array(bs).std(0)
    C[nm] = {n_: dict(coef=float(b), se_subject_boot=float(e)) for n_, b, e in zip(names, beta, se)}
    log("잔차 회귀", nm, {n_: (round(float(b), 4), round(float(e), 4)) for n_, b, e in zip(names, beta, se)})
# 움직임 구간
q = np.nanpercentile(img[ok], [33, 67])
bins = dict(static=ok & (img <= q[0]), moderate=ok & (img > q[0]) & (img <= q[1]), moving=ok & (img > q[1]))
sp = np.linalg.norm(vel, axis=1)
bins.update(toward_cam=ok & (vel[:, 0] < -0.2), away_cam=ok & (vel[:, 0] > 0.2), left=ok & (vel[:, 1] > 0.2), right=ok & (vel[:, 1] < -0.2), still=ok & (sp < 0.05))
C["bins"] = {k: dict(n=int(m.sum()), median_xy=np.median(r[m], 0).round(4).tolist() if m.sum() else None, mean_speed=float(np.nanmean(sp[m])) if m.sum() else None) for k, m in bins.items()}
C["img_speed_terciles"] = q.tolist()
if bins["toward_cam"].sum() > 50 and bins["away_cam"].sum() > 50:
    dv = np.median(vel[bins["away_cam"], 0]) - np.median(vel[bins["toward_cam"], 0]); dr = np.median(r[bins["away_cam"], 0]) - np.median(r[bins["toward_cam"], 0])
    C["walk_direction_dt_estimate_s"] = float(-dr / dv)
log("구간", json.dumps(C["bins"]))
# 피험자 상수
psub = {f"S{s:02d}": np.median(r[sid == s], 0).round(4).tolist() for s in SUBJ}
arr = np.array(list(psub.values()))
C["per_subject_median_xy"] = psub; C["between_subject_sd_xy"] = arr.std(0, ddof=1).tolist()
# 높이 진단 (가정한 tz 기준) — 물리 치우침은 변환에 넣지 않음, 보고만
okz = ok & np.isfinite(zr) & np.isfinite(zh) & np.isfinite(zt)
Hz = dict(radar_minus_hip_median=float(np.median(zr[okz] - zh[okz])), radar_minus_torso_median=float(np.median(zr[okz] - zt[okz])),
          radar_z_median=float(np.median(zr[okz])), hip_z_median=float(np.median(zh[okz])), torso_z_median=float(np.median(zt[okz])))
Xz = np.c_[np.ones(okz.sum()), Pall["b"][okz, 0] - 2.4, zt[okz] - np.median(zt[okz])]
bz = huber_ls(Xz, zr[okz] - zt[okz])
Hz["regress_(radar_z−torso_z)"] = dict(const=float(bz[0]), slope_vs_range=float(bz[1]), slope_vs_torso_z=float(bz[2]),
                                        note="거리 기울기 = pitch 오차 후보 (rad), 몸통 높이 기울기 + 1 = 고도 압축 s_el 후보")
Hz["per_subject_radar_minus_hip"] = {f"S{s:02d}": round(float(np.median((zr - zh)[okz & (sid == s)])), 4) for s in SUBJ}
C["height"] = Hz
log("높이", json.dumps({k: v for k, v in Hz.items() if k != "per_subject_radar_minus_hip"}))
V["residual_structure"] = C

# ---------------------------------------------------------------- D. 이전 변환과 비교 (파라미터만)
mo = json.loads(str(np.load(f"{OLD}/S01.npz", allow_pickle=False)["meta"]))
Ro, to = np.array(mo["extrinsic"]["R"]), np.array(mo["extrinsic"]["t"])
dR = Rf @ Ro.T; ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))


def ypr(Rm):
    """방 좌표에서 레이더 보어사이트(센서 +y)·위(센서 +z) 방향으로 yaw/pitch/roll."""
    by = Rm[:, 1]; bz_ = Rm[:, 2]
    yaw = np.degrees(np.arctan2(by[1], by[0])); pitch = np.degrees(np.arcsin(np.clip(by[2], -1, 1)))
    xr = Rm[:, 0]; roll = np.degrees(np.arcsin(np.clip(xr[2], -1, 1)))
    return dict(yaw_boresight_deg=float(yaw), pitch_boresight_up_deg=float(pitch), roll_deg=float(roll))


c0s = {s: np.load(f"{WORK}/radar_track_S{s:02d}.npz")["c0"] for s in SUBJ}
mapd = {f"S{s:02d}": ((Rf @ c + tf) - (Ro @ c + to)).round(4).tolist() for s, c in c0s.items()}
md = np.array(list(mapd.values()))
V["old_transform_comparison"] = dict(old_source=mo["extrinsic"]["source"], old_R=Ro.tolist(), old_t=to.tolist(), new_R=Rf.tolist(), new_t=tf.tolist(),
                                     rotation_diff_deg=ang, old_angles=ypr(Ro), new_angles=ypr(Rf), translation_diff_new_minus_old=(tf - to).tolist(),
                                     mapped_c0_diff_new_minus_old_per_subject=mapd, mapped_c0_diff_median=np.median(md, 0).tolist(),
                                     note="c0 = 피험자별 레이더 점 중앙 (센서 좌표). 두 변환이 사람 위치의 레이더 점을 방 어디에 놓는지의 차. 라벨 미사용")
log("이전 변환 비교", json.dumps({k: V["old_transform_comparison"][k] for k in ["rotation_diff_deg", "old_angles", "new_angles", "translation_diff_new_minus_old", "mapped_c0_diff_median"]}))
jdump(V, f"{WORK}/validation.json")

# ---------------------------------------------------------------- B. 주입 복원 (원 점에서 궤적부터 다시)
RAW = {}


def load_raw(s):
    st = np.load(f"{WORK}/stereo_S{s:02d}.npz", allow_pickle=False); T = int(st["T"]); f0 = int(st["frame0"])
    pts, rf, has, dup = attach_frames(read_radar_csv(s), T, f0)
    return s, pts, has & ~dup


def rot_axis(axis, deg):
    a = np.radians(deg); c, s_ = np.cos(a), np.sin(a)
    if axis == "z":
        return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    return np.array([[1.0, 0, 0], [0, c, -s_], [0, s_, c]])


def inj_track(args):
    s, pts, uniq, case = args
    T = len(pts); P2 = [None] * T
    Rz_ = rot_axis(case.get("axis", "z"), case.get("deg", 0.0)); d = np.array(case.get("d", [0.0, 0.0, 0.0])); k = int(case.get("shift", 0))
    for t in range(T):
        if pts[t] is None:
            continue
        q = pts[t].copy(); q[:, :3] = (q[:, :3] - d) @ Rz_            # p_inj = Rᵀ (p − d)  (행 벡터: (p−d) R)
        P2[t] = q
    u2 = uniq.copy()
    if k:
        P2 = [None] * k + P2[:T - k]; u2 = np.r_[np.zeros(k, bool), uniq[:T - k]]
    tr = build_track(P2, u2)
    return s, tr, u2


if __name__ == "__main__":
    with Pool(10) as pool:
        for s, pts, uniq in pool.map(load_raw, SUBJ):
            RAW[s] = (pts, uniq)
    log("원 점 적재")
    cases = [dict(name="baseline"), dict(name="yaw+3", deg=3.0), dict(name="shift_x+0.10_y+0.05", d=[0.10, 0.05, 0.0]),
             dict(name="yaw-2_shift_x-0.05_y-0.10", deg=-2.0, d=[-0.05, -0.10, 0.0]), dict(name="time+2frames", shift=2),
             dict(name="pitch+5_sensitivity", axis="x", deg=5.0)]
    INJ = {}
    base = None
    for case in cases:
        with Pool(10) as pool:
            res = pool.map(inj_track, [(s, RAW[s][0], RAW[s][1], case) for s in SUBJ])
        Dc = {}
        for s, tr, u2 in res:
            d = dict(D[s]); a = tr["dyn"].copy(); a[~u2] = np.nan; d["a3"] = a; d["npts"] = np.where(u2, tr["n_gate"], np.nan).astype(float); Dc[s] = d
        # 시간 (점 수 ↔ 2D 속도)
        lg = []
        for s in SUBJ:
            r_ = lag_xcorr(smooth3(Dc[s]["img"]), smooth3(Dc[s]["npts"]), 8)
            if r_["peak"] >= 0.2:
                lg.append(r_["best_sub"])
        lag_med = float(np.median(lg)); Lc = int(np.round(lag_med)) if abs(lag_med) >= 0.5 else 0
        P_ = pairs(Dc, {s: Lc for s in SUBJ}, R0)
        f = fit_extra_fast(P_, Ca, R0, psi0=0.0, tau0=None, n_outer=4)          # 배치 탐색과 같은 출발 (ψ=0, τ=중앙값)
        row = dict(lag_median_frames=lag_med, lag_applied=Lc, psi_deg=float(np.degrees(f["psi"])), tau=f["tau"].tolist(), med_r=float(np.median(np.linalg.norm(f["r"], axis=1))))
        if base is None:
            base = row
        else:
            dd = np.array(case.get("d", [0, 0, 0.0])); dl = case.get("deg", 0.0) if case.get("axis", "z") == "z" else 0.0
            exp_psi = base["psi_deg"] + dl
            exp_tau = np.array(base["tau"]) + (rotz(np.radians(base["psi_deg"])) @ R0 @ dd)[:2]
            row["expected"] = dict(psi_deg=exp_psi, tau=exp_tau.tolist(), lag=base["lag_median_frames"] + case.get("shift", 0))
            row["error"] = dict(psi_deg=row["psi_deg"] - exp_psi, tau=(np.array(row["tau"]) - exp_tau).tolist(), lag_frames=lag_med - row["expected"]["lag"],
                                lat_at_2p4=float(((rot2(f["psi"]) @ (R0 @ [0, 2.4, 0])[:2] + f["tau"]) - (rot2(np.radians(exp_psi)) @ (R0 @ [0, 2.4, 0])[:2] + exp_tau))[1]))
        INJ[case["name"]] = row
        log("주입", case["name"], json.dumps(row))
    V["injection_recovery"] = dict(cases=cases, results=INJ, note="원 점에 주입 → 궤적 재구성 → 시간·BEV 적합 재실행. 기준 = 주입 없는 같은 절차 결과")
    jdump(V, f"{WORK}/validation.json")
    log("VALIDATE_DONE")
