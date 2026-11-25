"""v3 mRI: 정답 없는 레이더 → 방 정합 (BEV 3-DoF + 장착 제약).

입력: work/stereo_Sxx.npz (mri_stereo_track), work/radar_track_Sxx.npz (mri_radar_track). 정답·라벨 없음.
1. 시간: 외부 파라미터와 무관한 움직임 신호의 상호상관 (레이더 게이트 점 수 ↔ 2D 키포인트 속도; 보조: 평균 |도플러|).
   피험자별 전체·1000 프레임 창·앞뒤 절반. 공통 지연 = 피험자 추정의 중앙값 → 정수 프레임으로 반올림해 적용.
2. 축 배치: 행렬식 +1 인 24 가지 R0 각각에서 연속 yaw ψ 와 (τx, τy) 를 가중 강건 적합 (3 프레임마다 부분 표본).
   선택 = BEV 잔차 중앙값 최소. 레이더 위치 τ 가 책상(카메라 근처) 사전 범위 |τ| < 1 m 인지 함께 보고.
   진단: 최적 배치의 x 부호 반전(거울, 행렬식 −1)도 적합해 비교만 함 (채택 안 함).
3. 최종: 모든 프레임으로 적합. 총 공분산 = 스테레오(프레임별 이방성) + 레이더 고주파(센서 좌표 → 방) + Σ_extra(잔차에서 추정).
   위치 기반 시간 재확인 (공통 지연 L ∈ [−3, 3] 에서 강건 비용).
4. 반복성: 피험자별 적합 + 피험자별 블록 부트스트랩, 한 피험자 빼기 (LOSO), 앞·뒤 절반 따로 적합.
5. 불확도: 2단 블록 부트스트랩 (피험자 복원추출 → 피험자 안 87 프레임(≈10 s) 블록 복원추출), B = 200.
6. z, pitch, roll: 장착 제약 (같은 책상, 수평). z = 카메라1 높이 + Δh, Δh = 0 (가정, σ 0.10 m). v3 바닥 파일에 mRI 가 있으면 보고만.
출력: work/registration_<variant>.json
"""
import sys, json, time, os
sys.path.insert(0, "/home/interns/unisense/code/v3")
import numpy as np
from mri_radar_common import (load_subject, radar_hf_cov, pairs, fit_extra_fast, bev_fit_fast, perms24, rotz, rot2, smooth3, lag_xcorr,
                              load_spec, WORK, SUBJ, jdump)

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "dyn"
QUICK = len(sys.argv) > 2 and sys.argv[2] == "quick"
B_BOOT, B_SUBJ, BLOCK = (200, 100, 87) if not QUICK else (20, 10, 87)
DH_ASSUMED, DH_SIGMA = 0.0, 0.10
rng = np.random.default_rng(20261124)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
cams, cam_h, _ = load_spec()
D = {s: load_subject(s, VARIANT) for s in SUBJ}
Ca = radar_hf_cov(D)
out = dict(variant=VARIANT, radar_hf_sigma_sensor_m=np.sqrt(np.diag(Ca)).round(4).tolist(), radar_hf_cov_sensor=Ca.round(6).tolist())
log("레이더 고주파 σ (센서 x, y, z)", np.sqrt(np.diag(Ca)).round(4))

# ---------------------------------------------------------------- 1. 시간
lag_rows = {}
for s, d in D.items():
    x = smooth3(d["img"]); y = smooth3(d["npts"]); y2 = smooth3(d["dopE"]); T = d["T"]; h = T // 2
    whole = lag_xcorr(x, y, 8); dop = lag_xcorr(x, y2, 8)
    first, second = lag_xcorr(x[:h], y[:h], 8), lag_xcorr(x[h:], y[h:], 8)
    win = []
    for a0 in range(0, T - 999, 500):
        r = lag_xcorr(x[a0:a0 + 1000], y[a0:a0 + 1000], 8); win.append([a0, round(r["best_sub"], 3), round(r["peak"], 3)])
    gw = [w[1] for w in win if w[2] >= 0.3]
    lag_rows[s] = dict(npts_whole=round(whole["best_sub"], 3), npts_peak=round(whole["peak"], 3), dop_whole=round(dop["best_sub"], 3), dop_peak=round(dop["peak"], 3),
                       first=round(first["best_sub"], 3), first_peak=round(first["peak"], 3), second=round(second["best_sub"], 3), second_peak=round(second["peak"], 3),
                       win_median=round(float(np.median(gw)), 3) if gw else None, win_n_good=len(gw), win_iqr=round(float(np.subtract(*np.percentile(gw, [75, 25]))), 3) if len(gw) > 1 else None,
                       windows=win)
    log(f"S{s:02d} 지연(프레임) 점수 {whole['best_sub']:+.2f} (ρ {whole['peak']:.2f})  도플러 {dop['best_sub']:+.2f} (ρ {dop['peak']:.2f})  앞 {first['best_sub']:+.2f} 뒤 {second['best_sub']:+.2f}  창 중앙 {lag_rows[s]['win_median']} (n {len(gw)})")
good = [r["npts_whole"] for r in lag_rows.values() if r["npts_peak"] >= 0.2]
lag_med = float(np.median(good)); lag_int = int(np.round(lag_med)) if abs(lag_med) >= 0.5 else 0
fps = D[1]["fps"]
out["time"] = dict(signal="radar gated point count vs 2D keypoint speed (smoothed 3)", sign="L>0: radar frame t+L ↔ camera frame t (radar late)",
                   per_subject=lag_rows, median_frames=round(lag_med, 3), median_ms=round(1e3 * lag_med / fps, 1), n_subjects_used=len(good),
                   spread_frames_iqr=round(float(np.subtract(*np.percentile(good, [75, 25]))), 3), applied_integer_lag=lag_int,
                   applied_note="공통 정수 지연. |중앙값| < 0.5 프레임이면 0 (팩 프레임 대응을 바꾸지 않음)")
log("공통 지연 중앙값", round(lag_med, 3), "프레임 =", round(1e3 * lag_med / fps, 1), "ms → 적용 정수", lag_int)
LAGS = {s: lag_int for s in SUBJ}

# ---------------------------------------------------------------- 2. 축 배치 24 가지
perm_rows = []; best = None
for R0, perm, sg in perms24():
    P = pairs(D, LAGS, R0); sub = np.arange(0, len(P["b"]), 3)
    res = fit_extra_fast(P, Ca, R0, n_outer=3, sub=sub)
    rn = np.linalg.norm(res["r"], axis=1)
    q = P["a"][sub, :2] @ rot2(res["psi"]).T + res["tau"]; bb = P["b"][sub]; sid = P["sid"][sub]
    lc, xc = [], []
    for s in SUBJ:
        m = sid == s
        for k, lst in [(1, lc), (0, xc)]:
            u = q[m, k] - q[m, k].mean(); v = bb[m, k] - bb[m, k].mean(); lst.append(float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum() + 1e-12)))
    up = int(np.flatnonzero(R0[2])[0])
    row = dict(perm=perm, sign=sg, up_axis=["x", "y", "z"][up], up_sign=int(R0[2, up]), psi_deg=round(float(np.degrees(res["psi"])), 3), tau=res["tau"].round(4).tolist(),
               med_r=round(float(np.median(rn)), 4), p90_r=round(float(np.percentile(rn, 90)), 4), extra_sigma=res["extra_sigma"].round(4).tolist(),
               lat_corr_med=round(float(np.median(lc)), 3), depth_corr_med=round(float(np.median(xc)), 3), desk_prior_ok=bool(np.linalg.norm(res["tau"]) < 1.0))
    perm_rows.append(row); log("배치", perm, sg, json.dumps({k: row[k] for k in ["up_axis", "up_sign", "psi_deg", "tau", "med_r", "lat_corr_med", "depth_corr_med"]}))
    if best is None or row["med_r"] < best[0]["med_r"]:
        best = (row, R0, res)
out["perm_search"] = perm_rows
row, R0, res0 = best
log("최적 배치", row["perm"], row["sign"], "ψ", row["psi_deg"], "τ", row["tau"], "med|r|", row["med_r"])
srt = sorted(perm_rows, key=lambda r: r["med_r"])
distinct = []
for r in srt:
    if not any(abs(r["med_r"] - q["med_r"]) < 1e-4 for q in distinct):
        distinct.append(r)
out["perm_best"] = dict(perm=row["perm"], sign=row["sign"], R0=R0.tolist(), n_equivalent=int(sum(abs(r["med_r"] - row["med_r"]) < 1e-4 for r in perm_rows)),
                        runner_up_distinct=distinct[1] if len(distinct) > 1 else None, margin_med_r=round(distinct[1]["med_r"] - row["med_r"], 4) if len(distinct) > 1 else None)

# 거울 진단 (채택 안 함)
M = R0.copy(); M[:, 0] *= -1
Pm = pairs(D, LAGS, M); subm = np.arange(0, len(Pm["b"]), 3); rm_ = fit_extra_fast(Pm, Ca, M, n_outer=3, sub=subm)
qm = Pm["a"][subm, :2] @ rot2(rm_["psi"]).T + rm_["tau"]; lcm = []
for s in SUBJ:
    m = Pm["sid"][subm] == s; u = qm[m, 1] - qm[m, 1].mean(); v = Pm["b"][subm][m, 1] - Pm["b"][subm][m, 1].mean(); lcm.append(float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum() + 1e-12)))
out["mirror_diagnostic"] = dict(R0_mirror=M.tolist(), det=-1, psi_deg=round(float(np.degrees(rm_["psi"])), 3), tau=rm_["tau"].round(4).tolist(),
                                med_r=round(float(np.median(np.linalg.norm(rm_["r"], axis=1))), 4), lat_corr_med=round(float(np.median(lcm)), 3), note="채택하지 않음 (반사 금지). 비교용")
log("거울 진단", out["mirror_diagnostic"])

# ---------------------------------------------------------------- 3. 최종 적합 (모든 프레임)
P = pairs(D, LAGS, R0)
fin = fit_extra_fast(P, Ca, R0, psi0=res0["psi"], tau0=res0["tau"], n_outer=5)
EX = fin["extra_sigma"]; CaR = fin["Ca_room_xy"]
psi, tau = fin["psi"], fin["tau"]
rn = np.linalg.norm(fin["r"], axis=1)
log("최종 ψ", round(np.degrees(psi), 3), "τ", tau.round(4), "Σ_extra σ", EX.round(4), "med|r|", round(float(np.median(rn)), 4), "n", len(rn))
# 위치 기반 시간 재확인
pos_lag = []
for L in range(-3, 4):
    PL = pairs(D, {s: L for s in SUBJ}, R0)
    rL = bev_fit_fast(PL["a"][:, :2], PL["b"], PL["Sb"], np.diag(EX ** 2), psi0=psi, tau0=tau, Ca_room=CaR)
    hub = np.where(np.sqrt(rL["m"]) <= 2, rL["m"], 4 * np.sqrt(rL["m"]) - 4)
    pos_lag.append(dict(L=L, med_r=round(float(np.median(np.linalg.norm(rL["r"], axis=1))), 4), mean_huber=round(float(hub.mean()), 4), n=int(rL["n"]),
                        psi_deg=round(float(np.degrees(rL["psi"])), 3), tau=rL["tau"].round(4).tolist()))
log("위치 기반 지연", [(r["L"], r["mean_huber"], r["med_r"]) for r in pos_lag])
out["time"]["position_based_scan"] = pos_lag
out["time"]["position_based_best"] = min(pos_lag, key=lambda r: r["mean_huber"])["L"]

R = rotz(psi) @ R0
tz = cam_h + DH_ASSUMED
t = np.array([tau[0], tau[1], tz])
out["final"] = dict(psi_deg=float(np.degrees(psi)), psi_rel_R0_deg=float(np.degrees(psi)), tau=tau.tolist(), R=R.tolist(), t=t.tolist(), gn_cov=fin["cov"].tolist(),
                    extra_sigma=EX.tolist(), med_r=float(np.median(rn)), p90_r=float(np.percentile(rn, 90)), n_pairs=int(len(rn)),
                    r_median_xy=np.median(fin["r"], 0).tolist(), huber_inlier_frac=float((fin["w"] >= 1).mean()),
                    z_rule=dict(camera1_height_m=cam_h, delta_h_assumed_m=DH_ASSUMED, delta_h_sigma_m=DH_SIGMA, pitch_deg=0.0, roll_deg=0.0,
                                note="같은 책상·수평 장착 가정. 방 z 는 정답 없는 ViTPose 발목 바닥 (사양 mri_v03)"))
# 바닥 파일 확인 (있으면 보고만)
fl = "/home/interns/unisense/dev/packs/v3/floors.json"
out["floors_json"] = dict(exists=os.path.exists(fl))
if os.path.exists(fl):
    try:
        F = json.load(open(fl)); out["floors_json"]["mri_keys"] = [k for k in F if "mri" in k.lower()] if isinstance(F, dict) else None
    except Exception as e:
        out["floors_json"]["error"] = str(e)


# 3-DoF 를 "피험자 거리에서의 옆 오프셋" 으로도 보고 (yaw–옆 이동 상관 대비)
def lat_at(psi_, tau_, rr=2.4):
    """레이더 보어사이트 거리 rr 의 점이 방 y 에서 어디에 놓이는가 (= τy + rr·sin(ψ − ψ_nominal)) ."""
    p_sensor_room = (R0 @ np.array([0.0, rr, 0.0]))[:2]       # R0 로 옮긴 보어사이트 점 (회전 전)
    q = rot2(psi_) @ p_sensor_room + tau_
    return float(q[1]), float(q[0])


out["final"]["boresight_2p4m_room_xy"] = list(lat_at(psi, tau)[::-1])

# ---------------------------------------------------------------- 4. 반복성: 피험자별, LOSO, 절반
def boot_fit(P_, psi_, tau_, idx):
    r_ = bev_fit_fast(P_["a"][idx, :2], P_["b"][idx], P_["Sb"][idx], np.diag(EX ** 2), psi0=psi_, tau0=tau_, Ca_room=CaR, iters=15)
    return np.r_[r_["psi"], r_["tau"]]


def blocks_of(n, L):
    return [np.arange(i, min(i + L, n)) for i in range(0, n, L)]


per = {}
for s in SUBJ:
    Ps = pairs(D, LAGS, R0, subj=[s])
    fs = fit_extra_fast(Ps, Ca, R0, psi0=psi, tau0=tau, n_outer=3)
    bl = blocks_of(len(Ps["b"]), BLOCK); bs = []
    for b_ in range(B_SUBJ):
        pick = rng.integers(0, len(bl), len(bl)); idx = np.concatenate([bl[i] for i in pick])
        r_ = bev_fit_fast(Ps["a"][idx, :2], Ps["b"][idx], Ps["Sb"][idx], np.diag(fs["extra_sigma"] ** 2), psi0=fs["psi"], tau0=fs["tau"], Ca_room=CaR, iters=15)
        bs.append(np.r_[r_["psi"], r_["tau"]])
    bs = np.array(bs); C = np.cov(bs.T)
    la, xa = lat_at(fs["psi"], fs["tau"])
    lab = np.array([lat_at(b[0], b[1:])[0] for b in bs])
    per[s] = dict(psi_deg=float(np.degrees(fs["psi"])), tau=fs["tau"].tolist(), boot_sd=[float(np.degrees(np.sqrt(C[0, 0]))), float(np.sqrt(C[1, 1])), float(np.sqrt(C[2, 2]))],
                  boot_cov=C.tolist(), extra_sigma=fs["extra_sigma"].tolist(), med_r=float(np.median(np.linalg.norm(fs["r"], axis=1))), n=int(fs["n"]),
                  lat_at_2p4=la, lat_at_2p4_sd=float(lab.std()), r_median_xy=np.median(fs["r"], 0).tolist())
    log(f"S{s:02d} ψ {per[s]['psi_deg']:.2f}±{per[s]['boot_sd'][0]:.2f}°  τ ({fs['tau'][0]:+.3f}±{per[s]['boot_sd'][1]:.3f}, {fs['tau'][1]:+.3f}±{per[s]['boot_sd'][2]:.3f})  옆@2.4m {la:+.3f}±{lab.std():.3f}  med|r| {per[s]['med_r']:.3f}")
th = np.array([[np.radians(per[s]["psi_deg"]), *per[s]["tau"]] for s in SUBJ])
mean_th = np.array([psi, *tau])
chi2 = float(sum((th[i] - mean_th) @ np.linalg.inv(np.array(per[s]["boot_cov"])) @ (th[i] - mean_th) for i, s in enumerate(SUBJ)))
lat = np.array([per[s]["lat_at_2p4"] for s in SUBJ]); lat_sd = np.array([per[s]["lat_at_2p4_sd"] for s in SUBJ])
out["per_subject"] = {f"S{s:02d}": v for s, v in per.items()}
out["repeatability"] = dict(psi_deg_sd=float(np.degrees(th[:, 0].std(ddof=1))), tau_x_sd=float(th[:, 1].std(ddof=1)), tau_y_sd=float(th[:, 2].std(ddof=1)),
                            psi_deg_range=[float(np.degrees(th[:, 0].min())), float(np.degrees(th[:, 0].max()))],
                            tau_x_range=[float(th[:, 1].min()), float(th[:, 1].max())], tau_y_range=[float(th[:, 2].min()), float(th[:, 2].max())],
                            lat_at_2p4_sd=float(lat.std(ddof=1)), lat_at_2p4_median_boot_sd=float(np.median(lat_sd)),
                            median_boot_sd=dict(psi_deg=float(np.median([per[s]["boot_sd"][0] for s in SUBJ])), tau_x=float(np.median([per[s]["boot_sd"][1] for s in SUBJ])),
                                                tau_y=float(np.median([per[s]["boot_sd"][2] for s in SUBJ]))),
                            chi2_vs_pooled=chi2, chi2_dof=3 * (len(SUBJ) - 1),
                            corr_psi_tauy=float(np.corrcoef(th[:, 0], th[:, 2])[0, 1]))
log("반복성", json.dumps(out["repeatability"]))

loso = []
for s in SUBJ:
    Pl = pairs(D, LAGS, R0, subj=[q for q in SUBJ if q != s])
    rl = bev_fit_fast(Pl["a"][:, :2], Pl["b"], Pl["Sb"], np.diag(EX ** 2), psi0=psi, tau0=tau, Ca_room=CaR)
    loso.append(dict(left_out=f"S{s:02d}", psi_deg=float(np.degrees(rl["psi"])), tau=rl["tau"].tolist(), lat_at_2p4=lat_at(rl["psi"], rl["tau"])[0]))
L_ = np.array([[np.radians(r["psi_deg"]), *r["tau"]] for r in loso])
out["loso"] = dict(rows=loso, psi_deg_sd=float(np.degrees(L_[:, 0].std(ddof=1))), tau_sd=L_[:, 1:].std(0, ddof=1).tolist(),
                   max_dev=dict(psi_deg=float(np.degrees(np.abs(L_[:, 0] - psi).max())), tau=np.abs(L_[:, 1:] - tau).max(0).tolist()))
log("LOSO", json.dumps({k: v for k, v in out["loso"].items() if k != "rows"}))

halves = {}
for nm, fr in [("first", lambda s, t_, T: t_ < T // 2), ("second", lambda s, t_, T: t_ >= T // 2)]:
    Ph = pairs(D, LAGS, R0, frames=fr)
    rh = fit_extra_fast(Ph, Ca, R0, psi0=psi, tau0=tau, n_outer=3)
    halves[nm] = dict(psi_deg=float(np.degrees(rh["psi"])), tau=rh["tau"].tolist(), extra_sigma=rh["extra_sigma"].tolist(), n=int(rh["n"]), lat_at_2p4=lat_at(rh["psi"], rh["tau"])[0])
halves["diff"] = dict(psi_deg=halves["second"]["psi_deg"] - halves["first"]["psi_deg"], tau=(np.array(halves["second"]["tau"]) - np.array(halves["first"]["tau"])).tolist(),
                      lat_at_2p4=halves["second"]["lat_at_2p4"] - halves["first"]["lat_at_2p4"])
out["halves"] = halves
log("절반", json.dumps(halves["diff"]))

# ---------------------------------------------------------------- 5. 2단 블록 부트스트랩 (공통 변환)
sub_idx = {s: np.flatnonzero(P["sid"] == s) for s in SUBJ}
sub_blocks = {s: [sub_idx[s][b_] for b_ in blocks_of(len(sub_idx[s]), BLOCK)] for s in SUBJ}
bs = []
for b_ in range(B_BOOT):
    ss = rng.choice(SUBJ, len(SUBJ), replace=True); idx = []
    for s in ss:
        bl = sub_blocks[s]; pick = rng.integers(0, len(bl), len(bl)); idx.append(np.concatenate([bl[i] for i in pick]))
    bs.append(boot_fit(P, psi, tau, np.concatenate(idx)))
    if (b_ + 1) % 50 == 0:
        log("부트스트랩", b_ + 1)
bs = np.array(bs); C = np.cov(bs.T)
lab = np.array([lat_at(b[0], b[1:])[0] for b in bs])
out["bootstrap"] = dict(B=B_BOOT, block_frames=BLOCK, scheme="two-level: subjects with replacement, then 87-frame blocks within subject",
                        sd=dict(psi_deg=float(np.degrees(np.sqrt(C[0, 0]))), tau_x=float(np.sqrt(C[1, 1])), tau_y=float(np.sqrt(C[2, 2]))),
                        cov=C.tolist(), corr=(C / np.sqrt(np.outer(np.diag(C), np.diag(C)))).tolist(),
                        ci95=dict(psi_deg=np.degrees(np.percentile(bs[:, 0], [2.5, 97.5])).tolist(), tau_x=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(), tau_y=np.percentile(bs[:, 2], [2.5, 97.5]).tolist()),
                        lat_at_2p4_sd=float(lab.std()))
log("부트스트랩 σ", json.dumps(out["bootstrap"]["sd"]), "옆@2.4m σ", round(float(lab.std()), 4))
jdump(out, f"{WORK}/registration_{VARIANT}.json")
log("REG_DONE")
