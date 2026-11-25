"""v3 mRI: 2D ViTPose 키포인트(두 카메라)만으로 정답 없는 스테레오 사람 궤적을 만든다.

절차
1. 각 카메라에서 좌우 평균 2D 점 (골반 중점 = 11·12, 어깨 중점 = 5·6, 몸통 중심 = 두 중점의 평균). 좌우 뒤바뀜에 둔감.
   점수 > 0.3 인 키포인트만.
2. 두 광선 가중 최소제곱 삼각측량 + 공분산 (좁은 기선 → 깊이 방향으로 길쭉한 이방성).
   광선 각 잡음 σ_θ 은 에피폴라 잔차(기선 ⟂ 성분)의 MAD 로 정답 없이 추정 (σ_θ = σ_epi/√2, 두 카메라 같다고 가정).
3. 거부: 카메라1 깊이 < 0.5 m 또는 > 6 m (뒤·먼 삼각측량), |에피폴라| > 4σ_epi, 골반 높이 ∉ [−0.2, 1.8] m.
4. 시간 중앙값 평활 (창 5 프레임 ≈ 0.57 s). 원 궤적 − 평활 궤적의 축별 고주파 잔차로 잡음 배율을 잼
   (모형 σ 대비, 깊이·옆·위 축 따로; 1 보다 작으면 1 — 고주파 잔차는 시간 상관 오차를 못 봄).
   (모형 공분산 + 0.03 m 등방 바닥) 마할라노비스 > 3.5 인 프레임을 버리고 다시 평활.
   깊이의 느린(시간 상관) 오차는 여기서 못 잼 → 정합 단계에서 잔차로 κ 배율을 추정.
5. 시간 정합용 움직임 신호: 두 카메라 17 키포인트 2D 속도 평균 → 몸통 깊이/f 로 m/s 환산 (외부 파라미터 무관).
출력: dev/packs/v3/mri_radar/work/stereo_Sxx.npz, work/stereo_summary.json
"""
import sys, os, json, time
sys.path.insert(0, "/home/interns/unisense/code/v3")
import numpy as np
from mri_radar_common import (load_spec, load_kp, triangulate, mid2d, nan_median_filter, WORK, SUBJ, jdump,
                              L_SH, R_SH, L_HIP, R_HIP, SCORE_MIN)

W_MED = 5
SIG_FLOOR = 0.03       # m, 이상치 판정용 등방 바닥 (사람 몸통의 프레임 간 실제 움직임·흔들림)
os.makedirs(WORK, exist_ok=True)
t0 = time.time()
cams, cam_h, fps_spec = load_spec()
base = cams[1]["t"] - cams[0]["t"]
print("기선 (방 좌표)", base.round(4), "길이", round(float(np.linalg.norm(base)), 4), "카메라 높이", round(cam_h, 4), flush=True)


def pts2d(K):
    out = {}
    for ci, (kk, ss) in enumerate([("kp_px", "kp_score"), ("kp_px_cam2", "kp_score_cam2")]):
        kp, sc = K[kk], K[ss]
        hm, hok = mid2d(kp, sc, L_HIP, R_HIP); sm, sok = mid2d(kp, sc, L_SH, R_SH)
        out[ci] = dict(hip=(hm, hok), sh=(sm, sok), torso=(0.5 * (hm + sm), hok & sok))
    return out


# ---- 1차: 에피폴라 잔차로 σ_θ 추정 (모든 피험자, 2D 만)
KP, META, P2 = {}, {}, {}
for s in SUBJ:
    KP[s], META[s] = load_kp(s); P2[s] = pts2d(KP[s])
sig = {}
for nm in ["hip", "sh", "torso"]:
    e_all = []
    for s in SUBJ:
        (u1, ok1), (u2, ok2) = P2[s][0][nm], P2[s][1][nm]; ok = ok1 & ok2
        X, C, epi, dep = triangulate(u1[ok], u2[ok], cams, np.array([2e-3, 2e-3]))
        good = (dep > 0.5) & (dep < 6.0)
        e_all.append(epi[good])
    e = np.concatenate(e_all); med = np.median(e); mad = 1.4826 * np.median(np.abs(e - med))
    sig[nm] = dict(epi_median_mrad=round(1e3 * float(med), 3), epi_sigma_mrad=round(1e3 * float(mad), 3), sigma_theta_mrad=round(1e3 * float(mad) / np.sqrt(2), 3), n=int(len(e)),
                   frac_gt_4sigma=round(float((np.abs(e - med) > 4 * mad).mean()), 4))
    print("에피폴라", nm, sig[nm], flush=True)

# ---- 2차: 삼각측량 + 거부 + 평활 + 잡음 배율
summary = dict(sigma_theta=sig, baseline_m=float(np.linalg.norm(base)), camera_height_m=cam_h, window=W_MED, subjects={})
resid_all = {nm: [] for nm in ["torso"]}
TR = {}
for s in SUBJ:
    T = META[s]["T"]; rec = {}
    for nm in ["hip", "sh", "torso"]:
        (u1, ok1), (u2, ok2) = P2[s][0][nm], P2[s][1][nm]; ok = ok1 & ok2
        st = sig[nm]["sigma_theta_mrad"] * 1e-3; e_med = sig[nm]["epi_median_mrad"] * 1e-3; e_sig = sig[nm]["epi_sigma_mrad"] * 1e-3
        X = np.full((T, 3), np.nan); C = np.full((T, 3, 3), np.nan); EP = np.full(T, np.nan); DP = np.full(T, np.nan)
        ii = np.flatnonzero(ok)
        if len(ii):
            x, c, epi, dep = triangulate(u1[ii], u2[ii], cams, np.array([st, st]))
            X[ii], C[ii], EP[ii], DP[ii] = x, c, epi, dep
        rej = dict(no_kp=int((~ok).sum()))
        bad = ok & ~((DP > 0.5) & (DP < 6.0)); rej["depth"] = int(bad.sum())
        bad2 = ok & ~bad & (np.abs(EP - e_med) > 4 * e_sig); rej["epipolar"] = int(bad2.sum())
        keep = ok & ~bad & ~bad2
        if nm == "hip":
            bad3 = keep & ~((X[:, 2] > -0.2) & (X[:, 2] < 1.8)); rej["height"] = int(bad3.sum()); keep &= ~bad3
        X[~keep] = np.nan; C[~keep] = np.nan
        rec[nm] = dict(X=X, C=C, keep=keep, rej=rej)
    # 골반 높이로 거부된 프레임은 몸통에서도 버림 (몸통 = 골반·어깨 중점의 평균이므로)
    bad = rec["torso"]["keep"] & ~rec["hip"]["keep"]
    rec["torso"]["rej"]["hip_rejected"] = int(bad.sum()); rec["torso"]["keep"] = rec["torso"]["keep"] & ~bad
    rec["torso"]["X"][bad] = np.nan; rec["torso"]["C"][bad] = np.nan
    TR[s] = rec

# 축별 잡음 배율: 원 − 평활 잔차를 모형 공분산 고유축(깊이·옆·위)에 투영
def eig_axes(C):
    w, V = np.linalg.eigh(C)                                   # 오름차순: [옆 또는 위, …, 깊이]
    return w, V


scale_rows = []
for s in SUBJ:
    X = TR[s]["torso"]["X"]; C = TR[s]["torso"]["C"]; Xs, nv = nan_median_filter(X, W_MED)
    ok = np.isfinite(X).all(1) & np.isfinite(Xs).all(1) & (nv >= 4)
    if ok.sum() < 50:
        continue
    w, V = eig_axes(C[ok]); r = (X[ok] - Xs[ok]); proj = np.einsum("nij,ni->nj", V, r)     # 고유축 좌표
    scale_rows.append((proj, np.sqrt(w)))
proj = np.concatenate([p for p, _ in scale_rows]); sd = np.concatenate([q for _, q in scale_rows])
# 중앙값 평활(창 5) 잔차의 분산은 백색 잡음일 때 σ²·(1 + 0.64⁻¹/5 − …) ≈ 1.16 σ² — 보정 계수 1/√1.16
emp = 1.4826 * np.median(np.abs(proj), 0) / np.sqrt(1.16); mod = np.median(sd, 0)
scale = np.maximum(emp / mod, 1.0)                              # 키우기만 함: 고주파 잔차는 시간 상관 오차를 못 보므로 줄이는 근거가 안 됨
print("모형 σ (고유축 작은→큰) m", mod.round(4), "고주파 경험 σ", emp.round(4), "배율", scale.round(2), flush=True)
summary["noise_scale_eigaxes"] = dict(model_sigma_m=mod.round(4).tolist(), empirical_sigma_m=emp.round(4).tolist(), scale=scale.round(3).tolist(),
                                      note="고유축 오름차순 (옆/위 → 깊이). 시간 고주파 잔차 기반 — 시간 상관 오차와 상수 치우침은 안 보임")

for s in SUBJ:
    T = META[s]["T"]; fps = META[s]["fps"]; out = {}
    for nm in ["hip", "sh", "torso"]:
        X = TR[s][nm]["X"].copy(); C = TR[s][nm]["C"].copy()
        # 고유축 배율 적용 (모든 점 종류에 몸통 배율 사용)
        okc = np.isfinite(C).all((1, 2))
        if okc.any():
            w, V = np.linalg.eigh(C[okc]); w = w * scale[None] ** 2; C[okc] = np.einsum("nij,nj,nkj->nik", V, w, V)
        Xs, nv = nan_median_filter(X, W_MED)
        okm = np.isfinite(X).all(1) & np.isfinite(Xs).all(1)
        r = X - Xs; m = np.full(T, np.nan)
        Cf = C[okm] + (SIG_FLOOR ** 2) * np.eye(3)[None]                # 몸 흔들림·실제 움직임 바닥 σ (거부 판정에만)
        m[okm] = np.einsum("ni,nij,nj->n", r[okm], np.linalg.inv(Cf), r[okm])
        out_r = okm & (m > 3.5 ** 2); X[out_r] = np.nan; C[out_r] = np.nan
        Xs, nv = nan_median_filter(X, W_MED)
        valid = np.isfinite(Xs).all(1) & (nv >= 3)
        Cs = np.full((T, 3, 3), np.nan)
        # 평활 궤적 공분산 ≈ 창 안 원 공분산 평균 × (π/2)/n_valid (중앙값 효율, 백색 잡음 가정)
        Cpad = np.concatenate([np.full((2, 3, 3), np.nan), C, np.full((2, 3, 3), np.nan)])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Cm = np.nanmean(np.stack([Cpad[i:i + T] for i in range(W_MED)], 0), 0)
        Cs[valid] = Cm[valid] * (np.pi / 2) / nv[valid][:, None, None]
        Xs[~valid] = np.nan
        out[nm] = dict(Xs=Xs, Cs=Cs, X=X, C=C, n_raw=int(np.isfinite(X).all(1).sum()), n_smooth=int(valid.sum()), n_outlier=int(out_r.sum()), rej=TR[s][nm]["rej"])
    # 움직임 신호 (외부 파라미터 무관): 2D 키포인트 속도 → m/s
    K = KP[s]; spd = []
    for kk, ss, ci in [("kp_px", "kp_score", 0), ("kp_px_cam2", "kp_score_cam2", 1)]:
        kp, sc = K[kk], K[ss]; ok = (sc > SCORE_MIN); d = np.linalg.norm(np.diff(kp, axis=0), axis=-1); okd = ok[1:] & ok[:-1]
        d[~okd] = np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                spd.append(np.nanmean(d, 1))
    f = 0.5 * (cams[0]["K"][0, 0] + cams[1]["K"][0, 0])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pxs = np.nanmean(np.stack(spd, 0), 0)
            depth = np.nanmedian(out["torso"]["Xs"][:, 0] - cams[0]["t"][0])
    img_speed = np.r_[np.nan, pxs * depth / f * fps]                         # (T,) m/s, 프레임 t−1→t
    np.savez_compressed(f"{WORK}/stereo_S{s:02d}.npz", torso=out["torso"]["Xs"], torso_cov=out["torso"]["Cs"], torso_raw=out["torso"]["X"], torso_raw_cov=out["torso"]["C"],
                        hip=out["hip"]["Xs"], hip_cov=out["hip"]["Cs"], sh=out["sh"]["Xs"], img_speed=img_speed, fps=fps, frame0=META[s]["frame0"], T=T)
    Xs = out["torso"]["Xs"]; v = np.isfinite(Xs).all(1)
    sd = np.sqrt(np.einsum("nii->ni", out["torso"]["Cs"][v])) if v.any() else np.zeros((0, 3))
    summary["subjects"][f"S{s:02d}"] = dict(T=T, fps=fps, frame0=META[s]["frame0"],
        torso=dict(n_raw=out["torso"]["n_raw"], n_smooth=out["torso"]["n_smooth"], n_outlier=out["torso"]["n_outlier"], rej=out["torso"]["rej"]),
        hip_valid=out["hip"]["n_smooth"], torso_xyz_pct=dict(p5=np.nanpercentile(Xs[v], 5, 0).round(3).tolist(), p50=np.nanpercentile(Xs[v], 50, 0).round(3).tolist(), p95=np.nanpercentile(Xs[v], 95, 0).round(3).tolist()),
        torso_sigma_med_m=np.median(sd, 0).round(4).tolist() if len(sd) else None,
        hip_z_median=round(float(np.nanmedian(out["hip"]["Xs"][:, 2])), 3))
    print(f"S{s:02d}", json.dumps(summary["subjects"][f"S{s:02d}"], ensure_ascii=False), flush=True)
jdump(summary, f"{WORK}/stereo_summary.json")
print("STEREO_DONE", round(time.time() - t0, 1), "s", flush=True)
