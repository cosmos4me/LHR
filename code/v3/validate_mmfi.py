"""v3 MM-Fi 정답 없는 검증 (2026-11-24). 정답을 읽지 않는다.
 (a) 깊이 기준의 세션 편향: 세션별 배경 Δ(1/Z) (cache_v1 bg_med, 방 중앙 배경 대비) — D435 시차 오프셋이면 거리와 무관한 상수.
     LiDAR·레이더의 세션별 앞 (x3) 편차와의 상관.
 (b) LiDAR 반복성: 녹화·세션·LOSO·방 사이 (리그 강체성), 바닥 법선으로 본 LiDAR–카메라 회전 일관성.
 (c) LiDAR hold-out: 녹화 앞 절반으로 ICP → 뒤 절반 잔차, 방 단위 홀수/짝수 세션.
 (d) LiDAR 주입 복원: 원시 LiDAR 에 (yaw +3°, pitch +1°, 이동 (+0.05, −0.08, +0.03) m) 주입 → 바닥 + 앵커 탐색 + ICP 재실행.
 (e) 닫힘: 레이더↔LiDAR 직접 적합 (깊이 없이) 과 LiDAR→방 ∘ 레이더→LiDAR vs 레이더→방. 수직도 LiDAR 사람 높이로 H 재추정.
 (f) 레이더 검증 요약 (fit.json).
출력: dev/packs/v3/mmfi_radar/_reg/validation.json. 사용: python -I validate_mmfi.py"""
import sys, os, json, zlib
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
import mmfi_common as mc
import reg_lidar_mmfi as RL
import fit_radar_mmfi as FR
from reg_radar_mmfi import R0, M, QP

X3 = 3.1
LR = json.load(open(mc.WORK_L / "lidar_room.json")); RF_ = json.load(open(mc.WORK_R / "fit.json"))


# ---------------- (a) 깊이 세션 편향 ----------------
def depth_bias():
    out = {}
    for env in mc.ENVS:
        recs = mc.recordings((env,)); B = []
        for e, nm in recs:
            Z = mc.SafeNpz(mc.CACHE / e / f"{nm}.npz", mc.ALLOW_CACHE); z = Z["bg_med"].astype(float); v = Z["bg_valid"]; z[(v < 0.5) | ~np.isfinite(z)] = np.nan; B.append(z)
        B = np.array(B)
        with np.errstate(all="ignore"):
            ref = np.nanmedian(B, 0)
        per = {}
        for (e, nm), z in zip(recs, B):
            m = np.isfinite(z) & np.isfinite(ref) & (np.abs(z - ref) < 0.3) & (ref > 1.0) & (ref < 6); m[:, 55:105] = False
            d = 1 / z[m] - 1 / ref[m]; zz = ref[m]
            per[nm] = (np.median(d), np.median(d[zz < 2.5]) if (zz < 2.5).sum() > 50 else np.nan, np.median(d[zz > 3.5]) if (zz > 3.5).sum() > 50 else np.nan)
        subs = sorted({nm[:3] for nm in per}); S = {}
        for s in subs:
            a = np.array([v for k, v in per.items() if k[:3] == s])
            S[s] = dict(dinv=float(np.median(a[:, 0])), dinv_near=float(np.nanmedian(a[:, 1])), dinv_far=float(np.nanmedian(a[:, 2])), dZ_at_3p1=float(-X3 ** 2 * np.median(a[:, 0])),
                        disparity_px=float(19.175 * np.median(a[:, 0])))
        lid = LR[env]["per_subject"]; rad = RF_[env]["per_subject"]
        xs = np.array([S[s]["dZ_at_3p1"] for s in subs]); xl = np.array([lid[s]["x3"] for s in subs]) - LR[env]["x3"]
        xr = np.array([rad[s]["tx"] + X3 * np.cos(np.radians(rad[s]["psi_deg"])) for s in subs]); xr = xr - np.median(xr)
        out[env] = dict(per_subject=S, corr_dZ_vs_lidar_x3=float(np.corrcoef(xs, xl)[0, 1]), slope_lidar=float(np.polyfit(xs, xl, 1)[0]),
                        corr_dZ_vs_radar_x3=float(np.corrcoef(xs, xr)[0, 1]), slope_radar=float(np.polyfit(xs, xr, 1)[0]),
                        corr_lidar_vs_radar_x3=float(np.corrcoef(xl, xr)[0, 1]),
                        lidar_x3_dev=dict(zip(subs, xl.round(4).tolist())), radar_x3_dev=dict(zip(subs, xr.round(4).tolist())))
        print(env, "깊이 Δ(1/Z) 세션", {s: round(S[s]["dZ_at_3p1"], 3) for s in subs}, "corr LiDAR %.2f radar %.2f, LiDAR↔radar %.2f" % (
            out[env]["corr_dZ_vs_lidar_x3"], out[env]["corr_dZ_vs_radar_x3"], out[env]["corr_lidar_vs_radar_x3"]), flush=True)
    return out


# ---------------- (b) LiDAR 반복성·리그 ----------------
def lidar_repeat():
    out = {}
    for env in mc.ENVS:
        L = LR[env]; ps = L["per_subject"]; lo = L["loso"]
        f = lambda d, k: float(np.std([v[k] for v in d.values()]))
        out[env] = dict(rec_spread=L["rec_spread"], per_subject_std=dict(psi_deg=f(ps, "psi_deg"), x3=f(ps, "x3"), y3=f(ps, "y3")),
                        loso_std=dict(psi_deg=f(lo, "psi_deg"), x3=f(lo, "x3"), y3=f(lo, "y3")), sigma=L["sigma"])
    rooms = dict(psi_deg=[np.degrees(LR[e]["psi"]) for e in mc.ENVS], x3=[LR[e]["x3"] for e in mc.ENVS], y3=[LR[e]["y3"] for e in mc.ENVS], hL=[LR[e]["hL"] for e in mc.ENVS])
    out["across_rooms"] = {k: dict(values=np.round(v, 4).tolist(), std=float(np.std(v))) for k, v in rooms.items()}
    # 바닥 법선: LiDAR 좌표 n_L^e vs 카메라 좌표 n_C^e (= R_d2r 의 셋째 행) → 한 회전으로 맞춰지는가 (기울기 변화의 일관성)
    nL, nC = [], []
    for e in mc.ENVS:
        nL.append(np.array(LR[e]["nL"])); nm = mc.recordings((e,))[0][1]; P = mc.SafeNpz(mc.DPK / e / f"{nm}.npz", mc.ALLOW_DEPTH_PACK); nC.append(np.asarray(P["R_d2r"], float)[2])
    nL, nC = np.array(nL), np.array(nC); H = nL.T @ nC; U_, S_, Vt = np.linalg.svd(H); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U_.T))]); R = Vt.T @ D @ U_.T
    res = np.degrees(np.arccos(np.clip(((nL @ R.T) * nC).sum(1), -1, 1)))
    tiltL = np.degrees(np.arccos(np.clip(nL @ nL.mean(0) / np.linalg.norm(nL.mean(0)), -1, 1))); tiltC = np.degrees(np.arccos(np.clip(nC @ nC.mean(0) / np.linalg.norm(nC.mean(0)), -1, 1)))
    out["rig_floor_normals"] = dict(resid_deg=res.round(3).tolist(), lidar_tilt_change_deg=tiltL.round(3).tolist(), camera_tilt_change_deg=tiltC.round(3).tolist(),
                                    note="LiDAR 바닥 법선과 카메라 바닥 법선이 방마다 같은 회전으로 이어지면 (잔차 작음) 두 센서는 같은 강체 플랫폼에서 함께 기울어짐")
    print("LiDAR 방 사이:", out["across_rooms"], "바닥 법선 잔차", out["rig_floor_normals"], flush=True)
    return out


# ---------------- (c, d) LiDAR hold-out / 주입 ----------------
def lidar_rec_tests(job):
    env, nm, inject = job
    A = dict(np.load(mc.WORK_L / f"A_{env}_{nm}.npz", allow_pickle=False)); F = mc.lidar_frames(env, nm)
    D, Rd, td, zc = mc.depth_person(env, nm, drop_edge=True); T = int(A["T"]); Rt, o = A["Rt"], A["o"]
    C = np.load(mc.WORK_L / f"C_{env}_{nm}.npz", allow_pickle=False); psi0, t0 = float(C["psi"]), C["t"]
    res = dict(env=env, nm=nm)
    # hold-out: 앞 절반 적합 → 뒤 절반 잔차, 뒤 절반 자체 적합과 비교
    g1 = RL.gather_tr(F, D, A, psi0, t0, 0, range(0, T // 2), Rt, o); g2 = RL.gather_tr(F, D, A, psi0, t0, 0, range(T // 2, T), Rt, o)
    p1, t1, m1, _ = RL.icp(g1[0], g1[2], g1[1], g1[3], [psi0, *t0]); p2, t2, m2, _ = RL.icp(g2[0], g2[2], g2[1], g2[3], [psi0, *t0])
    _, _, m12, _ = RL.icp(g2[0], g2[2], g2[1], g2[3], [p1, *t1], iters=0)
    P3 = lambda p, t: np.array([X3 * np.cos(p) + t[0], X3 * np.sin(p) + t[1]])
    res["holdout"] = dict(resid_second_half_own=m2, resid_second_half_from_first=m12, d_psi_deg=float(np.degrees(p1 - p2)), d_p3=(P3(p1, t1) - P3(p2, t2)).tolist())
    if inject:
        dpsi, dpitch, dt = np.radians(3.0), np.radians(1.0), np.array([0.05, -0.08, 0.03])
        cy, sy = np.cos(dpitch), np.sin(dpitch); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rzz = mc.Rz(dpsi); Rinj = Rzz @ Ry
        Fi = [f @ Rinj.T + dt for f in F]                                   # 주입: x' = Rinj x + dt
        nL, cL, inl, sig, ext = RL.floor_lidar(Fi); Rl = mc.rot_a_to_b(nL, np.array([0, 0, 1.0]))
        Ai = dict(A); Ai.update(Rl=Rl, cL=cL)
        Hh = np.zeros((RL.NB, RL.NB)); cd = A["cd"]
        for k in range(0, T, 2):
            if not np.isfinite(cd[k]).all():
                continue
            q = Fi[k] @ Rl.T; h = q[:, 2] + cL; m = (h > 0.3) & (h < 1.9); r = cd[k] - q[m, :2]; i = np.floor((r + RL.HB) / RL.HSTEP).astype(int); ok = (i >= 0).all(1) & (i < RL.NB).all(1)
            np.add.at(Hh, (i[ok, 0], i[ok, 1]), 1.0)
        tc, _, _ = RL.peak(Hh, rmax=1.2)
        g = RL.gather(Fi, D, Ai, tc, 0, range(2, T - 2, 2), Rt, o); pi_, ti_, _, _ = RL.icp(g[0], g[2], g[1], g[3], [0.0, *tc])
        for _ in range(2):
            g = RL.gather_tr(Fi, D, Ai, pi_, ti_, 0, range(T), Rt, o); pi_, ti_, mi_, _ = RL.icp(g[0], g[2], g[1], g[3], [pi_, *ti_])
        # 원래 사람 점 (원시) x 에 대해: 원래 변환 T0(x) vs 주입 후 변환 Ti(Rinj x + dt) — 방 좌표 차이
        def to_room(Fr, Rl_, cL_, psi, t):
            q = Fr @ Rl_.T; c, s = np.cos(psi), np.sin(psi); return np.c_[q[:, :2] @ np.array([[c, s], [-s, c]]) + t, q[:, 2] + cL_]
        d = []
        for k in range(0, T, 10):
            x = F[k]; q = x @ A["Rl"].T; h = q[:, 2] + float(A["cL"]); c, s = np.cos(psi0), np.sin(psi0); anchor = (cd[k] - t0) @ np.array([[c, -s], [s, c]])
            idx = RL.extract(q[:, :2], h, anchor) if np.isfinite(cd[k]).all() else []
            if len(idx) < 15:
                continue
            X0 = to_room(x[idx], A["Rl"], float(A["cL"]), psi0, t0); Xi = to_room(Fi[k][idx], Rl, cL, pi_, ti_); d.append(np.median(Xi - X0, 0))
        d = np.array(d)
        res["inject"] = dict(injected=dict(yaw_deg=3.0, pitch_deg=1.0, t=dt.tolist()), recovered_yaw_minus_orig_deg=float(np.degrees(pi_ - psi0)),
                             floor_tilt_change_deg=float(np.degrees(np.arccos(np.clip(nL @ (Rinj @ A["nL"]), -1, 1)))), h_change=float(cL - float(A["cL"])),
                             room_point_error_med=np.median(d, 0).tolist() if len(d) else None, room_point_error_abs_max=np.abs(d).max(0).tolist() if len(d) else None)
    return res


# ---------------- (e) 닫힘 ----------------
def closure_room(env):
    """레이더 공칭 중심 ↔ LiDAR 사람 중심 (방 바닥 수평 L' 좌표) 직접 적합 (깊이 안 씀; 게이트만 방 레이더 변환 사용)."""
    L = LR[env]; Rl_room = np.array(L["Rl"]); hL = L["hL"]; f = RF_[env]["fit"]; psiR = np.radians(f["psi_deg"]); tR = np.array([f["tx"], f["ty"]]); HR = f["H"]
    cR, sR = np.cos(psiR), np.sin(psiR); R2R = np.array([[cR, -sR], [sR, cR]]); psiL, tL = L["psi"], np.array([L["tx"], L["ty"]]); cL_, sL_ = np.cos(psiL), np.sin(psiL); R2L = np.array([[cL_, -sL_], [sL_, cL_]])
    rows, sess, vrows = [], [], []
    for e, nm in mc.recordings((env,)):
        A = np.load(mc.WORK_L / f"A_{e}_{nm}.npz", allow_pickle=False); C = np.load(mc.WORK_L / f"C_{e}_{nm}.npz", allow_pickle=False)
        if not bool(C["ok"]):
            continue
        Ra = dict(np.load(mc.WORK_R / f"A_{e}_{nm}.npz", allow_pickle=False)); Lr, dop, g, off = FR.nominal(Ra)
        pts, poff = C["pts"].astype(float), C["off"]; Rl_rec, cL_rec = A["Rl"], float(A["cL"])
        for k in range(0, int(A["T"]), 2):
            p = pts[poff[k]:poff[k + 1]]
            if len(p) < 15:
                continue
            q = (np.c_[p[:, :2], p[:, 2] - cL_rec] @ Rl_rec) @ Rl_room.T; cl = np.median(q[:, :2], 0); zl = q[:, 2] + hL     # L' 수평, 높이
            m = np.arange(off[k], off[k + 1]); m = m[g[m]]
            if len(m) < 3:
                continue
            Q = Lr[m]; Qr = Q[:, :2] @ R2R.T + tR; clr = cl @ R2L.T + tL                                # 게이트: 방 좌표에서 LiDAR 사람 근처
            keep = (np.linalg.norm(Qr - clr, axis=1) < 0.7) & (Q[:, 2] + HR > -0.2) & (Q[:, 2] + HR < 2.3)
            if keep.sum() < 3:
                continue
            u = cl / np.linalg.norm(cl); rows.append((*np.median(Q[keep, :2], 0), *(cl + FR.LOS_PRIOR * u))); sess.append(nm[:3])
            vrows.append((*np.quantile(Q[keep, 2], QP), *np.quantile(zl, QP)))
    rows = np.array(rows); vrows = np.array(vrows); sess = np.array(sess)
    xRL, _ = FR.fit_h(np.c_[np.zeros(len(rows)), rows], (0.0, 0.0, 0.0))
    # 수직: LiDAR 사람 높이 분위수를 기준으로 V2 → H_RL (레이더 3 m 유효 높이, LiDAR 바닥 기준)
    vr = np.c_[np.zeros((len(vrows), 5)), vrows]; vfit = FR.fit_v(vr)
    # 합성: 레이더 → L' (xRL) → 방 (LiDAR 변환)
    psiC = psiL + xRL[0]; tC = R2L @ xRL[1:3] + tL
    P3 = lambda p, t: np.array([X3 * np.cos(p) + t[0], X3 * np.sin(p) + t[1]])
    dp3 = P3(psiC, tC) - P3(psiR, tR)
    # 부트스트랩 (세션 군집)
    rng = np.random.default_rng(int(env[1:]) + 7); subs = sorted(set(sess)); bs = []
    for _ in range(200):
        pick = rng.choice(subs, len(subs)); idx = np.concatenate([np.flatnonzero(sess == s) for s in pick]); xb, _ = FR.fit_h(np.c_[np.zeros(len(idx)), rows[idx]], xRL)
        pc = psiL + xb[0]; tc_ = R2L @ xb[1:3] + tL; bs.append([*(P3(pc, tc_) - P3(psiR, tR)), np.degrees(pc - psiR)])
    bs = np.array(bs)
    # 세션별 레이더−LiDAR 앞 오프셋 (깊이 편향이 원인이면 세션 사이 일정해야 함)
    per_s = {}
    for s in subs:
        ii = np.flatnonzero(sess == s); xs, _ = FR.fit_h(np.c_[np.zeros(len(ii)), rows[ii]], xRL); pc = psiL + xs[0]; tc_ = R2L @ xs[1:3] + tL
        per_s[s] = (P3(pc, tc_) - P3(psiR, tR)).round(4).tolist()
    sigL = LR[env]["sigma"]; sigR = RF_[env]["boot_rec"]
    comb_x = float(np.sqrt(sigL["x3"] ** 2 + sigR["sigma_tx"] ** 2 + bs[:, 0].std() ** 2)); comb_y = float(np.sqrt(sigL["y3"] ** 2 + sigR["sigma_lat_at_3m"] ** 2 + bs[:, 1].std() ** 2))
    out = dict(radar_to_lidar=dict(psi_deg=float(np.degrees(xRL[0])), t=xRL[1:3].tolist()), closure_p3=dp3.tolist(), closure_yaw_deg=float(np.degrees(psiC - psiR)),
               closure_sigma_boot=bs.std(0).tolist(), combined_sigma_x=comb_x, combined_sigma_y=comb_y,
               H_from_lidar_v2=vfit["H_v2"], H_from_lidar_v1=vfit["H_v1"], H_from_depth=f["H"], H_from_depth_v1=f["H_v1"], s_from_lidar=vfit["s"],
               per_subject_dp3=per_s, per_subject_dx_std=float(np.std([v[0] for v in per_s.values()])), n=int(len(rows)))
    print(env, "닫힘 Δp3 (m)", np.round(dp3, 3), "Δψ %.2f°" % out["closure_yaw_deg"], "결합 σ (x, y)", round(comb_x, 3), round(comb_y, 3),
          "| H: depth V2 %.3f / LiDAR V2 %.3f, depth V1 %.3f / LiDAR V1 %.3f" % (f["H"], vfit["H_v2"], f["H_v1"], vfit["H_v1"]), "세션 Δx std %.3f" % out["per_subject_dx_std"], flush=True)
    return env, out


if __name__ == "__main__":
    V = {}
    V["depth_session_bias"] = depth_bias()
    V["lidar_repeat"] = lidar_repeat()
    recs = mc.recordings(); sub = [r for i, r in enumerate(recs) if i % 5 == 0]
    jobs = [(e, nm, (i % 9 == 0)) for i, (e, nm) in enumerate(sub)]
    with Pool(int(os.environ.get("V3_NP", "24"))) as p:
        T = p.map(lidar_rec_tests, jobs)
        CL = dict(p.map(closure_room, mc.ENVS))
    ho = np.array([[t["holdout"]["resid_second_half_own"], t["holdout"]["resid_second_half_from_first"], t["holdout"]["d_psi_deg"], *t["holdout"]["d_p3"]] for t in T])
    V["lidar_holdout"] = dict(n=len(T), resid_own_med=float(np.nanmedian(ho[:, 0])), resid_from_first_med=float(np.nanmedian(ho[:, 1])), ratio_med=float(np.nanmedian(ho[:, 1] / ho[:, 0])),
                              abs_d_psi_deg_med=float(np.nanmedian(np.abs(ho[:, 2]))), abs_d_x3_med=float(np.nanmedian(np.abs(ho[:, 3]))), abs_d_y3_med=float(np.nanmedian(np.abs(ho[:, 4]))),
                              abs_d_y3_p90=float(np.nanpercentile(np.abs(ho[:, 4]), 90)))
    inj = [t["inject"] for t in T if "inject" in t]
    E = np.array([x["room_point_error_med"] for x in inj if x["room_point_error_med"] is not None])
    V["lidar_injection"] = dict(n=len(inj), injected=inj[0]["injected"] if inj else None, recovered_yaw_minus_orig_deg=[round(x["recovered_yaw_minus_orig_deg"], 3) for x in inj],
                                floor_tilt_change_deg=[round(x["floor_tilt_change_deg"], 3) for x in inj], h_change=[round(x["h_change"], 4) for x in inj],
                                room_point_error_abs_med=np.median(np.abs(E), 0).tolist() if len(E) else None, room_point_error_abs_max=np.abs(E).max(0).tolist() if len(E) else None)
    # 방 단위 hold-out (LiDAR): 홀수 세션 vs 짝수 세션 추정 차이
    ho_room = {}
    for env in mc.ENVS:
        R = LR[env]["rec"]; odd = [v for k, v in R.items() if int(k[1:3]) % 2 == 1]; even = [v for k, v in R.items() if int(k[1:3]) % 2 == 0]
        f = lambda rows: (np.median([r["psi_deg"] for r in rows]), np.median([r["x3"] for r in rows]), np.median([r["y3"] for r in rows]))
        a, b = f(odd), f(even); ho_room[env] = dict(d_psi_deg=float(a[0] - b[0]), d_x3=float(a[1] - b[1]), d_y3=float(a[2] - b[2]))
    V["lidar_holdout_room_odd_even"] = ho_room
    V["closure"] = CL
    V["radar"] = {e: {k: RF_[e][k] for k in ("axis_check", "time", "fit", "boot_rec", "repeat", "holdout", "injection")} for e in mc.ENVS}
    print("LiDAR hold-out", V["lidar_holdout"], "\n주입", V["lidar_injection"], "\n방 hold-out", ho_room, flush=True)
    json.dump(V, open(mc.WORK_R / "validation.json", "w"), indent=1, ensure_ascii=False, default=lambda o: None)
    print("VALIDATION DONE", flush=True)
