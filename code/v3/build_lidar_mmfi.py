"""v3 MM-Fi LiDAR 팩 조립 (2026-11-24). 입력 = reg_lidar_mmfi.py 의 녹화별 결과 (_reg/A_*, C_*). 정답을 읽지 않는다.
결정 (근거는 notes/v3_mmfi.md):
 - LiDAR 바닥은 녹화 사이 매우 안정 (방 안 pitch·roll 퍼짐 ≤ 0.02°, 높이 ≤ 2 mm) → 방 중앙값 바닥 → 방 z = 0 (환경 깊이 바닥) 에 맞춤.
   녹화별 깊이 바닥 (bg_med) 은 세션마다 D435 시차 오프셋 (1/32 px 계단) 때문에 기울어 보이므로 목표로 쓰지 않고 진단으로만 기록.
 - (ψ, tx, ty) 는 방 단위 강체: 녹화별 추정에서 yaw 는 옆 이동과 거의 퇴화 (녹화별 ψ 퍼짐 약 2°, 3 m 옆 오프셋 퍼짐 약 1 cm).
   그래서 3.1 m 앞 기준점의 방 좌표 p3 = T(3.1, 0) 를 녹화마다 구하고, 옆 성분 = 녹화 중앙값, 앞 성분 = 세션 중앙값들의 중앙값
   (세션별 깊이 시차 오프셋에 강건), ψ = 녹화 중앙값. 세션 군집 부트스트랩으로 σ.
 - 시간: 방 평균 상대 잔차가 τ = 0 에서 최소 → 이동 없음.
 - 사람 점: reg 단계에서 깊이 사람 앵커로 다시 뽑은 점 (E04 추출 실패 수정) 을 방 변환으로 옮겨 v02 와 같은 가상 핀홀 uvz (640×480, f = 400) 로 저장.
   라벨 칸은 형식만 유지 (0, mask False). 라벨은 깊이 팩 mmfi_v02_specA 에서 읽는다 (ms_cache2 와 같음).
출력: dev/packs/v3/mmfi_lidar/{env}_{nm}.npz, mmfi_lidar_v3_spec.json, _reg/lidar_room.json. 사용: python -I build_lidar_mmfi.py"""
import sys, os, json, zlib
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
import mmfi_common as mc

X3 = 3.1                                   # MM-Fi 피험자 거리 (플랫폼에서 약 3 m)
N_PTS, K_V = 256, [400.0, 400.0, 320.0, 240.0]
SPEC = mc.OUT_L / "mmfi_lidar_v3_spec.json"


def load_rec(env, nm):
    A = np.load(mc.WORK_L / f"A_{env}_{nm}.npz", allow_pickle=False); C = np.load(mc.WORK_L / f"C_{env}_{nm}.npz", allow_pickle=False)
    return A, C


def room_solution():
    sol = {}
    for env in mc.ENVS:
        recs = mc.recordings((env,)); R = []
        for e, nm in recs:
            A, C = load_rec(e, nm)
            if not bool(C["ok"]):
                continue
            psi = float(C["psi"]); t = C["t"]; p3 = np.array([X3 * np.cos(psi) + t[0], X3 * np.sin(psi) + t[1]])
            R.append(dict(nm=nm, sub=nm[:3], nL=A["nL"], cL=float(A["cL"]), psi=psi, t=t, p3=p3, tau=int(C["tau"]), tau_curve=C["tau_curve"],
                          corr=C["corr"].tolist(), med=float(C["med"]), boot_sig=C["boot"].std(0).tolist(), floor_d=json.loads(str(A["floor_d"])), floor_l=json.loads(str(A["floor_l"]))))
        nL = np.median([r["nL"] for r in R], 0); nL /= np.linalg.norm(nL); hL = float(np.median([r["cL"] for r in R]))
        subs = sorted({r["sub"] for r in R})

        def est(rows):
            psi = float(np.median([r["psi"] for r in rows])); y3 = float(np.median([r["p3"][1] for r in rows]))
            x3 = float(np.median([np.median([r["p3"][0] for r in rows if r["sub"] == s]) for s in sorted({r["sub"] for r in rows})]))
            return psi, x3, y3
        psi, x3, y3 = est(R); tx, ty = x3 - X3 * np.cos(psi), y3 - X3 * np.sin(psi)
        rng = np.random.default_rng(int(env[1:])); bs = []
        for _ in range(500):
            pick = rng.choice(subs, len(subs)); rows = [r for s in pick for r in R if r["sub"] == s]
            # 같은 세션이 두 번 뽑히면 세션 중앙값 계산에서 한 번만 세지 않도록 이름을 바꿔 붙임
            rows2 = []
            for i, s in enumerate(pick):
                rows2 += [dict(r, sub=f"{s}#{i}") for r in R if r["sub"] == s]
            p_, x_, y_ = est(rows2); bs.append([p_, x_, y_, x_ - X3 * np.cos(p_), y_ - X3 * np.sin(p_)])
        bs = np.array(bs)
        per_s = {}
        for s in subs:
            rr = [r for r in R if r["sub"] == s]; ps, xs, ys = float(np.median([r["psi"] for r in rr])), float(np.median([r["p3"][0] for r in rr])), float(np.median([r["p3"][1] for r in rr]))
            per_s[s] = dict(psi_deg=np.degrees(ps), x3=xs, y3=ys, tx=xs - X3 * np.cos(ps), ty=ys - X3 * np.sin(ps), n=len(rr),
                            depth_floor_tilt_deg=float(np.median([r["floor_d"]["tilt_deg"] for r in rr if r["floor_d"]])),
                            depth_floor_z_at_person=float(np.median([r["floor_d"]["z_at_person"] for r in rr if r["floor_d"]])))
        loso = {}
        for s in subs:
            p_, x_, y_ = est([r for r in R if r["sub"] != s]); loso[s] = dict(psi_deg=np.degrees(p_), x3=x_, y3=y_)
        # τ: 방 평균 상대 잔차
        cur = np.array([r["tau_curve"][:, 4] / r["tau_curve"][2, 4] for r in R if len(r["tau_curve"]) == 5])
        fl_dev = np.array([[np.degrees(np.arccos(np.clip(r["nL"] @ nL, -1, 1))), r["cL"] - hL] for r in R])
        Rl = mc.rot_a_to_b(nL, np.array([0, 0, 1.0]))
        sol[env] = dict(psi=psi, tx=float(tx), ty=float(ty), x3=x3, y3=y3, nL=nL.tolist(), hL=hL, Rl=Rl.tolist(), tau=0,
                        floor_pitch_deg=float(np.degrees(np.arctan2(nL[0], nL[2]))), floor_roll_deg=float(np.degrees(np.arctan2(nL[1], nL[2]))),
                        sigma=dict(psi_deg=float(np.degrees(bs[:, 0].std())), x3=float(bs[:, 1].std()), y3=float(bs[:, 2].std()), tx=float(bs[:, 3].std()), ty=float(bs[:, 4].std()),
                                   floor_angle_deg=float(np.median(fl_dev[:, 0]) * 1.4826), hL=float(1.4826 * np.median(np.abs(fl_dev[:, 1])))),
                        tau_rel_residual=dict(zip(map(str, (-2, -1, 0, 1, 2)), np.mean(cur, 0).round(4).tolist())),
                        rec_spread=dict(psi_deg_std=float(np.degrees(np.std([r["psi"] for r in R]))), x3_std=float(np.std([r["p3"][0] for r in R])), y3_std=float(np.std([r["p3"][1] for r in R])),
                                        floor_dev_deg_p95=float(np.percentile(fl_dev[:, 0], 95)), floor_dev_h_p95=float(np.percentile(np.abs(fl_dev[:, 1]), 95)),
                                        n_floor_outlier=int(((fl_dev[:, 0] > 0.3) | (np.abs(fl_dev[:, 1]) > 0.01)).sum())),
                        per_subject=per_s, loso=loso,
                        rec={r["nm"]: dict(psi_deg=float(np.degrees(r["psi"])), tx=float(r["t"][0]), ty=float(r["t"][1]), x3=float(r["p3"][0]), y3=float(r["p3"][1]), tau=r["tau"],
                                           corr_lidar_depth_xy=r["corr"], icp_med_m=r["med"], boot_sigma=r["boot_sig"], floor_dev_deg=float(d[0]), floor_dev_h=float(d[1]),
                                           depth_floor=r["floor_d"], lidar_floor=r["floor_l"]) for r, d in zip(R, fl_dev)})
        print(env, "ψ %.2f° tx %.3f ty %.3f | p3 (%.3f, %.3f) | σ ψ %.2f° x3 %.3f y3 %.3f | floor pitch %.2f roll %.2f hL %.3f | τ curve %s" % (
            np.degrees(psi), tx, ty, x3, y3, sol[env]["sigma"]["psi_deg"], sol[env]["sigma"]["x3"], sol[env]["sigma"]["y3"], sol[env]["floor_pitch_deg"], sol[env]["floor_roll_deg"], hL,
            sol[env]["tau_rel_residual"]), flush=True)
    return sol


def write_pack(job):
    env, nm, S = job; out = mc.OUT_L / f"{env}_{nm}.npz"
    A, C = load_rec(env, nm); T = int(A["T"])
    Rl_rec, cL_rec = A["Rl"], float(A["cL"]); Rl, hL = np.array(S["Rl"]), S["hL"]; psi, t = S["psi"], np.array([S["tx"], S["ty"]])
    c, s = np.cos(psi), np.sin(psi); R2 = np.array([[c, -s], [s, c]])
    sens = np.array([t[0], t[1], hL]); fwd = np.array([c, s, 0.0]); right = np.array([s, -c, 0.0]); down = np.array([0, 0, -1.0]); Rd = np.stack([right, down, fwd], 1)
    rng = np.random.default_rng(zlib.crc32(f"mmfi_lidar_v3/{env}/{nm}".encode())); fx, fy, cx, cy = K_V
    uvz = np.zeros((T, N_PTS, 3), np.float32); valid = np.zeros((T, N_PTS), bool); n_raw = np.zeros(T, np.int32)
    if bool(C["ok"]):
        pts, off = C["pts"].astype(float), C["off"]
        for k in range(T):
            p = pts[off[k]:off[k + 1]]
            if not len(p):
                continue
            raw = np.c_[p[:, :2], p[:, 2] - cL_rec] @ Rl_rec                     # 녹화 수평 좌표 → LiDAR 원시 좌표
            q = raw @ Rl.T; X = np.c_[q[:, :2] @ R2.T + t, q[:, 2] + hL]       # 방 바닥·방 강체 변환
            cam = (X - sens) @ Rd; cam = cam[cam[:, 2] > 0.3]; n_raw[k] = len(cam)
            if not len(cam):
                continue
            pick = rng.choice(len(cam), min(N_PTS, len(cam)), replace=False); cc_ = cam[pick]
            uvz[k, :len(pick)] = np.c_[fx * cc_[:, 0] / cc_[:, 2] + cx, fy * cc_[:, 1] / cc_[:, 2] + cy, cc_[:, 2]]; valid[k, :len(pick)] = True
    sid = int(nm[1:3])
    meta = dict(dataset="mmfi_lidar", env=env, subject=f"U{sid:02d}", action=nm[4:], T=int(T), fps=mc.FPS, label_family="h36m17", label_source="mmfi_ir_stereo_hrnet_triangulated",
                kp_family=None, has_rgb=False, canonical=True, camera_height_m=float(hL),
                room_frame="MM-Fi depth room (env depth floor z=0, RGB optical axis x); LiDAR GT-free: floor (roll/pitch/z) + depth-person ICP (x/y/yaw), per-room rigid",
                spec=str(SPEC), version="v3", eval_only=False, source=f"extracted_lidar/{env}/{nm[:3]}/{nm[4:]}",
                labels_included=False, labels_from=f"dev/packs/mmfi_v02_specA/{env}/{nm}.npz",
                extrinsic=dict(source=dict(roll_pitch_z="floor", x_y_yaw="depth_person", time="depth_person_icp_residual"), psi_deg=float(np.degrees(psi)), t=[float(t[0]), float(t[1]), float(hL)],
                               floor_normal_lidar=S["nL"], sigma=S["sigma"], tau_frames=0), person_extraction="depth-person anchor (cylinder 0.7 m, 0.2 m components within 0.35 m), floor+0.05–2.3 m")
    z13 = np.zeros((T, 13), np.float32)
    np.savez_compressed(out, kp_px=np.zeros((T, 17, 2), np.float32), kp_valid=np.zeros((T, 17), bool), pts_uvz=uvz, pts_edge=np.zeros((T, N_PTS), bool), pts_valid=valid,
                        pts_n_raw=n_raw, pts_n_mixed=np.zeros(T, np.int32), comp_size=n_raw, labels=np.zeros((T, 13, 3), np.float32), label_mask=np.zeros((T, 13), bool),
                        label_quality=np.zeros((T, 13), np.uint8), joint_speed=z13, R_c2r=Rd, t_c2r=sens, R_d2r=Rd, t_d2r=sens, meta=json.dumps(meta, ensure_ascii=False))
    return env, nm, f"T {T} pts/frame {n_raw[n_raw > 0].mean() if (n_raw > 0).any() else 0:.0f} empty {(n_raw == 0).mean():.3f}"


if __name__ == "__main__":
    sol = room_solution(); json.dump(sol, open(mc.WORK_L / "lidar_room.json", "w"), indent=1, ensure_ascii=False, default=float)
    spec = dict(dataset="mmfi_lidar", version="v3", spec_path=str(SPEC), fps=mc.FPS, label_family="h36m17", label_source="mmfi_ir_stereo_hrnet_triangulated", rgb=None,
                depth=dict(type="tof", sensor="MM-Fi LiDAR point cloud (virtual pinhole for storage)", image_wh=[640, 480], K=K_V, f_px=K_V[0], sigma0_m=0.02, k_rel=0.0,
                           sigma_pix_px=float(1 / np.sqrt(12)), quant_step_m=0.0, corr_len_px=4.0, n_eff_per_256=64, provenance="사전: LiDAR 거리 잡음 약 2 cm (제조사 사양 수준), 점 사이 상관 작음 (v02 와 같음)"),
                room=dict(source="v3 정답 없음: LiDAR 바닥 → 환경 깊이 바닥 (z, roll, pitch); 깊이 사람 점 ICP → (x, y, yaw), 방 단위 강체 (code/v3/reg_lidar_mmfi.py, build_lidar_mmfi.py)",
                          per_room={e: dict(psi_deg=float(np.degrees(v["psi"])), t=[v["tx"], v["ty"], v["hL"]], sigma=v["sigma"]) for e, v in sol.items()}), eval_only=False)
    SPEC.write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    jobs = [(e, nm, sol[e]) for e, nm in mc.recordings()]
    with Pool(int(os.environ.get("V3_NP", "24"))) as p:
        for env, nm, st in p.imap_unordered(write_pack, jobs):
            print("W", env, nm, st, flush=True)
    print("LIDAR PACKS DONE", len(jobs), flush=True)
