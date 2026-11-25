"""v3 MM-Fi 매니페스트 (2026-11-24): 방별 변환 (LiDAR·레이더), 출처 태그, 부트스트랩 σ, 녹화별 진단, 정답 없는 검증 요약.
입력: _reg/lidar_room.json, _reg/fit.json, _reg/validation.json (모두 정답 없이 만든 것). 출력: dev/packs/v3/mmfi_manifest.json.
사용: python -I make_manifest_mmfi.py"""
import sys, json, math
sys.path.insert(0, "/home/interns/unisense/code/v3")
import numpy as np
import mmfi_common as mc
from reg_radar_mmfi import R0, M


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


if __name__ == "__main__":
    LR = json.load(open(mc.WORK_L / "lidar_room.json")); RF = json.load(open(mc.WORK_R / "fit.json")); V = json.load(open(mc.WORK_R / "validation.json"))
    man = dict(version="v3", dataset="mmfi", created="2026-11-24", gt_used_in_build=False,
               gt_guard="빌드 코드 (reg_*, fit_radar, build_*, validate) 는 mmfi_common.SafeNpz 허용 목록으로만 npz 를 읽음 (labels/gt/label_*/joint* 거부). 라벨은 audit_mmfi.py 만 읽고 그 출력은 변환에 쓰지 않음.",
               room_frame="MM-Fi 깊이 방 좌표: z = 환경 깊이 바닥 (정답 없음, adapt_mmfi floor_for_env), x = RGB 광축 수평 투영, 원점 = RGB 카메라 아래 바닥. 깊이 팩 R_d2r/t_d2r 그대로.",
               code=dict(common="code/v3/mmfi_common.py", lidar_reg="code/v3/reg_lidar_mmfi.py", lidar_build="code/v3/build_lidar_mmfi.py", radar_feat="code/v3/reg_radar_mmfi.py",
                         radar_fit="code/v3/fit_radar_mmfi.py", radar_build="code/v3/build_radar_mmfi.py", validate="code/v3/validate_mmfi.py", audit="code/v3/audit_mmfi.py"),
               packs=dict(lidar="dev/packs/v3/mmfi_lidar/{env}_{nm}.npz (mmfi_lidar_v02 형식; labels 칸은 비움 → 깊이 팩 라벨 사용)", radar="dev/packs/v3/mmfi_radar/{env}/{nm}.npz (mmfi_v03_radar 형식)",
                          lidar_spec="dev/packs/v3/mmfi_lidar/mmfi_lidar_v3_spec.json"),
               lidar={}, radar={}, recordings={}, validation=V)
    for e in mc.ENVS:
        L = LR[e]; Rl = np.array(L["Rl"]); R = mc.Rz(L["psi"]) @ Rl; t = [L["tx"], L["ty"], L["hL"]]
        man["lidar"][e] = dict(transform=dict(R=R.tolist(), t=t, convention="X_room = R x_lidar + t", psi_deg=math.degrees(L["psi"]), floor_normal_lidar=L["nL"], height_m=L["hL"],
                                              floor_pitch_deg=L["floor_pitch_deg"], floor_roll_deg=L["floor_roll_deg"], p3_room=[L["x3"], L["y3"]], tau_frames=0),
                               source=dict(roll_pitch_z="floor", x_y_yaw="depth_person", time="depth_person (ICP residual vs τ)"),
                               sigma_boot=L["sigma"], tau_rel_residual=L["tau_rel_residual"], rec_spread=L["rec_spread"], per_subject=L["per_subject"], loso=L["loso"],
                               granularity="per room (rigid rig); per-recording / per-subject estimates kept as diagnostics")
        f = RF[e]; nm0 = mc.recordings((e,))[0][1]; Rd = np.asarray(np.load(mc.WORK_R / f"A_{e}_{nm0}.npz")["R_d2r"], float)
        psi = math.radians(f["fit"]["psi_deg"]); Rr = mc.Rz(psi) @ Rd @ M @ R0
        man["radar"][e] = dict(transform=dict(R=Rr.tolist(), t=[f["fit"]["tx"], f["fit"]["ty"], f["fit"]["H"]], convention="X_room = R x_raw + t (x_raw = 원시 레이더 x y z)", det=float(np.linalg.det(Rr)),
                                              psi_deg=f["fit"]["psi_deg"], H_3m_effective=f["fit"]["H"], H_choice=f["fit"]["H_choice"], H_alt_v2=f["fit"]["H_v2"], x3_room=f["fit"]["x3"], y3_room=f["fit"]["y3"], elev_slope_s=f["fit"]["elev_slope_s"],
                                              los_prior_m=f["fit"]["los_prior_m"], R0_raw_to_nominal=R0.tolist(), tau_applied_frames=0),
                               source=dict(axes="axis_sign_corr", roll_pitch="floor+mounting", x_y_yaw="depth_person", z="depth_person_quantile", time="velocity"),
                               sigma_boot=f["boot_rec"], time=f["time"], axis_check=f["axis_check"], repeat=f["repeat"], holdout=f["holdout"], injection=f["injection"],
                               per_subject=f["per_subject"], z_uncertainty=dict(H_v2=f["fit"]["H_v2"], H_v1=f["fit"]["H_v1"], diff_v2_v1=f["fit"]["H_v2"] - f["fit"]["H_v1"],
                                                                                H_from_lidar_v2=V["closure"][e]["H_from_lidar_v2"], sigma_boot_v2=f["boot_rec"]["sigma_H_v2"]),
                               yaw_alt_via_lidar_deg=math.degrees(LR[e]["psi"]) + V["closure"][e]["radar_to_lidar"]["psi_deg"],
                               granularity="per room")
        for nm, d in L["rec"].items():
            man["recordings"][f"{e}/{nm}"] = dict(lidar=d, radar=f["rec"].get(nm), depth_session_dZ_at_3p1=V["depth_session_bias"][e]["per_subject"][nm[:3]]["dZ_at_3p1"])
    json.dump(clean(man), open(mc.OUT / "mmfi_manifest.json", "w"), indent=1, ensure_ascii=False)
    print("MANIFEST", mc.OUT / "mmfi_manifest.json", "recordings", len(man["recordings"]), flush=True)
