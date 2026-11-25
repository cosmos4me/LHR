"""v3 mRI 레이더 팩 쓰기: dev/packs/v3/mri_radar/Sxx.npz + manifest.json.

형식은 dev/packs/mri_v03_radar 와 같음: radar_pts (N,5) float32 = 방 x y z (mri_v03), 도플러 m/s, 세기; radar_off (T+1) int64; meta (JSON 문자열).
meta["room_frame"] = "mri_v03" (ms_cache2 assert), meta["extrinsic"]["source"] 와 meta["source_tag"] 에 정답 없는 출처 표시.
프레임 대응은 v0.3 과 같음 (t = Camera Frame − frame0, 한 정렬 프레임에 레이더 프레임 여럿이면 마지막, 중복 프레임 유지) + 적용 지연 L.
점검: 이전 팩과 프레임별 점 수 동일, 이전 변환의 역변환으로 되돌린 점 = CSV 점 (라벨 미사용).
"""
import sys, os, json, time
sys.path.insert(0, "/home/interns/unisense/code/v3")
import numpy as np
from mri_radar_common import read_radar_csv, attach_frames, PK, OUT, OLD, WORK, SUBJ, jdump

t0 = time.time()
REG = json.load(open(f"{WORK}/registration_dyn.json"))
VAL = json.load(open(f"{WORK}/validation.json")) if os.path.exists(f"{WORK}/validation.json") else None
REG_ALL = json.load(open(f"{WORK}/registration_all.json")) if os.path.exists(f"{WORK}/registration_all.json") else None
R = np.array(REG["final"]["R"]); t = np.array(REG["final"]["t"]); L = int(REG["time"]["applied_integer_lag"])
CARD = dict(device="TI IWR1443 (mRI)", doppler_step_mps=0.356, frames_per_s=8.7, columns="x y z (방 m), 도플러 m/s, 세기")
TAGS = dict(yaw="stereo_person", x="stereo_person", y="stereo_person", z="mount_assumption+floor", pitch="mount_assumption+floor", roll="mount_assumption+floor",
            time="velocity", axis_permutation="stereo_person")
rep = REG["repeatability"]; bs = REG["bootstrap"]
WITHIN = json.load(open(f"{WORK}/validation_within.json")) if os.path.exists(f"{WORK}/validation_within.json") else None
fps = 8.695652173913043
# 시간 차이 추정 셋 (모두 정답 없음): 움직임 에너지 상호상관, 속도 회귀 (잔차 = −v·Δt), 위치 기반 지연 탐색 (포물선 꼭짓점)
pl = {r_["L"]: r_["mean_huber"] for r_ in REG["time"]["position_based_scan"]}; kb = min(pl, key=pl.get)
den = pl[kb - 1] - 2 * pl[kb] + pl[kb + 1]; pos_sub = kb + (0.5 * (pl[kb - 1] - pl[kb + 1]) / den if den > 0 else 0.0)
DT = dict(motion_xcorr_s=REG["time"]["median_frames"] / fps, position_scan_parabolic_s=pos_sub / fps)
if VAL is not None:
    rs = VAL["residual_structure"]
    DT["velocity_regression_x_s"] = -rs["res_x_depth"]["vel_x (m per m/s = −Δt s)"]["coef"]; DT["velocity_regression_y_s"] = -rs["res_y_lateral"]["vel_y (m per m/s = −Δt s)"]["coef"]
DT_REC = float(np.median(list(DT.values())))
# 병렬로 만든 v3 정답 없는 mRI 바닥 (스테레오 발 접촉, code/v3/floor_mri.py) 이 있으면 mri_v03 방 바닥과의 차이를 기록만 함
FLOOR = None
fl = "/home/interns/unisense/dev/packs/v3/floor/mri.json"
if os.path.exists(fl):
    Fp = json.load(open(fl))["pooled"]
    FLOOR = dict(file=fl, angle_change_deg=Fp["angle_change_deg"], height_change_m=Fp["height_change_m"], sigma_tilt_total_deg=Fp["sigma_tilt_total_deg"],
                 sigma_h_total_m=Fp["sigma_h_total_m"], note="v3 바닥이 mri_v03 방 바닥과 거의 같음 → 방 z 정의·레이더 z 규칙 유지 (바꾸지 않음)")
EXT = dict(source="v3_gtfree_stereo_person_bev3dof_mount_level", source_tags=TAGS, uses_labels=False, R=R.round(6).tolist(), t=t.round(6).tolist(),
           R0_axis_perm=REG["perm_best"]["R0"], psi_deg=round(REG["final"]["psi_deg"], 4), tau_xy=np.round(REG["final"]["tau"], 5).tolist(),
           z_rule=REG["final"]["z_rule"], floor_check_v3=None, time_lag_frames_applied=L, time_lag_estimate_ms=REG["time"]["median_ms"],
           time_offset_estimates_s=DT, time_offset_recommended_s=DT_REC,
           time_offset_note="레이더가 카메라 키포인트보다 늦음 (+). 1 프레임(115 ms) 미만이라 프레임 대응은 바꾸지 않음. 토큰 dt 에 더할 값으로만 기록",
           bootstrap_sd=bs["sd"], per_subject_sd=dict(psi_deg=rep["psi_deg_sd"], tau_x=rep["tau_x_sd"], tau_y=rep["tau_y_sd"], lateral_at_2p4m=rep["lat_at_2p4_sd"]),
           fit_subjects="S01–S20 공통 (스테레오 2D 키포인트 + 레이더 점만, 라벨 없음)")
EXT["floor_check_v3"] = None if FLOOR is None else {k: v for k, v in FLOOR.items() if k != "note"}
os.makedirs(OUT, exist_ok=True)
checks = {}
for s in SUBJ:
    P = np.load(f"{PK}/S{s:02d}.npz", allow_pickle=False); m = json.loads(str(P["meta"])); T, f0 = int(m["T"]), int(m["frame0"])
    pts, rf, has, dup = attach_frames(read_radar_csv(s), T, f0)
    rows = [np.zeros((0, 5), np.float32)] * T
    for k in range(T):
        j = k + L
        if 0 <= j < T and pts[j] is not None:
            q = pts[j].copy(); q[:, :3] = pts[j][:, :3] @ R.T + t; rows[k] = q.astype(np.float32)
    off = np.r_[0, np.cumsum([len(r_) for r_ in rows])].astype(np.int64)
    meta = dict(dataset="mri", subject=f"S{s:02d}", T=T, frame0=f0, card=CARD, extrinsic=EXT, version="v3-radar", room_frame="mri_v03",
                source_tag="gtfree_stereo_person_bev3dof", frame_map=f"pack t ← radar of Camera Frame (t + {L}) + frame0; v0.3 assignment (last radar frame per aligned frame)",
                dup_frames=int(dup.sum()), built=time.strftime("%Y-%m-%d %H:%M"))
    np.savez_compressed(f"{OUT}/S{s:02d}.npz", radar_pts=np.concatenate(rows), radar_off=off, meta=json.dumps(meta, ensure_ascii=False))
    # 점검 (이전 팩: radar_pts, radar_off, meta 만)
    O = np.load(f"{OLD}/S{s:02d}.npz", allow_pickle=False); mo = json.loads(str(O["meta"])); Ro, to = np.array(mo["extrinsic"]["R"]), np.array(mo["extrinsic"]["t"])
    same_off = bool(np.array_equal(O["radar_off"], off)) if L == 0 else None
    back = (O["radar_pts"][:, :3].astype(float) - to) @ Ro                  # Roᵀ (q − t)
    ref = np.concatenate([pts[k][:, :3] if pts[k] is not None else np.zeros((0, 3)) for k in range(T)])
    err = float(np.abs(back - ref).max()) if len(back) == len(ref) else None
    newp = np.load(f"{OUT}/S{s:02d}.npz", allow_pickle=False)
    assert json.loads(str(newp["meta"])).get("room_frame") == "mri_v03"
    checks[f"S{s:02d}"] = dict(n_pts=int(off[-1]), frames_with_pts=int((np.diff(off) > 0).sum()), same_offsets_as_v03=same_off, old_inverse_vs_csv_max_abs_m=err,
                               doppler_intensity_equal_v03=bool(np.allclose(O["radar_pts"][:, 3:], newp["radar_pts"][:, 3:])) if same_off else None)
    print(f"S{s:02d}", json.dumps(checks[f"S{s:02d}"]), flush=True)

man = dict(dataset="mri", what="v3 mRI TI 레이더 → 방(mri_v03) 정합, 정답 없음", built=time.strftime("%Y-%m-%d %H:%M"),
           code=["/home/interns/unisense/code/v3/mri_radar_common.py", "/home/interns/unisense/code/v3/mri_stereo_track.py", "/home/interns/unisense/code/v3/mri_radar_track.py",
                 "/home/interns/unisense/code/v3/mri_radar_register.py", "/home/interns/unisense/code/v3/mri_radar_validate.py", "/home/interns/unisense/code/v3/mri_radar_write.py"],
           inputs=dict(radar_csv="dev/data/mri/extracted/dataset_release/aligned_data/radar/singleframe/subjectN.csv (데이터만, python -I)",
                       packs_read_keys=["kp_px", "kp_score", "kp_px_cam2", "kp_score_cam2", "meta"], spec_keys=["rgb.K", "rgb.K_cam2", "room.*"],
                       not_read=["labels", "labels_cam", "label_mask", "label_quality", "action", "mRI pose_labels / GT / pose_2d / refined_gt_kps", "spec rgb.sigma_model (학습 피험자 정답으로 맞춘 값)"]),
           transform=dict(q_room="R @ p_sensor + t", R=R.tolist(), t=t.tolist(), R0_axis_perm=REG["perm_best"]["R0"], psi_deg=REG["final"]["psi_deg"], tau_xy=REG["final"]["tau"],
                          boresight_2p4m_room_xy=REG["final"]["boresight_2p4m_room_xy"], z_rule=REG["final"]["z_rule"], extra_sigma_xy=REG["final"]["extra_sigma"]),
           source_tags=TAGS, time=dict(applied_integer_lag_frames=L, offset_estimates_s=DT, offset_recommended_s=DT_REC, estimate_median_frames=REG["time"]["median_frames"], estimate_median_ms=REG["time"]["median_ms"],
                                       iqr_frames=REG["time"]["spread_frames_iqr"], position_based_best=REG["time"]["position_based_best"]),
           bootstrap=REG["bootstrap"], per_subject=REG["per_subject"], repeatability=REG["repeatability"], loso={k: v for k, v in REG["loso"].items() if k != "rows"},
           halves=REG["halves"], perm_best=REG["perm_best"], mirror_diagnostic=REG["mirror_diagnostic"],
           variant_all_points=None if REG_ALL is None else dict(psi_deg=REG_ALL["final"]["psi_deg"], tau=REG_ALL["final"]["tau"], boresight_2p4m_room_xy=REG_ALL["final"]["boresight_2p4m_room_xy"],
                                                                 med_r=REG_ALL["final"]["med_r"]),
           validation=None if VAL is None else dict(held_out=VAL.get("held_out"), injection_recovery=VAL.get("injection_recovery", {}).get("results"),
                                                    old_transform_comparison={k: v for k, v in VAL["old_transform_comparison"].items() if k != "mapped_c0_diff_new_minus_old_per_subject"}),
           validation_within_subject=WITHIN, floor_check=FLOOR,
           checks=checks, files=[f"S{s:02d}.npz" for s in SUBJ], work_dir="work/ (중간 궤적, registration_*.json, validation.json)")
jdump(man, f"{OUT}/manifest.json")
print("WRITE_DONE", round(time.time() - t0, 1), "s", flush=True)
