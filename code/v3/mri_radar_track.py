"""v3 mRI: 레이더 사람 궤적 (센서 좌표, 외부 파라미터 없이).

1. 원본 CSV (singleframe) 를 팩 프레임 t = Camera Frame − frame0 에 붙임 (v0.3 과 같은 대응). 같은 레이더 프레임이
   다음 정렬 프레임에 다시 붙은 경우(중복)는 궤적·시간 신호에서 뺌 (팩에는 v0.3 처럼 그대로 둠).
2. 정적 클러터 점검: 0.1 m 복셀 점유율 (고유 레이더 프레임 기준) ≥ 0.30 인 복셀을 정적으로 봄.
3. 녹화 중심 c0 (좌표별 중앙값) 에서 1.5 m 넘게 먼 점 제거 (방위 ±90° 가짜 점, 유령점). 3D 거리라 축 배치와 무관.
4. 1차: 프레임별 좌표별 절사 평균 (25–75 %; TI 방위가 sinθ 1/32 칸으로 양자화되어 중앙값은 칸에 붙음) → 창 9 중앙값 = 거친 궤적.
5. 2차: 거친 궤적에서 1.0 m 안의 점 → 절사 평균. dyn = |도플러| ≥ 1칸 (기본), all = 모든 점 (비교용). 점 < 3 이면 결측. 창 5 중앙값 평활.
6. 시간 신호 (외부 파라미터 무관): 게이트 안 점 수 n_gate, 평균 |도플러|.
출력: work/radar_track_Sxx.npz, work/radar_track_summary.json   (함수는 mri_radar_common.build_track — 주입 시험과 공유)
"""
import sys, json, time
sys.path.insert(0, "/home/interns/unisense/code/v3")
import numpy as np
from mri_radar_common import read_radar_csv, attach_frames, build_track, TRACK_PAR, WORK, SUBJ, jdump

t0 = time.time()
summary = dict(params=TRACK_PAR, subjects={})
for s in SUBJ:
    st = np.load(f"{WORK}/stereo_S{s:02d}.npz", allow_pickle=False); T = int(st["T"]); f0 = int(st["frame0"])
    pts, rf, has, dup = attach_frames(read_radar_csv(s), T, f0); uniq = has & ~dup
    tr = build_track(pts, uniq)
    np.savez_compressed(f"{WORK}/radar_track_S{s:02d}.npz", dyn=tr["dyn"], all=tr["all"], dyn_raw=tr["dyn_raw"], all_raw=tr["all_raw"], n_dyn=tr["n_dyn"], n_all=tr["n_all"],
                        dop_energy=tr["dop_energy"], n_gate=tr["n_gate"], uniq=uniq, dup=dup, has=has, radar_frame=rf, c0=tr["c0"], T=T, frame0=f0)
    rec = dict(T=T, frames_with_radar=int(has.sum()), unique_radar_frames=int(uniq.sum()), dup_frames=int(dup.sum()), pts_total=tr["n_pts"], c0=tr["c0"].round(3).tolist(),
               static=dict(max_voxel_occupancy=round(tr["max_occ"], 3), n_static_voxels=tr["n_static_vox"]), frac_far_gate0=round(tr["frac_far"], 4),
               track_valid=dict(dyn=int(np.isfinite(tr["dyn"]).all(1).sum()), all=int(np.isfinite(tr["all"]).all(1).sum())),
               pts_med=dict(dyn=float(np.median(tr["n_dyn"][uniq])), all=float(np.median(tr["n_all"][uniq]))))
    summary["subjects"][f"S{s:02d}"] = rec
    print(f"S{s:02d}", json.dumps(rec, ensure_ascii=False), flush=True)
jdump(summary, f"{WORK}/radar_track_summary.json")
print("TRACK_DONE", round(time.time() - t0, 1), "s", flush=True)
