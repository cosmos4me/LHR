"""v3 MM-Fi 레이더 팩 조립 (2026-11-24). 입력 = fit_radar_mmfi.py 의 방 변환 (_reg/fit.json). 정답을 읽지 않는다.
모든 녹화 (ms_cache2 가 읽는 1080 개, E04 S31·S32 포함) 의 원시 레이더 점을 방 좌표로 옮겨 mmfi_v03_radar 와 같은 형식으로 저장:
  radar_pts (N, 5) = 방 x y z, 도플러 (원시 열 4, m/s), 세기 (원시 열 3); radar_off (T+1); meta.
  X = Rz(ψ) · (R_d2r · M · R0 · x_raw) + (tx, ty, H). 점 게이트 없음 (v03 과 같음). 시간 이동 없음 (|τ| < 1 프레임).
사용: python -I build_radar_mmfi.py"""
import sys, os, json
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
import mmfi_common as mc
from reg_radar_mmfi import R0, M

CARD = dict(device="TI IWR6843 (MM-Fi)", doppler_step_mps=0.6043, range_res_m=None, frames_per_s=10,
            note="점 = 5 열 (x y z 방 좌표 m, 도플러 m/s, 세기); MM-Fi 공개 레이더 프레임은 인접 프레임 누적 (0.5 s, 5 프레임)")


def transform(env, F, Rd):
    f = F[env]["fit"]; psi = np.radians(f["psi_deg"]); c, s = np.cos(psi), np.sin(psi); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    R = Rz @ np.asarray(Rd, float) @ M @ R0; t = np.array([f["tx"], f["ty"], f["H"]])
    return R, t


def one(job):
    env, nm, F = job; out = mc.OUT_R / env / f"{nm}.npz"
    A = np.load(mc.WORK_R / f"A_{env}_{nm}.npz", allow_pickle=False); T = int(A["T"]); P = A["pts"].astype(float); off = A["off"]
    R, t = transform(env, F, A["R_d2r"])
    assert abs(np.linalg.det(R) - 1) < 1e-6
    X = P[:, :3] @ R.T + t; pts = np.c_[X, P[:, 4], P[:, 3]].astype(np.float32)
    f = F[env]
    meta = dict(dataset="mmfi", env=env, seq=nm, T=T, card=CARD, version="v3-radar",
                extrinsic=dict(source=dict(axes="axis_sign_corr (lateral corr with depth person; det=+1)", roll_pitch="floor+mounting (camera-parallel: depth-floor camera tilt)",
                                           x_y_yaw="depth_person", z="depth_person_quantile (V1: radar point height median = depth person height median, b_z prior 0; 3 m effective height; V2 compression-model value kept as H_v2_alt)", time="velocity (doppler-energy xcorr)"),
                               R=R.round(6).tolist(), t=t.round(5).tolist(), R0_raw_to_nominal=R0.tolist(), psi_deg=f["fit"]["psi_deg"], H=f["fit"]["H"], H_v2_alt=f["fit"]["H_v2"], H_rule=f["fit"]["H_choice"],
                               los_prior_m=f["fit"]["los_prior_m"], sigma=f["boot_rec"], tau_frames_measured=f["time"]["tau_frames"], tau_applied_frames=0,
                               gt_used=False))
    np.savez_compressed(out, radar_pts=pts, radar_off=np.asarray(off, np.int64), meta=json.dumps(meta, ensure_ascii=False))
    return env, nm, f"T {T} pts {len(pts)}"


if __name__ == "__main__":
    F = json.load(open(mc.WORK_R / "fit.json"))
    for e in mc.ENVS:
        (mc.OUT_R / e).mkdir(parents=True, exist_ok=True)
    jobs = [(e, nm, {e: F[e]}) for e, nm in mc.recordings()]
    with Pool(int(os.environ.get("V3_NP", "24"))) as p:
        for env, nm, st in p.imap_unordered(one, jobs):
            print("W", env, nm, st, flush=True)
    print("RADAR PACKS DONE", len(jobs), flush=True)
