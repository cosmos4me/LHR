"""v3 MM-Fi 레이더 (TI IWR6843) → 방 정답 없는 재정합 (2026-11-24). 정답·라벨·IR 을 읽지 않는다 (mmfi_common.SafeNpz).
원시 레이더 좌표에서 시작: 열 x (앞) y z snr doppler. 축 부호는 정답 없이 정한다 (녹화 안 옆·높이 상관 부호) → 고유 회전 R0 (det = +1).
레이더 → 방:  X = Rz(ψ) · (R_d2r · M · R0 · x) + (tx, ty, H)
  - R_d2r · M: 레이더가 깊이 카메라와 평행하게 달렸다는 장착 제약 (pitch·roll 은 깊이 바닥이 정한 카메라 기울기를 그대로 씀; 레이더 고도로 기울기를 추정하지 않음).
  - (ψ, tx, ty): 레이더 사람 점 중심 궤적 ↔ 깊이 사람 점 궤적 (+ 물리 사전값: 레이더 중심은 깊이 보이는 표면보다 시선 방향으로 0.10 m 뒤 = 번역기 HALF_BODY), Huber IRLS.
  - H (3 m 유효 높이): 거리 다양성이 없어 높이·pitch 가 분리되지 않음 → "3 m 유효 높이" 하나로 보고. 두 관례를 계산:
      V2 고도 압축 모형 q_r = s (q_d − H) + H (압축 중심 = 레이더 보어사이트 높이) — 분위수 회귀 기울기·절편에서 H = −a/s.
      V1 중앙값 일치 (b̂_z = 0): 레이더 점 높이 중앙값 = 깊이 사람 높이 중앙값.
  - 시간: 레이더 |도플러| 합 ↔ 깊이 원본 프레임 차분 에너지 (5 프레임 중심 평활; 레이더는 0.5 s 누적), 그리고 중심 속력 ↔ 깊이 중심 속력 (Peršić: 외부 파라미터 무관).
단계: stageA (녹화마다 특징 추출, 병렬) → 방 단위 적합·검증은 fit_radar_mmfi.py.
사용: python -I reg_radar_mmfi.py stageA [env ...]"""
import sys, os, json
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
import mmfi_common as mc

R0 = np.diag([1.0, -1.0, -1.0])           # 원시 → 공칭 (x 앞, y 왼쪽, z 위): x 축 둘레 180° 회전 (det +1). 부호 근거는 fit 단계에서 다시 검사
M = np.array([[0, -1.0, 0], [0, 0, -1.0], [1.0, 0, 0]])   # 공칭 레이더 축 → 깊이 카메라 축 (x 앞→cam z, y 왼→cam −x, z 위→cam −y)
QP = np.arange(0.1, 0.91, 0.1)


def stageA(job):
    env, nm = job; out = mc.WORK_R / f"A_{env}_{nm}.npz"
    if out.exists():
        return env, nm, "skip"
    D, Rd, td, zc = mc.depth_person(env, nm); T = len(D)
    R = mc.radar_frames(env, nm, T)
    cd = np.full((T, 2), np.nan); dq = np.full((T, len(QP)), np.nan); dn = np.zeros(T, np.int32); dmed = np.full((T, 3), np.nan)
    for k in range(T):
        dn[k] = len(D[k])
        if len(D[k]) >= 30:
            cd[k] = np.median(D[k][:, :2], 0); dq[k] = np.quantile(D[k][:, 2], QP); dmed[k] = np.median(D[k], 0)
    ed = mc.depth_motion_energy(env, nm, zc)
    off = np.r_[0, np.cumsum([len(r) for r in R])]; pts = np.concatenate(R) if off[-1] else np.zeros((0, 5))
    np.savez_compressed(out, pts=pts.astype(np.float32), off=off, cd=cd, dq=dq, dn=dn, dmed=dmed, ed=ed, zc=zc, R_d2r=Rd, t_d2r=td, T=T)
    return env, nm, f"ok T {T} radar pts/frame {off[-1] / max(T, 1):.1f} depth ok {np.isfinite(cd).all(1).mean():.2f} ed ok {np.isfinite(ed).mean():.2f}"


if __name__ == "__main__":
    mc.WORK_R.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1]; envs = tuple(sys.argv[2:]) or mc.ENVS
    recs = mc.recordings(envs)
    if stage == "stageA":
        with Pool(int(os.environ.get("V3_NP", "24"))) as p:
            for env, nm, st in p.imap_unordered(stageA, recs):
                print("A", env, nm, st, flush=True)
    print("RADAR", stage, "DONE", flush=True)
