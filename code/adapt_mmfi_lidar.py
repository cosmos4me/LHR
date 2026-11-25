"""MM-Fi LiDAR 검증 세트 (평가 전용, 2026-11-11; 사용자: "ITOP 말고 다른 데이터셋도 검증용으로"). 학습에 쓰지 않는다.
의미: 같은 MM-Fi 사람·동작·정답에 '처음 보는 깊이 센서 (LiDAR 점군)'. MM-Fi 를 학습한 겹에서는 몸·라벨은 본 것이고 센서만 새것 → 센서 몫만 따로 본다.
- 입력: dev/data/mmfi/extracted_lidar/Exx/Syy/Azz/lidar/frameNNN.bin (float64 xyz, LiDAR 좌표: x 앞, z 위, 바닥 약 z = −1.2). 점군에는 바닥과 사람이 함께 있다.
- 바닥 (정답 없음): 시퀀스마다 낮은 점들의 평면 (반복 최소제곱) → 수평 맞춤 회전 + 센서 높이.
- 사람 점 (정답 없음): 바닥 위 5 cm – 2.3 m, 바닥 위 점들의 수평 중앙값에서 1.0 m 안.
- 방 좌표 수평 맞춤 (yaw, 수평 이동, 표면→몸 중심 거리 δ): 환경 (E01–E03) 마다 train 피험자의 정답 골반으로 최소제곱 (사용자 허용: 센서 카드 값은 그 데이터셋 train 정답으로 맞춰도 됨). 시험 피험자 정답은 쓰지 않는다.
- 저장: 가상 핀홀 (640×480, f = 400 px) 의 (u, v, z), 256점, 정답 = MM-Fi 팩 라벨을 MM-Fi 와 같은 정준 몸으로 (어댑터 δ 를 MM-Fi 깊이 카메라 방향으로 미리 적용).
- 분할: MM-Fi E1_cross_subject 의 test 피험자 9명만 팩으로 만든다 (MM-Fi 깊이 칸과 같은 사람).
사용: python adapt_mmfi_lidar.py"""
import json, sys, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
CODE = Path("/home/interns/unisense/code"); sys.path.insert(0, str(CODE))
LID = Path("/home/interns/unisense/dev/data/mmfi/extracted_lidar"); PK = Path("/home/interns/unisense/dev/packs/mmfi_v02_specA")
OUT = Path("/home/interns/unisense/dev/packs/mmfi_lidar_v01"); SPEC = CODE / "specs" / "mmfi_lidar_v01.json"; CAL = OUT / "calib.json"
N_PTS, FPS, K_V = 256, 10.0, [400.0, 400.0, 320.0, 240.0]
DELTA = 0.10                                                                                      # 보이는 표면 중앙값 → 골반 수평 거리 (= physics_baseline.HALF_BODY)
SPL = json.loads((CODE / "splits" / "mmfi.json").read_text())["E1_cross_subject"]; ENVS = ("E01", "E02", "E03")
import os
ALL = os.environ.get("LIDAR_ALL", "0") == "1"                                                    # 11-23 LHR: 모든 피험자 · E04 포함, 학습에도 씀 (평가 전용 해제)
if ALL:
    OUT = Path("/home/interns/unisense/dev/packs/mmfi_lidar_v02"); SPEC = CODE / "specs" / "mmfi_lidar_v02.json"; CAL = OUT / "calib.json"; ENVS = ("E01", "E02", "E03", "E04")
CALSUB = {"E04": ["S33"]}                                                                          # E04 보정 피험자 (레이더 결합과 같음)


def frames(env, s, a):
    d = LID / env / s / a / "lidar"
    fs = sorted(d.glob("frame*.bin"), key=lambda f: int(f.stem[5:])) if d.exists() else []
    return [np.fromfile(f, dtype=np.float64).reshape(-1, 3) for f in fs]


def floor(fr):
    """낮은 점 평면 z = αx + βy + γ (LiDAR 좌표) → 단위 법선 n (위쪽), c (n·x + c = 0 이 바닥)."""
    P = np.concatenate([f[f[:, 2] < np.percentile(f[:, 2], 40)] for f in fr[:: max(1, len(fr) // 30)]])
    keep = np.ones(len(P), bool)
    for _ in range(4):
        A = np.c_[P[keep, 0], P[keep, 1], np.ones(keep.sum())]; coef, *_ = np.linalg.lstsq(A, P[keep, 2], rcond=None)
        r = P[:, 2] - (np.c_[P[:, 0], P[:, 1], np.ones(len(P))] @ coef); keep = np.abs(r) < max(0.02, 2.5 * np.median(np.abs(r[keep])))
    n = np.array([-coef[0], -coef[1], 1.0]); s = np.linalg.norm(n); return n / s, -coef[2] / s


def level_rot(n):
    z = np.array([0, 0, 1.0]); v = np.cross(n, z); s, c = np.linalg.norm(v), n @ z
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s; th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def person(f, Rl, c):
    q = f @ Rl.T; h = q[:, 2] + c; m = (h > 0.05) & (h < 2.3)
    if m.sum() < 10:
        return q[:0], h[:0]
    med = np.median(q[m, :2], 0); m &= np.linalg.norm(q[:, :2] - med, axis=1) < 1.0
    return q[m], h[m]


def rot2(y):
    c, s = np.cos(y), np.sin(y); return np.array([[c, -s], [s, c]])


def calibrate(env):
    """train 피험자: 사람 점 수평 중앙값 (수평 맞춘 LiDAR) ↔ 정답 골반 (방). g = R(yaw)(c + δ ĉ) + t."""
    C, G = [], []
    subs = sorted({p.name for p in (LID / env).iterdir()} & set(CALSUB.get(env, SPL["train"])))
    for s in subs:
        for a in sorted(p.name for p in (LID / env / s).iterdir())[::3]:
            fr = frames(env, s, a); pk = PK / env / f"{s}_{a}.npz"
            if not fr or not pk.exists():
                continue
            P = np.load(pk); y = P["labels"].astype(float); ok = P["label_mask"][:, 7] & P["label_mask"][:, 8]; n, c = floor(fr); Rl = level_rot(n)
            for k in range(0, min(len(fr), len(y)), 10):
                q, h = person(fr[k], Rl, c)
                if len(q) >= 20 and ok[k]:
                    C.append(np.median(q[:, :2], 0)); G.append(0.5 * (y[k, 7, :2] + y[k, 8, :2]))
    C, G = np.array(C), np.array(G)
    u = C / np.linalg.norm(C, axis=1, keepdims=True)                                          # δ = 몸 두께 절반 0.10 m 로 고정 (사전; 사람이 늘 비슷한 방향에 서서 δ 와 이동이 구별되지 않음)
    def res(p):
        return ((C + DELTA * u) @ rot2(p[0]).T + p[1:3] - G).ravel()
    sol = least_squares(res, x0=[0.0, 0.0, 0.0], loss="soft_l1", f_scale=0.05); r = res(sol.x).reshape(-1, 2)
    return dict(yaw=float(sol.x[0]), t=[float(sol.x[1]), float(sol.x[2])], delta=DELTA, n=int(len(C)), resid_med_m=float(np.median(np.linalg.norm(r, axis=1))))


def process(job):
    env, s, a, cal = job; out = OUT / f"{env}_{s}_{a}.npz"; pk = PK / env / f"{s}_{a}.npz"
    fr = frames(env, s, a)
    if out.exists() or not fr or not pk.exists():
        return f"{env}/{s}/{a}", "skip"
    P = np.load(pk); m = json.loads(P["meta"].item()); T = min(len(fr), len(P["labels"])); n, c = floor(fr); Rl = level_rot(n)
    R2 = rot2(cal["yaw"]); sens = np.array([cal["t"][0], cal["t"][1], c])                    # 센서 위치 (방): 수평 = R(yaw)·0 + t, 높이 = c
    fwd = np.r_[R2 @ np.array([1.0, 0.0]), 0.0]; right = np.r_[R2 @ np.array([0.0, -1.0]), 0.0]; down = np.array([0, 0, -1.0])
    Rd = np.stack([right, down, fwd], 1)                                                     # 가상 카메라 → 방 (열 = 카메라 x, y, z)
    rng = np.random.default_rng(zlib.crc32(f"mmfi_lidar/{env}/{s}/{a}".encode())); fx, fy, cx, cy = K_V
    uvz = np.zeros((T, N_PTS, 3), np.float32); valid = np.zeros((T, N_PTS), bool); n_raw = np.zeros(T, np.int32)
    for k in range(T):
        q, h = person(fr[k], Rl, c)
        if not len(q):
            continue
        X = np.c_[q[:, :2] @ R2.T + np.array(cal["t"]), h]; cam = (X - sens) @ Rd; cam = cam[cam[:, 2] > 0.3]; n_raw[k] = len(cam)
        if not len(cam):
            continue
        pick = rng.choice(len(cam), min(N_PTS, len(cam)), replace=False); cc_ = cam[pick]
        uvz[k, :len(pick)] = np.c_[fx * cc_[:, 0] / cc_[:, 2] + cx, fy * cc_[:, 1] / cc_[:, 2] + cy, cc_[:, 2]]; valid[k, :len(pick)] = True
    import lhr_data as LD
    dl, dh = LD.adapter_of("mmfi"); y = P["labels"][:T].astype(float); u = y - np.asarray(P["t_d2r"], float); u /= np.linalg.norm(u, axis=-1, keepdims=True).clip(1e-6)
    sh = np.full(13, dl); sh[0] = dh; yc = y + sh[None, :, None] * u                          # MM-Fi 와 같은 정준 몸 (MM-Fi 깊이 카메라 방향 δ)
    sid = int(s[1:]); meta = dict(dataset="mmfi_lidar", env=env, subject=f"U{sid:02d}", action=a, T=int(T), fps=FPS, label_family="h36m17", label_source="mmfi_ir_stereo_hrnet_triangulated",
                                  kp_family=None, has_rgb=False, canonical=True, camera_height_m=float(c), room_frame="MM-Fi room (labels); LiDAR leveled by own floor, yaw/shift fit on train subjects",
                                  spec=str(SPEC), version="v0.2" if ALL else "v0.1", eval_only=not ALL, source=f"extracted_lidar/{env}/{s}/{a}")
    q = np.asarray(P["label_quality"][:T]); sp = np.linalg.norm(np.gradient(yc, axis=0), axis=-1) * FPS
    np.savez_compressed(out, kp_px=np.zeros((T, 17, 2), np.float32), kp_valid=np.zeros((T, 17), bool), pts_uvz=uvz, pts_edge=np.zeros((T, N_PTS), bool), pts_valid=valid,
                        pts_n_raw=n_raw, pts_n_mixed=np.zeros(T, np.int32), comp_size=n_raw, labels=yc.astype(np.float32), label_mask=np.asarray(P["label_mask"][:T]),
                        label_quality=q, joint_speed=sp.astype(np.float32), R_c2r=Rd, t_c2r=sens, R_d2r=Rd, t_d2r=sens, meta=json.dumps(meta))
    return f"{env}/{s}/{a}", f"ok T {T} 점 {float(n_raw.mean()):.0f} 높이 {c:.2f} 발 {float(np.median(yc[:, 11:13, 2])):.2f}"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cal = json.loads(CAL.read_text()) if CAL.exists() else {}
    for env in ENVS:
        if env not in cal:
            cal[env] = calibrate(env); CAL.write_text(json.dumps(cal, indent=1)); print("calib", env, cal[env], flush=True)
    spec = dict(dataset="mmfi_lidar", version="v0.1", spec_path=str(SPEC), fps=FPS, label_family="h36m17", label_source="mmfi_ir_stereo_hrnet_triangulated", rgb=None,
                depth=dict(type="tof", sensor="MM-Fi LiDAR point cloud (virtual pinhole for storage)", image_wh=[640, 480], K=K_V, f_px=K_V[0], sigma0_m=0.02, k_rel=0.0,
                           sigma_pix_px=float(1 / np.sqrt(12)), quant_step_m=0.0, corr_len_px=4.0, n_eff_per_256=64, provenance="사전: LiDAR 거리 잡음 약 2 cm (제조사 사양 수준), 점 사이 상관 작음"),
                room=dict(source="MM-Fi 방 좌표 (라벨); 바닥 = LiDAR 자체 평면, 수평 맞춤 = train 피험자 정답 골반 (calib.json)"), eval_only=not ALL)
    SPEC.write_text(json.dumps(spec, indent=1))
    jobs = [(env, s, a, cal[env]) for env in ENVS for s in sorted((set(p.name for p in (LID / env).iterdir()) if ALL else set(SPL["test"])) & {p.name for p in (LID / env).iterdir()}) for a in sorted(p.name for p in (LID / env / s).iterdir())]
    with Pool(12) as p:
        for k, st in p.imap_unordered(process, jobs):
            print(k, st, flush=True)
    print("MMFI LIDAR PACKS DONE", len(jobs), flush=True)
