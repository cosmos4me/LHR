"""mmBody 어댑터 (변환기 v0.1): 선택 캐시(dev/data/mmbody/cache_v1, mmbody_fetch.py) → 시퀀스별 팩 (dev/packs/mmbody_v01/{train|test}/{이름}.npz).
- 입력: master Azure Kinect 점군(depth_pcl/master, 데이터셋이 레이더 좌표로 준 것). RGB 영상은 아직 받지 않음 → kp_valid = False.
  주의 (감사 notes/mmbody_audit.json + 논문 4.1절): 점군은 "정답 골격에서 자동으로 만든 상자"로 이미 잘려 있다.
  가장 낮은 점 = 가장 낮은 발 관절 + 1.0–1.1 cm (32 시퀀스 모두). → 사람 분할은 정답 보조(오라클)이며 meta seg_leak = True로 표시.
  occlusion 장면 2개는 점군이 모든 프레임에서 비어 있다 (깊이 조건 평가 불가). smoke_0은 프레임당 약 94점.
- 정답: SMPL-X 관절 55개 (MoSh++ 맞춤) → 공통 13관절 (머리 = head 15, 어깨 16/17, 팔꿈치 18/19, 손목 20/21, 엉덩이 1/2, 무릎 4/5, 발목 7/8).
- 방 좌표: OptiTrack 보정(calib.txt optitrack: 모캡 → 레이더)의 바닥 (모캡 y 위, 바닥 y = 0) — OctoNet과 같은 규약 (모캡 바닥 z = 0).
  점군의 아래쪽 경계는 정답으로 잘렸으므로 바닥 추정에 쓰지 않는다. 카메라 = master Kinect (calib kinect_master: 카메라 → 레이더, 공칭 장착).
  원점 = 카메라 바로 아래 바닥점, x = 광축의 바닥 투영 (conv_common.room_frame).
- 점 저장: (u, v, z)를 가상 핀홀 K_V (Azure Kinect NFOV 640×576 공칭)로 저장 — 점이 이미 3D라 무손실. 256점 표본.
  가장자리 표시: 각도 공간 반지름 3.5 px 안 이웃 수가 프레임 중앙값의 절반 미만(점군 경계)이거나 이웃 깊이 범위 > 0.2 m.
- 시간: 레이더 프레임 기준 약 14 FPS (논문 3.1). 분할: 공식 train/test, val = train 중 피험자 한 명(β 묶음).
- 피험자: SMPL-X β 평균이 0.05 안에서 같은 시퀀스 = 같은 사람 (감사). train 피험자와 test 피험자는 겹치지 않음.
사용: python adapt_mmbody.py [--spec-only]"""
import json, sys, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_common as cc

CACHE = Path("/home/interns/unisense/dev/data/mmbody/cache_v1")
OUT = Path("/home/interns/unisense/dev/packs/mmbody_v01")
CODE = Path(__file__).resolve().parent
SPEC_PATH = CODE / "specs" / "mmbody_v01.json"
AUDIT = json.loads(Path("/home/interns/unisense/notes/mmbody_audit.json").read_text())
N_PTS, FPS = 256, 14.0
K_V = [504.0, 504.0, 320.0, 288.0]           # 가상 핀홀 (Azure Kinect NFOV unbinned 공칭)
R_NB, Z_EDGE = 3.5, 0.2                      # 가장자리: 이웃 반지름 [px], 이웃 깊이 범위 [m]
SMPLX13 = [15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
VAL_GROUP = ["train/sequence_8", "train/sequence_9"]


def subjects():
    """β 묶음 → 피험자 번호 (감사 순서대로 B01…)."""
    return {n: i + 1 for i, g in enumerate(AUDIT["beta_groups(<0.05)"]) for n in g}


def geometry(calib):
    """레이더 ↔ master 카메라 ↔ 방. 반환 Rm, tm (카메라 → 레이더), R_c2r, t_c2r, 카메라 높이."""
    Rm, tm = np.array(calib["kinect_master"]["R"], float), np.array(calib["kinect_master"]["t"], float)
    Ro, to = np.array(calib["optitrack"]["R"], float), np.array(calib["optitrack"]["t"], float)
    n_r = Ro[:, 1]; c_r = -n_r @ to                                                # 레이더 좌표 높이 = n_r·X + c_r (모캡 y 위)
    n_c = Rm.T @ n_r; c_c = n_r @ tm + c_r                                         # 카메라 좌표 바닥: n_c·X + c_c = 0
    R, t = cc.room_frame(n_c, c_c)
    return Rm, tm, R, t, float(c_c)


def edge_flags(cam, pick, f):
    """cam (N,3) 카메라 좌표 점, pick 표본 번호 → 가장자리 bool."""
    ang = cam[:, :2] / cam[:, 2:3]; tr = cKDTree(ang); r = R_NB / f
    nb = tr.query_ball_point(ang[pick], r)
    cnt = np.array([len(x) for x in nb]); zr = np.array([np.ptp(cam[x, 2]) if len(x) else 0.0 for x in nb])
    full = np.median([len(x) for x in tr.query_ball_point(ang[np.random.default_rng(0).choice(len(ang), min(200, len(ang)), replace=False)], r)])
    return (cnt < 0.5 * full) | (zr > Z_EDGE)


def process(f):
    split, name = f.parent.name, f.stem; key = f"{split}/{name}"
    out = OUT / split / f"{name}.npz"
    if out.exists():
        return key, "skip", None
    z = np.load(f); calib = json.loads(str(z["calib"])); T = len(z["frame"])
    Rm, tm, R, t, h = geometry(calib)
    J = z["joints"].astype(float)[:, SMPLX13]
    lab = ((J - tm) @ Rm) @ R.T + t
    o, X = z["dpcl_off"], z["dpcl_xyz"]
    rng = np.random.default_rng(zlib.crc32(f"mmbody/{key}".encode()))
    uvz = np.zeros((T, N_PTS, 3), np.float32); edge = np.zeros((T, N_PTS), bool); valid = np.zeros((T, N_PTS), bool); n_raw = np.zeros(T, np.int32)
    fx, fy, cx, cy = K_V
    for k in range(T):
        P = X[o[k]:o[k + 1]].astype(float)
        if len(P) == 0:
            continue
        cam = (P - tm) @ Rm; cam = cam[cam[:, 2] > 0.3]; n_raw[k] = len(cam)
        if not len(cam):
            continue
        pick = rng.choice(len(cam), min(N_PTS, len(cam)), replace=False); c = cam[pick]
        uvz[k, :len(pick)] = np.c_[fx * c[:, 0] / c[:, 2] + cx, fy * c[:, 1] / c[:, 2] + cy, c[:, 2]]
        edge[k, :len(pick)] = edge_flags(cam, pick, fx) if len(cam) > 20 else True; valid[k, :len(pick)] = True
    q = np.zeros((T, 13), np.uint8); q[1:][np.all(J[1:] == J[:-1], -1)] |= 1
    sp = np.linalg.norm(np.gradient(J, axis=0), axis=-1) * FPS
    sid = subjects()[key]
    meta = dict(dataset="mmbody", env=name.split("__")[0] if split == "test" else "train", subject=f"B{sid:02d}", sequence=key, split=split,
                part="val" if key in VAL_GROUP else ("train" if split == "train" else "test"), T=int(T), fps=FPS,
                label_family="smplx55", label_source="optitrack_mosh++_smplx", kp_family=None, has_rgb=False,
                seg_leak=True, seg_source="dataset depth_pcl/master cropped with GT-skeleton boxes (paper 4.1; lowest point = lowest foot joint + 1 cm)",
                depth_empty=bool(n_raw.max() == 0), camera_height_m=h, room_frame="optitrack floor (calib)", spec=str(SPEC_PATH), version="v0.1",
                license="mmBody: non-commercial academic research only; no redistribution", seed=int(zlib.crc32(f"mmbody/{key}".encode())))
    OUT.joinpath(split).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, kp_px=np.zeros((T, 17, 2), np.float32), kp_valid=np.zeros((T, 17), bool),
                        pts_uvz=uvz, pts_edge=edge, pts_valid=valid, pts_n_raw=n_raw, pts_n_mixed=np.zeros(T, np.int32), comp_size=n_raw,
                        labels=lab.astype(np.float32), labels_world=z["joints"].astype(np.float32)[:, SMPLX13], label_mask=np.ones((T, 13), bool),
                        label_quality=q, joint_speed=sp.astype(np.float32), R_c2r=R, t_c2r=t, R_d2r=R, t_d2r=t, frame=z["frame"], meta=json.dumps(meta))
    return key, "ok", dict(T=T, h=h, foot_z=float(np.median(lab[:, 11:13, 2].min(1))), head_z=float(np.median(lab[:, 0, 2])),
                           pts=float(np.mean(n_raw)), edge=float(edge[valid].mean()) if valid.any() else None)


def write_spec(sig):
    S_ = dict(dataset="mmbody", version="v0.1", spec_path=str(SPEC_PATH), fps=FPS, label_family="smplx55", label_source="optitrack_mosh++_smplx",
              rgb=None, depth=dict(type="tof", sensor="Azure Kinect DK master (dataset point cloud, radar frame; virtual pinhole for storage)",
                                   image_wh=[640, 576], K=K_V, f_px=K_V[0], sigma0_m=sig["sigma0_m"], k_rel=sig["k_rel"],
                                   sigma_pix_px=float(1 / np.sqrt(12)), quant_step_m=0.001, corr_len_px=8.0, n_eff_per_256=31,
                                   provenance=sig["provenance"]),
              room=dict(source="calib.txt optitrack (MoCap -> radar), floor = MoCap y = 0; camera = kinect_master (nominal mount)"),
              audit="notes/mmbody_audit.json", seg_leak=True)
    SPEC_PATH.write_text(json.dumps(S_, indent=1))


if __name__ == "__main__":
    sig = json.loads(Path("/home/interns/unisense/notes/mmbody_sigma_gtfree.json").read_text())
    write_spec(sig)
    if "--spec-only" in sys.argv:
        sys.exit(0)
    files = sorted(CACHE.glob("*/*.npz")); rep = {}
    with Pool(8) as p:
        for key, st, info in p.imap_unordered(process, files):
            rep[key] = info; print(key, st, json.dumps(info), flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1)); print("packs done", len(files))
