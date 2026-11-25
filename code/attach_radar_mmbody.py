"""11-23 다중 센서 LHR — 데이터 1단계: mmBody Arbe 4D 레이더를 LHR 팩 (mmbody_v01, 방 좌표) 에 붙임.
변환 (정확, 데이터셋 보정값): 레이더 → 카메라 X_c = Rmᵀ (X_r − tm) (kinect_master: 카메라 → 레이더), 카메라 → 방 = 팩 R_c2r, t_c2r.
사람 점 고르기 = 신탁 조건 (깊이 depth_pcl 과 같은 정보 수준: 데이터셋이 정답 뼈대 상자로 자름): Arbe 점 중 몸 점은 4–5 % 뿐이고 정답 없는 규칙 (정지 배경 칸 제거, |v|>0, 움직임 중심) 은 몸 근처 비율 0.05–0.17 (11-23 시험).
→ 정답 뼈대 축 정렬 상자 + 0.3 m 안, 세기 상위 512 점. meta.person_select.oracle = true; mmBody 레이더는 학습 원천으로만 쓰고 정답 없는 주장 시험에는 쓰지 않음.
출력: dev/packs/mmbody_v02_radar/{train,test}/<이름>.npz (radar_pts (N,5) = 방 x y z, 도플러 m/s, 세기 dB; radar_off; meta). 검사: 변환 확인 (labels_world → 방 = labels), 몸 0.3 m 안 점 비율 (정답은 검사에만)."""
import os, json, glob
import numpy as np
U = "/home/interns/unisense"; SRC = f"{U}/dev/data/mmbody/cache_v1"; PK = f"{U}/dev/packs/mmbody_v01"; OUT = f"{U}/dev/packs/mmbody_v02_radar"
BN = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12), (1, 2)]
CARD = dict(device="Arbe Phoenix 4D (mmBody)", doppler_step_mps=0.071, frames_per_s=14, columns="x y z (방 m), 도플러 m/s (원본 열 4), 세기 dB (원본 열 5); 원본 열 3 은 전부 0")
VOX, OCC, CAP, PAD = 0.10, 0.60, 512, 0.30


def seg_frac(P, J, r=0.3):
    if len(P) == 0:
        return np.nan
    A = J[[a for a, _ in BN]]; d = J[[b for _, b in BN]] - A; L2 = (d * d).sum(-1).clip(1e-9)
    t = (((P[:, None] - A[None]) * d[None]).sum(-1) / L2[None]).clip(0, 1); D = np.linalg.norm(P[:, None] - (A[None] + t[..., None] * d[None]), axis=-1)
    return float((D.min(1) < r).mean())


rep = {}
for part in ("train", "test"):
    os.makedirs(f"{OUT}/{part}", exist_ok=True)
    for f in sorted(glob.glob(f"{PK}/{part}/*.npz")):
        nm = os.path.basename(f); P = np.load(f, allow_pickle=False); Z = np.load(f"{SRC}/{part}/{nm}", allow_pickle=False); c = json.loads(str(Z["calib"]))
        Rm, tm = np.array(c["kinect_master"]["R"], float), np.array(c["kinect_master"]["t"], float); Rc, tc = P["R_c2r"].astype(float), P["t_c2r"].astype(float)
        to_room = lambda X: ((X - tm) @ Rm) @ Rc.T + tc                                           # (X − tm) Rm = Rmᵀ(X − tm) 의 행 벡터 형
        chk = float(np.median(np.linalg.norm(to_room(P["labels_world"].reshape(-1, 3).astype(float)) - P["labels"].reshape(-1, 3), axis=1)))
        idx = {int(fr): i for i, fr in enumerate(Z["frame"])}; T = len(P["frame"]); ro = Z["radar_off"]; RD = Z["radar_data"]          # 한 번만 풂 (npz 는 접근마다 전체 압축 해제)
        bg = set()
        rows = []; fr_ = []; nraw = []; rng = np.random.default_rng(0)
        for t_, fr in enumerate(P["frame"]):
            i = idx.get(int(fr))
            if i is None:
                rows.append(np.zeros((0, 5), np.float32)); continue
            r = RD[ro[i]:ro[i + 1]].astype(float); q = to_room(r[:, :3]); nraw.append(len(q))
            J_ = P["labels"][t_].astype(float); lo, hi = J_.min(0) - PAD, J_.max(0) + PAD; keep = np.all((q >= lo) & (q <= hi), 1)          # 신탁: 정답 뼈대 상자 + PAD
            q, rr = q[keep], r[keep]
            if len(q) > CAP:
                o = np.argsort(-rr[:, 5])[:CAP]; q, rr = q[o], rr[o]
            rows.append(np.c_[q, rr[:, 4], rr[:, 5]].astype(np.float32))
            if rng.random() < 0.03 and len(q):
                fr_.append(seg_frac(q, P["labels"][t_].astype(float)))
        off = np.r_[0, np.cumsum([len(x) for x in rows])]
        meta = dict(dataset="mmbody", part=part, seq=nm[:-4], T=T, card=CARD, extrinsic=dict(source="dataset calib (kinect_master) + pack R_c2r", labels_world_to_room_check_mm=round(chk * 1000, 2)),
                    person_select=dict(method="GT skeleton box + 0.3 m, top-512 by power (same information level as dataset depth_pcl crop)", pad_m=PAD, cap=CAP, oracle=True, gt_free=False), version="v0.2-radar")
        np.savez_compressed(f"{OUT}/{part}/{nm}", radar_pts=np.concatenate(rows), radar_off=off, meta=json.dumps(meta, ensure_ascii=False))
        rep[f"{part}/{nm[:-4]}"] = dict(check_mm=round(chk * 1000, 2), raw_pts_med=float(np.median(nraw)) if nraw else 0, kept_pts_med=float(np.median(np.diff(off))), near_body_frac=round(float(np.nanmean(fr_)), 3) if fr_ else None, bg_voxels=len(bg))
        print(part, nm, json.dumps(rep[f"{part}/{nm[:-4]}"], ensure_ascii=False), flush=True)
json.dump(rep, open(f"{OUT}/attach_report.json", "w"), indent=1, ensure_ascii=False); print("ATTACH_DONE")
