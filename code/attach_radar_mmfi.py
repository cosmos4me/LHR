"""11-23 다중 센서 LHR — 데이터 1단계 (①' 세상 / ① 번역기 입력): MM-Fi 실측 TI 레이더를 LHR 팩 (mmfi_v02_specA, 방 좌표) 에 붙임.
레이더 캐시 (runs/xproj/cache/mmfi_radar_hfix.npz, E04 는 mmfi_radar_e04.npz) 는 레이더 기준 G 좌표; 팩 labels 는 방 좌표. 같은 정답 관절이므로
방마다 강체 변환 G→방 (Kabsch, 적합 피험자 = 학습 피험자; E04 는 S33 하나) 을 맞추고 레이더 점을 옮김. 변환 출처는 meta 에 'label_kabsch' 로 표시 (나중에 정답 없는 레이더↔깊이 정합으로 교체).
출력: dev/packs/mmfi_v03_radar/E0x/Sxx_Axx.npz (radar_pts (N,5) = 방 x y z, 도플러 m/s, 세기; radar_off (T+1); meta) — 팩 본체는 그대로 두고 옆에 붙이는 파일.
검사: 방마다 Kabsch 관절 잔차, 레이더 점이 정답 뼈 0.3 m 안에 있는 비율 (G 좌표 vs 방 좌표 — 같아야 함). 사용: python attach_radar_mmfi.py"""
import os, json, glob
import numpy as np
U = "/home/interns/unisense"; PK = f"{U}/dev/packs/mmfi_v02_specA"; OUT = f"{U}/dev/packs/mmfi_v03_radar"
TR = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26]; CAL = {1: TR, 2: TR, 3: TR, 4: [33]}
BN = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12), (1, 2)]
CARD = dict(device="TI IWR6843 (MM-Fi)", doppler_step_mps=0.6043, range_res_m=None, frames_per_s=10, note="점 = 5 열 (x y z 방 좌표 m, 도플러 m/s, 세기); 0.5 s 누적은 캐시 단계에서 이미 됨 여부 확인 필요")


def seg_frac(P, J, r=0.3):
    if len(P) == 0:
        return np.nan
    A = J[[a for a, _ in BN]]; d = J[[b for _, b in BN]] - A; L2 = (d * d).sum(-1).clip(1e-9)
    t = (((P[:, None] - A[None]) * d[None]).sum(-1) / L2[None]).clip(0, 1); D = np.linalg.norm(P[:, None] - (A[None] + t[..., None] * d[None]), axis=-1)
    return float((D.min(1) < r).mean())


def kabsch(X, Y):
    mx, my = X.mean(0), Y.mean(0); H = (X - mx).T @ (Y - my); U_, S, Vt = np.linalg.svd(H); Dg = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U_.T))]); R = Vt.T @ Dg @ U_.T
    return R, my - R @ mx


rep = {}
for e in (1, 2, 3, 4):
    F = dict(np.load(f"{U}/runs/xproj/cache/{'mmfi_radar_hfix' if e < 4 else 'mmfi_radar_e04'}.npz", allow_pickle=False)); room = (F["subj"] - 1) // 10 + 1
    m = room == e; seqs = np.unique(F["seq"][m]); Xs, Ys = [], []
    for s in seqs:
        sj = int(s[1:3])
        if sj not in CAL[e]:
            continue
        P = np.load(f"{PK}/E0{e}/{s}.npz", allow_pickle=False); ii = np.flatnonzero(m & (F["seq"] == s)); fr = F["frame"][ii]
        ok = P["label_mask"][fr].all(1) & F["M"][ii].all(1)
        Xs.append(F["J"][ii][ok].reshape(-1, 3)); Ys.append(P["labels"][fr][ok].reshape(-1, 3))
    X, Y = np.concatenate(Xs).astype(float), np.concatenate(Ys).astype(float); R, t = kabsch(X, Y); res = np.linalg.norm(X @ R.T + t - Y, axis=1)
    os.makedirs(f"{OUT}/E0{e}", exist_ok=True); fg, fr_ = [], []; rng = np.random.default_rng(e); nseq = 0
    for s in seqs:
        P = np.load(f"{PK}/E0{e}/{s}.npz", allow_pickle=False); T = P["labels"].shape[0]; ii = np.flatnonzero(m & (F["seq"] == s)); ii = ii[np.argsort(F["frame"][ii])]
        rows = [np.zeros((0, 5), np.float32)] * T
        for i in ii:
            q = F["pts"][F["off"][i]:F["off"][i + 1]].astype(float); q2 = q.copy(); q2[:, :3] = q[:, :3] @ R.T + t; rows[int(F["frame"][i])] = q2.astype(np.float32)
            if rng.random() < 0.02 and len(q):
                fg.append(seg_frac(q[:, :3], F["J"][i].astype(float))); fr_.append(seg_frac(q2[:, :3], P["labels"][int(F["frame"][i])].astype(float)))
        off = np.r_[0, np.cumsum([len(r) for r in rows])]
        meta = dict(dataset="mmfi", env=f"E0{e}", seq=s, T=T, card=CARD, extrinsic=dict(source="label_kabsch", fit_subjects=CAL[e], R=R.round(6).tolist(), t=t.round(6).tolist(), joint_resid_mm_med=round(float(np.median(res)) * 1000, 1)), version="v0.3-radar")
        np.savez_compressed(f"{OUT}/E0{e}/{s}.npz", radar_pts=np.concatenate(rows), radar_off=off, meta=json.dumps(meta, ensure_ascii=False)); nseq += 1
    rep[f"E0{e}"] = dict(seqs=nseq, kabsch_resid_mm_med=round(float(np.median(res)) * 1000, 1), kabsch_resid_mm_p90=round(float(np.percentile(res, 90)) * 1000, 1),
                         near_body_frac_G=round(float(np.nanmean(fg)), 3), near_body_frac_room=round(float(np.nanmean(fr_)), 3), R_det=round(float(np.linalg.det(R)), 4), t=t.round(3).tolist())
    print(f"E0{e}", json.dumps(rep[f"E0{e}"], ensure_ascii=False), flush=True)
json.dump(rep, open(f"{OUT}/attach_report.json", "w"), indent=1, ensure_ascii=False); print("ATTACH_DONE")
