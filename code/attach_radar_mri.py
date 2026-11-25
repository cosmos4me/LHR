"""11-23 다중 센서 LHR — 데이터 1단계: mRI TI 레이더 (CSV, 프레임별 점: X Y Z 도플러 세기, Camera Frame = 정렬 프레임 번호) 를 LHR 팩 (mri_v02, 방 좌표) 에 붙임.
프레임 대응: 팩 t = Camera Frame − frame0 (팩 meta). 레이더 → 방 변환은 공개값이 없어 학습 피험자 (splits/mri.json train) 의 점 ↔ 정답 뼈대 ICP 로 맞춤:
행렬식 +1 인 축 배치 24 가지를 모두 출발점으로 (반사 금지 — MM-Fi m0 교훈), 잘라낸 (70 %) 점–뼈 최근접 Kabsch 반복 20 회, 중앙 거리 최소를 고름. 출처 'label_icp_train' 표시 (나중에 정답 없는 정합으로 교체).
출력: dev/packs/mri_v03_radar/Sxx.npz (radar_pts (N,5) = 방 x y z, 도플러 m/s, 세기; radar_off (T+1); meta). 검사: 피험자별 몸 0.3 m 안 점 비율 (학습/시험 피험자 따로)."""
import os, csv, json, itertools
import numpy as np
U = "/home/interns/unisense"; CSV = f"{U}/dev/data/mri/extracted/dataset_release/aligned_data/radar/singleframe"; PK = f"{U}/dev/packs/mri_v03"; OUT = f"{U}/dev/packs/mri_v03_radar"
SPL = json.load(open(f"{U}/code/splits/mri.json")); TRAIN = [int(str(s).lstrip("Ssubject")) for s in SPL["train"]]
BN = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12), (1, 2)]
CARD = dict(device="TI IWR1443 (mRI)", doppler_step_mps=0.356, frames_per_s=8.7, columns="x y z (방 m), 도플러 m/s, 세기")
os.makedirs(OUT, exist_ok=True)


def read_csv(s):
    out = {}
    with open(f"{CSV}/subject{s}.csv") as fh:
        for r in csv.DictReader(fh):
            c = int(float(r["Camera Frame"])); out.setdefault(c, {})
            out[c].setdefault(int(r["Frame #"]), []).append([float(r["X"]), float(r["Y"]), float(r["Z"]), float(r["Doppler"]), float(r["Intensity"])])
    return {c: np.array(v[max(v)], float) for c, v in out.items()}                    # 같은 정렬 프레임에 레이더 프레임 여럿이면 마지막


def closest(P, J):
    A = J[[a for a, _ in BN]]; d = J[[b for _, b in BN]] - A; L2 = (d * d).sum(-1).clip(1e-9)
    t = (((P[:, None] - A[None]) * d[None]).sum(-1) / L2[None]).clip(0, 1); Q = A[None] + t[..., None] * d[None]; D = np.linalg.norm(P[:, None] - Q, axis=-1); k = D.argmin(1)
    return Q[np.arange(len(P)), k], D[np.arange(len(P)), k]


def kabsch(X, Y):
    mx, my = X.mean(0), Y.mean(0); U_, S, Vt = np.linalg.svd((X - mx).T @ (Y - my)); Dg = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U_.T))]); R = Vt.T @ Dg @ U_.T
    return R, my - R @ mx


subj = sorted(int(f[7:-4]) for f in os.listdir(CSV) if f.startswith("subject")); RAD = {s: read_csv(s) for s in subj}; PKS = {}
for s in subj:
    P = np.load(f"{PK}/S{s:02d}.npz", allow_pickle=False); PKS[s] = (P["labels"].astype(float), P["label_mask"], json.loads(str(P["meta"]))["frame0"])
# ICP 맞춤용 표본 (학습 피험자)
rng = np.random.default_rng(0); smp = []
for s in TRAIN:
    L, M, f0 = PKS[s]; cs = [c for c in RAD[s] if 0 <= c - f0 < len(L) and M[c - f0].all() and len(RAD[s][c]) >= 5]
    for c in rng.choice(cs, min(250, len(cs)), replace=False):
        smp.append((RAD[s][c][:, :3], L[c - f0]))
best = None
for perm in itertools.permutations(range(3)):
    for sg in itertools.product([1, -1], repeat=3):
        R = np.zeros((3, 3))
        for i in range(3):
            R[i, perm[i]] = sg[i]
        if np.linalg.det(R) < 0:
            continue
        X0 = np.concatenate([p for p, _ in smp]); Y0 = np.concatenate([np.repeat(j[[7, 8]].mean(0, keepdims=True), len(p), 0) for p, j in smp]); t = np.median(Y0 - X0 @ R.T, 0)
        for it in range(20):
            Xs, Ys = [], []
            for p, j in smp:
                q = p @ R.T + t; c_, d_ = closest(q, j); keep = d_ <= np.percentile(d_, 70); Xs.append(p[keep]); Ys.append(c_[keep])
            R, t = kabsch(np.concatenate(Xs), np.concatenate(Ys))
        med = float(np.median(np.concatenate([closest(p @ R.T + t, j)[1] for p, j in smp])))
        print("출발", perm, sg, "→ 중앙 거리", round(med, 3), flush=True)
        if best is None or med < best[0]:
            best = (med, R, t, perm, sg)
med, R, t, perm, sg = best; print("최적", perm, sg, round(med, 3), "det", round(float(np.linalg.det(R)), 4), "t", t.round(3), flush=True)
rep = {}
for s in subj:
    L, M, f0 = PKS[s]; T = len(L); rows = [np.zeros((0, 5), np.float32)] * T; fr = []
    for c, p in RAD[s].items():
        if 0 <= c - f0 < T:
            q = p.copy(); q[:, :3] = p[:, :3] @ R.T + t; rows[c - f0] = q.astype(np.float32)
            if rng.random() < 0.05 and M[c - f0].all():
                fr.append(float((closest(q[:, :3], L[c - f0])[1] < 0.3).mean()))
    off = np.r_[0, np.cumsum([len(r) for r in rows])]
    meta = dict(dataset="mri", subject=f"S{s:02d}", T=T, frame0=f0, card=CARD, extrinsic=dict(source="label_icp_train", fit_subjects=TRAIN, R=R.round(6).tolist(), t=t.round(6).tolist(), median_dist_m=round(med, 4), start_perm=list(perm), start_sign=list(sg)), version="v0.3-radar")
    np.savez_compressed(f"{OUT}/S{s:02d}.npz", radar_pts=np.concatenate(rows), radar_off=off, meta=json.dumps(meta, ensure_ascii=False))
    rep[f"S{s:02d}"] = dict(train=s in TRAIN, frames_with_radar=int(sum(len(r) > 0 for r in rows)), T=T, pts_med=float(np.median([len(r) for r in rows if len(r)])), near_body_frac=round(float(np.mean(fr)), 3) if fr else None)
    print(f"S{s:02d}", json.dumps(rep[f"S{s:02d}"], ensure_ascii=False), flush=True)
json.dump(dict(extrinsic=dict(R=R.tolist(), t=t.tolist(), median_dist_m=med), subjects=rep), open(f"{OUT}/attach_report.json", "w"), indent=1, ensure_ascii=False); print("ATTACH_DONE")
