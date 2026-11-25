"""11-25 덤프(pred.npz)에서 피험자 묶음별 MPJPE / rootrel / 골반 3D (정답 정상 프레임). 사용: python subset_eval.py <run_dir>"""
import sys, json
import numpy as np
run = sys.argv[1]; d = np.load(f"{run}/pred.npz"); P, G = d["P"].astype(float), d["G"].astype(float)
s = np.array([str(x).split(":")[-1] for x in d["sub"]]); SP = json.load(open("/home/interns/unisense/code/splits/mri_devfinal.json"))
pg, pp = G[:, [7, 8]].mean(1), P[:, [7, 8]].mean(1); ok = (np.linalg.norm(G - pg[:, None], axis=-1) < 1.2).all(1)
e = np.linalg.norm(P - G, axis=-1).mean(1) * 1000; rr = np.linalg.norm((P - pp[:, None]) - (G - pg[:, None]), axis=-1).mean(1) * 1000; p3 = np.linalg.norm(pp - pg, axis=1) * 1000
SP["fold3_C"] = ["S04", "S08", "S12", "S16", "S20"]   # C (mRI 만 4겹) 겹 3 시험 피험자 — C 와 같은 피험자 비교용
for name in ["dev", "fold3_C"]:  # final 은 규칙상 따로 보지 않음
    m = np.isin(s, SP[name]) & ok
    print(f"{name:13s} 피험자 {len(SP[name]):2d} 프레임 {m.sum():6d} | MPJPE {e[m].mean():6.1f} rootrel {rr[m].mean():6.1f} 골반3D {p3[m].mean():6.1f}")
m = ok; print(f"{'all':13s} 피험자 {len(np.unique(s)):2d} 프레임 {m.sum():6d} | MPJPE {e[m].mean():6.1f} rootrel {rr[m].mean():6.1f} 골반3D {p3[m].mean():6.1f}")
