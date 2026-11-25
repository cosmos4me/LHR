"""번역기 물리 추정 v2 (학습 없음): phys_v1의 기준점·관절 추정에 물리 공분산 Σ를 더한다 → dev/packs/phys_v2/{ds}/...npz
조건 순서 [rgb, depth, both]. a3 (T,3,3), x3 (T,3,13,3), sa (T,3,3,3) 기준점 Σ, sx (T,3,13,3,3) 관절 Σ.
Σ는 광선 기준으로 만든다: Σ = σ_∥² uuᵀ + σ_⊥² (I − uuᵀ) + σ_def² I (u = 카메라 → 점 단위벡터).
  - 2D 잡음 σ_px: notes/sigma_px_gtfree.json (GT 없음; MM-Fi는 평활된 키포인트라 과소 → 사전 하한 PX_MIN).
  - 단안(MM-Fi rgb): 발목 높이 불확실 σ_h → 발목 광선 거리 오차 σ_h/|n·d|, 정면 평면 가정의 몸 깊이 σ_fp. σ_⊥ = Z·σ_px/f.
  - 깊이 기준점(사람 점 중앙값 + 몸 두께 절반): σ_∥² = σ_z(Z)²/N_eff + σ_thick², σ_⊥ = HALF_BODY (보이는 점 중심 ↔ 골반).
  - 희소 깊이 관절(MM-Fi both): σ_∥² = σ_z(Z)² + (OFF/2)² + 선택 점 깊이 퍼짐²/n, σ_⊥ = Z·σ_px/f.
  - 삼각측량(mRI): Σ = (Σ_i (I − d_i d_iᵀ)/σ_i²)⁻¹, σ_i = Z_i·σ_px/f_i.
  - 시간 일관성(움직임 연속성, 데이터에서): 기준점이 시간 중앙값에서 벗어난 거리 τ를 등방으로 더한다.
  - 빈 프레임을 가까운 프레임으로 채웠으면 V_FILL·|Δt|를 등방으로 더한다 (사람 이동 속도 사전).
  - 골반 기준점 Σ = (Σ_왼엉덩이 + Σ_오른엉덩이)/2 (완전 상관 가정, 보수적).
사전 상수는 모두 문헌·기존 번역기 상수에서 왔고 GT로 맞추지 않았다.
사용: python phys_features_v2.py [mmfi mri octonet]"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import conv_common as cc
import physics_baseline as pb
from phys_features import fill_nearest, hips

OUT = Path("/home/interns/unisense/dev/packs/phys_v2")
SPX = json.loads(Path("/home/interns/unisense/notes/sigma_px_gtfree.json").read_text())
PX_MIN = 2.0          # 2D 검출 잡음 하한 [px] (사전)
SIG_DEF = 0.05        # 라벨 체계별 관절 정의 차이 [m] (사전, 표면→관절 오프셋 크기 수준)
SIG_H = 0.04          # 발목 높이 불확실 [m] (ANKLE_H의 50%)
SIG_FP = pb.HALF_BODY  # 정면 평면 가정의 몸 깊이 [m]
SIG_THICK = 0.05      # 몸 두께 절반의 불확실 [m] (HALF_BODY의 50%)
V_FILL = 1.0          # 채운 프레임의 위치 불확실 증가 [m/s] (보행 속도 사전)
I3 = np.eye(3)
C17 = [[3, 4], 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def ray_cov(u, s_par, s_perp):
    uu = u[..., :, None] * u[..., None, :]
    return (s_par ** 2)[..., None, None] * uu + (s_perp ** 2)[..., None, None] * (I3 - uu) + SIG_DEF ** 2 * I3


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cov13(S):
    """(...,17,3,3) → (...,13,3,3); 머리 = 두 귀 Σ 평균 (보수적)."""
    return np.stack([S[..., m, :, :].mean(-3) if isinstance(m, list) else S[..., m, :, :] for m in C17], -3)


def hip_cov(S13):
    return 0.5 * (S13[:, 7] + S13[:, 8])


def iso(s):
    return (s ** 2)[..., None, None] * I3


def tconsist(a):
    """기준점이 시간 중앙값에서 벗어난 거리 (데이터에서 본 움직임 연속성 위반)."""
    d = np.linalg.norm(a - pb.tmed(a), axis=-1)
    return np.nan_to_num(d, nan=0.0)


def depth_root_cov(p, ok, o, a, dmodel):
    """사람 점 중앙값 기준점의 Σ. 깊이 잡음은 점 수의 유효 독립 개수(사양 n_eff_per_256)로 줄인다."""
    u = unit(a - o); Z = np.linalg.norm(a - o, axis=-1)
    n = ok.sum(1).astype(float); neff = np.maximum(1.0, n * dmodel.get("n_eff_per_256", 31) / 256.0)
    s_par = np.sqrt(cc.axial_sigma(Z, dmodel) ** 2 / neff + SIG_THICK ** 2)
    return ray_cov(u, s_par, np.full_like(Z, pb.HALF_BODY))


def mono_cov(P, spec, X, n=np.array([0, 0, 1.0])):
    """단안 발목–바닥 + 정면 평면 해의 관절 Σ (17관절)."""
    o, d = pb.rays(P["kp_px"], spec["rgb"]["K"], P["R_c2r"], P["t_c2r"])
    f = 0.5 * (spec["rgb"]["K"][0] + spec["rgb"]["K"][1])
    nd = d[:, 15:17] @ n
    t = (pb.ANKLE_H - (o @ n)) / nd
    okk = (nd < -0.05) & (t > 0.3) & (t < 10); t = np.where(okk, t, np.inf); k = np.argmin(t, 1)
    da = d[np.arange(len(d)), 15 + k]; nda = np.abs(da @ n)
    ax = np.asarray(P["R_c2r"])[:, 2]; fwd = ax - (ax @ n) * n; fwd /= np.linalg.norm(fwd)
    dj = np.maximum(d @ fwd, 0.1)                                                    # (T,17)
    s_a = (SIG_H / np.maximum(nda, 0.05)) * (da @ fwd)                               # 발목 기준 평면 위치 오차 (fwd 방향)
    s_par = np.sqrt(s_a[:, None] ** 2 + SIG_FP ** 2) / dj
    Z = np.linalg.norm(X - o, axis=-1)
    spx = np.maximum(np.asarray(SPX["mmfi"]), PX_MIN)
    return ray_cov(d, s_par, Z * spx / f)


def sparse_cov(P, spec, X):
    """MM-Fi 희소 깊이 관절의 Σ (17관절)."""
    o, d = pb.rays(P["kp_px"], spec["rgb"]["K"], P["R_c2r"], P["t_c2r"])
    p, ok = pb.depth_points(P, spec["depth"])
    v = p - o; t = np.einsum("tjc,tnc->tjn", d, v)
    sel = ok[:, None, :] & ((v ** 2).sum(-1)[:, None, :] - t ** 2 < pb.TAU ** 2) & (t > 0)
    tt = np.where(sel, t, np.nan); med = np.nanmedian(tt, -1)
    mad = 1.4826 * np.nanmedian(np.abs(tt - med[..., None]), -1); cnt = np.maximum(sel.sum(-1), 1)
    Z = np.linalg.norm(X - o, axis=-1); f = 0.5 * (spec["rgb"]["K"][0] + spec["rgb"]["K"][1])
    s_par = np.sqrt(cc.axial_sigma(np.nan_to_num(Z, nan=3.0), spec["depth"]) ** 2 + (pb.OFF / 2) ** 2 + np.nan_to_num(mad) ** 2 / cnt)
    spx = np.maximum(np.asarray(SPX["mmfi"]), PX_MIN)
    return ray_cov(d, s_par, np.nan_to_num(Z, nan=3.0) * spx / f)


def tri_cov(o1, d1, o2, d2, X, f1, f2, spx):
    def info(o, d, f):
        Z = np.linalg.norm(X - o, axis=-1); s = Z * spx / f
        return (I3 - d[..., :, None] * d[..., None, :]) / (s ** 2)[..., None, None]
    A = info(o1, d1, f1) + info(o2, d2, f2)
    A = np.where(np.isfinite(A).all((-1, -2))[..., None, None], A, I3)
    return np.linalg.inv(A) + SIG_DEF ** 2 * I3


def finish(a, S, fps):
    """빈 기준점 채우기 + 채운 거리·시간 불일치 Σ 추가."""
    ok = np.isfinite(a).all(-1)
    if not ok.any():
        return a, S, int(len(a))
    idx = np.where(ok)[0]; j = idx[np.abs(np.arange(len(a))[:, None] - idx[None]).argmin(1)]
    gap = np.abs(np.arange(len(a)) - j) / fps
    a2, S2 = a[j], S[j] + iso(V_FILL * gap)
    S2 = S2 + iso(tconsist(a2))
    return a2, S2, int((~ok).sum())


def one(job):
    ds, f = job
    P = np.load(f); T = len(P["labels"]); m = json.loads(P["meta"].item())
    a3 = np.full((T, 3, 3), np.nan); x3 = np.full((T, 3, 13, 3), np.nan)
    sa = np.tile(I3, (T, 3, 1, 1)); sx = np.tile(I3, (T, 3, 13, 1, 1))
    spec = json.loads(pb.SPECS[ds].read_text()); fps = float(spec.get("fps", 10))
    if ds == "mmfi":
        o, d = pb.rays(P["kp_px"], spec["rgb"]["K"], P["R_c2r"], P["t_c2r"])
        Xm, _ = pb.mono_floor(o, d, P["R_c2r"], np.array([0, 0, 1.0]), 0.0)
        x3[:, 0] = cc.to_common13(Xm, "coco17"); a3[:, 0] = hips(x3[:, 0])
        sx[:, 0] = cov13(mono_cov(P, spec, Xm)); sa[:, 0] = hip_cov(sx[:, 0])
        p, ok = pb.depth_points(P, spec["depth"]); od = np.asarray(P["t_d2r"], float)
        a3[:, 1] = pb.person_root(p, ok, od); sa[:, 1] = depth_root_cov(p, ok, od, a3[:, 1], spec["depth"])
        Xs = pb.mmfi_sparse(P, spec, pb.OFF)
        x3[:, 2] = cc.to_common13(Xs, "coco17"); sx[:, 2] = cov13(sparse_cov(P, spec, Xs))
        hb = hips(x3[:, 2]); hv = np.isfinite(hb).all(-1)
        a3[:, 2] = np.where(hv[:, None], hb, a3[:, 1]); sa[:, 2] = np.where(hv[:, None, None], hip_cov(sx[:, 2]), sa[:, 1])
    elif ds == "mri":
        R12, T12 = np.asarray(spec["rgb"]["R_cam1_to_cam2"]), np.asarray(spec["rgb"]["t_cam1_to_cam2"])
        Rc, tc = np.asarray(P["R_c2r"]), np.asarray(P["t_c2r"], float)
        o1, d1 = pb.rays(P["kp_px"], spec["rgb"]["K"], Rc, tc)
        o2, d2 = pb.rays(P["kp_px_cam2"], spec["rgb"]["K_cam2"], Rc @ R12.T, Rc @ (-R12.T @ T12) + tc)
        s1, s2 = P["kp_score"].astype(float), P["kp_score_cam2"].astype(float)
        X17 = pb.triangulate(o1, d1, s1 ** 2, o2, d2, s2 ** 2); X17[(s1 < pb.MIN_SCORE) | (s2 < pb.MIN_SCORE)] = np.nan
        K1, K2 = np.asarray(spec["rgb"]["K"]), np.asarray(spec["rgb"]["K_cam2"])
        S17 = tri_cov(o1, d1, o2, d2, np.nan_to_num(X17, nan=3.0), 0.5 * (K1[0, 0] + K1[1, 1]), 0.5 * (K2[0, 0] + K2[1, 1]), np.asarray(SPX["mri"]))
        x3[:, 0] = cc.to_common13(X17, "coco17"); a3[:, 0] = hips(x3[:, 0])
        sx[:, 0] = cov13(S17); sa[:, 0] = hip_cov(sx[:, 0])
    else:
        dm = spec["depth"][f"{m['env']}_{m['day']}"] if ds == "octonet" else spec["depth"]   # mmbody: 단일 사양
        p, ok = pb.depth_points(P, dm); od = np.asarray(P["t_d2r"], float)
        a3[:, 1] = pb.person_root(p, ok, od); sa[:, 1] = depth_root_cov(p, ok, od, a3[:, 1], dm)
    filled = {}
    for c in range(3):
        if np.isfinite(a3[:, c]).any():
            a3[:, c], sa[:, c], filled[c] = finish(a3[:, c], sa[:, c], fps)
    sa = np.where(np.isfinite(sa).all((-1, -2))[..., None, None], sa, I3)
    sx = np.where(np.isfinite(sx).all((-1, -2))[..., None, None], sx, I3)
    rel = f.relative_to(pb.PACKS[ds])
    out = OUT / ds / rel; out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, a3=a3.astype(np.float32), x3=x3.astype(np.float32), sa=sa.astype(np.float32), sx=sx.astype(np.float32))
    return ds, str(rel), filled


if __name__ == "__main__":
    dss = sys.argv[1:] or ["mmfi", "mri", "octonet"]
    jobs = []
    for ds in dss:
        files = sorted(pb.PACKS[ds].glob("E*/*.npz" if ds == "mmfi" else "*.npz"))
        jobs += [(ds, f) for f in files]
    stats = {}
    with Pool(16) as pool:
        for ds, rel, filled in pool.imap_unordered(one, jobs):
            s = stats.setdefault(ds, dict(files=0, filled={0: 0, 1: 0, 2: 0}))
            s["files"] += 1
            for c, n in filled.items():
                s["filled"][c] += n
    print(json.dumps(stats, indent=1))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    (OUT / "READY").touch()
