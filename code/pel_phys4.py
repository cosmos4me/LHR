"""11-25 pel_phys4 = pel_phys3 + (3) 레이더 단서도 정합 (ms_cache3.unify_radar) 과 같은 사람 문: 프레임마다 움직이는 점의 수평 중앙값에서 0.8 m 안 점만 평균 (유령·잡음 점 6–7 % 제외; 진단 mRI 시선 치우침 −68 → −21 mm).
11-25 pel_phys3 = pel_phys2 + (1) 바닥 단서에 발목·바닥 높이 불확실성 σ_h = 3 cm (광선이 바닥과 스치는 각이면 시선 방향으로 길어짐: σ = σ_h·|d_xy|/|d_z|; 설계 문서 2.2 "발목 광선 ∩ 바닥 +13–22 cm" 처리)
  (2) 깊이와 LiDAR 가 함께 있으면 같은 몸 표면을 보는 상관 단서라 한 번만 셈 (정보 절반씩; 카메라 2대 규칙과 같은 원리. 감사: 깊이+LiDAR 68% 적중 0.26).
11-24 pel_phys2 = pel_phys + 다중 카메라 결합 규칙 (notes/v3_rgb_bias_diag.md §2): 카메라 2대 엉덩이 광선 → 삼각측량 점 하나 (시선 σ = r·σΔ/sinγ, 수직 σ = r·σΔ, σΔ = 3 mrad 검출기 사전값),
상관된 바닥·크기 단서는 한 번만 (카메라 0 우선). 퇴화 삼각측량 (sinγ<0.01, 거리 0.5–8 m 밖, 카메라 뒤) 은 기존 광선 2개로 되돌림. 카메라 1대면 pel_phys 와 같음.
11-24 골반만 찾기 (사용자: "다른 관절은 찾으려 하지 말고 골반 위치만, 번역기·인코더·풀이기로").
골반 = 엉덩이 L/R (H13 7, 8) 의 중점, 방 좌표 (원점 = 카메라 아래 바닥, x 정면, z 위, m).

① 번역기 (물리, 학습 없음, 상수는 사전값): 모달리티마다 골반에 대한 가우스 정보 (Λ = Σ⁻¹, η = Λ μ). 한 방향만 아는 측정은 그 방향 정보만 줌.
  rgb_ray   카메라마다 엉덩이 중점 광선: 광선에 수직인 두 방향만, σ⊥ = σθ · 거리 (σθ = 15 mrad)
  rgb_floor 카메라마다 발목 중점 광선 ∩ 발목 높이 평면 (0.08 m) → 수평 위치만: σ² = 몸 기울기 0.12² + 각도 오차 (바닥과 스치는 각이면 길어짐)
  rgb_size  카메라마다 어깨 중점–엉덩이 중점 광선 사이 각 α, 몸통 길이 0.50 m → 거리 r = 0.5/α, 광선 방향만, σ_r = 0.25 r
  depth / lidar 사람 점 평균 + 센서에서 먼 쪽으로 몸 반두께 0.10 m, 높이 −0.07 m (보이는 표면 중심 ≈ 0.57 H, 엉덩이 ≈ 0.52 H); σ = 0.08, 0.08, 0.10
  radar     움직이는 점 (정지 표시 아님) 3개 이상 평균; σ = 0.20, 0.20, 0.35 (팔 쪽 쏠림 25–55 cm, 고도 거침)
  prior     높이만 z ~ N(0.85, 0.35²) (서기 0.9 · 앉기 0.5 · 눕기 0.2; 몸 지식), 수평 사전 없음 (위치는 외우지 않음)
③ 풀이기: 정보 합 → μ, Σ. 광선 σ⊥ 는 첫 추정 거리로 한 번 더 계산 (2회). 선택: 시퀀스 안 Σ 가중 1.1 s 평활.
사용: python pel_phys.py  → 물리 단독 감사 (데이터셋 × 모달리티 조합) runs/pel1/phys_audit.json"""
import sys, os, json, math, itertools
import numpy as np
import torch
import torch.nn.functional as F

SIG_TH, H_ANK, SIG_LEAN, L_TORSO, SIZE_REL = 0.015, 0.08, 0.12, 0.50, 0.25
B_THICK, DZ_C, SIG_D = 0.10, 0.07, (0.08, 0.08, 0.10)
SIG_R = (0.20, 0.20, 0.35); Z_PRIOR, SIG_ZP = 0.85, 0.35
SIG_DELTA = 0.003
SIG_H = 0.03   # pel_phys3: 발목 관절 높이·바닥 높이 불확실성 (m)
COMPS = ["rgb_tri", "rgb_ray0", "rgb_ray1", "rgb_floor0", "rgb_floor1", "rgb_size0", "rgb_size1", "depth", "lidar", "radar"]
MOD_OF = {"rgb_tri": "rgb", "rgb_ray0": "rgb", "rgb_ray1": "rgb", "rgb_floor0": "rgb", "rgb_floor1": "rgb", "rgb_size0": "rgb", "rgb_size1": "rgb", "depth": "depth", "lidar": "lidar", "radar": "radar"}


def _ray_info(o, d, sig):
    I = torch.eye(3, device=o.device).expand(*o.shape[:-1], 3, 3); L = (I - d[..., :, None] * d[..., None, :]) / sig[..., None, None] ** 2
    return L, (L @ o[..., None])[..., 0]


def _cloud_info(pts, m, org, sig, dz, push):
    w = m.float(); n = w.sum(1); c = (pts * w[..., None]).sum(1) / n.clamp(min=1)[:, None]
    u = c[:, :2] - org[:, :2]; u = u / torch.linalg.norm(u, dim=1, keepdim=True).clamp(min=1e-3); mu = c.clone(); mu[:, :2] += push * u; mu[:, 2] -= dz
    U = torch.zeros(len(c), 3, 3, device=c.device); U[:, :2, 0] = u; U[:, 0, 1], U[:, 1, 1] = -u[:, 1], u[:, 0]; U[:, 2, 2] = 1
    L = U @ torch.diag(torch.tensor([1 / s ** 2 for s in sig], device=c.device)) @ U.transpose(1, 2)
    return L, (L @ mu[..., None])[..., 0], n


def components(ray, rgbm, dep, dv, lid, lv, rad, rv, org, rguess=None):
    """입력은 배치 텐서 (가린 모달리티는 마스크가 False). 출력 {성분: (Λ (B,3,3), η (B,3), 유효 (B,))}."""
    B = ray.shape[0]; out = {}; dev = ray.device
    o, d = ray[..., :3], ray[..., 3:6]
    for c in range(2):
        hv = rgbm[:, c, 7] & rgbm[:, c, 8]; oc = o[:, c, 7]; dh = F.normalize(d[:, c, 7] + d[:, c, 8], dim=-1)
        r = torch.full((B,), 3.0, device=dev) if rguess is None else torch.linalg.norm(rguess - oc, dim=-1).clamp(0.5, 10.0)
        L, e = _ray_info(oc, dh, SIG_TH * r); out[f"rgb_ray{c}"] = (L, e, hv)
        av = rgbm[:, c, 11] & rgbm[:, c, 12]; da = F.normalize(d[:, c, 11] + d[:, c, 12], dim=-1); t = (H_ANK - oc[:, 2]) / da[:, 2].clamp(max=-1e-3)
        fv = av & hv & (da[:, 2] < -0.02) & (t > 0.3) & (t < 12); A = oc + t[:, None] * da
        uh = F.normalize(da[:, :2], dim=-1); wh = torch.stack([-uh[:, 1], uh[:, 0]], -1); sa = SIG_TH * t / da[:, 2].abs().clamp(min=0.02); sp = SIG_TH * t
        sh = SIG_H * torch.linalg.norm(da[:, :2], dim=-1) / da[:, 2].abs().clamp(min=0.02)                    # pel_phys3 (1)
        Sh = SIG_LEAN ** 2 * torch.eye(2, device=dev) + (sa ** 2 + sh ** 2)[:, None, None] * uh[:, :, None] * uh[:, None, :] + sp[:, None, None] ** 2 * wh[:, :, None] * wh[:, None, :]
        L = torch.zeros(B, 3, 3, device=dev); L[:, :2, :2] = torch.linalg.inv(Sh); out[f"rgb_floor{c}"] = (L, (L @ A[..., None])[..., 0], fv)
        sv = hv & rgbm[:, c, 1] & rgbm[:, c, 2]; ds_ = F.normalize(d[:, c, 1] + d[:, c, 2], dim=-1); al = torch.arccos((ds_ * dh).sum(-1).clamp(-1, 1)).clamp(min=0.02)
        rr = L_TORSO / al; L = dh[:, :, None] * dh[:, None, :] / (SIZE_REL * rr)[:, None, None] ** 2; out[f"rgb_size{c}"] = (L, (L @ (oc + rr[:, None] * dh)[..., None])[..., 0], sv & (rr < 12))
    # 다중 카메라 결합 규칙: 두 카메라 엉덩이 광선이 모두 유효하면 삼각측량 점 하나로 (광선 2개 대신), 바닥·크기 단서는 카메라 0 것만
    h0, h1 = rgbm[:, 0, 7] & rgbm[:, 0, 8], rgbm[:, 1, 7] & rgbm[:, 1, 8]
    o0, o1 = o[:, 0, 7], o[:, 1, 7]; d0, d1 = F.normalize(d[:, 0, 7] + d[:, 0, 8], dim=-1), F.normalize(d[:, 1, 7] + d[:, 1, 8], dim=-1)
    w = o0 - o1; a_, b_, c_ = (d0 * d0).sum(-1), (d0 * d1).sum(-1), (d1 * d1).sum(-1); dd_, ee_ = (d0 * w).sum(-1), (d1 * w).sum(-1); den = (a_ * c_ - b_ * b_).clamp(min=1e-9)
    s0, s1 = (b_ * ee_ - c_ * dd_) / den, (a_ * ee_ - b_ * dd_) / den; X = 0.5 * ((o0 + s0[:, None] * d0) + (o1 + s1[:, None] * d1))
    sg = torch.linalg.norm(torch.cross(d0, d1, dim=-1), dim=-1); r_ = torch.linalg.norm(X - 0.5 * (o0 + o1), dim=-1)
    st = h0 & h1 & (sg > 0.01) & (r_ > 0.5) & (r_ < 8.0) & (s0 > 0.3) & (s1 > 0.3)
    u_ = F.normalize(d0 + d1, dim=-1); sp = SIG_DELTA * r_.clamp(0.5, 8.0); sl = sp / sg.clamp(min=0.01)
    I3 = torch.eye(3, device=dev).expand(B, 3, 3); P_ = u_[:, :, None] * u_[:, None, :]
    Lt = (I3 - P_) / sp[:, None, None] ** 2 + P_ / sl[:, None, None] ** 2; Lt = Lt * st[:, None, None].float()
    out["rgb_tri"] = (Lt, (Lt @ torch.nan_to_num(X)[..., None])[..., 0], st)
    for nm in ("rgb_ray0", "rgb_ray1"):
        L_, e_, v_ = out[nm]; out[nm] = (L_, e_, v_ & ~st)
    for k_ in ("floor", "size"):
        L_, e_, v_ = out[f"rgb_{k_}1"]; out[f"rgb_{k_}1"] = (L_, e_, v_ & ~(h0 & out[f"rgb_{k_}0"][2]))
    L, e, n = _cloud_info(dep, dv, org, SIG_D, DZ_C, B_THICK); out["depth"] = (L, e, n >= 10)
    L, e, n = _cloud_info(lid, lv, org, SIG_D, DZ_C, B_THICK); out["lidar"] = (L, e, n >= 10)
    both = (out["depth"][2] & out["lidar"][2]).float(); fct = 1.0 - 0.5 * both                         # pel_phys3 (2)
    for k_ in ("depth", "lidar"):
        L_, e_, v_ = out[k_]; out[k_] = (L_ * fct[:, None, None], e_ * fct[:, None], v_)
    dyn = rv & (rad[..., 5] < 0.5); Xn = torch.where(dyn[..., None], rad[..., :3], torch.full_like(rad[..., :3], float("nan"))); md = torch.nan_to_num(torch.nanmedian(Xn[..., :2], dim=1).values)   # pel_phys4 (3)
    dyn = dyn & (torch.linalg.norm(rad[..., :2] - md[:, None], dim=-1) < 0.8); w = dyn.float(); n = w.sum(1); c_ = (rad[..., :3] * w[..., None]).sum(1) / n.clamp(min=1)[:, None]
    L = torch.diag(torch.tensor([1 / s ** 2 for s in SIG_R], device=dev)).expand(B, 3, 3).clone(); out["radar"] = (L, (L @ c_[..., None])[..., 0], n >= 3)
    return out


def solve(comps, use, B, dev):
    """③ 정보 합. use = 쓸 성분 이름 목록. 반환 μ (B,3), Σ (B,3,3), 정보 있는 성분 수 (B,)."""
    L = torch.zeros(B, 3, 3, device=dev); e = torch.zeros(B, 3, device=dev); k = torch.zeros(B, device=dev)
    L[:, 2, 2] += 1 / SIG_ZP ** 2; e[:, 2] += Z_PRIOR / SIG_ZP ** 2; L += 1e-4 * torch.eye(3, device=dev)
    for nm in use:
        Lc, ec, v = comps[nm]; vf = v.float(); L = L + Lc * vf[:, None, None]; e = e + ec * vf[:, None]; k = k + vf
    S = torch.linalg.inv(L); return (S @ e[..., None])[..., 0], S, k


def physics(T, keep, idx=None):
    """T = 텐서 사전 (ray, dep, dv, lid, lv, rad, rv, org). keep = 모달리티 집합 ('rgb1' = 카메라 0만). 두 번 풀이 (광선 σ 거리 갱신)."""
    g = (lambda k: T[k]) if idx is None else (lambda k: T[k][idx])
    ray = g("ray").float(); rgbm = ray[..., 6] > 0
    if not ({"rgb", "rgb1"} & keep):
        rgbm = torch.zeros_like(rgbm)
    if "rgb1" in keep and "rgb" not in keep:
        rgbm = rgbm.clone(); rgbm[:, 1] = False
    dv = g("dv") & ("depth" in keep); lv = g("lv") & ("lidar" in keep); rv = g("rv") & ("radar" in keep)
    args = (ray, rgbm, g("dep").float()[..., :3], dv, g("lid").float(), lv, g("rad").float(), rv, g("org").float())
    B, dev = ray.shape[0], ray.device
    c1 = components(*args); mu, S, k = solve(c1, COMPS, B, dev)
    c2 = components(*args, rguess=mu); mu, S, k = solve(c2, COMPS, B, dev)
    return mu, S, k, c2


def smooth(mu, S, seq, t, win=1.1):
    """시퀀스 안 ±win/2 s 창에서 정보 가중 평균 (Σ⁻¹ 가중)."""
    mu, S = np.asarray(mu, float), np.asarray(S, float); L = np.linalg.inv(S); e = (L @ mu[..., None])[..., 0]; out = mu.copy()
    for q in np.unique(seq):
        ii = np.flatnonzero(seq == q); ii = ii[np.argsort(t[ii], kind="stable")]; tt = t[ii]
        cL = np.concatenate([np.zeros((1, 3, 3)), np.cumsum(L[ii], 0)]); ce = np.concatenate([np.zeros((1, 3)), np.cumsum(e[ii], 0)])
        lo = np.searchsorted(tt, tt - win / 2, "left"); hi = np.searchsorted(tt, tt + win / 2, "right")
        out[ii] = np.linalg.solve(cL[hi] - cL[lo], (ce[hi] - ce[lo])[..., None])[..., 0]
    return out


def metrics(mu, S, g, k=None):
    err = mu - g; e3 = np.linalg.norm(err, axis=1); d2 = np.einsum("bi,bij,bj->b", err, np.linalg.inv(S), err)
    return dict(n=int(len(g)), pel_3d=round(float(e3.mean()) * 1000, 1), med=round(float(np.median(e3)) * 1000, 1), pel_h=round(float(np.linalg.norm(err[:, :2], axis=1).mean()) * 1000, 1),
                pel_v=round(float(np.abs(err[:, 2]).mean()) * 1000, 1), bias=[round(float(x) * 1000, 1) for x in err.mean(0)], cover68=round(float((d2 < 3.53).mean()), 3),
                sig_med=round(float(np.median(np.sqrt(np.trace(S, axis1=1, axis2=2)))) * 1000, 1))


if __name__ == "__main__":
    C = os.environ.get("CACHE", "/home/interns/unisense/dev/packs/ms_cache_v2"); OUT = os.environ.get("AUDIT_OUT", "/home/interns/unisense/runs/pel1"); os.makedirs(OUT, exist_ok=True); torch.set_num_threads(32)
    KS = ("ray", "dep", "dv", "lid", "lv", "rad", "rv", "org", "y", "ym", "seq", "t")
    REP = {}
    for ds in ["mri", "mmfi", "octonet", "mmbody", "itop"]:
        Z = np.load(f"{C}/{ds}.npz", allow_pickle=False); D = {k: Z[k] for k in KS}; ok = D["ym"][:, 7] & D["ym"][:, 8]; D = {k: v[ok] for k, v in D.items()}
        T = {k: torch.from_numpy(D[k]) for k in ("ray", "dep", "dv", "lid", "lv", "rad", "rv", "org")}; G = D["y"][:, [7, 8]].mean(1).astype(float)
        av = dict(rgb=(D["ray"][..., 6] > 0).any((1, 2)), depth=D["dv"].any(1), lidar=D["lv"].any(1), radar=D["rv"].any(1))
        present = [m for m in ["rgb", "depth", "lidar", "radar"] if av[m].mean() > 0.05]
        subs = [frozenset(c) for r in range(1, len(present) + 1) for c in itertools.combinations(present, r)] + ([frozenset({"rgb1"}), frozenset({"rgb1", "radar"})] if "rgb" in present else [])
        REP[ds] = {}
        for sub in subs:
            need = [("rgb" if m == "rgb1" else m) for m in sub]; sel = np.flatnonzero(np.all([av[m] for m in need], 0))
            if len(sel) < 50:
                continue
            mus, Ss = [], []
            for i in range(0, len(sel), 20000):
                mu, S, k, _ = physics(T, set(sub), torch.from_numpy(sel[i:i + 20000])); mus.append(mu.numpy()); Ss.append(S.numpy())
            mu, S = np.concatenate(mus).astype(float), np.concatenate(Ss).astype(float); r = metrics(mu, S, G[sel])
            ms = smooth(mu, S, D["seq"][sel], D["t"][sel]); r["pel_3d_sm"] = round(float(np.linalg.norm(ms - G[sel], axis=1).mean()) * 1000, 1)
            name = "+".join(sorted(sub)); REP[ds][name] = r
            print(f"{ds:8s} {name:24s} n {r['n']:6d} 3D {r['pel_3d']:6.1f} (평활 {r['pel_3d_sm']:6.1f}) 수평 {r['pel_h']:6.1f} 수직 {r['pel_v']:6.1f} 치우침 {r['bias']} 68%적중 {r['cover68']:.2f} σ중앙 {r['sig_med']}", flush=True)
        # 성분 하나씩 (같은 프레임, 전부 켠 상태에서 각 성분 단독 + 사전)
        sel = np.flatnonzero(np.all([av[m] for m in present], 0))[:20000]
        _, _, _, cp = physics(T, set(present), torch.from_numpy(sel)); B = len(sel)
        TH = {1: 1.0, 2: 2.28, 3: 3.53}
        for nm in COMPS:
            Lc, ec, v = cp[nm]; v = v.numpy()
            if v.mean() < 0.05:
                continue
            Lc, ec, g = Lc.double()[v], ec.double()[v], torch.from_numpy(G[sel][v]); Pi = torch.linalg.pinv(Lc); m = (Pi @ ec[..., None])[..., 0]; P = Pi @ Lc
            dlt = g - m; pe = (P @ dlt[..., None])[..., 0]; d2 = (dlt[:, None] @ Lc @ dlt[..., None])[:, 0, 0]; rk = int(round(float(torch.diagonal(P, dim1=1, dim2=2).sum(-1).median())))
            r = dict(n=int(v.sum()), rank=rk, err_informed=round(float(torch.linalg.norm(pe, dim=1).mean()) * 1000, 1), bias=[round(float(x) * 1000, 1) for x in (-pe).mean(0)], cover68=round(float((d2 < TH[rk]).double().mean()), 3))
            REP[ds][f"comp:{nm}"] = r
            print(f"{ds:8s}   성분 {nm:11s} n {r['n']:6d} 정보 방향 수 {rk} 그 방향 오차 {r['err_informed']:7.1f} 치우침 {r['bias']} 68%적중 {r['cover68']:.2f}", flush=True)
    json.dump(REP, open(f"{OUT}/phys_audit.json", "w"), indent=1, ensure_ascii=False); print("PHYS_DONE")
