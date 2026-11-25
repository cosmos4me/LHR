"""모델 v4 (인자 그래프): 학습된 사전 × 물리 측정 인자를 미분 가능한 가우스–뉴턴으로 푼다. 학습하는 것은 사전뿐이다.
- 사전 인자: model_v3 망(기준점 상대 토큰 + 물리 x̂·Σ 입력)의 (μ_p, Σ_p). 사전 망은 학습 때 본 모달리티의 토큰만 쓴다
  (처음 보는 모달리티는 아래 물리 인자로만 들어간다 → 학습된 해석이 분포 밖에서 쓰이지 않는다).
- 측정 인자 (사양·GT 없는 상수, 강건 Huber, IRLS):
  · RGB 광선 (시점마다; mRI는 두 시점): ‖(I − ddᵀ)(X − o)‖ / σ⊥, σ⊥² = (Z·σ_px/f)² + SIG_LAB²  (σ_px: notes/sigma_px_gtfree.json)
  · 깊이 표면: 관절은 가장 가까운 사람 표면 점에서 카메라 반대쪽으로 분절 반지름 r_j만큼 뒤 (보이는 표면 뒤의 관절)
  · 바닥: 낮은 발목 높이 = ANKLE_H (접촉, 강건), 두 발목 모두 바닥 관통 금지
  · 기준점(깊이가 있을 때): 두 엉덩이 중점 ~ N(a, Σ_a)
  · 뼈 길이: 시퀀스 상수 ℓ (번역기 x̂에서 GT 없이: mRI 삼각측량, MM-Fi 희소 깊이 — 그 조건에서만) 없으면 학습된 인구 평균 ℓ̄
- 해: X0 = μ_p에서 N_IT번 가우스–뉴턴 (39×39 정보 행렬). 출력 Σ = 마지막 정보 행렬 역의 관절 블록 (라플라스).
- 라벨 정의 오프셋: 학습 데이터셋(= 라벨 체계)별 관절 오프셋 b (몸 좌표, 각 축 ±B_MAX). 처음 보는 데이터셋은 0.
- 손실: 최종 (μ, Σ)의 β-NLL + MPJPE + AUX·MPJPE(사전 평균).
GPU 0만 사용. 사용: python model_v4.py --train mmfi,mri,octonet --eval mmfi,mri,octonet --out RUN [--iters 4]"""
import json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_common as cc
import model_v0 as m0
import model_v1 as m1
import model_v3 as m3
import physics_baseline as pb

SPX = json.loads(Path("/home/interns/unisense/notes/sigma_px_gtfree.json").read_text())
PX_MIN = 2.0                      # 2D 잡음 하한 [px] (MM-Fi 평활 키포인트, 사전)
SIG_LAB = 0.03                    # 검출기 키포인트 정의·의미 오차 [m] (사전)
SIG_SRF = 0.03                    # 표면→관절 오프셋 불확실 [m] (사전)
GATE_SRF = 0.30                   # 표면 점 연관 문턱 [m]
SIG_FLOOR, SIG_PEN = 0.04, 0.01   # 발목 접촉·관통 σ [m]
SIG_BONE, SIG_BONE_POP = 0.02, 0.04
B_MAX = 0.10                      # 라벨 정의 오프셋 한계 [m]
AUX = 0.3
N_IT = m1.arg("--iters", 4, int)
HUB = 2.5                         # Huber 문턱 [σ]
# 공통 13관절: 머리, 어깨 L/R, 팔꿈치 L/R, 손목 L/R, 엉덩이 L/R, 무릎 L/R, 발목 L/R
BONES = [(1, 3), (3, 5), (2, 4), (4, 6), (7, 9), (9, 11), (8, 10), (10, 12), (1, 2), (7, 8), (1, 7), (2, 8)]
BI, BJ = [b[0] for b in BONES], [b[1] for b in BONES]
ELL0 = [0.30, 0.25, 0.30, 0.25, 0.42, 0.40, 0.42, 0.40, 0.34, 0.22, 0.50, 0.50]   # 인구 평균 시작값 [m] (학습됨)
R_J = torch.tensor([0.08, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.09, 0.09, 0.05, 0.05, 0.04, 0.04])  # 분절 반지름 [m] (physics_baseline.OFF)
H13 = [[3, 4], 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


# ---------------- 데이터: v3 + 광선 인자 (시점 2개까지), 깊이 카메라 위치, 시퀀스 뼈 길이 ----------------
def rays13(kp, score, K, R, t, spx, ok_min):
    kp13 = np.stack([kp[:, m].mean(1) if isinstance(m, list) else kp[:, m] for m in H13], 1)
    s13 = np.stack([np.asarray(spx)[m].mean() if isinstance(m, list) else np.asarray(spx)[m] for m in H13])
    if score is None:
        ok = np.isfinite(kp13).all(-1)
    else:
        sc = np.stack([score[:, m].min(1) if isinstance(m, list) else score[:, m] for m in H13], 1); ok = sc >= ok_min
    o, d = pb.rays(kp13, K, R, t)
    f = 0.5 * (pb.kvec(K)[0] + pb.kvec(K)[1])
    out = np.concatenate([np.broadcast_to(o, d.shape), d, np.broadcast_to((s13 / f)[None, :, None], d.shape[:-1] + (1,))], -1)
    return np.nan_to_num(out).astype(np.float32), ok & np.isfinite(out).all(-1)


def seq_bones(x):
    """(T,13,3) 물리 관절 추정 → 시퀀스 뼈 길이 중앙값 (12,), 유효 프레임 20개 미만이면 NaN."""
    L = np.linalg.norm(x[:, BI] - x[:, BJ], axis=-1)
    n = np.isfinite(L).sum(0)
    return np.where(n >= 20, np.nanmedian(L, 0), np.nan)


def build(ds, part, step=1):
    D = m3.build(ds, part, step)
    spec = json.loads(pb.SPECS[ds].read_text())
    RV, RO, DC, BL = [], [], [], []
    for f in m1.pack_files(ds, m1.split_of(ds, part)):
        P = np.load(f); T = len(P["labels"]); Q = np.load(m3.PHYS / ds / f.relative_to(m1.PACKS[ds]))
        rv = np.zeros((T, 2, 13, 7), np.float32); ro = np.zeros((T, 2, 13), bool); bl = np.full((T, 3, 12), np.nan, np.float32)
        if ds == "mmfi":
            rv[:, 0], ro[:, 0] = rays13(P["kp_px"], None, spec["rgb"]["K"], P["R_c2r"], P["t_c2r"], np.maximum(SPX["mmfi"], PX_MIN), 0)
            bl[:, 2] = seq_bones(Q["x3"][:, 2])
        elif ds == "mri":
            R12, T12 = np.asarray(spec["rgb"]["R_cam1_to_cam2"]), np.asarray(spec["rgb"]["t_cam1_to_cam2"])
            Rc, tc = np.asarray(P["R_c2r"]), np.asarray(P["t_c2r"], float)
            rv[:, 0], ro[:, 0] = rays13(P["kp_px"], P["kp_score"], spec["rgb"]["K"], Rc, tc, SPX["mri"], pb.MIN_SCORE)
            rv[:, 1], ro[:, 1] = rays13(P["kp_px_cam2"], P["kp_score_cam2"], spec["rgb"]["K_cam2"], Rc @ R12.T, Rc @ (-R12.T @ T12) + tc, SPX["mri"], pb.MIN_SCORE)
            bl[:, 0] = seq_bones(Q["x3"][:, 0])
        sl = slice(0, None, step)
        RV.append(rv[sl]); RO.append(ro[sl]); BL.append(bl[sl]); DC.append(np.broadcast_to(np.asarray(P["t_d2r"], np.float32), (len(rv[sl]), 3)))
    D["rv"] = torch.from_numpy(np.concatenate(RV)); D["rvok"] = torch.from_numpy(np.concatenate(RO))
    D["bl3"] = torch.from_numpy(np.concatenate(BL)); D["dcam"] = torch.from_numpy(np.ascontiguousarray(np.concatenate(DC)))
    assert len(D["rv"]) == len(D["y"])
    return D


# ---------------- 가우스–뉴턴 ----------------
def huber(m, k=HUB):
    return torch.where(m <= k, torch.ones_like(m), k / m.clamp(min=1e-9)).detach()


def acc(H5, i, j, val):
    """H5[:, i[n], j[n]] += val[:, n] (중복 인덱스 누적)."""
    B = H5.shape[0]; bidx = torch.arange(B, device=H5.device)[:, None].expand(-1, len(i))
    H5.index_put_((bidx, i[None].expand(B, -1), j[None].expand(B, -1)), val, accumulate=True)


def add_block(H5, g5, j, Hb, gb):
    """관절 j (정수 또는 (B,) 텐서) 블록에 더한다. Hb (B,3,3), gb (B,3)."""
    i = torch.arange(H5.shape[0], device=H5.device)
    H5[i, j, j] += Hb; g5[i, j] += gb


def solve(mu_p, Lp, F, iters=N_IT):
    """F: 인자 입력 dict. 반환 X (B,13,3), Σ (B,13,3,3)."""
    B = mu_p.shape[0]; dev = mu_p.device; I3 = torch.eye(3, device=dev)
    Lam_p = torch.cholesky_inverse(Lp)
    X = mu_p
    rj = R_J.to(dev)
    for it in range(iters + 1):
        H5 = torch.zeros(B, 13, 13, 3, 3, device=dev); g5 = torch.zeros(B, 13, 3, device=dev)
        jj = torch.arange(13, device=dev)
        # 사전
        e = X - mu_p; m = torch.sqrt((e[..., None, :] @ Lam_p @ e[..., None]).squeeze(-1).squeeze(-1).clamp(min=1e-12))
        w = huber(m, 3.0)[..., None, None]
        H5[:, jj, jj] += w * Lam_p; g5 += (w * Lam_p @ e[..., None])[..., 0]
        # 광선
        for v in range(2):
            ok = F["rvok"][:, v] & F["ray_on"][:, None]
            if not bool(ok.any()):
                continue
            o, d, sa = F["rv"][:, v, :, 0:3], F["rv"][:, v, :, 3:6], F["rv"][:, v, :, 6]
            q = X - o; Z = (q * d).sum(-1).clamp(min=0.3)
            s2 = (Z * sa) ** 2 + SIG_LAB ** 2
            Pm = I3 - d[..., :, None] * d[..., None, :]
            r = (Pm @ q[..., None])[..., 0]; m = r.norm(dim=-1) / s2.sqrt()
            w = (ok.float() * huber(m))[..., None, None] / s2[..., None, None]
            H5[:, jj, jj] += w * Pm; g5 += (w * Pm @ q[..., None])[..., 0]
        # 깊이 표면
        if bool(F["pt_on"].any()):
            p, pok = F["p"], F["pok"]
            dist = torch.cdist(X.detach(), p.detach()); dist = torch.where(pok[:, None, :], dist, torch.full_like(dist, 1e3))
            dmin, k = dist.min(-1)
            pn = torch.gather(p, 1, k[..., None].expand(-1, -1, 3)); san = torch.gather(F["s_ax"], 1, k)
            u = pn - F["dcam"][:, None]; u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            tgt = pn + rj[None, :, None] * u
            e = X - tgt; s2 = san ** 2 + SIG_SRF ** 2; m = e.norm(dim=-1) / s2.sqrt()
            ok = (dmin < GATE_SRF) & F["pt_on"][:, None]
            w = (ok.float() * huber(m) / s2)[..., None, None]
            H5[:, jj, jj] += w * I3; g5 += (w[..., 0] * e)
            # 기준점 (두 엉덩이 중점)
            e = 0.5 * (X[:, 7] + X[:, 8]) - F["a"]; La = torch.cholesky_inverse(F["La"])
            m = torch.sqrt((e[:, None] @ La @ e[..., None])[:, 0, 0].clamp(min=1e-12))
            w = (F["pt_on"].float() * huber(m, 3.0))[:, None, None]
            for i_ in (7, 8):
                for j_ in (7, 8):
                    H5[:, i_, j_] += 0.25 * w * La
                g5[:, i_] += 0.5 * (w * La @ e[..., None])[..., 0]
        # 바닥 (z = 0, 위쪽 +z)
        z = X[:, 11:13, 2]; low = z.argmin(1)
        ez = z.gather(1, low[:, None])[:, 0] - pb.ANKLE_H
        w = huber(ez.abs() / SIG_FLOOR, 1.5) / SIG_FLOOR ** 2
        Hb = torch.zeros(B, 3, 3, device=dev); Hb[:, 2, 2] = w; gb = torch.zeros(B, 3, device=dev); gb[:, 2] = w * ez
        add_block(H5, g5, 11 + low, Hb, gb)
        for a_ in (11, 12):
            pen = X[:, a_, 2] - (pb.ANKLE_H - 0.03); act = (pen < 0).float() / SIG_PEN ** 2
            Hb = torch.zeros(B, 3, 3, device=dev); Hb[:, 2, 2] = act; gb = torch.zeros(B, 3, device=dev); gb[:, 2] = act * pen
            add_block(H5, g5, a_, Hb, gb)
        # 뼈 길이
        diff = X[:, BI] - X[:, BJ]; Lb = diff.norm(dim=-1).clamp(min=1e-4); u = diff / Lb[..., None]
        ell = torch.where(torch.isfinite(F["bl"]), F["bl"], F["ell_pop"][None].expand(B, -1))
        sb = torch.where(torch.isfinite(F["bl"]), torch.full_like(Lb, SIG_BONE), torch.full_like(Lb, SIG_BONE_POP))
        e = (Lb - ell) / sb; w = huber(e.abs()) / sb ** 2
        uu = w[..., None, None] * u[..., :, None] * u[..., None, :]; gu = (w * e * sb)[..., None] * u
        bi, bj = torch.tensor(BI, device=dev), torch.tensor(BJ, device=dev)
        acc(H5, bi, bi, uu); acc(H5, bj, bj, uu); acc(H5, bi, bj, -uu); acc(H5, bj, bi, -uu)
        g5.index_add_(1, bi, gu); g5.index_add_(1, bj, -gu)
        H = H5.permute(0, 1, 3, 2, 4).reshape(B, 39, 39) + 1e-6 * torch.eye(39, device=dev)
        if it == iters:
            break
        X = X - torch.linalg.solve(H, g5.reshape(B, 39, 1)).reshape(B, 13, 3)
    S = torch.diagonal(torch.linalg.inv(H).reshape(B, 13, 3, 13, 3), dim1=1, dim2=3).permute(0, 3, 1, 2)   # 관절 대각 블록 (B,13,3,3)
    return X, 0.5 * (S + S.transpose(-1, -2))


# ---------------- 모델 ----------------
class NetV4(nn.Module):
    def __init__(self, seen_rgb, seen_depth, tag_mode="spec"):
        super().__init__()
        self.prior = m3.NetV3(tag_mode=tag_mode)
        self.b = nn.Parameter(torch.zeros(3, 13, 3))                           # 라벨 체계(데이터셋)별 정의 오프셋 (몸 좌표)
        self.ell_pop = nn.Parameter(torch.tensor(ELL0))
        self.register_buffer("seen", torch.tensor([seen_rgb, seen_depth]))
        self.register_buffer("fam", torch.zeros(3, dtype=torch.bool))

    def nu(self):
        return self.prior.nu()

    def offset(self, X, ds):
        """몸 좌표 (가로 = 엉덩이 좌→우의 수평 성분, 위 = +z) 오프셋 → 방 좌표. 학습 안 한 라벨 체계는 0."""
        ex = X[:, 7] - X[:, 8]; ex = torch.stack([ex[:, 0], ex[:, 1], torch.zeros_like(ex[:, 0])], -1)
        ex = ex / ex.norm(dim=-1, keepdim=True).clamp(min=1e-6); ez = torch.zeros_like(ex); ez[:, 2] = 1
        ey = torch.cross(ez, ex, dim=-1); R = torch.stack([ex, ey, ez], -1)            # (B,3,3) 열 = 몸 축
        known = ds < self.fam.numel(); di = torch.where(known, ds, torch.zeros_like(ds))          # 모델이 모르는 데이터셋 번호(예: mmbody 3) = 처음 보는 체계
        b = B_MAX * torch.tanh(self.b[di] / B_MAX) * (self.fam[di] & known)[:, None, None].float()
        return (R[:, None] @ b[..., None])[..., 0]


KEYS = m3.KEYS + ("rv", "rvok", "bl3", "dcam")


def run_model(model, b, ro, po, phys=False):
    cond = m3.cond_of(ro, po)
    a, x, xv, none, Sa, Sx = m3.pick(b["a3"], b["x3"], b["sa"], b["sx"], cond)
    La = torch.linalg.cholesky(Sa); Lx = torch.linalg.cholesky(Sx)
    rp, pp = ro & model.seen[0], po & model.seen[1]                                  # 사전 망은 본 모달리티의 토큰만
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mu_p, Lp = model.prior(b["ray"], rp, b["pts"], pp, a, x, xv, La, Lx, b["ds"])
    mu_p, Lp = mu_p.float(), Lp.float()
    i = torch.arange(len(cond), device=cond.device)
    F = dict(rv=b["rv"], rvok=b["rvok"], ray_on=ro.any(1), pt_on=po.any(1), p=b["pts"][..., 0:3], pok=po,
             s_ax=torch.exp(b["pts"][..., 6]), dcam=b["dcam"], a=a, La=La, bl=b["bl3"][i, cond], ell_pop=model.ell_pop)
    X, S = solve(mu_p, Lp, F)
    off = model.offset(X, b["ds"])
    L = torch.linalg.cholesky(S + 1e-8 * torch.eye(3, device=S.device))
    out = (X + off, L, none, mu_p + off)
    if phys:
        hv = xv[:, 7] & xv[:, 8]
        out = out + (torch.where(hv[:, None], 0.5 * (x[:, 7] + x[:, 8]), a), torch.where(hv[:, None, None], 0.5 * (Sx[:, 7] + Sx[:, 8]), Sa))
    return out


ACCUM = m1.arg("--accum", 1, int)


def train(model, Ds, steps, bs, dev, dist, log, beta=0.5, lam=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    names = list(Ds); t0 = time.time()
    for it in range(steps):
        parts = []
        for nm in names:
            D = Ds[nm]; idx = torch.randint(0, len(D["y"]), (bs // len(names),), device=D["y"].device)
            pb_ = {kk: D[kk][idx] for kk in KEYS}
            pb_["pts"], pb_["pok"] = m1.pad_pts(pb_["pts"]), m1.pad_pts(pb_["pok"])
            parts.append(pb_)
        b = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        ro, po = m1.masks(b["rok"], b["pok"], True)
        none = m3.pick(b["a3"], b["x3"], b["sa"], b["sx"], m3.cond_of(ro, po))[3]
        W = (b["ymask"] & ((b["q"] & 1) == 0) & ~none[:, None]).float().sum().clamp(min=1)   # 배치 전체 정규화 → 조각으로 나눠도 기울기 같음
        opt.zero_grad(set_to_none=True); loss = 0.0
        for sl in torch.arange(len(b["y"]), device=b["y"].device).chunk(ACCUM):            # --accum: 메모리만 줄인다 (기본 1 = 예전과 같음)
            bc = {k: v[sl] for k, v in b.items()}
            mu, L, none_c, mu_p = run_model(model, bc, ro[sl], po[sl])
            w = (bc["ymask"] & ((bc["q"] & 1) == 0) & ~none_c[:, None]).float()
            nll, logdet = m0.nll_terms(mu, L, bc["y"], model.nu() if dist == "t" else None)
            lc = ((torch.exp(beta * logdet / 3).detach() * nll + lam * torch.linalg.norm(bc["y"] - mu, dim=-1)
                   + AUX * torch.linalg.norm(bc["y"] - mu_p, dim=-1)) * w).sum() / W
            lc.backward(); loss += lc.item()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if it % 500 == 0 or it == steps - 1:
            log(f"step {it} loss {loss:.4f} nu {model.nu().item():.2f} {(it + 1) / (time.time() - t0):.1f} it/s")
    return time.time() - t0


@torch.no_grad()
def forward_all(model, D, dev, cname, bs=512):
    out = {k: [] for k in ("mu", "S", "y", "w", "cam", "none", "rp", "Srp", "mup")}
    for s in range(0, len(D["y"]), bs):
        sl = slice(s, min(len(D["y"]), s + bs))
        b = {k: D[k][sl].to(dev) for k in KEYS + ("cam",)}
        b["pts"], b["pok"] = m1.pad_pts(b["pts"]), m1.pad_pts(b["pok"])
        ro = b["rok"] & (cname != "depth_only"); po = b["pok"] & (cname != "rgb_only")
        mu, L, none, mup, rp, Srp = run_model(model, b, ro, po, phys=True)
        for k, v in (("mu", mu), ("S", L @ L.transpose(-1, -2)), ("y", b["y"]), ("cam", b["cam"]), ("none", none), ("rp", rp), ("Srp", Srp), ("mup", mup)):
            out[k].append(v)
        out["w"].append(b["ymask"] & ((b["q"] & 1) == 0))
    return {k: torch.cat(v) for k, v in out.items()}


BPAIRS = [(0, 2), (1, 3), (4, 6), (5, 7), (10, 11)]


@torch.no_grad()
def evaluate(model, D, dev, dist, scales=None):
    model.eval(); res = {}
    nu = float(model.nu()) if dist == "t" else None
    T3, T1 = m0.thresholds(dist, nu)
    has_r, has_p = bool(D["rok"].any()), bool(D["pok"].any())
    for cname in [c for c, ok in (("rgb_only", has_r), ("depth_only", has_p), ("both", has_r and has_p)) if ok]:
        A = forward_all(model, D, dev, cname)
        r = A["y"] - A["mu"]; S = A["S"]
        u = A["y"] - A["cam"][:, None]; u = u / u.norm(dim=-1, keepdim=True)
        rp = (r * u).sum(-1); root = 0.5 * (r[:, 7] + r[:, 8])
        d2 = (r.unsqueeze(-2) @ torch.linalg.solve(S, r.unsqueeze(-1))).squeeze(-1).squeeze(-1)
        w = A["w"].float(); mm = lambda x: float((x * w).sum() / w.sum())
        bl = lambda X: torch.stack([(X[:, i] - X[:, j]).norm(dim=-1) for i, j in BONES], -1)
        lp, lg = bl(A["mu"]), bl(A["y"]); wb = torch.stack([A["w"][:, i] & A["w"][:, j] for i, j in BONES], -1).float()
        res[cname] = dict(mpjpe_abs_mm=mm(r.norm(dim=-1)) * 1000, root_mm=mm(root.norm(dim=-1)[:, None].expand(-1, 13)) * 1000,
                          rel_mm=mm((r - root[:, None]).norm(dim=-1)) * 1000, ray_axis_mm=mm(rp.abs()) * 1000,
                          perp_mm=mm((r - rp[..., None] * u).norm(dim=-1)) * 1000, prior_abs_mm=mm((A["y"] - A["mup"]).norm(dim=-1)) * 1000,
                          bone_err_mm=float(((lp - lg).abs() * wb).sum() / wb.sum()) * 1000,
                          asym_mm=float(torch.stack([(lp[:, a] - lp[:, b]).abs() for a, b in BPAIRS], -1).mean()) * 1000,
                          cov3d={k: mm((d2 <= v).float()) for k, v in T3.items()}, n=int(A["w"].sum()),
                          no_anchor_frac=float(A["none"].float().mean()))
        gr = 0.5 * (A["y"][:, 7] + A["y"][:, 8]); er = gr - A["rp"]; wr = (A["w"][:, 7] & A["w"][:, 8]).float()
        res[cname]["root_phys_mm"] = float((er.norm(dim=-1) * wr).sum() / wr.sum()) * 1000
        res[cname]["root_model_hips_mm"] = float(((gr - 0.5 * (A["mu"][:, 7] + A["mu"][:, 8])).norm(dim=-1) * wr).sum() / wr.sum()) * 1000
        if scales and cname in scales:
            k = scales[cname]; res[cname]["recal_k"] = k
            res[cname]["cov3d_recal"] = {kk: mm((d2 / k <= v).float()) for kk, v in T3.items()}
    model.train()
    return res


@torch.no_grad()
def recal(model, D, dev, dist):
    from scipy import stats
    nu = float(model.nu()) if dist == "t" else None
    ideal = 3 * stats.f.ppf(0.5, 3, nu) if dist == "t" else stats.chi2.ppf(0.5, 3)
    model.eval(); out = {}
    has_r, has_p = bool(D["rok"].any()), bool(D["pok"].any())
    for cname in [c for c, ok in (("rgb_only", has_r), ("depth_only", has_p), ("both", has_r and has_p)) if ok]:
        A = forward_all(model, D, dev, cname); r = A["y"] - A["mu"]
        d2 = (r.unsqueeze(-2) @ torch.linalg.solve(A["S"], r.unsqueeze(-1))).squeeze(-1).squeeze(-1)
        out[cname] = float(d2[A["w"]].median() / ideal)
    model.train()
    return out


def main():
    arg = m1.arg
    tr_names = arg("--train", "mmfi,mri,octonet").split(","); ev_names = arg("--eval", "mmfi,mri,octonet").split(",")
    steps, bs, dist, tag_mode = arg("--steps", 8000, int), arg("--bs", 510, int), arg("--dist", "t"), arg("--tags", "spec")
    out = Path(arg("--out")); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a"); log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    seed = arg("--seed", 0, int); torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda"); t0 = time.time()
    Dtr = {n: {k: v.to(dev) for k, v in build(n, "train").items()} for n in tr_names}
    Dva = {n: build(n, "val", step=2) for n in ev_names}; Dte = {n: build(n, "test", step=2) for n in ev_names}
    log(f"data built {time.time() - t0:.0f}s: " + ", ".join(f"{n} train {len(Dtr[n]['y'])}" for n in tr_names))
    seen_rgb = any(bool(Dtr[n]["rok"].any()) for n in tr_names); seen_depth = any(bool(Dtr[n]["pok"].any()) for n in tr_names)
    model = NetV4(seen_rgb, seen_depth, tag_mode).to(dev)
    for n in tr_names:
        model.fam[m1.DSID[n]] = True
    secs = train(model, Dtr, steps, bs, dev, dist, log)
    res = {"meta": dict(model="v4", train=tr_names, eval=ev_names, steps=steps, bs=bs, dist=dist, tags=tag_mode, seed=seed, iters=N_IT,
                        seen_rgb=seen_rgb, seen_depth=seen_depth, packs={k: str(v) for k, v in m1.PACKS.items()}, phys=str(m3.PHYS),
                        ell_pop=[round(float(v), 3) for v in model.ell_pop], train_seconds=secs, nu=float(model.nu()) if dist == "t" else None)}
    for n in ev_names:
        k = recal(model, Dva[n], dev, dist)
        res[n] = evaluate(model, Dte[n], dev, dist, scales=k)
        for c, r in res[n].items():
            log(f"{n} {c}: abs {r['mpjpe_abs_mm']:.1f} root {r['root_mm']:.1f} rel {r['rel_mm']:.1f} prior {r['prior_abs_mm']:.1f} "
                f"bone {r['bone_err_mm']:.1f} asym {r['asym_mm']:.1f} phys-root {r['root_phys_mm']:.1f} recal {[round(v, 2) for v in r.get('cov3d_recal', {}).values()]}")
    (out / "metrics.json").write_text(json.dumps(res, indent=1)); torch.save(model.state_dict(), out / "model.pt")
    log(f"train time {secs / 60:.1f} min")


if __name__ == "__main__":
    main()
