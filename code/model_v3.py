"""모델 v3 (물리 Σ 단위 잔차): v2와 같지만 번역기의 물리 공분산 Σ_phys (dev/packs/phys_v2, phys_features_v2.py)를 쓴다.
- 관절 물리 추정 x̂가 있으면 μ = x̂ + L·(K·tanh(r/K)), L = chol(Σ_phys): 보정은 물리 불확실성 단위(σ)로 표현되고 ±Kσ로 묶인다.
  → "kσ만큼 옮겨라"는 무단위 보정이라 σ가 다른 센서·데이터셋으로 옮겨 간다. 정확한 측정(작은 σ)은 거의 못 움직인다.
- x̂가 없으면 μ = a + L_a·(K·tanh(s/K)) + 몸 상대 위치 (각 축 ±BODY, 두 엉덩이 중점 = 0으로 중심화).
  s는 두 엉덩이 질의에서 나온 골반 이동량 → 골반은 기준점 Σ_a의 ±Kσ 안에 묶인다.
- 질의 입력에 x̂ − a, 유효 표시, chol(Σ_x)·chol(Σ_a)의 log 대각·비대각을 넣어 모델이 측정 신뢰도를 안다.
- 출력 Σ = (L_x L_net)(L_x L_net)ᵀ: 물리 Σ에 대한 상대 배율을 배운다 (x̂ 없는 관절의 L_x = chol(Σ_a + SIG_REL² I)).
- 평가에 물리 단독 골반 오차(root_phys_mm)와 물리 Σ_a 포함률을 함께 기록한다.
GPU 0만 사용. 사용: python model_v3.py --train mmfi,mri,octonet --eval mmfi,mri,octonet --out RUN [--tags spec|none|dataset_id|shuffled]"""
import json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_v0 as m0
import model_v1 as m1

m1.PACKS.update(mri=Path("/home/interns/unisense/dev/packs/mri_v03"), octonet=Path("/home/interns/unisense/dev/packs/octonet_v06"))
PHYS = Path("/home/interns/unisense/dev/packs/phys_v2")
K_SIG = 3.0                       # 잔차 한계 [σ_phys]
BODY = 1.0                        # 골반 기준 관절 상대 위치 한계 [m] (성인 골반–손목 최대 거리 수준, 사전)
SIG_REL = 0.15                    # x̂ 없는 관절의 기준 Σ에 더하는 상대 자세 불확실 [m] (사전)
H_IDX = 14                        # 광선·점 토큰에서 카메라 높이 태그 자리
arg = m1.arg


def build(ds, part, step=1):
    D = m1.build(ds, part, "rays", step)
    A = {k: [] for k in ("a3", "x3", "sa", "sx")}
    for f in m1.pack_files(ds, m1.split_of(ds, part)):
        Q = np.load(PHYS / ds / f.relative_to(m1.PACKS[ds]))
        for k in A:
            A[k].append(Q[k][::step])
    for k in A:
        D[k] = torch.from_numpy(np.concatenate(A[k]))
    assert len(D["a3"]) == len(D["y"])
    return D


def cond_of(ro, po):
    r, p = ro.any(1), po.any(1)
    return torch.where(r & p, 2, torch.where(p, 1, 0))


def pick(a3, x3, sa, sx, cond):
    """조건별 기준점·물리 관절 추정과 Σ. 없으면 both → depth → rgb 대체.
    반환 a (B,3), x (B,13,3: 없는 관절은 a), 유효 (B,13), 기준점 없음 (B,), Σ_a (B,3,3), Σ_x (B,13,3,3: 없는 관절은 Σ_a + SIG_REL² I)."""
    i = torch.arange(len(cond), device=cond.device)
    a, Sa = a3[i, cond], sa[i, cond]
    for c in (1, 0):
        bad = ~torch.isfinite(a).all(-1)
        a = torch.where(bad[:, None], a3[i, c], a); Sa = torch.where(bad[:, None, None], sa[i, c], Sa)
    none = ~torch.isfinite(a).all(-1); a = torch.nan_to_num(a)
    x = x3[i, cond]; xv = torch.isfinite(x).all(-1)
    Sref = Sa[:, None] + SIG_REL ** 2 * torch.eye(3, device=a.device)
    Sx = torch.where(xv[..., None, None], sx[i, cond], Sref)
    return a, torch.where(xv[..., None], x, a[:, None]), xv, none, Sa, Sx


def chol_feat(L):
    """(...,3,3) 하삼각 → log 대각 3 + 비대각 3."""
    return torch.cat([torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), L[..., 1, 0:1], L[..., 2, 0:1], L[..., 2, 1:2]], -1)


class NetV3(m0.Net):
    def __init__(self, d=192, layers=4, heads=6, tag_mode="spec"):
        super().__init__(d, layers, heads)
        self.ray = m0.MLP(m1.RAY_F, d, d); self.pt = m0.MLP(m1.PT_F, d, d); self.phys = m0.MLP(16, d, d)
        self.ds_emb = nn.Embedding(3, d); self.tag_mode = tag_mode
        self.mu[2].bias.data.zero_()                                                    # 잔차 0 = 물리 추정에서 출발
        self.rel = m0.MLP(d, d, 3); self.root = nn.Linear(d, 3)
        nn.init.zeros_(self.root.weight); nn.init.zeros_(self.root.bias)
        self.chol[2].bias.data = torch.tensor([0.5413, 0.5413, 0.5413, 0, 0, 0])        # softplus = 1 → 출력 Σ = 물리 Σ에서 출발

    def forward(self, ray, ray_on, pts, pt_on, a, x, xv, La, Lx, ds=None):
        ray, pts = ray.clone(), pts.clone()
        ray[..., 8:11] -= a[:, None]; pts[..., 0:3] -= a[:, None]                       # 기준점 기준 상대 좌표
        ray[..., H_IDX] = a[:, None, 2]; pts[..., H_IDX] = a[:, None, 2]                # 카메라 높이 대신 기준점 높이 (바닥 위치)
        if self.tag_mode == "none":
            ray = torch.cat([ray[..., :11], torch.zeros_like(ray[..., 11:])], -1); pts = torch.cat([pts[..., :9], torch.zeros_like(pts[..., 9:])], -1)
        B = ray.shape[0]
        ids = torch.arange(17, device=ray.device)
        tr = self.ray(ray) + self.kp_id(ids)[None] + self.typ.weight[1]
        tp = self.pt(pts) + self.typ.weight[2]
        pf = torch.cat([x - a[:, None], xv[..., None].to(x.dtype), chol_feat(Lx), chol_feat(La)[:, None].expand(-1, 13, -1)], -1)
        tq = self.query[None].expand(B, -1, -1) + self.typ.weight[0] + self.phys(pf)
        if self.tag_mode == "dataset_id" and ds is not None:
            e = self.ds_emb(ds)[:, None]; tr, tp, tq = tr + e, tp + e, tq + e
        h = torch.cat([tq, tr, tp], 1)
        pad = torch.cat([torch.zeros(B, 13, dtype=torch.bool, device=ray.device), ~ray_on, ~pt_on], 1)
        h = self.norm(self.enc(h, src_key_padding_mask=pad)[:, :13])
        h = h.float()
        r = K_SIG * torch.tanh(self.mu(h) / K_SIG)
        mu_x = x + (Lx @ r[..., None])[..., 0]                                          # 물리 관절 추정 + Σ 단위 잔차
        s = K_SIG * torch.tanh(self.root(0.5 * (h[:, 7] + h[:, 8])) / K_SIG)
        rel = BODY * torch.tanh(self.rel(h) / BODY); rel = rel - 0.5 * (rel[:, 7:8] + rel[:, 8:9])
        mu_a = (a + (La @ s[..., None])[..., 0])[:, None] + rel                         # 기준점 + Σ_a 단위 골반 이동 + 몸 상대 위치
        mu = torch.where(xv[..., None], mu_x, mu_a); c = self.chol(h)
        L = torch.zeros(B, 13, 3, 3, device=ray.device, dtype=c.dtype)
        L[..., 0, 0] = F.softplus(c[..., 0]) + 1e-3; L[..., 1, 1] = F.softplus(c[..., 1]) + 1e-3; L[..., 2, 2] = F.softplus(c[..., 2]) + 1e-3
        L[..., 1, 0] = c[..., 3]; L[..., 2, 0] = c[..., 4]; L[..., 2, 1] = c[..., 5]
        return mu, Lx @ L


KEYS = ("ray", "rok", "pts", "pok", "y", "ymask", "q", "ds", "a3", "x3", "sa", "sx")


def run_model(model, b, ro, po, phys=False):
    a, x, xv, none, Sa, Sx = pick(b["a3"], b["x3"], b["sa"], b["sx"], cond_of(ro, po))
    La = torch.linalg.cholesky(Sa); Lx = torch.linalg.cholesky(Sx)                  # x̂ 없는 관절의 Σ_x는 이미 Σ_a + SIG_REL² I
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mu, L = model(b["ray"], ro, b["pts"], po, a, x, xv, La, Lx, b["ds"])
    if phys:
        hv = xv[:, 7] & xv[:, 8]
        return mu.float(), L.float(), none, torch.where(hv[:, None], 0.5 * (x[:, 7] + x[:, 8]), a), torch.where(hv[:, None, None], 0.5 * (Sx[:, 7] + Sx[:, 8]), Sa)
    return mu.float(), L.float(), none


def train(model, Ds, steps, bs, dev, dist, log, beta=0.5, lam=1.0, shuffle_tags=False):
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
        if shuffle_tags:
            perm = torch.randperm(len(b["y"]), device=dev)
            b["ray"] = torch.cat([b["ray"][..., :11], b["ray"][perm][..., 11:]], -1); b["pts"] = torch.cat([b["pts"][..., :9], b["pts"][perm][..., 9:]], -1)
        ro, po = m1.masks(b["rok"], b["pok"], True)
        mu, L, none = run_model(model, b, ro, po)
        w = (b["ymask"] & ((b["q"] & 1) == 0) & ~none[:, None]).float()
        nll, logdet = m0.nll_terms(mu, L, b["y"], model.nu() if dist == "t" else None)
        loss = ((torch.exp(beta * logdet / 3).detach() * nll + lam * torch.linalg.norm(b["y"] - mu, dim=-1)) * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if it % 500 == 0 or it == steps - 1:
            log(f"step {it} loss {loss.item():.4f} nu {model.nu().item():.2f} {(it + 1) / (time.time() - t0):.1f} it/s")
    return time.time() - t0


@torch.no_grad()
def forward_all(model, D, dev, cname, bs=512):
    out = {k: [] for k in ("mu", "S", "y", "w", "cam", "none", "rp", "Srp")}
    for s in range(0, len(D["y"]), bs):
        sl = slice(s, min(len(D["y"]), s + bs))
        b = {k: D[k][sl].to(dev) for k in KEYS + ("cam",)}
        b["pts"], b["pok"] = m1.pad_pts(b["pts"]), m1.pad_pts(b["pok"])
        ro = b["rok"] & (cname != "depth_only"); po = b["pok"] & (cname != "rgb_only")
        mu, L, none, rp, Srp = run_model(model, b, ro, po, phys=True)
        out["rp"].append(rp); out["Srp"].append(Srp); out["mu"].append(mu); out["S"].append(L @ L.transpose(-1, -2)); out["y"].append(b["y"]); out["cam"].append(b["cam"])
        out["w"].append(b["ymask"] & ((b["q"] & 1) == 0)); out["none"].append(none)
    return {k: torch.cat(v) for k, v in out.items()}


T3_G68 = 3.5059                    # χ²₃ 68% (물리 Σ는 가우스)


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
        res[cname] = dict(mpjpe_abs_mm=mm(r.norm(dim=-1)) * 1000, root_mm=mm(root.norm(dim=-1)[:, None].expand(-1, 13)) * 1000,
                          rel_mm=mm((r - root[:, None]).norm(dim=-1)) * 1000, ray_axis_mm=mm(rp.abs()) * 1000,
                          perp_mm=mm((r - rp[..., None] * u).norm(dim=-1)) * 1000,
                          cov3d={k: mm((d2 <= v).float()) for k, v in T3.items()}, n=int(A["w"].sum()),
                          no_anchor_frac=float(A["none"].float().mean()))
        gr = 0.5 * (A["y"][:, 7] + A["y"][:, 8]); er = gr - A["rp"]; wr = (A["w"][:, 7] & A["w"][:, 8]).float()
        dr = (er[:, None] @ torch.linalg.solve(A["Srp"], er[..., None]))[:, 0, 0]
        res[cname]["root_phys_mm"] = float((er.norm(dim=-1) * wr).sum() / wr.sum()) * 1000
        res[cname]["root_model_hips_mm"] = float(((gr - 0.5 * (A["mu"][:, 7] + A["mu"][:, 8])).norm(dim=-1) * wr).sum() / wr.sum()) * 1000
        res[cname]["phys_root_cov68"] = float(((dr <= T3_G68) .float() * wr).sum() / wr.sum())
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
    tr_names = arg("--train", "mmfi,mri,octonet").split(","); ev_names = arg("--eval", "mmfi,mri,octonet").split(",")
    steps, bs, dist, tag_mode = arg("--steps", 8000, int), arg("--bs", 510, int), arg("--dist", "t"), arg("--tags", "spec")
    out = Path(arg("--out")); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a"); log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    seed = arg("--seed", 0, int); torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda"); t0 = time.time()
    Dtr = {n: {k: v.to(dev) for k, v in build(n, "train").items()} for n in tr_names}
    Dva = {n: build(n, "val", step=2) for n in ev_names}; Dte = {n: build(n, "test", step=2) for n in ev_names}
    log(f"data built {time.time() - t0:.0f}s: " + ", ".join(f"{n} train {len(Dtr[n]['y'])}" for n in tr_names) + " | "
        + ", ".join(f"{n} test {len(Dte[n]['y'])}" for n in ev_names))
    model = NetV3(tag_mode=tag_mode).to(dev)
    secs = train(model, Dtr, steps, bs, dev, dist, log, shuffle_tags=(tag_mode == "shuffled"))
    res = {"meta": dict(model="v3", few=m1.FEW, few_subjects={d: sorted(m1.split_of(d, "train")) for d in m1.FEW}, train=tr_names, eval=ev_names, steps=steps, bs=bs, dist=dist, tags=tag_mode, seed=seed,
                        packs={k: str(v) for k, v in m1.PACKS.items()}, phys=str(PHYS), k_sig=K_SIG, body=BODY, sig_rel=SIG_REL,
                        train_seconds=secs, nu=float(model.nu()) if dist == "t" else None)}
    for n in ev_names:
        k = recal(model, Dva[n], dev, dist)
        res[n] = evaluate(model, Dte[n], dev, dist, scales=k)
        for c, r in res[n].items():
            log(f"{n} {c}: abs {r['mpjpe_abs_mm']:.1f} root {r['root_mm']:.1f} rel {r['rel_mm']:.1f} ray {r['ray_axis_mm']:.1f} "
                f"perp {r['perp_mm']:.1f} phys-root {r['root_phys_mm']:.1f} (cov68 {r['phys_root_cov68']:.2f}) recal {[round(v, 2) for v in r.get('cov3d_recal', {}).values()]} no-anchor {r['no_anchor_frac']:.3f}")
    (out / "metrics.json").write_text(json.dumps(res, indent=1)); torch.save(model.state_dict(), out / "model.pt")
    log(f"train time {secs / 60:.1f} min")


if __name__ == "__main__":
    main()
