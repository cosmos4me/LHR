"""E1 파일럿 모델 v0: 토큰(광선 17 + 깊이 점 256) → 공유 트랜스포머 → 공통 13관절 μ(방 좌표, m) + 전체 공분산 Σ.
- 단일 프레임(T = 1) 파일럿. 모달리티 가리기(RGB만 / 깊이만 / 둘 다)로 한 모델을 학습.
- 손실: β-NLL(β = 0.5, 전체 공분산) + MPJPE 항. 정답 정지 관절-프레임(label_quality bit 1)은 손실에서 뺀다.
- 평가(시험 피험자): 조건별 절대 MPJPE, 루트(엉덩이 중점) 오차, 루트 기준 MPJPE, 광선 방향·수직 축별 오차,
  3D χ²₃ 포함률(50/68.27/90/95%: d² ≤ 2.366/3.527/6.251/7.815), 광선 축 1D 포함률, 이방성 σ∥/σ⊥.
출력 분포: --dist gauss(기본) | t (다변량 Student-t, ν 학습; 포함률 기준은 d²/3 ~ F(3, ν), 광선 축은 t_ν).
학습 조건: --train-cond mixed(기본, 공유 모델) | rgb | depth | both (조건 전용 비교 모델).
재보정: 검증 피험자에서 조건별 분산 배율 k를 맞춰 시험 포함률을 원시·재보정 둘 다 보고.
GPU 0만 사용. 사용: python model_v0.py --packs DIR --split E1 --steps 8000 --out RUNDIR [--dist t] [--train-cond rgb] [--seed 1]"""
import json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_common as cc

CHI3 = {"50": 2.366, "68": 3.527, "90": 6.251, "95": 7.815}
Z1 = {"50": 0.674, "68": 1.0, "90": 1.645, "95": 1.960}


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


# ---------------- 데이터: 팩 → 토큰 (numpy, 미리 계산) ----------------
def tokenize_pack(P, spec):
    sm = spec["rgb"]["sigma_model"]
    kp = P["kp_px"].astype(float)
    sig = cc.keypoint_sigma_px(kp, sm["g_per_joint"], sm["beta_s"], spec["fps"])
    o, d, s_rad, s_tan, e_rad = cc.ray_tokens(kp, spec["rgb"]["K"], P["R_c2r"], P["t_c2r"], sig)
    T = len(kp)
    ray = np.concatenate([d, e_rad, np.log(s_rad)[..., None], np.log(s_tan)[..., None],
                          np.broadcast_to(o, (T, 17, 3))], -1)                                  # (T,17,11)
    uvz = P["pts_uvz"].astype(float); dep = spec["depth"]
    fx, fy, cx, cy = dep["K"]
    a = np.stack([(uvz[..., 0] - cx) / fx, (uvz[..., 1] - cy) / fy, np.ones_like(uvz[..., 2])], -1)
    an = np.linalg.norm(a, axis=-1, keepdims=True)
    R = np.asarray(P["R_d2r"]); p = (a * uvz[..., 2:3]) @ R.T + P["t_d2r"]
    ax = (a / an) @ R.T                                                                         # 깊이 잡음 주축(광선) 방향
    s_ax = cc.axial_sigma(uvz[..., 2], dep) * an[..., 0]
    s_lat = uvz[..., 2] * dep["sigma_pix_px"] / (0.5 * (fx + fy))
    pts = np.concatenate([p, ax, np.log(s_ax)[..., None], np.log(s_lat)[..., None], P["pts_edge"][..., None]], -1)  # (T,256,9)
    return ray.astype(np.float32), pts.astype(np.float32), P["pts_valid"]


def build(packs, subjects, step=1):
    spec = None; R, Pt, V, Y, M, Q, G, O = [], [], [], [], [], [], [], []
    for f in sorted(Path(packs).glob("E*/*.npz")):
        if f.name.split("_")[0] not in subjects:
            continue
        P = np.load(f)
        if spec is None:
            spec = json.loads(Path(json.loads(P["meta"].item())["spec"]).read_text())
        ray, pts, val = tokenize_pack(P, spec)
        sl = slice(0, None, step)
        R.append(ray[sl]); Pt.append(pts[sl]); V.append(val[sl]); Y.append(P["labels"][sl]); M.append(P["label_mask"][sl])
        Q.append(P["label_quality"][sl]); O.append(np.broadcast_to(P["t_c2r"], (len(ray[sl]), 3)))
        G.append(np.full(len(ray[sl]), int(f.name[1:3])))
    cat = lambda x: torch.from_numpy(np.ascontiguousarray(np.concatenate(x)))
    return dict(ray=cat(R), pts=cat(Pt), pval=cat(V), y=cat(Y).float(), ymask=cat(M), q=cat(Q), cam=cat(O).float(),
                subj=cat(G)), spec


# ---------------- 모델 ----------------
class MLP(nn.Sequential):
    def __init__(self, i, h, o):
        super().__init__(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class Net(nn.Module):
    def __init__(self, d=192, layers=4, heads=6):
        super().__init__()
        self.ray = MLP(11, d, d); self.kp_id = nn.Embedding(17, d); self.pt = MLP(9, d, d)
        self.typ = nn.Embedding(3, d); self.query = nn.Parameter(torch.randn(13, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers); self.norm = nn.LayerNorm(d)
        self.mu = MLP(d, d, 3); self.chol = MLP(d, d, 6)
        nn.init.zeros_(self.mu[2].weight); self.mu[2].bias.data = torch.tensor([3.0, 0.0, 0.9])   # 방 좌표 사람 위치 근처에서 시작
        nn.init.zeros_(self.chol[2].weight); self.chol[2].bias.data = torch.tensor([-2.0, -2.0, -2.0, 0, 0, 0])
        self.log_nu_m2 = nn.Parameter(torch.tensor(float(np.log(2.0))))                          # ν = 2 + exp(·), 시작 ν = 4 (t 출력일 때만 사용)

    def nu(self):
        return 2.0 + torch.exp(self.log_nu_m2)

    def forward(self, ray, ray_on, pts, pt_on):
        B = ray.shape[0]
        ids = torch.arange(17, device=ray.device)
        tr = self.ray(ray) + self.kp_id(ids)[None] + self.typ.weight[1]
        tp = self.pt(pts) + self.typ.weight[2]
        tq = self.query[None].expand(B, -1, -1) + self.typ.weight[0]
        x = torch.cat([tq, tr, tp], 1)
        pad = torch.cat([torch.zeros(B, 13, dtype=torch.bool, device=ray.device), ~ray_on, ~pt_on], 1)
        h = self.norm(self.enc(x, src_key_padding_mask=pad)[:, :13])
        mu = self.mu(h)
        c = self.chol(h)
        L = torch.zeros(B, 13, 3, 3, device=ray.device, dtype=c.dtype)
        L[..., 0, 0] = F.softplus(c[..., 0]) + 1e-3; L[..., 1, 1] = F.softplus(c[..., 1]) + 1e-3; L[..., 2, 2] = F.softplus(c[..., 2]) + 1e-3
        L[..., 1, 0] = c[..., 3]; L[..., 2, 0] = c[..., 4]; L[..., 2, 1] = c[..., 5]
        return mu, L


def nll_terms(mu, L, y, nu=None):
    """가우스(nu=None) 또는 다변량 Student-t(척도 행렬 LLᵀ, 자유도 nu)의 음의 로그우도 (상수항은 t에서만 nu 의존분 포함)."""
    r = (y - mu).unsqueeze(-1)
    z = torch.linalg.solve_triangular(L, r, upper=False).squeeze(-1)
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    d2 = (z ** 2).sum(-1)
    if nu is None:
        return 0.5 * d2 + 0.5 * logdet, logdet
    c = torch.lgamma(nu / 2) - torch.lgamma((nu + 3) / 2) + 1.5 * torch.log(nu * np.pi)
    return c + 0.5 * logdet + 0.5 * (nu + 3) * torch.log1p(d2 / nu), logdet


COND = {"rgb": 0, "depth": 1, "both": 2}


def modality_masks(B, pval, dev, train, train_cond="mixed"):
    """학습 조건: mixed = RGB만 25% / 깊이만 25% / 둘 다 50% (공유 모델), rgb·depth·both = 그 조건 전용 모델."""
    ray_on = torch.ones(B, 17, dtype=torch.bool, device=dev); pt_on = pval.clone()
    cond = torch.full((B,), 2, device=dev)                                                     # 0 RGB만, 1 깊이만, 2 둘 다
    if train and train_cond == "mixed":
        r = torch.rand(B, device=dev); cond[r < 0.25] = 0; cond[(r >= 0.25) & (r < 0.5)] = 1
    elif train:
        cond[:] = COND[train_cond]
    return ray_on, pt_on, cond


def apply_cond(ray_on, pt_on, cond):
    ray_on = ray_on & (cond != 1)[:, None]; pt_on = pt_on & (cond != 0)[:, None]
    return ray_on, pt_on


# ---------------- 학습·평가 ----------------
def train(model, D, steps, bs, dev, beta=0.5, lam=1.0, log=None, dist="gauss", train_cond="mixed"):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    N = len(D["y"]); t0 = time.time()
    for it in range(steps):
        idx = torch.randint(0, N, (bs,), device=D["y"].device)
        ray, pts, pval, y = (D[k][idx].to(dev, non_blocking=True) for k in ("ray", "pts", "pval", "y"))
        w = (D["ymask"][idx] & ((D["q"][idx] & 1) == 0)).to(dev).float()                        # 정지 정답 관절 제외
        ray_on, pt_on, cond = modality_masks(bs, pval, dev, True, train_cond)
        ray_on, pt_on = apply_cond(ray_on, pt_on, cond)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, L = model(ray, ray_on, pts, pt_on)
        mu, L = mu.float(), L.float()
        nll, logdet = nll_terms(mu, L, y, model.nu() if dist == "t" else None)
        bw = torch.exp(beta * logdet / 3).detach()                                                # β-NLL: (det Σ)^(β/3)
        mpjpe = torch.linalg.norm(y - mu, dim=-1)
        loss = ((bw * nll + lam * mpjpe) * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if log and (it % 500 == 0 or it == steps - 1):
            log(f"step {it} loss {loss.item():.4f} mpjpe {(mpjpe * w).sum().item() / w.sum().item() * 1000:.1f} mm nu {model.nu().item():.2f} "
                f"{(it + 1) / (time.time() - t0):.1f} it/s")
    return time.time() - t0


def thresholds(dist, nu):
    """명목 수준별 문턱: 3D는 d² (가우스 χ²₃, t는 3·F⁻¹(q; 3, ν)), 광선 축 1D는 |z| (가우스 정규, t는 t_ν)."""
    qs = {"50": 0.5, "68": 0.6827, "90": 0.9, "95": 0.95}
    if dist == "t":
        return ({k: 3 * stats.f.ppf(q, 3, nu) for k, q in qs.items()}, {k: stats.t.ppf(0.5 + q / 2, nu) for k, q in qs.items()})
    return ({k: stats.chi2.ppf(q, 3) for k, q in qs.items()}, {k: stats.norm.ppf(0.5 + q / 2) for k, q in qs.items()})


@torch.no_grad()
def predict_d2(model, D, dev, c, bs=2048):
    """조건 c에서 깨끗한 정답 관절의 마할라노비스 d² (재보정용)."""
    out = []
    for s in range(0, len(D["y"]), bs):
        sl = slice(s, min(len(D["y"]), s + bs)); B = sl.stop - sl.start
        ray, pts, pval, y = (D[k][sl].to(dev) for k in ("ray", "pts", "pval", "y"))
        ro, po, _ = modality_masks(B, pval, dev, False); ro, po = apply_cond(ro, po, torch.full((B,), c, device=dev))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, L = model(ray, ro, pts, po)
        S = L.float() @ L.float().transpose(-1, -2); r = y - mu.float()
        d2 = (r.unsqueeze(-2) @ torch.linalg.solve(S, r.unsqueeze(-1))).squeeze(-1).squeeze(-1)
        w = (D["ymask"][sl] & ((D["q"][sl] & 1) == 0)).to(dev)
        out.append(d2[w])
    return torch.cat(out)


def recal_scales(model, Dva, dev, dist):
    """검증 피험자에서 조건별 전역 분산 배율 k = median(d²) / 이상적 중앙값 (가우스 χ²₃, t는 3·F(3, ν))."""
    model.eval()
    nu = float(model.nu()) if dist == "t" else None
    ideal = 3 * stats.f.ppf(0.5, 3, nu) if dist == "t" else stats.chi2.ppf(0.5, 3)
    k = {c: float(predict_d2(model, Dva, dev, i).median() / ideal) for c, i in (("rgb_only", 0), ("depth_only", 1), ("both", 2))}
    model.train()
    return k


@torch.no_grad()
def evaluate(model, D, dev, bs=2048, dist="gauss", scales=None):
    model.eval(); out = {}
    nu = float(model.nu()) if dist == "t" else None
    T3, T1 = thresholds(dist, nu)
    N = len(D["y"])
    for cname, c in (("rgb_only", 0), ("depth_only", 1), ("both", 2)):
        acc = {k: [] for k in ("err", "root", "rel", "par", "perp", "d2", "z1", "aniso", "w", "wclean", "subj")}
        for s in range(0, N, bs):
            sl = slice(s, min(N, s + bs)); B = sl.stop - sl.start
            ray, pts, pval, y, cam = (D[k][sl].to(dev) for k in ("ray", "pts", "pval", "y", "cam"))
            ray_on, pt_on, _ = modality_masks(B, pval, dev, False)
            ray_on, pt_on = apply_cond(ray_on, pt_on, torch.full((B,), c, device=dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, L = model(ray, ray_on, pts, pt_on)
            mu, L = mu.float(), L.float(); S = L @ L.transpose(-1, -2)
            r = y - mu
            u = y - cam[:, None]; u = u / u.norm(dim=-1, keepdim=True)                            # 관절별 카메라 광선 방향
            rp = (r * u).sum(-1); rperp = (r - rp[..., None] * u).norm(dim=-1)
            root = 0.5 * (r[:, 7] + r[:, 8])
            d2 = (r.unsqueeze(-2) @ torch.linalg.solve(S, r.unsqueeze(-1))).squeeze(-1).squeeze(-1)
            s_par = torch.sqrt((u.unsqueeze(-2) @ S @ u.unsqueeze(-1)).squeeze(-1).squeeze(-1))
            s_tot = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1)
            s_perp = torch.sqrt(((s_tot - s_par ** 2) / 2).clamp(min=1e-12))
            acc["err"].append(r.norm(dim=-1)); acc["root"].append(root.norm(dim=-1)[:, None].expand(-1, 13))
            acc["rel"].append((r - root[:, None]).norm(dim=-1)); acc["par"].append(rp.abs()); acc["perp"].append(rperp)
            acc["d2"].append(d2); acc["z1"].append((rp / s_par).abs()); acc["aniso"].append(s_par / s_perp)
            acc["w"].append(D["ymask"][sl].to(dev)); acc["wclean"].append(D["ymask"][sl].to(dev) & ((D["q"][sl].to(dev) & 1) == 0))
            acc["subj"].append(D["subj"][sl].to(dev)[:, None].expand(-1, 13))
        A = {k: torch.cat(v) for k, v in acc.items()}
        m = lambda x, w: float((x * w).sum() / w.sum())
        res = {}
        for wn in ("w", "wclean"):
            w = A[wn].float()
            res[wn] = dict(mpjpe_abs_mm=m(A["err"], w) * 1000, root_mm=m(A["root"], w) * 1000, rel_mm=m(A["rel"], w) * 1000,
                           ray_axis_mm=m(A["par"], w) * 1000, perp_mm=m(A["perp"], w) * 1000,
                           cov3d={k: m((A["d2"] <= v).float(), w) for k, v in T3.items()},
                           cov_ray1d={k: m((A["z1"] <= v).float(), w) for k, v in T1.items()},
                           d2_median=float(A["d2"][A[wn]].median()), aniso_median=float(A["aniso"][A[wn]].median()))
            if scales:
                k = scales[cname]
                res[wn]["recal_k"] = k
                res[wn]["cov3d_recal"] = {kk: m((A["d2"] / k <= v).float(), w) for kk, v in T3.items()}
                res[wn]["cov_ray1d_recal"] = {kk: m((A["z1"] / k ** 0.5 <= v).float(), w) for kk, v in T1.items()}
        subj = A["subj"][:, 0].unique()
        res["per_subject_mpjpe_mm"] = {int(sj): m(A["err"][A["subj"][:, 0] == sj], A["w"][A["subj"][:, 0] == sj].float()) * 1000
                                       for sj in subj}
        out[cname] = res
    out["nu"] = nu
    model.train()
    return out


def main():
    packs, split = arg("--packs"), arg("--split", "E1")
    steps, bs, out = arg("--steps", 8000, int), arg("--bs", 512, int), Path(arg("--out"))
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    sp = json.loads((Path(__file__).resolve().parent / "splits" / "mmfi.json").read_text())["E1_cross_subject"]
    torch.manual_seed(arg("--seed", 0, int)); np.random.seed(arg("--seed", 0, int))
    t0 = time.time()
    Dtr, spec = build(packs, set(sp["train"])); Dte, _ = build(packs, set(sp["test"]), step=2)
    Dva, _ = build(packs, set(sp["val"]), step=2)
    log(f"data: train {len(Dtr['y'])} frames, val {len(Dva['y'])}, test {len(Dte['y'])} frames, build {time.time() - t0:.0f}s")
    dev = torch.device("cuda")
    Dtr = {k: v.to(dev) for k, v in Dtr.items()}                                              # 학습 데이터 GPU 상주 (약 1.5 GB)
    model = Net().to(dev)
    log(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    dist, tcond = arg("--dist", "gauss"), arg("--train-cond", "mixed")
    secs = train(model, Dtr, steps, bs, dev, log=log, dist=dist, train_cond=tcond)
    k = recal_scales(model, Dva, dev, dist)
    res = evaluate(model, Dte, dev, dist=dist, scales=k)
    res["meta"] = dict(packs=packs, split=split, steps=steps, bs=bs, train_seconds=secs, spec=spec["spec_id"],
                       version="model_v0 single-frame", dist=dist, train_cond=tcond, seed=arg("--seed", 0, int))
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    torch.save(model.state_dict(), out / "model.pt")
    for c in ("rgb_only", "depth_only", "both"):
        r = res[c]["wclean"]
        log(f"{c}: abs {r['mpjpe_abs_mm']:.1f} root {r['root_mm']:.1f} rel {r['rel_mm']:.1f} ray-axis {r['ray_axis_mm']:.1f} "
            f"perp {r['perp_mm']:.1f} | cov3d {r['cov3d']} | cov_ray {r['cov_ray1d']} | d2med {r['d2_median']:.2f} "
            f"aniso {r['aniso_median']:.2f} | recal k {r.get('recal_k', float('nan')):.2f} cov3d_recal {r.get('cov3d_recal')}")
    log(f"train time {secs / 60:.1f} min")


if __name__ == "__main__":
    main()
