"""모델 v1 (E4용 다중 데이터셋): 여러 데이터셋의 팩을 공통 토큰으로 읽어 하나의 모델로 학습·평가한다.
- 데이터셋: mmfi(v0.2 사양 A, E1 분할), mri(v0.2), octonet(v0.5). 데이터셋마다 있는 모달리티만 토큰이 된다
  (mRI는 깊이 없음, OctoNet은 RGB 없음).
- 토큰 (model_v0와 같은 물리 특징 + 사양 태그):
  - 광선 11 + 태그 8: log f, log 해상도 폭, log fps, 카메라 높이, log ν, log β, 검출기 범주 2.
  - 점 9 + 태그 6: log (f·b), log 양자화 간격, log f_d, 센서 범주 2, 카메라 높이.
- 입력·태그 방식 (E4·E5 비교): --input rays(기본) | pixels (K 없이 화소 좌표 / 해상도),
  --tags spec(기본) | none | dataset_id | shuffled (데이터셋 간 태그 교환, 지름길 검정).
- 학습: 데이터셋 균형 표본화(배치마다 데이터셋을 같은 확률로), 모달리티 가리기(있는 것 안에서), Student-t 또는 가우스 출력,
  β-NLL + MPJPE, 정지 정답 제외.
- 평가: 데이터셋 × 조건(rgb_only / depth_only / both, 가능한 것만)별 절대 오차 분해, 광선 축 오차, 포함률(원시·재보정), 이방성.
GPU 0만 사용. 사용: python model_v1.py --train mmfi,mri,octonet --eval mmfi,mri,octonet --out RUN [--dist t] [--tags spec] [--input rays]"""
import json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_common as cc
import model_v0 as m0

CODE = Path(__file__).resolve().parent
PACKS = {"mmfi": Path("/home/interns/unisense/dev/packs/mmfi_v02_specA"), "mri": Path("/home/interns/unisense/dev/packs/mri_v02"),
         "octonet": Path("/home/interns/unisense/dev/packs/octonet_v05")}
PACKS["mmbody"] = Path("/home/interns/unisense/dev/packs/mmbody_v01")                  # 2026-11-02 (DSID는 모델 통합 때)
DET = {"hrnet": 0, "vitpose_b": 1}
DSID = {"mmfi": 0, "mri": 1, "octonet": 2, "mmbody": 3}   # 3 이상 = 저장된 모델(크기 3)에서는 처음 보는 체계로 처리
RAY_F, PT_F = 11 + 8, 9 + 6


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def kvec(K):
    K = np.asarray(K, float)
    return [K[0, 0], K[1, 1], K[0, 2], K[1, 2]] if K.ndim == 2 else list(K)


MMFI_SPLIT = "E1_cross_subject"     # "E04_new_env": E04 새 환경 시험 (진단)


FEW = dict(x.split(":") for x in arg("--few", "").split(",") if x)   # H9: "--few mmfi:1" = 대상 train 피험자 k명만 (seed로 고름)


def few_pick(ds, subj):
    """train 피험자 집합 → k명 (정렬 후 seed·(n//2) 위치부터; seed 0/1이 다른 사람·다른 환경이 되게)."""
    k = int(FEW[ds]); L = sorted(subj); n = len(L); s0 = int(arg("--seed", 0, int)) * (n // 2) % n
    return {L[(s0 + i) % n] for i in range(k)}


def split_of(ds, part):
    out = _split_of(ds, part)
    return few_pick(ds, out) if part == "train" and ds in FEW else out


def _split_of(ds, part):
    if ds == "mmfi":
        return set(json.loads((CODE / "splits" / "mmfi.json").read_text())[MMFI_SPLIT][part])
    s = json.loads((CODE / "splits" / f"{ds}.json").read_text())[part]
    if ds == "mmbody":                                                               # 시퀀스 이름 ("train/sequence_0")
        return set(s)
    return {f"S{x:02d}" for x in s} if ds == "mri" else {f"U{x:02d}" for x in s}


def pack_files(ds, subjects):
    if ds == "mmbody":
        return [f for f in sorted(PACKS[ds].glob("*/*.npz")) if f"{f.parent.name}/{f.stem}" in subjects]
    if ds == "mmfi":
        return [f for f in sorted(PACKS[ds].glob("E*/*.npz")) if f.name.split("_")[0] in subjects]
    if ds == "mri":
        return [f for f in sorted(PACKS[ds].glob("S*.npz")) if f.stem in subjects]
    out = []
    for f in sorted(PACKS[ds].glob("*.npz")):
        meta = json.loads(np.load(f)["meta"].item())
        if meta["subject"] in subjects and not meta.get("multi_person_action"):
            out.append(f)
    return out


def tokens(P, spec, ds, input_mode):
    meta = json.loads(P["meta"].item()); T = len(P["labels"])
    ray = np.zeros((T, 17, RAY_F), np.float32); ray_ok = np.zeros((T, 17), bool)
    pts = np.zeros((T, 256, PT_F), np.float32); pt_ok = np.zeros((T, 256), bool)
    h = float(np.asarray(P["t_c2r"])[2])
    if spec.get("rgb") and (ds != "octonet"):
        rg = spec["rgb"]; sm = rg["sigma_model"]; K = kvec(rg["K"]); fps = spec["fps"]
        kp = P["kp_px"].astype(float)
        sig = cc.keypoint_sigma_px(kp, sm["g_per_joint"], sm["beta_s"], fps)
        o, d, s_rad, s_tan, e_rad = cc.ray_tokens(kp, K, P["R_c2r"], P["t_c2r"], sig)
        W = rg["image_wh"][0]
        if input_mode == "pixels":                                   # 비교 모델: K 없이 화소 좌표(해상도로 정규화)만
            base = np.concatenate([kp[..., :1] / W, kp[..., 1:2] / rg["image_wh"][1], np.log(sig / W)[..., None],
                                   np.zeros((T, 17, 8))], -1)
        else:
            base = np.concatenate([d, e_rad, np.log(s_rad)[..., None], np.log(s_tan)[..., None], np.broadcast_to(o, (T, 17, 3))], -1)
        det = np.zeros(2); det[DET.get(meta.get("detector", "hrnet"), 0)] = 1
        tag = np.r_[np.log(0.5 * (K[0] + K[1]) / 1000), np.log(W / 1000), np.log(fps / 10), h, np.log(sm["nu"]), np.log(sm["beta_s"]), det]
        ray = np.concatenate([base, np.broadcast_to(tag, (T, 17, 8))], -1).astype(np.float32)
        ray_ok = np.isfinite(ray).all(-1); ray = np.nan_to_num(ray)
    dep = spec.get("depth")
    if dep:
        dm = dep[f"{meta['env']}_{meta['day']}"] if ds == "octonet" else dep
        K = kvec(dm["K"]); fx, fy, cx, cy = K
        uvz = P["pts_uvz"].astype(float)
        a = np.stack([(uvz[..., 0] - cx) / fx, (uvz[..., 1] - cy) / fy, np.ones_like(uvz[..., 2])], -1)
        an = np.linalg.norm(a, axis=-1, keepdims=True)
        R = np.asarray(P["R_d2r"]); p = (a * uvz[..., 2:3]) @ R.T + P["t_d2r"]
        ax = (a / an) @ R.T
        s_ax = cc.axial_sigma(np.maximum(uvz[..., 2], 0.3), dm) * an[..., 0]
        s_lat = np.maximum(uvz[..., 2], 0.3) * dm["sigma_pix_px"] / (0.5 * (fx + fy))
        C = dm.get("stereo_constant_px_m") or (dm["f_px"] * dm["baseline_m"] if dm["type"] == "stereo" else 20.0)   # ToF: 기준선 없음 → 중립값 (형식은 styp)
        styp = np.zeros(2); styp[0 if dm["type"] == "stereo" else 1] = 1
        tag = np.r_[np.log(C / 20), np.log(max(dm.get("quant_step_m", 0.001), 1e-4) / 0.001), np.log(0.5 * (fx + fy) / 400), styp, h]
        pts = np.concatenate([p, ax, np.log(s_ax)[..., None], np.log(s_lat)[..., None], P["pts_edge"][..., None],
                              np.broadcast_to(tag, (T, 256, 6))], -1).astype(np.float32)
        pt_ok = P["pts_valid"].astype(bool) & np.isfinite(pts).all(-1); pts = np.nan_to_num(pts)
    return ray, ray_ok, pts, pt_ok


def build(ds, part, input_mode, step=1):
    subjects = split_of(ds, part); files = pack_files(ds, subjects)
    spec = None; L = {k: [] for k in ("ray", "rok", "pts", "pok", "y", "ymask", "q", "cam", "subj")}
    for f in files:
        P = np.load(f)
        if spec is None:
            spec = json.loads(Path(json.loads(P["meta"].item())["spec"]).read_text())
        ray, rok, pts, pok = tokens(P, spec, ds, input_mode)
        sl = slice(0, None, step); n = len(ray[sl])
        L["ray"].append(ray[sl]); L["rok"].append(rok[sl]); L["pts"].append(pts[sl]); L["pok"].append(pok[sl])
        L["y"].append(P["labels"][sl]); L["ymask"].append(P["label_mask"][sl] & np.isfinite(P["labels"][sl]).all(-1))
        L["q"].append(P["label_quality"][sl]); L["cam"].append(np.broadcast_to(np.asarray(P["t_c2r"], np.float32), (n, 3)))
        sj = json.loads(P["meta"].item())["subject"]; L["subj"].append(np.full(n, int(sj[1:])))
    D = {k: torch.from_numpy(np.ascontiguousarray(np.concatenate(v))) for k, v in L.items()}
    D["y"] = torch.nan_to_num(D["y"].float()); D["ds"] = torch.full((len(D["y"]),), DSID[ds])
    if not bool(D["pok"].any()):                                                     # 깊이 없는 데이터셋: 점 텐서를 1개로 줄여 메모리 절약 (배치 때 채움)
        D["pts"], D["pok"] = D["pts"][:, :1].contiguous(), D["pok"][:, :1].contiguous()
    return D


def pad_pts(x, n=256):
    """점 토큰 수를 n으로 맞춘다 (깊이 없는 데이터셋은 1개만 저장)."""
    if x.shape[1] == n:
        return x
    pad = torch.zeros(x.shape[0], n - x.shape[1], *x.shape[2:], dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], 1)


class NetV1(m0.Net):
    def __init__(self, d=192, layers=4, heads=6, tag_mode="spec"):
        super().__init__(d, layers, heads)
        self.ray = m0.MLP(RAY_F, d, d); self.pt = m0.MLP(PT_F, d, d)
        self.ds_emb = nn.Embedding(3, d); self.tag_mode = tag_mode

    def forward(self, ray, ray_on, pts, pt_on, ds=None):
        if self.tag_mode == "none":
            ray = torch.cat([ray[..., :11], torch.zeros_like(ray[..., 11:])], -1); pts = torch.cat([pts[..., :9], torch.zeros_like(pts[..., 9:])], -1)
        B = ray.shape[0]
        ids = torch.arange(17, device=ray.device)
        tr = self.ray(ray) + self.kp_id(ids)[None] + self.typ.weight[1]
        tp = self.pt(pts) + self.typ.weight[2]
        tq = self.query[None].expand(B, -1, -1) + self.typ.weight[0]
        if self.tag_mode == "dataset_id" and ds is not None:
            e = self.ds_emb(ds)[:, None]; tr, tp, tq = tr + e, tp + e, tq + e
        x = torch.cat([tq, tr, tp], 1)
        pad = torch.cat([torch.zeros(B, 13, dtype=torch.bool, device=ray.device), ~ray_on, ~pt_on], 1)
        h = self.norm(self.enc(x, src_key_padding_mask=pad)[:, :13])
        mu = self.mu(h); c = self.chol(h)
        L = torch.zeros(B, 13, 3, 3, device=ray.device, dtype=c.dtype)
        L[..., 0, 0] = F.softplus(c[..., 0]) + 1e-3; L[..., 1, 1] = F.softplus(c[..., 1]) + 1e-3; L[..., 2, 2] = F.softplus(c[..., 2]) + 1e-3
        L[..., 1, 0] = c[..., 3]; L[..., 2, 0] = c[..., 4]; L[..., 2, 1] = c[..., 5]
        return mu, L


def masks(rok, pok, train):
    """학습: 둘 다 있으면 RGB만 25% / 깊이만 25% / 둘 다 50%. 하나만 있으면 그것."""
    B = rok.shape[0]; has_r, has_p = rok.any(1), pok.any(1)
    cond = torch.full((B,), 2, device=rok.device)
    cond[has_r & ~has_p] = 0; cond[~has_r & has_p] = 1
    if train:
        both = has_r & has_p; r = torch.rand(B, device=rok.device)
        cond[both & (r < 0.25)] = 0; cond[both & (r >= 0.25) & (r < 0.5)] = 1
    return rok & (cond != 1)[:, None], pok & (cond != 0)[:, None]


def train(model, Ds, steps, bs, dev, dist, log, beta=0.5, lam=1.0, shuffle_tags=False):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    names = list(Ds); t0 = time.time()
    for it in range(steps):
        per = [bs // len(names)] * len(names)                                           # 데이터셋 균형 표본화
        parts = []
        for nm, k in zip(names, per):
            D = Ds[nm]; idx = torch.randint(0, len(D["y"]), (k,), device=D["y"].device)
            pb = {kk: D[kk][idx] for kk in ("ray", "rok", "pts", "pok", "y", "ymask", "q", "ds")}
            pb["pts"], pb["pok"] = pad_pts(pb["pts"]), pad_pts(pb["pok"])
            parts.append(pb)
        b = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        if shuffle_tags:                                                                # 태그 교환 (지름길 검정): 배치 안에서 태그만 섞음
            perm = torch.randperm(len(b["y"]), device=dev)
            b["ray"] = torch.cat([b["ray"][..., :11], b["ray"][perm][..., 11:]], -1); b["pts"] = torch.cat([b["pts"][..., :9], b["pts"][perm][..., 9:]], -1)
        ro, po = masks(b["rok"], b["pok"], True)
        w = (b["ymask"] & ((b["q"] & 1) == 0)).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, L = model(b["ray"], ro, b["pts"], po, b["ds"])
        mu, L = mu.float(), L.float()
        nll, logdet = m0.nll_terms(mu, L, b["y"], model.nu() if dist == "t" else None)
        loss = ((torch.exp(beta * logdet / 3).detach() * nll + lam * torch.linalg.norm(b["y"] - mu, dim=-1)) * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if it % 500 == 0 or it == steps - 1:
            log(f"step {it} loss {loss.item():.4f} nu {model.nu().item():.2f} {(it + 1) / (time.time() - t0):.1f} it/s")
    return time.time() - t0


@torch.no_grad()
def evaluate(model, D, dev, dist, scales=None, bs=512):
    model.eval(); res = {}
    nu = float(model.nu()) if dist == "t" else None
    T3, T1 = m0.thresholds(dist, nu)
    has_r, has_p = bool(D["rok"].any()), bool(D["pok"].any())
    conds = [c for c, ok in (("rgb_only", has_r), ("depth_only", has_p), ("both", has_r and has_p)) if ok]
    for cname in conds:
        A = {k: [] for k in ("err", "root", "rel", "par", "perp", "d2", "z1", "aniso", "w")}
        for s in range(0, len(D["y"]), bs):
            sl = slice(s, min(len(D["y"]), s + bs))
            b = {k: D[k][sl].to(dev) for k in ("ray", "rok", "pts", "pok", "y", "ymask", "q", "cam", "ds")}
            b["pts"], b["pok"] = pad_pts(b["pts"]), pad_pts(b["pok"])
            ro = b["rok"] & (cname != "depth_only"); po = b["pok"] & (cname != "rgb_only")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, L = model(b["ray"], ro, b["pts"], po, b["ds"])
            mu, L = mu.float(), L.float(); S = L @ L.transpose(-1, -2); r = b["y"] - mu
            u = b["y"] - b["cam"][:, None]; u = u / u.norm(dim=-1, keepdim=True)
            rp = (r * u).sum(-1); root = 0.5 * (r[:, 7] + r[:, 8])
            d2 = (r.unsqueeze(-2) @ torch.linalg.solve(S, r.unsqueeze(-1))).squeeze(-1).squeeze(-1)
            s_par = torch.sqrt((u.unsqueeze(-2) @ S @ u.unsqueeze(-1)).squeeze(-1).squeeze(-1))
            s_perp = torch.sqrt(((torch.diagonal(S, dim1=-2, dim2=-1).sum(-1) - s_par ** 2) / 2).clamp(min=1e-12))
            A["err"].append(r.norm(dim=-1)); A["root"].append(root.norm(dim=-1)[:, None].expand(-1, 13))
            A["rel"].append((r - root[:, None]).norm(dim=-1)); A["par"].append(rp.abs()); A["perp"].append((r - rp[..., None] * u).norm(dim=-1))
            A["d2"].append(d2); A["z1"].append((rp / s_par).abs()); A["aniso"].append(s_par / s_perp)
            A["w"].append(b["ymask"] & ((b["q"] & 1) == 0))
        A = {k: torch.cat(v) for k, v in A.items()}; w = A["w"].float()
        mm = lambda x: float((x * w).sum() / w.sum())
        r_ = dict(mpjpe_abs_mm=mm(A["err"]) * 1000, root_mm=mm(A["root"]) * 1000, rel_mm=mm(A["rel"]) * 1000,
                  ray_axis_mm=mm(A["par"]) * 1000, perp_mm=mm(A["perp"]) * 1000,
                  cov3d={k: mm((A["d2"] <= v).float()) for k, v in T3.items()}, cov_ray1d={k: mm((A["z1"] <= v).float()) for k, v in T1.items()},
                  aniso_median=float(A["aniso"][A["w"]].median()), n=int(A["w"].sum()))
        if scales and cname in scales:
            k = scales[cname]; r_["recal_k"] = k
            r_["cov3d_recal"] = {kk: mm((A["d2"] / k <= v).float()) for kk, v in T3.items()}
        res[cname] = r_
    model.train()
    return res


@torch.no_grad()
def recal(model, D, dev, dist):
    from scipy import stats
    nu = float(model.nu()) if dist == "t" else None
    ideal = 3 * stats.f.ppf(0.5, 3, nu) if dist == "t" else stats.chi2.ppf(0.5, 3)
    out = {}
    r = evaluate_d2(model, D, dev)
    for c, d2 in r.items():
        out[c] = float(d2.median() / ideal)
    return out


@torch.no_grad()
def evaluate_d2(model, D, dev, bs=512):
    model.eval(); out = {}
    has_r, has_p = bool(D["rok"].any()), bool(D["pok"].any())
    for cname in [c for c, ok in (("rgb_only", has_r), ("depth_only", has_p), ("both", has_r and has_p)) if ok]:
        L_ = []
        for s in range(0, len(D["y"]), bs):
            sl = slice(s, min(len(D["y"]), s + bs))
            b = {k: D[k][sl].to(dev) for k in ("ray", "rok", "pts", "pok", "y", "ymask", "q", "ds")}
            b["pts"], b["pok"] = pad_pts(b["pts"]), pad_pts(b["pok"])
            ro = b["rok"] & (cname != "depth_only"); po = b["pok"] & (cname != "rgb_only")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, L = model(b["ray"], ro, b["pts"], po, b["ds"])
            S = L.float() @ L.float().transpose(-1, -2); r = b["y"] - mu.float()
            d2 = (r.unsqueeze(-2) @ torch.linalg.solve(S, r.unsqueeze(-1))).squeeze(-1).squeeze(-1)
            L_.append(d2[b["ymask"] & ((b["q"] & 1) == 0)])
        out[cname] = torch.cat(L_)
    model.train()
    return out


def main():
    tr_names = arg("--train", "mmfi,mri,octonet").split(","); ev_names = arg("--eval", "mmfi,mri,octonet").split(",")
    steps, bs, dist = arg("--steps", 8000, int), arg("--bs", 510, int), arg("--dist", "t")
    tag_mode, input_mode = arg("--tags", "spec"), arg("--input", "rays")
    out = Path(arg("--out")); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a"); log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    seed = arg("--seed", 0, int); torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda"); t0 = time.time()
    Dtr = {n: {k: v.to(dev) for k, v in build(n, "train", input_mode).items()} for n in tr_names}
    Dva = {n: build(n, "val", input_mode, step=2) for n in ev_names}; Dte = {n: build(n, "test", input_mode, step=2) for n in ev_names}
    log(f"data built {time.time() - t0:.0f}s: " + ", ".join(f"{n} train {len(Dtr[n]['y'])}" for n in tr_names) + " | "
        + ", ".join(f"{n} test {len(Dte[n]['y'])}" for n in ev_names))
    model = NetV1(tag_mode=tag_mode).to(dev)
    secs = train(model, Dtr, steps, bs, dev, dist, log, shuffle_tags=(tag_mode == "shuffled"))
    res = {"meta": dict(train=tr_names, eval=ev_names, steps=steps, bs=bs, dist=dist, tags=tag_mode, input=input_mode, seed=seed,
                        train_seconds=secs, nu=float(model.nu()) if dist == "t" else None)}
    for n in ev_names:
        k = recal(model, Dva[n], dev, dist)
        res[n] = evaluate(model, Dte[n], dev, dist, scales=k)
        for c, r in res[n].items():
            log(f"{n} {c}: abs {r['mpjpe_abs_mm']:.1f} root {r['root_mm']:.1f} rel {r['rel_mm']:.1f} ray {r['ray_axis_mm']:.1f} "
                f"perp {r['perp_mm']:.1f} cov3d {[round(v, 2) for v in r['cov3d'].values()]} recal {[round(v, 2) for v in r.get('cov3d_recal', {}).values()]} "
                f"aniso {r['aniso_median']:.2f}")
    (out / "metrics.json").write_text(json.dumps(res, indent=1)); torch.save(model.state_dict(), out / "model.pt")
    log(f"train time {secs / 60:.1f} min")


if __name__ == "__main__":
    main()
