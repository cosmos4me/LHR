"""R1 평가 모델 (2026-11-13): 레이더 점 5프레임 → 13관절 (좌표 G, m). 학습 = MM-Fi 실측 train 또는 투영 변형 (mmBody → MM-Fi 형), 시험 = MM-Fi 실측 test.
모델: 점별 MLP + 프레임 임베딩 → 트랜스포머 인코더 (4층, d 128) → 관절 질의 13개 디코더 (2층) → 가운데 프레임 관절. 손실 L1.
입력 특징 --feat xyz (기본, EMDUL 과 같은 조건) | xyzdi (+ 도플러, 세기). 프레임당 점 N=128 (복원 추출, 없으면 가림).
출력 OUT/metrics.json (MPJPE·PA·골반 mm, 피험자별) + OUT/test_err.npz. 사용:
python xproj_train.py --train mmfi|<npz> --out DIR [--test mmfi|<npz>] [--calib 1] [--dann λ] [--pseudo <npz>] [--pred_unl] [--floor fix|res --train_h 1.01] [--ctr median] [--flip p] [--scale_aug a] [--feat xyz] [--steps 15000] [--seed 0] [--subj ..] [--mix <npz>]"""
import argparse, os, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--out", required=True); ap.add_argument("--feat", default="xyz"); ap.add_argument("--load", default=""); ap.add_argument("--el_aug", default=""); ap.add_argument("--acts", default=""); ap.add_argument("--cache", default=os.environ.get("XCACHE", "mmfi_radar.npz")); ap.add_argument("--h0", type=float, default=None); ap.add_argument("--unc", type=int, default=0); ap.add_argument("--feats", type=int, default=0); ap.add_argument("--feats_subj", default="")   # 11-22: --load 와 함께 특징 저장 (학습 창 일부·시험 창) — 새로움 점수용   # 11-22 LHR: 관절별 불확실도 (Laplace) + 사전 자세와 정밀도 가중 결합   # 11-22: --load 평가 때 학습 때 H0 그대로   # 11-22: --acts A01,A03,.. → MM-Fi 학습 프레임을 이 동작들로 제한 (처음 보는 동작 시험용)   # 11-22: 고도 응답 무작위화 "smin,smax,omin_deg,omax_deg" (레이더 좌표 고도 el′ = s·el + o, 거리·방위 유지)   # 11-22: --load net.pt → 학습 없이 평가 (역방향 점검용)
ap.add_argument("--steps", type=int, default=15000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--data", default="/home/interns/unisense/runs/xproj"); ap.add_argument("--W", type=int, default=5); ap.add_argument("--N", type=int, default=128)
ap.add_argument("--cap", type=int, default=0)                                   # 연기 시험용: 시험 창 수 제한 (0 = 전부)
ap.add_argument("--subj", default="")                                          # MM-Fi 학습 피험자 부분집합 (예: 1,11,21), 빈칸 = train 전부
ap.add_argument("--mix", default="")                                           # 섞을 투영 데이터 (배치 절반씩)
ap.add_argument("--test", default="mmfi")                                      # 시험 데이터: mmfi (E1 test) | <npz 경로> (예: mmBody 원본 test → A→A 상한)
ap.add_argument("--dann", type=float, default=0.0)                             # R6: 라벨 없는 MM-Fi train (18명) 과 특징 정렬 (기울기 반전, 가중 λ), 0 = 끔
ap.add_argument("--pseudo", default="")                                        # R6: 의사 라벨 npz (idx, J) — MM-Fi train 창에 붙여 배치 절반으로 섞음
ap.add_argument("--pred_unl", action="store_true")                             # R6: 학습 뒤 MM-Fi train 18명 창 예측 저장 (OUT/pred_unl.npz, 교사용)
ap.add_argument("--floor", default="")                                         # R7: 골반 높이 = 바닥 + 학습 데이터 골반 높이 H0. fix = 골반 z 를 H0 로 고정 (점에서 안 읽음), res = H0 + 학습 잔차. 빈칸 = 기존
ap.add_argument("--train_h", type=float, default=1.01)                         # R7: h 열이 없는 npz (A′) 의 배치 레이더 높이 (TI21R 은 S01 배치 → 1.01)
ap.add_argument("--anchor", default="")                                     # 11-23 R64 [P1]: "K,thr" — 골반 수평 기준점 = 앞뒤 K 프레임 (같은 시퀀스) 안 |도플러| < thr 점의 중앙값 (몸통: 호흡·흔들림뿐; 움직이는 팔 점 제외), 모자라면 그 구간 전체 점 중앙값
ap.add_argument("--ctr", default="mean")                                       # R8: 수평 기준점 mean | median (유령·한쪽 팔 점에 강건)
ap.add_argument("--flip", type=float, default=0.0)                             # R8: 학습 때 좌우 반전 확률 (y → −y, 좌우 관절 교환) — 한쪽 팔 동작 보강
ap.add_argument("--scale_aug", type=float, default=0.0)                        # R8: 학습 때 체형 크기 s ~ U(1−a, 1+a) (수평은 골반 기준, 높이는 바닥 기준; --floor 필요)
ap.add_argument("--pbs", type=int, default=256); ap.add_argument("--plab", type=float, default=0.0); ap.add_argument("--plab_root", type=int, default=0)   # 11-22 R57: 점별 부위 라벨 보조 손실 무게 (가상만, 없으면 −1 무시) / 1 = 몸통·다리로 판정된 점의 중심을 골반 출발값으로
ap.add_argument("--calib", default="")                                         # 출력 어댑터 맞춤 피험자 (MM-Fi train, 예: 1): 관절 affine 결합 (ACAE 식, 행 합 1) + 위치 치우침 3

LASTLAB = None
A = ap.parse_args(); torch.manual_seed(A.seed); rng = np.random.default_rng(A.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); D = Path(A.data); OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True)
FD = {"xyz": 3, "xyzd": 4}.get(A.feat, 5)                                   # 11-23 R64: xyzd = + 도플러만 (세기 제외)
LASTOFF = None
SPL = {"train": [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26], "test": [8, 9, 10, 18, 19, 20, 28, 29, 30]}   # MM-Fi E1_cross_subject


def load(path, subj_set=None):
    Z = np.load(path); n = len(Z["J"]); keep = np.ones(n, bool)
    if subj_set is not None:
        keep = np.isin(Z["subj"], subj_set) & Z["M"].all(1)
        if A.acts and set(subj_set) & set(SPL["train"]) == set(subj_set):          # 학습 쪽에만 동작 제한
            keep &= np.isin(np.array([str(q).split("_")[1] for q in Z["seq"]]), A.acts.split(","))
    seq = Z["seq"]; starts = np.r_[True, seq[1:] != seq[:-1]]; sid = np.cumsum(starts) - 1
    hh = Z["h"].astype(np.float32) if "h" in Z.files else np.full(n, A.train_h, np.float32)          # 레이더 높이 (MM-Fi: m0 적합값)
    return dict(dop=Z["pts"][:, 3].astype(np.float32) if Z["pts"].shape[1] > 3 else None, pts=Z["pts"][:, :FD].astype(np.float32), off=Z["off"], J=Z["J"].astype(np.float32), sid=sid, keep=keep, h=hh,
                subj=Z["subj"] if "subj" in Z.files else np.zeros(n, int), plab=Z["plab"].astype(np.int64) if "plab" in Z.files else np.full(len(Z["pts"]), -1, np.int64))


def windows(S):
    """가운데 프레임이 keep 이고 같은 시퀀스 안에서 W 프레임이 모두 있는 위치."""
    h = A.W // 2; i = np.arange(len(S["J"])); ok = S["keep"].copy()
    for d in range(-h, h + 1):
        j = (i + d).clip(0, len(i) - 1); ok &= (S["sid"][j] == S["sid"]) & (i + d >= 0) & (i + d < len(i))
    return np.flatnonzero(ok)


LR = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]


def batch(S, centers, aug=False):
    global LASTLAB
    h = A.W // 2; B = len(centers); X = np.zeros((B, A.W, A.N, FD), np.float32); Mk = np.ones((B, A.W, A.N), bool); Lb = np.full((B, A.W, A.N), -1, np.int64)
    for b, c in enumerate(centers):
        for w in range(A.W):
            k = c - h + w; p = S["pts"][S["off"][k]:S["off"][k + 1]]
            if len(p):
                ii = rng.integers(0, len(p), A.N); X[b, w] = p[ii]; Mk[b, w] = False; Lb[b, w] = S["plab"][S["off"][k] + ii]
    LASTLAB = Lb
    Y = S["J"][centers].copy()
    if A.anchor:                                                                   # 11-23 R64: 몸통 기준점 − 창 점 평균 (수평), 증강과 함께 변환
        global LASTOFF
        K_, th_ = A.anchor.split(","); K_, th_ = int(K_), float(th_); off_ = np.zeros((B, 2), np.float32)
        for b, c in enumerate(centers):
            ks = [k for k in range(c - K_, c + K_ + 1) if 0 <= k < len(S["sid"]) and S["sid"][k] == S["sid"][c]]
            q = np.concatenate([S["pts"][S["off"][k]:S["off"][k + 1], :2] for k in ks]); dv = np.concatenate([S["dop"][S["off"][k]:S["off"][k + 1]] for k in ks])
            if len(q) == 0 or (~Mk[b]).sum() == 0:
                continue
            sel = np.abs(dv) < th_; anc = np.median(q[sel], 0) if sel.sum() >= 3 else np.median(q, 0); off_[b] = anc - X[b][~Mk[b]][:, :2].mean(0)
    if aug and A.el_aug:
        s0, s1, o0, o1 = map(float, A.el_aug.split(",")); sc = rng.uniform(s0, s1, B); oo = np.radians(rng.uniform(o0, o1, B))
        xyz = X[..., :3].astype(np.float64); r = np.linalg.norm(xyz, axis=-1).clip(1e-6); el = np.arcsin(np.clip(xyz[..., 2] / r, -1, 1)); az = np.arctan2(xyz[..., 1], xyz[..., 0])
        el2 = sc[:, None, None] * el + oo[:, None, None]; X[..., 0] = r * np.cos(el2) * np.cos(az); X[..., 1] = r * np.cos(el2) * np.sin(az); X[..., 2] = r * np.sin(el2)
    if A.floor:                                                                    # 바닥 좌표 (z + 레이더 높이): 입력·정답 모두
        hh = S["h"][centers]; X[..., 2] += hh[:, None, None]; Y[..., 2] += hh[:, None]
    if aug and A.flip > 0:                                                         # 좌우 반전 (레이더 시선축 기준 거울상)
        f = rng.random(B) < A.flip; X[f, ..., 1] *= -1; Y[f, :, 1] *= -1
        if A.anchor:
            off_[f, 1] *= -1
        for a, b in LR:
            Y[f, a], Y[f, b] = Y[f, b].copy(), Y[f, a].copy()
    if aug and A.scale_aug > 0 and A.floor:                                        # 체형 크기: 수평은 골반 기준, 높이는 바닥(0) 기준
        sc = rng.uniform(1 - A.scale_aug, 1 + A.scale_aug, B).astype(np.float32); pxy = 0.5 * (Y[:, 7, :2] + Y[:, 8, :2])
        X[..., :2] = pxy[:, None, None] + (X[..., :2] - pxy[:, None, None]) * sc[:, None, None, None]; X[..., 2] *= sc[:, None, None]
        if A.anchor:
            off_ *= sc[:, None]
        Y[..., :2] = pxy[:, None] + (Y[..., :2] - pxy[:, None]) * sc[:, None, None]; Y[..., 2] *= sc[:, None]
    if A.anchor:
        LASTOFF = torch.from_numpy(off_).to(dev)
    return torch.from_numpy(X).to(dev), torch.from_numpy(Mk).to(dev), torch.from_numpy(Y).to(dev)


class Net(nn.Module):
    def __init__(s, d=128):
        super().__init__()
        s.pt = nn.Sequential(nn.Linear(FD, d), nn.GELU(), nn.Linear(d, d)); s.fe = nn.Embedding(A.W, d)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), 4)
        s.q = nn.Parameter(torch.randn(13, d) * 0.02)
        s.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), 2); s.head = nn.Linear(d, 6 if A.unc else 3)
        if A.plab > 0 or A.plab_root:
            s.ph = nn.Linear(d, 7)

    def forward(s, X, Mk, feat=False):
        B, W, N, _ = X.shape; ctr = (X[..., :3] * (~Mk)[..., None]).sum((1, 2)) / (~Mk).sum((1, 2)).clamp(min=1)[:, None]   # 점 중심 (원점 이동)
        if A.ctr == "median":                                                      # 좌표별 중앙값 (가린 칸 제외)
            Xn = X[..., :3].masked_fill(Mk[..., None], float("nan")).reshape(B, W * N, 3); md = torch.nanmedian(Xn, 1).values
            ctr = torch.where(torch.isnan(md), ctr, md)
        if A.floor:                                                                # 높이 기준은 점 중심이 아니라 H0 (바닥 위 골반 높이 사전값)
            ctr = torch.cat([ctr[:, :2], torch.full_like(ctr[:, 2:], H0)], 1)
        if A.anchor and LASTOFF is not None:                                       # 11-23 R64: 몸통 기준점으로 출발값 이동
            ctr = torch.cat([ctr[:, :2] + LASTOFF.to(ctr.dtype), ctr[:, 2:]], 1)
        Xc = X.clone(); Xc[..., :3] = Xc[..., :3] - ctr[:, None, None]
        t = s.pt(Xc) + s.fe.weight[None, :, None]; t = t.reshape(B, W * N, -1); m = Mk.reshape(B, W * N)
        m = m & ~(m.all(1, keepdim=True) & (torch.arange(W * N, device=X.device) == 0))     # 전부 비면 한 칸은 열어 둠
        h = s.enc(t, src_key_padding_mask=m); o = s.dec(s.q[None].expand(B, -1, -1), h, memory_key_padding_mask=m)
        if A.plab > 0 or A.plab_root:
            s.plog = s.ph(h)                                                     # (B, W·N, 7) 점 부위 로짓
            if A.plab_root:                                                      # 몸통·다리 점 가중 중심 (수평만) 으로 출발값 이동 — 움직이는 팔 쪽 쏠림 제거
                wb = (s.plog.float().softmax(-1)[..., [0, 3, 4]].sum(-1) * (~m).float()); xy = Xc[..., :2].reshape(B, W * N, 2)
                off = (wb[..., None] * xy).sum(1) / wb.sum(1, keepdim=True).clamp(min=1e-3); ctr = ctr + torch.cat([off, torch.zeros_like(off[:, :1])], 1)
        if feat:                                                                 # 영역 판별용: 관절 질의 출력 평균 + 점 토큰 평균
            v = (~m).float()[..., None]; return s.out(o, ctr), torch.cat([o.mean(1), (h * v).sum(1) / v.sum(1).clamp(min=1)], -1)
        return s.out(o, ctr)

    def out(s, o, ctr):
        hd = s.head(o); P = hd[..., :3] + ctr[:, None]
        if A.unc:
            s.logb = hd[..., 3:].clamp(-6, 1)                                      # 관절·좌표별 Laplace 척도 log b (m)
        if A.floor == "fix":                                                       # 골반 높이를 H0 로 고정: 관절 전체를 세로로 옮김 (자세의 높이 모양은 학습)
            P = torch.cat([P[..., :2], P[..., 2:] + (H0 - 0.5 * (P[:, 7:8, 2:] + P[:, 8:9, 2:]))], -1)
        return P


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def pa_err(P, G):
    """프레임별 Procrustes 정렬 뒤 관절 평균 거리."""
    out = []
    for p, g in zip(P, G):
        mp, mg = p.mean(0), g.mean(0); X, Y = p - mp, g - mg; U, S, Vt = np.linalg.svd(X.T @ Y); Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            U[:, -1] *= -1; Rm = U @ Vt
        sc = S.sum() / (X ** 2).sum(); out.append(np.linalg.norm(sc * X @ Rm + mg - g, axis=1).mean())
    return np.array(out)


if __name__ == "__main__":
    F = D / "cache" / A.cache                 # 11-22: XCACHE=mmfi_radar_hfix.npz (E01 h 발목 기준 보정)
    subj = [int(x) for x in A.subj.split(",")] if A.subj else SPL["train"]
    TR = load(F, subj) if A.train == "mmfi" else load(A.train); TE = load(F, SPL["test"]) if A.test == "mmfi" else load(A.test)
    MX = load(A.mix) if A.mix else None; ctr_mx = windows(MX) if MX is not None else None
    if A.pseudo:                                                                 # 의사 라벨: MM-Fi train 18명 창 (정답 대신 교사 예측)
        PZ = np.load(A.pseudo); MX = load(F, SPL["train"]); MX["J"] = MX["J"].copy(); MX["J"][PZ["idx"]] = PZ["J"]; ctr_mx = PZ["idx"][PZ["keep"]] if "keep" in PZ.files else PZ["idx"]
    UN = load(F, SPL["train"]) if (A.dann > 0 or A.pred_unl) else None; ctr_un = windows(UN) if UN is not None else None
    ctr_tr, ctr_te = windows(TR), windows(TE)
    if A.cap:
        ctr_te = ctr_te[np.linspace(0, len(ctr_te) - 1, A.cap).astype(int)]
    print("train windows", len(ctr_tr), "test windows", len(ctr_te), flush=True)
    global H0
    pz = lambda S, c: 0.5 * (S["J"][c, 7, 2] + S["J"][c, 8, 2]) + S["h"][c]
    H0 = float(np.median(pz(TR, ctr_tr))) if A.floor else 0.0
    if A.h0 is not None:
        H0 = A.h0                    # 주 학습 데이터 (--train) 의 라벨에서만: 섞기 때는 실측 쪽 정의를 따름
    print("floor", A.floor, "H0", round(H0, 3), flush=True)
    net = Net().to(dev); dc = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)).to(dev)
    opt = torch.optim.AdamW(list(net.parameters()) + list(dc.parameters()), 3e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=A.steps, pct_start=0.05); t0 = time.time()
    if A.load:
        net.load_state_dict(torch.load(A.load, map_location=dev))
    for it in range(0 if A.load else A.steps):
        if MX is None:
            X, Mk, Y = batch(TR, rng.choice(ctr_tr, A.bs), aug=True)
        else:
            b1, b2 = batch(TR, rng.choice(ctr_tr, A.bs // 2), aug=True), batch(MX, rng.choice(ctr_mx, A.bs - A.bs // 2), aug=True)
            X, Mk, Y = (torch.cat([u, v]) for u, v in zip(b1, b2))
        with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
            if A.dann > 0:
                Xu, Mu, _ = batch(UN, rng.choice(ctr_un, A.bs)); nb = len(X)
                Pall, fe = net(torch.cat([X, Xu]), torch.cat([Mk, Mu]), feat=True); P = Pall[:nb]
                lam = A.dann * (2 / (1 + np.exp(-10 * it / A.steps)) - 1); dl = dc(GRL.apply(fe.float(), lam)).squeeze(-1)
                ldom = nn.functional.binary_cross_entropy_with_logits(dl, torch.cat([torch.zeros(nb), torch.ones(len(Xu))]).to(dev))
            else:
                P = net(X, Mk); ldom = torch.zeros((), device=dev)
        if A.unc:
            lb = net.logb[:len(P)].float(); loss = ((P.float() - Y).abs() / lb.exp() + lb).mean() + ldom
        else:
            loss = (P.float() - Y).abs().mean() + ldom
        if A.plab > 0 and MX is None:
            lab_t = torch.from_numpy(LASTLAB).to(dev).reshape(len(P), -1)
            if (lab_t >= 0).any():
                loss = loss + A.plab * nn.functional.cross_entropy(net.plog[:len(P)].float().reshape(-1, 7), lab_t.reshape(-1), ignore_index=-1)
        loss = loss; opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        if it % 1000 == 0:
            print(f"step {it} loss {loss.item():.4f} dom {ldom.item():.3f} {(it + 1) / (time.time() - t0):.1f} it/s", flush=True)
    net.eval()
    if not A.load:
        torch.save(net.state_dict(), OUT / "net.pt")

    LB = []

    def predict(S, ctr):
        out = []; LB.clear()
        with torch.no_grad():
            for s in range(0, len(ctr), A.pbs):
                X, Mk, _ = batch(S, ctr[s:s + A.pbs]); o = net(X, Mk).float().cpu().numpy()
                if A.unc:
                    LB.append(net.logb.float().cpu().numpy())
                if A.floor:
                    o[..., 2] -= S["h"][ctr[s:s + A.pbs]][:, None]                      # 바닥 좌표 → 레이더 좌표
                out.append(o)
        return np.concatenate(out)

    def metrics(P, G, sj):
        e = np.linalg.norm(P - G, axis=-1).mean(1) * 1000
        dr = 0.5 * (P[:, 7] + P[:, 8]) - 0.5 * (G[:, 7] + G[:, 8]); root = np.linalg.norm(dr, axis=-1) * 1000; pa = pa_err(P, G) * 1000
        rel = np.linalg.norm((P - 0.5 * (P[:, 7:8] + P[:, 8:9])) - (G - 0.5 * (G[:, 7:8] + G[:, 8:9])), axis=-1).mean(1) * 1000   # 골반 정렬 MPJPE
        m = dict(mpjpe=float(e.mean()), pa=float(pa.mean()), root=float(root.mean()), rel=float(rel.mean()),
                 root_axis_mm=dict(zip(["x_fwd", "y_lat", "z_up"], (np.abs(dr).mean(0) * 1000).round(1).tolist())),
                 root_bias_mm=dict(zip(["x_fwd", "y_lat", "z_up"], (dr.mean(0) * 1000).round(1).tolist())),
                 per_subject={int(s): float(e[sj == s].mean()) for s in np.unique(sj)})
        return m, e, pa, root

    def fit_adapter(P, G, lam=1e-2):
        """관절 j 출력 = Σ_k w_jk P_k (Σ_k w_jk = 1, 좌표축 공유 → 회전·이동 등변) + 위치 치우침 b (G 좌표 3). 골반 예측 기준 상대 좌표로 최소제곱."""
        pel = 0.5 * (P[:, 7] + P[:, 8]); X = (P - pel[:, None]).transpose(0, 2, 1).reshape(-1, 13); W = np.zeros((13, 13))
        for j in range(13):
            y = (G[:, j] - pel).reshape(-1); W[j] = np.linalg.solve(X.T @ X + lam * np.eye(13), X.T @ y)
        f = lambda Q: pel_of(Q)[:, None] + np.einsum("jk,nkc->njc", W, Q - pel_of(Q)[:, None])
        b = (G - f(P)).mean((0, 1)); return (lambda Q: f(Q) + b), W, b

    pel_of = lambda Q: 0.5 * (Q[:, 7] + Q[:, 8])
    if A.feats:                                                                    # 특징: 관절 질의 출력 평균 + 점 토큰 평균 (256)
        def featz(S, ctr):
            out = []
            with torch.no_grad():
                for s in range(0, len(ctr), A.pbs):
                    X, Mk, _ = batch(S, ctr[s:s + A.pbs]); _, fe = net(X, Mk, feat=True); out.append(fe.float().cpu().numpy())
            return np.concatenate(out)
        sub = ctr_tr[np.random.default_rng(0).permutation(len(ctr_tr))[:20000]]
        extra = {}
        if A.feats_subj:                                                           # 새로움 문턱 맞춤용: 학습에서 뺀 사람들의 (학습 동작) 입력
            CSf = load(F, [int(x) for x in A.feats_subj.split(",")]); cc = windows(CSf); extra = dict(cal=featz(CSf, cc).astype(np.float16), cal_idx=cc)
        np.savez_compressed(OUT / "feats.npz", tr=featz(TR, sub).astype(np.float16), tr_idx=sub, te=featz(TE, ctr_te).astype(np.float16), te_idx=ctr_te, **extra)
    P = predict(TE, ctr_te); G = TE["J"][ctr_te]; sj = TE["subj"][ctr_te]; P_raw = P; LB_te = np.concatenate(LB) if A.unc else np.zeros(1)
    if A.unc:                                                                      # LHR 결합: 골반 기준 자세 = 사전 + w (예측 − 사전), w = b0² / (b0² + b²), 사전 = 학습 라벨 평균, b0 = 학습 라벨 평균 절대 편차
        bt = np.exp(LB_te); Rt = TR["J"][ctr_tr].astype(float); Rt = Rt - 0.5 * (Rt[:, 7:8] + Rt[:, 8:9]); pri = Rt.mean(0); b0 = np.abs(Rt - pri).mean(0)
        pp = 0.5 * (P[:, 7:8] + P[:, 8:9]); w = b0[None] ** 2 / (b0[None] ** 2 + bt ** 2); P = pp + pri[None] + w * ((P - pp) - pri[None])
    met0, e, pa, root = metrics(P, G, sj)
    met = dict(train=A.train, subj=A.subj, mix=A.mix, test=A.test, floor=A.floor, H0=H0, ctr=A.ctr, flip=A.flip, scale_aug=A.scale_aug, W=A.W, feat=A.feat, seed=A.seed, steps=A.steps, **met0, n_test=int(len(e)))
    if A.pred_unl:
        Pu = predict(UN, ctr_un); np.savez_compressed(OUT / "pred_unl.npz", idx=ctr_un, pred=Pu.astype(np.float32))
    if A.calib:
        CS = load(F, [int(x) for x in A.calib.split(",")]); cc = windows(CS); Pc = predict(CS, cc); ad, W, b = fit_adapter(Pc, CS["J"][cc])
        mc, _, _, _ = metrics(ad(P), G, sj); met["calib"] = dict(subj=A.calib, n_fit=int(len(cc)), bias_mm=(b * 1000).round(1).tolist(), w_diag=np.diag(W).round(3).tolist(), **mc)
        np.save(OUT / "adapter_W.npy", W)
    met["minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "metrics.json").write_text(json.dumps(met, indent=1)); np.savez_compressed(OUT / "test_err.npz", mpjpe=e, pa=pa, root=root, subj=sj, idx=ctr_te, pred=P.astype(np.float32), pred_raw=P_raw.astype(np.float32), logb=LB_te.astype(np.float32))
    print(json.dumps(met), flush=True)
