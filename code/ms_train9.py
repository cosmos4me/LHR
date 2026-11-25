"""11-25 v9 = v8full 기본 켬 + 흔들기 하한 1 cm (Σ 비례) + 번역기 pel_phys4 (레이더 사람 문) + ③ 풀이기 (평가: 망 출력 N(P, σn²) + 광선 옆 σ = σθ·거리 정보 합, 결과는 *_s 지표; σn 4 cm 는 v8 학습 데이터셋 검증 피험자 MM-Fi RGB+레이더 66 mm 에서, σθ 6 mrad 는 2D 검출 사전값 — mRI 로 맞춘 값 아님). 출력 runs/ms9.
11-25 v8full 옵션 (기본 끔 = v8 그대로): --phys pel_phys3 (번역기 Σ 두 가지 수정), --sigtok 1 (기준점 Σ 토큰 + Σ 비례 기준점 흔들기), --vview p (실측 RGB 표본도 p 확률로 무작위 장비 가상 시점으로), --adlos 1 (④ 어댑터에 데이터셋별 시선 방향 이동 항 + 어댑터 가중 감쇠 0).
11-25 v8 = v7 + 인코더·디코더 v1 핵심 (notes/reports/인코더 디코더 설계 근거.md 순서 1·2):
  --subdist 시험형 부분집합 표본: P(카메라+삼각측량+레이더만) = p_mri 0.4, P(카메라만) = p_cam 0.1, 나머지는 모달리티마다 유지 keep 0.7 (최소 하나)
  --dec2    디코더 질의가 3D 위치를 가짐: x0 = 기준점 + 템플릿, 층마다 x = sg(x_prev) + Δ (층별 머리, 0 초기화), 질의 = 관절 임베딩 + 장면 평균 + PE(x − 기준점), 모든 층에 같은 손실
  --valsub  학습 데이터셋마다 정렬 피험자 [2::5] 를 학습에서 빼고 따로 평가 (아는 데이터셋의 새 사람). 출력 runs/ms8.
11-24 v7 = v6 (pel_phys2) + 감사 수정: triangulate 방향 정규화·능선 1e-6 (C1), LiDAR 기준점 결합 포함 (M1), 기준점 시선 방향 흔들기 (C2), 가상 스테레오 상관 잡음 (M3), mRI 봉인 assert (M3 data), 기본 캐시 v3b, 출력 runs/ms7.
11-24 v5 = ms_train4 + --cache (기본 ms_cache_v3) + 레이더 밀도 무작위화 고침, 출력 runs/ms5.
11-24 토크나이저 v1 기준선 (보고서 notes/reports/다중 센서 토크나이저 설계 근거.md). ms_train3 + 다음 (기본 켬, 옵션으로 끔):
  --anchor  물리 골반 기준점 (pel_phys, 가린 뒤 입력만, GT 없음) 의 x·y 를 모든 좌표에서 뺌 + 출력도 기준점 기준, 학습 중 기준점 흔들기 --ajit (m)
  --band    sin·cos 파장 2 m–6.25 cm (기존 8 m–0.25 m)
  --plucker 광선 = [방향, four(기준점 기준 Plücker 모멘트), log 기준점 깊이, 관절] (기존 광선 위 2–5 m 네 점)
  --nosolo  한 데이터셋에만 있는 입력 제거: 깊이 색·색 있음 (mmBody), 레이더 정지 표시 (OctoNet)
  --dens    점 밀도 무작위화 (깊이·LiDAR 유지 비율, 조밀 레이더 → TI 수준 솎기)
  --vrig    가상 RGB 카메라 위치 무작위 (사람에서 1.5–5 m, 방위 ±70°, 높이 0.5–2.5 m)
출력 runs/ms4.

11-24 v4 = v3 + ① 삼각측량 Σ·유효성 (--trisig), ⑤ 좁은 기선 가상 카메라 (--vnarrow, 기선 8 cm–1.5 m 로그균등), 2D 큰 오차 꼬리 (--raytail) + ④ 라벨 어댑터 (--adapt, 데이터셋별 관례, 평균 0), 가상 광선은 공통 몸에서 (--vcanon), 평가에 보정 라벨 --calib 개 (_ex = 보정 피험자 뺀 0 라벨, _c = 보정 후). 출력 runs/ms4.
11-23 LHR 다중 센서 ② v3: 데이터셋 하나를 통째로 빼고 (--hold) 나머지 전체 + 모든 모달리티로 학습 → 뺀 데이터셋의 모든 입력으로 관절 (방 좌표 절대) 평가.
입력: dev/packs/ms_cache_v2/{mmfi,mri,mmbody,octonet,itop}.npz (ms_cache2.py). 모달리티: rgb 광선 · depth 점 (+색) · lidar 점 · radar 점 · wifi · imu (+ ① 삼각측량 토큰).
학습 방법:
- 모달리티 마스킹: 있는 모달리티마다 0.5 로 유지 (하나는 반드시).
- ⑤ 가상 RGB (--vrgb): RGB 없는 표본의 절반에 주 카메라 원점 (RGB 없으면 깊이 카메라) → 정답 관절 + 각도 잡음 (σ 4–12 mrad) 광선. 거리 다양한 데이터셋에서 RGB 를 보게 함 (정면 거리 암기 방지).
- ⑤ 가상 위치 이동 (--shift) · ⑤ 가상 카메라 2 (--vcam) · ① 삼각측량 토큰 (--tri), yaw ±15° · 좌우 반사 — 기하 보강을 받은 표본은 wifi · imu 를 가림 (방 고정 송수신기·몸 부착 센서와 기하가 어긋나므로).
- wifi · imu 는 학습 원천 통계로 표준화. 학습에 한 번도 없는 모달리티는 평가 'all' 에서 빼고 'all+unseen' 으로 따로 보고.
- 데이터셋 표본 가중 ∝ √프레임. 데이터셋 id 는 입력에 없음. 조기 종료 없음.
사용: python ms_train2.py --hold mri --out RUN [--seed 0] [--steps 20000]
11-24 추가: --train 원천 목록 (기본 = hold 를 뺀 나머지 전체). hold 가 원천에 있으면 --cv K --fold f 로 피험자 분할 (시험 = 정렬 피험자[f::K], 같은 데이터셋 기준선).
  --drop imu 처럼 모달리티를 학습·평가 모두에서 없는 것으로 처리. --evalprec fp32 = 평가를 bf16 없이.
  --load model.pt 는 학습 없이 평가만. 평가에 피험자별 MPJPE (per_sub) 와 골반 수평 1.1 s 평활 결과 (mpjpe_sm, pel_h_sm) 를 함께 기록."""
import sys, os, json, math, time, argparse, itertools
sys.path.insert(0, "/home/interns/unisense/code")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

ap = argparse.ArgumentParser(); ap.add_argument("--hold", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=192); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--evalstride", type=int, default=1)
for k in ("mask", "shift", "vcam", "tri", "vrgb"):
    ap.add_argument(f"--{k}", type=int, default=1)
ap.add_argument("--train", default=""); ap.add_argument("--cv", type=int, default=0); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--load", default="")
ap.add_argument("--drop", default=""); ap.add_argument("--evalprec", default="bf16", choices=["bf16", "fp32"]); ap.add_argument("--dump", type=int, default=0); ap.add_argument("--dumpall", type=int, default=0)
ap.add_argument("--adapt", type=int, default=1); ap.add_argument("--vcanon", type=int, default=1); ap.add_argument("--calib", type=int, default=100)
ap.add_argument("--cache", default="/home/interns/unisense/dev/packs/ms_cache_v3b"); ap.add_argument("--densfix", type=int, default=1); ap.add_argument("--test_subjects", default=""); ap.add_argument("--allow_mri", type=int, default=0); ap.add_argument("--anchor", type=int, default=1); ap.add_argument("--ajit", type=float, default=0.10); ap.add_argument("--band", type=int, default=1); ap.add_argument("--plucker", type=int, default=1)
ap.add_argument("--oracle", default="none", choices=["none", "anchor", "rays", "both"]); ap.add_argument("--evalsubs", default=""); ap.add_argument("--noval", type=int, default=0)   # 11-25 진단 전용 (정답을 평가 입력에 넣어 단계별 몫을 잼, 학습에는 안 씀)
ap.add_argument("--phys", default="pel_phys4"); ap.add_argument("--sigtok", type=int, default=1); ap.add_argument("--jmin", type=float, default=0.01); ap.add_argument("--jmax", type=float, default=0.25); ap.add_argument("--vview", type=float, default=0.25); ap.add_argument("--adlos", type=int, default=1); ap.add_argument("--solver", type=int, default=1); ap.add_argument("--snose", type=int, default=0); ap.add_argument("--sbone", type=float, default=0.0); ap.add_argument("--bonek", type=float, default=1.0); ap.add_argument("--dumpval", type=int, default=0); ap.add_argument("--sn", type=float, default=0.04); ap.add_argument("--sth", type=float, default=0.006)
ap.add_argument("--subdist", type=int, default=1); ap.add_argument("--p_mri", type=float, default=0.4); ap.add_argument("--p_cam", type=float, default=0.1); ap.add_argument("--keep", type=float, default=0.7); ap.add_argument("--dec2", type=int, default=1); ap.add_argument("--valsub", type=int, default=1); ap.add_argument("--valstride", type=int, default=2)
ap.add_argument("--nosolo", type=int, default=1); ap.add_argument("--dens", type=int, default=1); ap.add_argument("--vrig", type=int, default=1)
ap.add_argument("--trisig", type=int, default=1); ap.add_argument("--vnarrow", type=int, default=1); ap.add_argument("--raytail", type=float, default=0.05)   # ① 삼각측량 Σ·유효성, ⑤ 좁은 기선 가상 카메라, 2D 큰 오차 꼬리   # ④ 라벨 어댑터, 가상 RGB 를 공통 몸에서, 평가 보정 라벨 수
A = ap.parse_args(); DROP = set(filter(None, A.drop.split(","))); PP = importlib.import_module(A.phys)
C = A.cache; OUT = f"/home/interns/unisense/runs/ms9/{A.out}"; os.makedirs(OUT, exist_ok=True)
DSALL = ["mmfi", "mri", "mmbody", "octonet", "itop"]; assert A.hold in DSALL
torch.manual_seed(A.seed); np.random.seed(A.seed); dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
LOG = open(f"{OUT}/train.log", "a")
MODS = ["rgb", "depth", "lidar", "radar", "wifi", "imu"]


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


J_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
TZ = torch.tensor([1.60, 1.40, 1.40, 1.10, 1.10, 0.85, 0.85, 0.90, 0.90, 0.48, 0.48, 0.08, 0.08]); TL = torch.tensor([0.0, 0.17, -0.17, 0.20, -0.20, 0.20, -0.20, 0.11, -0.11, 0.11, -0.11, 0.11, -0.11])
BASE = torch.stack([torch.full((13,), 3.0), TL, TZ], -1)
KEYS = ("ray", "dep", "dv", "lid", "lv", "rad", "rv", "wifi", "wv", "imu", "iv", "org", "y", "ym")


def load(names):
    Ds = [np.load(f"{C}/{n}.npz", allow_pickle=False) for n in names]
    D = {k: np.concatenate([d[k] for d in Ds]) for k in KEYS + ("ds", "sub", "seq", "t", "act")}
    return D


def to_dev(D, norm):
    T = {}
    for k in KEYS:
        x = torch.from_numpy(D[k]); T[k] = x.to(dev) if x.dtype == torch.bool else x.to(dev, torch.float16 if k not in ("y", "org") else torch.float32)
    T["wifi"] = ((T["wifi"].float() - norm["wm"]) / norm["ws"]).half(); T["imu"] = ((T["imu"].float() - norm["im"]) / norm["is"]).half()
    return T


LS = torch.tensor([2.0, 1.0, 0.5, 0.25, 0.125, 0.0625] if A.band else [8.0, 4.0, 2.0, 1.0, 0.5, 0.25], device=dev)
FUSE = list(PP.COMPS)   # 감사 M1: v3 LiDAR 정합 고침 → LiDAR 도 기준점 결합에 포함


def anchor_of(B_, M, full=False):
    """① 물리 골반 기준점 (pel_phys 2회 풀이, 가린 뒤 입력만). 반환 (B,3) = (a_x, a_y, 0). full: (a, Σ (B,3,3), 성분 수 (B,))."""
    with torch.autocast("cuda", enabled=False), torch.no_grad():
        ray = B_["ray"].float(); args = (ray, M["rgb"], B_["dep"].float()[..., :3], M["depth"], B_["lid"].float(), M["lidar"], B_["rad"].float(), M["radar"], B_["org"].float())
        n = ray.shape[0]; a, _, _ = PP.solve(PP.components(*args), FUSE, n, ray.device); a, S, k = PP.solve(PP.components(*args, rguess=a), FUSE, n, ray.device)
        a = torch.nan_to_num(a); a[:, 2] = 0
    return (a, torch.nan_to_num(S, nan=1.0), k) if full else a


def four(x):
    a = 2 * math.pi * x[..., None] / LS
    return torch.cat([x, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], -1)


def triangulate(ray, rmask):
    with torch.autocast("cuda", enabled=False):
        r = ray.float(); o, d = r[..., :3], F.normalize(r[..., 3:6], dim=-1); w = rmask.float()[..., None, None]   # fp16 캐시 방향 다시 정규화
        Pm = torch.eye(3, device=r.device) - d[..., :, None] * d[..., None, :]
        Am = (w * Pm).sum(1); bv = (w * (Pm @ o[..., None])).sum(1)[..., 0]
        x = torch.linalg.solve(Am + 1e-6 * torch.eye(3, device=r.device), bv); ok = rmask.sum(1) >= 2
        sang = torch.linalg.norm(torch.cross(d[:, 0], d[:, 1], dim=-1), dim=-1) * ok
        dist = torch.linalg.norm(x - o[:, 0], dim=-1); lsig = torch.zeros_like(dist)
        if A.trisig:                                                            # ① 깊이 σ = σ⊥ / sin각 (σ⊥ = 거리의 0.6 %), 카메라 뒤·8 m 밖·sin각 < 0.02 는 버림
            front = (((x[:, None] - o) * d).sum(-1) > 0.3).all(1); ok = ok & front & (dist < 8.0) & (sang > 0.02)
            lsig = torch.log(0.006 * dist.clamp(0.3, 8.0) / sang.clamp(min=0.02)) * ok; sang = sang * ok
        x = torch.where(ok[..., None], x, torch.zeros_like(x))
    return x, ok, sang, lsig


class Net(nn.Module):
    def __init__(s, d=256, L=4, Ld=3, H=8):
        super().__init__()
        mlp = lambda i: nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, d))
        s.ray_in, s.dep_in, s.lid_in, s.rad_in, s.wifi_in, s.imu_in, s.tri_in = mlp(3 + 39 + 1 + 13 if A.plucker else 4 * 39 + 13), mlp(39 if A.nosolo else 39 + 4), mlp(39), mlp(39 + 2 if A.nosolo else 39 + 3), mlp(513), mlp(13 + 6), mlp(39 + 13 + 1 + (1 if A.trisig else 0))
        s.typ = nn.Parameter(torch.zeros(8 if A.sigtok else 7, d)); s.sig_in = mlp(4) if A.sigtok else None   # v8full: 기준점 Σ 토큰 (종류 7)
        s.q = nn.Parameter(torch.randn(13, d) * 0.02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, H, 2 * d, 0.1, batch_first=True, norm_first=True), L)
        s.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, H, 2 * d, 0.1, batch_first=True, norm_first=True), Ld)
        s.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))
        if A.dec2:                                                                   # v8: 질의 위치·층별 잔차 머리 (0 초기화 → 처음엔 x0 그대로)
            s.qpe = mlp(39); s.qctx = nn.Linear(d, d); s.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3)) for _ in range(Ld)])
            for hd in s.heads:
                nn.init.zeros_(hd[1].weight); nn.init.zeros_(hd[1].bias)
        s.register_buffer("base", BASE.clone() - (torch.tensor([3.0, 0.0, 0.0]) if A.anchor else 0.0)); s.register_buffer("eye", torch.eye(13)); s.register_buffer("eye6", torch.eye(6))

    def forward(s, B_, M):
        ray, dep, lid, rad, wifi, imu = (B_[k].float() for k in ("ray", "dep", "lid", "rad", "wifi", "imu")); B = ray.shape[0]
        if A.sigtok:                                                               # v8full: 기준점 Σ 를 시선·옆·높이 σ 로, 흔들기도 그 σ 에 비례
            a, Sa, ka = anchor_of(B_, M, True); hz = torch.tensor([1.0, 1.0, 0.0], device=a.device)
            uq = F.normalize((a - B_["org"].float()) * hz, dim=-1); wq = torch.stack([-uq[:, 1], uq[:, 0], torch.zeros_like(uq[:, 0])], -1)
            sq = lambda v: torch.sqrt(torch.einsum("bi,bij,bj->b", v, Sa, v).clamp(min=1e-6)); sl, sw, sz = sq(uq), sq(wq), Sa[:, 2, 2].clamp(min=1e-6).sqrt()
            if s.training and A.ajit > 0:
                a = a + uq * (torch.randn(B, 1, device=a.device) * sl.clamp(A.jmin, A.jmax)[:, None]) + wq * (torch.randn(B, 1, device=a.device) * sw.clamp(A.jmin, A.jmax)[:, None])
            tsig = s.sig_in(torch.stack([torch.log(sl), torch.log(sw), torch.log(sz), ka / 10.0], -1))[:, None] + s.typ[7]
        else:
            a = anchor_of(B_, M) if A.anchor else torch.zeros(B, 3, device=ray.device)
        if A.anchor and s.training and A.ajit > 0 and not A.sigtok:                # 기준점 흔들기 (수평만)
            a = a + torch.randn_like(a) * A.ajit * torch.tensor([1.0, 1.0, 0.0], device=a.device)
            u_ = F.normalize((a - B_["org"].float()) * torch.tensor([1.0, 1.0, 0.0], device=a.device), dim=-1); a = a + u_ * torch.randn(B, 1, device=a.device) * A.ajit   # 감사 C2: 시선 방향 추가 흔들기
        if "ga" in B_:                                                             # 진단: 기준점 = 정답 골반 (평가 전용)
            a = B_["ga"].float()
        r = ray.reshape(B, -1, 7); o, dd = r[..., :3] - a[:, None], r[..., 3:6]; oh = s.eye.repeat(ray.shape[1], 1)[None].expand(B, -1, -1)
        if A.plucker:                                                              # 기준점 기준 Plücker: 방향 + 모멘트 (o×d) + log 기준점 깊이
            tst = (-(o * dd).sum(-1, keepdim=True)).clamp(min=0.3)
            tr = s.ray_in(torch.cat([dd, four(torch.cross(o, dd, dim=-1)), torch.log(tst), oh], -1)) + s.typ[0]
        else:
            tr = s.ray_in(torch.cat([torch.cat([four(o + sd * dd) for sd in (2.0, 3.0, 4.0, 5.0)], -1), oh], -1)) + s.typ[0]
        dp = dep[..., :3] - a[:, None]; col = dep[..., 3:6]
        td = s.dep_in(four(dp) if A.nosolo else torch.cat([four(dp), col, (col.sum(-1, keepdim=True) > 0).float()], -1)) + s.typ[1]
        tl = s.lid_in(four(lid - a[:, None])) + s.typ[2]; rp = rad[..., :3] - a[:, None]
        tq = s.rad_in(torch.cat([four(rp), rad[..., 3:5] if A.nosolo else rad[..., 3:6]], -1)) + s.typ[3]
        tw = s.wifi_in(wifi)[:, None] + s.typ[4]; ti = s.imu_in(torch.cat([imu, s.eye6[None].expand(B, -1, -1)], -1)) + s.typ[5]
        x, ok, sang, lsig = triangulate(ray, M["rgb"]); tt = s.tri_in(torch.cat([four(torch.where(ok[..., None], x - a[:, None], x)), s.eye[None].expand(B, -1, -1), sang[..., None]] + ([lsig[..., None]] if A.trisig else []), -1)) + s.typ[6]
        tok = torch.cat([tr, td, tl, tq, tw, ti, tt] + ([tsig] if A.sigtok else []), 1)
        m = torch.cat([M["rgb"].reshape(B, -1), M["depth"], M["lidar"], M["radar"], M["wifi"][:, None], M["imu"][:, None].expand(-1, 6), ok & (A.tri > 0)] + ([torch.ones(B, 1, dtype=torch.bool, device=ray.device)] if A.sigtok else []), 1)
        empty = ~m.any(1)
        if empty.any():
            m = m.clone(); m[empty, 0] = True
        h = s.enc(tok, src_key_padding_mask=~m)
        if A.dec2:
            hm = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            t = s.q[None].expand(B, -1, -1) + s.qctx(hm)[:, None]; x = s.base[None].expand(B, -1, -1) + a[:, None]; outs = []
            for l, layer in enumerate(s.dec.layers):
                xd = x.detach(); t = layer(t + s.qpe(four(xd - a[:, None])), h, memory_key_padding_mask=~m); x = xd + s.heads[l](t).float(); outs.append(x)
            return torch.stack(outs, 0) if s.training else outs[-1]
        return s.base + s.head(s.dec(s.q[None].expand(B, -1, -1), h, memory_key_padding_mask=~m)) + a[:, None]


PAR = {0: None, 3: 1, 4: 2, 5: 3, 6: 4, 9: 7, 10: 8, 11: 9, 12: 10}                # 뼈 방향 항을 갖는 관절 ← 부모 (None = 어깨 중앙)
MIR = torch.tensor([1.0, -1.0, 1.0])


def body_R(Q):
    """몸 좌표계 (행 = 축): y = 엉덩이 축 (수평), z = 위, x = y × z."""
    h = Q[:, 8] - Q[:, 7]; yv = F.normalize(torch.stack([h[:, 0], h[:, 1], torch.zeros_like(h[:, 0])], -1), dim=-1, eps=1e-6)
    zv = torch.zeros_like(yv); zv[:, 2] = 1
    return torch.stack([torch.cross(yv, zv, dim=-1), yv, zv], 1)


def bone_u(Q):
    U = torch.zeros_like(Q)
    for j, pj in PAR.items():
        U[:, j] = F.normalize(Q[:, j] - (Q[:, [1, 2]].mean(1) if pj is None else Q[:, pj]), dim=-1, eps=1e-6)
    return U


def adapt_apply(P, O, a):
    """공통 몸 P → 라벨 관례: + 몸 좌표계 오프셋 O (B,13,3) + a (B,13) × 뼈 방향. 좌표계·뼈 방향은 P 에서 (기울기 막음)."""
    Q = P.detach()
    return P + torch.einsum("bic,bki->bkc", body_R(Q), O) + a[..., None] * bone_u(Q)


class Adapter(nn.Module):
    """④ 데이터셋별 라벨 관례. 좌우 대칭으로 묶고, 데이터셋 평균을 0 으로 (공통 몸 = 평균 관례). 데이터셋 id 는 출력 쪽에만 씀."""
    def __init__(s, nd):
        super().__init__(); s.O = nn.Parameter(torch.zeros(nd, 13, 3)); s.a = nn.Parameter(torch.zeros(nd, 13)); s.dl = nn.Parameter(torch.zeros(nd)) if A.adlos else None
        s.register_buffer("am", torch.tensor([1.0 if j in PAR else 0.0 for j in range(13)])); s.register_buffer("mir", MIR.clone())

    def eff(s):
        O = 0.5 * (s.O + s.O[:, J_SWAP] * s.mir); a = 0.5 * (s.a + s.a[:, J_SWAP]) * s.am
        return O - O.mean(0, keepdim=True), a - a.mean(0, keepdim=True)

    def forward(s, P, dsi, org=None):
        O, a = s.eff(); out = adapt_apply(P, O[dsi], a[dsi])
        if s.dl is not None and org is not None:                                   # v8full: 라벨이 라벨 카메라 시선을 따라 앞뒤로 밀린 관례 (평균 0)
            u = F.normalize((P.detach()[:, [7, 8]].mean(1) - org) * torch.tensor([1.0, 1.0, 0.0], device=P.device), dim=-1)
            out = out + (s.dl - s.dl.mean())[dsi][:, None, None] * u[:, None]
        return out


def fit_adapter_np(P, G):
    """평가용: 보정 라벨 (P 공통 몸, G 정답) 로 관절별 [I3 | 뼈방향] 최소제곱 → (O 13×3, a 13). 망은 그대로."""
    Pt = torch.from_numpy(P).float(); Rb = body_R(Pt).numpy(); U = bone_u(Pt).numpy(); E = np.einsum("nab,njb->nja", Rb, G - P)
    Ub = np.einsum("nab,njb->nja", Rb, U); O = np.zeros((13, 3)); a = np.zeros(13)
    for j in range(13):
        X = np.concatenate([np.eye(3)[None].repeat(len(P), 0), Ub[:, j, :, None] * (j in PAR)], 2).reshape(-1, 4)
        w = np.linalg.lstsq(X, E[:, j].reshape(-1), rcond=None)[0]; O[j], a[j] = w[:3], w[3]
    return O, a


def adapt_np(P, O, a):
    Pt = torch.from_numpy(P).float(); n = len(P)
    return adapt_apply(Pt, torch.from_numpy(O).float()[None].expand(n, -1, -1), torch.from_numpy(a).float()[None].expand(n, -1)).numpy().astype(float)


def batch(T, idx):
    return {k: T[k][idx] for k in KEYS + tuple(x for x in ("dsi", "ga") if x in T)}


def avail(B_):
    d = dict(rgb=B_["ray"][..., 6] > 0, depth=B_["dv"], lidar=B_["lv"], radar=B_["rv"], wifi=B_["wv"], imu=B_["iv"])
    for k in DROP:
        d[k] = torch.zeros_like(d[k])
    return d


def tail(n):
    """⑤ 2D 검출 실패 흉내: 관절마다 확률 raytail 로 5–30 cm 3D 잡음."""
    if A.raytail <= 0:
        return 0.0
    m = (torch.rand(n, 13, 1, device=dev) < A.raytail).float()
    return m * F.normalize(torch.randn(n, 13, 3, device=dev), dim=-1) * (0.05 + 0.25 * torch.rand(n, 13, 1, device=dev))


def augment(B_):
    """기하 보강 (표본 절반): yaw · 반사 · 위치 이동 → 이 표본은 wifi · imu 가림. 그다음 가상 RGB · 가상 카메라 2."""
    B_ = {k: v.clone() for k, v in B_.items()}; n = B_["y"].shape[0]; geo = torch.rand(n, device=dev) < 0.5
    ray, dep, lid, rad, y, ym, org = B_["ray"].float(), B_["dep"].float(), B_["lid"].float(), B_["rad"].float(), B_["y"], B_["ym"], B_["org"]
    th = (torch.rand(n, device=dev) * 2 - 1) * math.radians(15) * geo; c, s_ = torch.cos(th), torch.sin(th)
    R = torch.zeros(n, 3, 3, device=dev); R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s_, s_, c, 1
    flip = geo & (torch.rand(n, device=dev) < 0.5); Fm = torch.ones(n, 3, device=dev); Fm[flip, 1] = -1
    rot = lambda X: torch.einsum("bij,b...j->b...i", R, X * Fm.view(n, *([1] * (X.dim() - 2)), 3))
    ray[..., :3] = rot(ray[..., :3]); ray[..., 3:6] = rot(ray[..., 3:6]); dep[..., :3] = rot(dep[..., :3]); lid = rot(lid); rad[..., :3] = rot(rad[..., :3]); y = rot(y); org = rot(org[:, None])[:, 0]
    if flip.any():
        ray[flip] = ray[flip][:, :, J_SWAP]; y[flip] = y[flip][:, J_SWAP]; ym = ym.clone(); ym[flip] = ym[flip][:, J_SWAP]
    if A.shift:
        sh = geo & (torch.rand(n, device=dev) < 0.7); De = torch.zeros(n, 3, device=dev); pel = y[:, [7, 8]].mean(1)
        De[:, 0] = torch.maximum(torch.rand(n, device=dev) * 2.0 - 1.2, 1.3 - pel[:, 0]); De[:, 1] = torch.rand(n, device=dev) * 1.2 - 0.6; De = De * sh[:, None]
        o, d = ray[..., :3], ray[..., 3:6]; Y = y[:, None]; w = Y - o; e = o + (w * d).sum(-1, keepdim=True) * d - Y
        Yn = Y + De[:, None, None]; sc = torch.linalg.norm(Yn - o, dim=-1, keepdim=True) / torch.linalg.norm(w, dim=-1, keepdim=True).clamp(min=0.1)
        dn = F.normalize(Yn + e * sc - o, dim=-1); vv = (ray[..., 6] > 0) & ym[:, None]
        ray[..., 3:6] = torch.where(sh[:, None, None, None], dn, d); ray[..., 6] = torch.where(sh[:, None, None], vv.float(), ray[..., 6])
        dep[..., :3] = dep[..., :3] + De[:, None]; lid = lid + De[:, None] * B_["lv"][..., None]; rad[..., :3] = rad[..., :3] + De[:, None] * (rad[..., 5:6] < 0.5); y = y + De[:, None]
    yc = y
    if A.vcanon and adp is not None and "dsi" in B_:                            # 가상 광선은 라벨 관례를 뺀 공통 몸에서
        with torch.no_grad():
            yc = y - (adp(y, B_["dsi"], org.float()) - y)
    if A.vrgb:                                                                  # ⑤ 가상 RGB: RGB 없는 표본 절반, 주 카메라 원점 → 관절 + 각도 잡음
        no = ~(ray[..., 6] > 0).any((1, 2)); vv = (~no) & (torch.rand(n, device=dev) < A.vview) if (A.vview > 0 and A.vrig) else torch.zeros_like(no); vr = (no & (torch.rand(n, device=dev) < 0.5)) | vv
        if vr.any():
            oc = org.clone()
            if A.vrig:                                                             # ⑤ 가상 카메라 위치 무작위 (표본 70 %): 사람 골반에서 거리·방위·높이
                pel = yc[:, [7, 8]].mean(1); base_az = torch.atan2(org[:, 1] - pel[:, 1], org[:, 0] - pel[:, 0])
                az = base_az + (torch.rand(n, device=dev) * 2 - 1) * math.radians(70); D = 1.5 + 3.5 * torch.rand(n, device=dev)
                cand = torch.stack([pel[:, 0] + D * torch.cos(az), pel[:, 1] + D * torch.sin(az), 0.5 + 2.0 * torch.rand(n, device=dev)], -1)
                oc = torch.where(((torch.rand(n, device=dev) < 0.7) | vv)[:, None], cand, org)
            dist = torch.linalg.norm(yc - oc[:, None], dim=-1, keepdim=True); sig = (0.004 + 0.008 * torch.rand(n, 1, 1, device=dev)) * dist
            p = yc + torch.randn_like(yc) * sig + tail(n); d1 = F.normalize(p - oc[:, None], dim=-1)
            new = torch.cat([oc[:, None].expand(-1, 13, -1), d1, ym.float()[..., None]], -1); ray[:, 0] = torch.where(vr[:, None, None], new, ray[:, 0])
            if vv.any():                                                           # v8full: 실측 RGB 를 가상 시점으로 바꾼 표본은 둘째 카메라도 비움 (가상 둘째 카메라는 아래 vcam 이 다시 만듦)
                ray[:, 1, :, 6] = torch.where(vv[:, None], torch.zeros_like(ray[:, 1, :, 6]), ray[:, 1, :, 6])
    if A.vcam:
        v1 = ray[:, 0, :, 6] > 0; vc = (torch.rand(n, device=dev) < 0.5) & v1.any(1) & ~(ray[:, 1, :, 6] > 0).any(1)
        if vc.any():
            o1, d1 = ray[:, 0, :, :3], ray[:, 0, :, 3:6]; w1 = yc - o1; e1 = torch.linalg.norm(o1 + (w1 * d1).sum(-1, keepdim=True) * d1 - yc, dim=-1).clamp(0.005, 0.1)
            b = torch.stack([torch.rand(n, device=dev) * 0.6 - 0.3, torch.sign(torch.rand(n, device=dev) - 0.5) * ((0.08 * (1.5 / 0.08) ** torch.rand(n, device=dev)) if A.vnarrow else (0.2 + 1.3 * torch.rand(n, device=dev))), torch.rand(n, device=dev) * 0.6 - 0.3], -1)
            e1v = o1 + (w1 * d1).sum(-1, keepdim=True) * d1 - yc; rho = 0.95 * torch.rand(n, 1, 1, device=dev)          # 감사 M3: 두 카메라 검출 오차 상관 (같은 검출기)
            o2 = o1[:, :1] + b[:, None]; p2 = yc + rho * e1v + torch.sqrt(1 - rho ** 2) * torch.randn_like(yc) * e1[..., None] / math.sqrt(2) + tail(n); d2 = F.normalize(p2 - o2, dim=-1)
            new = torch.cat([o2.expand(-1, 13, -1), d2, (v1 & ym).float()[..., None]], -1); ray[:, 1] = torch.where(vc[:, None, None], new, ray[:, 1])
    if A.dens:                                                                     # 점 밀도 무작위화 (가리기만, 반복 없음)
        for k in ("dv", "lv"):
            keep = 0.15 + 0.85 * torch.rand(n, 1, device=dev); on = torch.rand(n, 1, device=dev) < 0.5
            B_[k] = B_[k] & ~(on & (torch.rand_like(B_[k].float()) > keep))
        nr = B_["rv"].sum(1, keepdim=True).float()
        if A.densfix:                                                            # 11-24 고침: 캐시 상한 96점이라 '>100' 조건이 작동 안 했음 → 기기 무관하게 표본 절반을 점 8–60 개로 솎음
            tgt = (8 + 52 * torch.rand(n, 1, device=dev)) / nr.clamp(min=1); on = torch.rand(n, 1, device=dev) < 0.5
        else:                                                                    # ms_train4 그대로 (비교용)
            dense = nr > 100; tgt = torch.where(dense & (torch.rand(n, 1, device=dev) < 0.5), (20 + 80 * torch.rand(n, 1, device=dev)) / nr.clamp(min=1), 0.3 + 0.7 * torch.rand(n, 1, device=dev))
            on = dense | (torch.rand(n, 1, device=dev) < 0.3)
        B_["rv"] = B_["rv"] & ~(on & (torch.rand_like(B_["rv"].float()) > tgt))
    B_.update(ray=ray, dep=dep, lid=lid, rad=rad, y=y, ym=ym, org=org)
    B_["wv"] = B_["wv"] & ~geo; B_["iv"] = B_["iv"] & ~geo
    return B_


def train_masks(B_, seen):
    av = avail(B_); n = B_["y"].shape[0]; has = torch.stack([av[k].reshape(n, -1).any(1) & seen[k] for k in MODS], 1)
    kp = (torch.rand(n, len(MODS), device=dev) < (A.keep if A.subdist else 0.5)) & has if A.mask else has.clone()
    if A.mask and A.subdist:                                                       # v8: 시험형 부분집합을 직접 뽑음
        u = torch.rand(n, device=dev); ir, iq = MODS.index("rgb"), MODS.index("radar")
        ml = (u < A.p_mri) & has[:, ir]; co = (u >= A.p_mri) & (u < A.p_mri + A.p_cam) & has[:, ir]
        kp[ml | co] = False; kp[ml | co, ir] = True; kp[ml, iq] = has[ml, iq]
    none = ~kp.any(1)
    if none.any():
        pick = torch.multinomial(has[none].float() + 1e-6, 1)[:, 0]; kp[none] = F.one_hot(pick, len(MODS)).bool() & has[none]
    M = {}
    for j, k in enumerate(MODS):
        M[k] = av[k] & kp[:, j].view(n, *([1] * (av[k].dim() - 1)))
    return M


def subset_masks(B_, keep):
    av = avail(B_); M = {}
    for k in MODS:
        M[k] = av[k] if (k in keep or (k == "rgb" and "rgb1" in keep)) else torch.zeros_like(av[k])
    if "rgb1" in keep:
        M["rgb"] = M["rgb"].clone(); M["rgb"][:, 1] = False
    return M


def smooth_root(P, seq, t, win=1.1):
    """골반 수평 (x, y) 을 시퀀스 안에서 ±win/2 s 이동 평균으로 바꾸고 모든 관절을 같은 만큼 옮김 (11-22 채택 지식, 사후)."""
    Ps = P.copy(); pel = P[:, [7, 8]].mean(1)
    for q in np.unique(seq):
        ii = np.flatnonzero(seq == q); ii = ii[np.argsort(t[ii], kind="stable")]; tt = t[ii]
        cs = np.r_[np.zeros((1, 2)), np.cumsum(pel[ii, :2], 0)]; lo = np.searchsorted(tt, tt - win / 2, "left"); hi = np.searchsorted(tt, tt + win / 2, "right")
        Ps[ii, :, :2] += ((cs[hi] - cs[lo]) / (hi - lo)[:, None] - pel[ii, :2])[:, None]
    return Ps


def pa(P, G):
    P0, G0 = P - P.mean(1, keepdims=True), G - G.mean(1, keepdims=True); out = []
    for p, g in zip(P0, G0):
        U_, S, Vt = np.linalg.svd(p.T @ g); Dg = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U_.T))]); R = Vt.T @ Dg @ U_.T; sc = (S * np.diag(Dg)).sum() / (p ** 2).sum()
        out.append(np.linalg.norm(sc * p @ R.T - g, axis=-1).mean())
    return np.array(out)


# ---------- 학습 (기본: hold 를 뺀 나머지 전체. hold 가 원천에 있으면 피험자 분할)
train_ds = A.train.split(",") if A.train else [d for d in DSALL if d != A.hold]; inside = A.hold in train_ds
assert "mri" not in train_ds or A.allow_mri, "mRI 는 봉인 시험 세트: 학습 금지 (참고선은 --allow_mri 1)"
assert all(d in DSALL for d in train_ds) and (not inside or A.cv >= 2 or A.test_subjects)
test_s = (A.test_subjects.split(",") if A.test_subjects else [str(x) for x in sorted(set(np.load(f"{C}/{A.hold}.npz", allow_pickle=False)["sub"]))[A.fold::A.cv]]) if inside else []
net = Net().to(dev); adp = None
if A.load:
    ck = torch.load(A.load, map_location=dev, weights_only=False); net.load_state_dict(ck["state"])
    norm = {k: v.to(dev) for k, v in ck["norm"].items()}; seen = {k: torch.tensor(v, device=dev) for k, v in ck["seen"].items()}; VAL = ck.get("val", {}); train_ds = ck.get("train_ds", train_ds)
    if ck.get("adapter") is not None and A.adapt:
        adp = Adapter(len(train_ds)).to(dev); adp.load_state_dict(ck["adapter"])
    log("load", A.load, "hold", A.hold, "학습에 있던 모달리티", ck["seen"], "원래 args", ck["args"])
else:
    D = load(train_ds)
    if inside:
        keep = ~((D["ds"] == A.hold) & np.isin(D["sub"], test_s)); D = {k: v[keep] for k, v in D.items()}; log("피험자 분할: 시험", test_s)
    VAL = {}
    if A.valsub:                                                                   # v8: 학습 데이터셋마다 피험자 [2::5] 를 뺌
        vk = np.zeros(len(D["y"]), bool)
        for d_ in train_ds:
            if inside and d_ == A.hold:
                continue
            VAL[d_] = [str(x) for x in sorted(set(D["sub"][D["ds"] == d_].tolist()))[2::5]]; vk |= (D["ds"] == d_) & np.isin(D["sub"].astype(str), VAL[d_])
        D = {k: v[~vk] for k, v in D.items()}; log("v8 검증 피험자 (학습에서 뺌)", VAL)
    n0 = len(D["y"])
    wv, iv = D["wv"], D["iv"]
    norm = dict(wm=torch.tensor(D["wifi"][wv].astype(np.float32).mean(0) if wv.any() else np.zeros(513, np.float32), device=dev),
                ws=torch.tensor(D["wifi"][wv].astype(np.float32).std(0) + 1e-3 if wv.any() else np.ones(513, np.float32), device=dev),
                im=torch.tensor(D["imu"][iv].astype(np.float32).mean(0) if iv.any() else np.zeros((6, 13), np.float32), device=dev),
                **{"is": torch.tensor(D["imu"][iv].astype(np.float32).std(0) + 1e-3 if iv.any() else np.ones((6, 13), np.float32), device=dev)})
    seen = {k: torch.tensor(bool(v), device=dev) for k, v in dict(rgb=(D["ray"][..., 6] > 0).any() or A.vrgb, depth=D["dv"].any(), lidar=D["lv"].any(), radar=D["rv"].any(), wifi=wv.any(), imu=iv.any()).items()}
    cnt = {d: int((D["ds"] == d).sum()) for d in train_ds}; w = np.array([math.sqrt(cnt[d]) / cnt[d] for d in D["ds"]]); w = torch.from_numpy(w / w.sum()).float().to(dev)
    dsi = np.array([train_ds.index(x) for x in D["ds"]]); T = to_dev(D, norm); T["dsi"] = torch.from_numpy(dsi).to(dev); del D
    log("hold", A.hold, "학습", cnt, "학습에 있는 모달리티", {k: bool(v) for k, v in seen.items()}, "args", vars(A))
    if A.adapt:
        adp = Adapter(len(train_ds)).to(dev)
    opt = torch.optim.AdamW([{"params": list(net.parameters())}] + ([{"params": list(adp.parameters()), "weight_decay": 0.0 if A.adlos else 0.05}] if adp is not None else []), lr=A.lr, weight_decay=0.05)
    sch = lambda it: min(1.0, (it + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(it, A.steps) / A.steps)); t0 = time.time(); run = 0.0
    for it in range(A.steps):
        for g in opt.param_groups:
            g["lr"] = A.lr * sch(it)
        idx = torch.multinomial(w, A.bs, replacement=True); B_ = augment(batch(T, idx)); M = train_masks(B_, seen); ym = B_["ym"]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            PL = net(B_, M).float()
        PL = PL if PL.dim() == 4 else PL[None]; tot = 0.0; okp = ym[:, 7] & ym[:, 8]
        for P in PL:                                                               # v8: 층마다 같은 손실 (마지막 층 값이 기록됨)
            if adp is not None:
                P = adp(P, B_["dsi"], B_["org"].float())
            e = torch.linalg.norm(P - B_["y"], dim=-1); loss = (e * ym).sum() / ym.sum().clamp(min=1)
            lp = (torch.linalg.norm(P[:, [7, 8]].mean(1) - B_["y"][:, [7, 8]].mean(1), dim=-1) * okp).sum() / okp.sum().clamp(min=1); tot = tot + (loss + lp) / len(PL)
        opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        if it % 1000 == 0 or it == A.steps - 1:
            log(f"it {it} loss(MPJPE) {run * 1000:.1f} mm, pelvis {lp.item() * 1000:.1f} mm, {time.time() - t0:.0f} s")
    if adp is not None:
        Oe, ae = adp.eff()
        for i, d in enumerate(train_ds):
            log("④", d, "시선 이동 mm", round(float((adp.dl - adp.dl.mean())[i]) * 1000, 1) if adp.dl is not None else None, "|O| mm", [round(float(v) * 1000) for v in torch.linalg.norm(Oe[i], dim=-1)], "a mm", [round(float(v) * 1000) for v in ae[i]])
    torch.save(dict(state=net.state_dict(), adapter=adp.state_dict() if adp is not None else None, train_ds=train_ds, norm={k: v.cpu() for k, v in norm.items()}, seen={k: bool(v) for k, v in seen.items()}, args=vars(A), val=VAL), f"{OUT}/model.pt"); del T; torch.cuda.empty_cache()

# ---------- 평가: 뺀 데이터셋 전체 (+ v8: 학습 데이터셋의 검증 피험자 = 아는 데이터셋의 새 사람)
net.eval()
R = dict(anchor=A.anchor, ajit=A.ajit, band=A.band, plucker=A.plucker, nosolo=A.nosolo, dens=A.dens, vrig=A.vrig, adapt=A.adapt, vcanon=A.vcanon, trisig=A.trisig, vnarrow=A.vnarrow, raytail=A.raytail, calib_n=A.calib,
         subdist=A.subdist, p_mri=A.p_mri, p_cam=A.p_cam, keep=A.keep, dec2=A.dec2, valsub=A.valsub, val_subjects=VAL, hold=A.hold, train=train_ds, test_subjects=test_s, load=A.load, drop=sorted(DROP), evalprec=A.evalprec, results={}, val={})


SBONES = [(1, 3), (2, 4), (3, 5), (4, 6), (7, 9), (8, 10), (9, 11), (10, 12), (1, 2), (7, 8), (1, 7), (2, 8)]


def ray_solve(P, ray, cams, seq=None):
    """③ 풀이기 (학습 없음): 관절마다 망 출력 사전 N(P, σn² I) + 카메라 광선 (광선에 수직인 두 방향만, σ = σθ · 거리) 정보 합.
    --snose 0: 머리 (관절 0) 는 광선 조건에서 뺌 (머리 관절 정의가 데이터셋마다 다름: 학습 어댑터의 코 오프셋이 가장 큼).
    --sbone σL > 0: 모든 관절 동시 풀이에 뼈 길이 항 (|Xc−Xp| − L)²/σL² 추가, L = 녹화별 망 예측 뼈 길이 중앙값 × bonek (bonek 는 학습 데이터셋 검증 피험자에서 잰 망의 뼈 길이 치우침)."""
    if not cams:
        return P
    if A.sbone > 0 and seq is not None:
        o_ = ray[..., :3].astype(float); d_ = ray[..., 3:6].astype(float); d_ = d_ / np.maximum(np.linalg.norm(d_, axis=-1, keepdims=True), 1e-9); v_ = (ray[..., 6] > 0).astype(float)
        msk = np.zeros(2); msk[cams] = 1; v_ = v_ * msk[None, :, None]
        if not A.snose:
            v_[:, :, 0] = 0
        Pm_ = np.eye(3)[None, None, None] - d_[..., :, None] * d_[..., None, :]; wr = v_ / (A.sth * np.linalg.norm(P[:, None] - o_, axis=-1).clip(0.5, 8)) ** 2
        L = np.zeros((len(P), len(SBONES)))
        for k, (a, b) in enumerate(SBONES):
            l = np.linalg.norm(P[:, b] - P[:, a], axis=1)
            for q in np.unique(seq):
                m = seq == q; L[m, k] = np.median(l[m]) * A.bonek
        X = P.copy(); lr = 0.5 / (1 / A.sn ** 2 + wr.sum(1).max() + 4 / A.sbone ** 2)
        for _ in range(400):
            g = (X - P) / A.sn ** 2 + (wr[..., None] * np.einsum("ncjab,ncjb->ncja", Pm_, X[:, None] - o_)).sum(1)
            for k, (a, b) in enumerate(SBONES):
                dv = X[:, b] - X[:, a]; ln = np.linalg.norm(dv, axis=1, keepdims=True).clip(1e-6); f = ((ln - L[:, k:k + 1]) / A.sbone ** 2) * dv / ln; g[:, b] += f; g[:, a] -= f
            X -= lr * g
        return X
    L = np.broadcast_to(np.eye(3) / A.sn ** 2, P.shape[:2] + (3, 3)).copy(); e = (P / A.sn ** 2)[..., None]
    for c in cams:
        o, d, v = ray[:, c, :, :3].astype(float), ray[:, c, :, 3:6].astype(float), ray[:, c, :, 6] > 0; d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        if not A.snose:
            v = v.copy(); v[:, 0] = False
        sr = A.sth * np.linalg.norm(P - o, axis=-1).clip(0.5, 8.0); Lc = (np.eye(3) - d[..., :, None] * d[..., None, :]) / sr[..., None, None] ** 2 * v[..., None, None]; L += Lc; e += Lc @ o[..., None]
    return np.linalg.solve(L, e)[..., 0]


def run_eval(H, tag, extra=(), calib=0, adsi=None, dump=False, prefix=""):
    if A.oracle in ("rays", "both"):                                               # 진단: 2D 광선 = 카메라 원점 → 정답 관절 (검출·정합 오차 0)
        H = dict(H); ry = H["ray"].astype(np.float32).copy(); o = ry[..., :3]; dg = H["y"][:, None].astype(np.float32) - o; dg /= np.maximum(np.linalg.norm(dg, axis=-1, keepdims=True), 1e-6)
        ry[..., 3:6] = np.where((ry[..., 6] > 0)[..., None], dg, ry[..., 3:6]); H["ray"] = ry
    TH = to_dev(H, norm); N = len(H["y"]); full = H["ym"].all(1)
    if A.oracle in ("anchor", "both"):
        ga = H["y"][:, [7, 8]].mean(1).astype(np.float32); ga[:, 2] = 0; TH["ga"] = torch.from_numpy(ga).to(dev)
    avh = {k: (v.reshape(N, -1).any(1).cpu().numpy()) for k, v in avail(batch(TH, torch.arange(N, device=dev))).items()}
    present = [k for k in MODS if avh[k].mean() > 0.05]; seen_p = [k for k in present if bool(seen[k])]; unseen_p = [k for k in present if not bool(seen[k])]
    subs = ["+".join(seen_p)] + (["+".join(present)] if unseen_p else []) + [k for k in seen_p] + ["+".join(x for x in seen_p if x != k) for k in seen_p if len(seen_p) > 2]
    subs += (["rgb+radar"] if "rgb" in seen_p and "radar" in seen_p else []) + list(extra)
    subs = list(dict.fromkeys(s for s in subs if s)); out = {}; cal = None; cs = sorted(set(H["sub"].tolist()))[0]
    if A.evalsubs:
        subs = [x for x in A.evalsubs.split(",") if x]
    with torch.no_grad():
        for sub in subs:
            keep = set(sub.split("+")); sel = full.copy()
            for k in keep:
                sel &= avh["rgb" if k == "rgb1" else k]
            idx = np.flatnonzero(sel)
            if len(idx) < 50:
                continue
            P = []
            for i in range(0, len(idx), 512):
                b = torch.from_numpy(idx[i:i + 512]).to(dev); B_ = batch(TH, b)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda" and A.evalprec == "bf16"):
                    P.append(net(B_, subset_masks(B_, keep)).float().cpu().numpy())
            P = np.concatenate(P); G = H["y"][idx].astype(float); e = np.linalg.norm(P - G, axis=-1); pp, pg = P[:, [7, 8]].mean(1), G[:, [7, 8]].mean(1)
            Ps = ray_solve(P, H["ray"][idx], [0, 1] if "rgb" in keep else ([0] if "rgb1" in keep else []), H["seq"][idx]) if A.solver else None
            r = out[sub] = dict(n=int(len(idx)), frac=round(len(idx) / N, 3), mpjpe=round(float(e.mean()) * 1000, 1), pa=round(float(pa(P[::5], G[::5]).mean()) * 1000, 1),
                                pel_h=round(float(np.linalg.norm(pp[:, :2] - pg[:, :2], axis=1).mean()) * 1000, 1), pel_v=round(float(np.abs(pp[:, 2] - pg[:, 2]).mean()) * 1000, 1),
                                pel_x_bias=round(float((pp[:, 0] - pg[:, 0]).mean()) * 1000, 1), pel_y_bias=round(float((pp[:, 1] - pg[:, 1]).mean()) * 1000, 1),
                                rootrel=round(float(np.linalg.norm((P - pp[:, None]) - (G - pg[:, None]), axis=-1).mean()) * 1000, 1), pel_3d=round(float(np.linalg.norm(pp - pg, axis=1).mean()) * 1000, 1))
            if A.dumpall or (prefix and A.dumpval):                                                       # 11-25: 조합마다 덤프 (단일 모달리티 비교)
                np.savez(f"{OUT}/{prefix}pred_{sub}.npz", P=P.astype(np.float32), G=G.astype(np.float32), sub=H["sub"][idx], act=H["act"][idx], seq=H["seq"][idx], t=H["t"][idx])
            if Ps is not None:                                                     # ③ 풀이기 결과 (*_s)
                pps = Ps[:, [7, 8]].mean(1); gk_ = (np.linalg.norm(G - pg[:, None], axis=-1) < 1.2).all(1)
                r.update(mpjpe_s=round(float(np.linalg.norm(Ps - G, axis=-1).mean()) * 1000, 1), rootrel_s=round(float(np.linalg.norm((Ps - pps[:, None]) - (G - pg[:, None]), axis=-1).mean()) * 1000, 1),
                         pel_3d_s=round(float(np.linalg.norm(pps - pg, axis=1).mean()) * 1000, 1), mpjpe_s_ok=round(float(np.linalg.norm(Ps - G, axis=-1)[gk_].mean()) * 1000, 1), pa_s=round(float(pa(Ps[::5], G[::5]).mean()) * 1000, 1))
            if dump and sub == subs[0]:
                np.savez(f"{OUT}/pred.npz", P=P.astype(np.float32), G=G.astype(np.float32), sub=H["sub"][idx], act=H["act"][idx], seq=H["seq"][idx], t=H["t"][idx], **({"Ps": Ps.astype(np.float32)} if Ps is not None else {}))
            gok = (np.linalg.norm(G - pg[:, None], axis=-1) < 1.2).all(1); Pk, Gk = P[gok], G[gok]; ppk, pgk = Pk[:, [7, 8]].mean(1), Gk[:, [7, 8]].mean(1)
            r.update(n_ok=int(gok.sum()), mpjpe_ok=round(float(np.linalg.norm(Pk - Gk, axis=-1).mean()) * 1000, 1),
                     rootrel_ok=round(float(np.linalg.norm((Pk - ppk[:, None]) - (Gk - pgk[:, None]), axis=-1).mean()) * 1000, 1),
                     pa_ok=round(float(pa(Pk, Gk).mean()) * 1000, 1), pel_3d_ok=round(float(np.linalg.norm(ppk - pgk, axis=1).mean()) * 1000, 1))
            if adsi is not None and adp is not None:                                  # v8: 아는 데이터셋은 학습한 ④ 관례 어댑터를 붙인 값도
                Pa = adp(torch.from_numpy(P).float().to(dev), torch.full((len(P),), adsi, device=dev, dtype=torch.long), torch.from_numpy(H["org"][idx].astype(np.float32)).to(dev)).cpu().numpy().astype(float); ppa = Pa[:, [7, 8]].mean(1)
                r.update(mpjpe_adp=round(float(np.linalg.norm(Pa - G, axis=-1).mean()) * 1000, 1), rootrel_adp=round(float(np.linalg.norm((Pa - ppa[:, None]) - (G - pg[:, None]), axis=-1).mean()) * 1000, 1),
                         pel_3d_adp=round(float(np.linalg.norm(ppa - pg, axis=1).mean()) * 1000, 1))
            if calib:
                ex = H["sub"][idx] != cs
                if cal is None and (~ex).sum() >= 10:
                    ci = np.flatnonzero(~ex); ci = ci[np.linspace(0, len(ci) - 1, min(calib, len(ci))).astype(int)]; cal = fit_adapter_np(P[ci], G[ci]); R["calib"] = dict(subject=str(cs), n=int(len(ci)), fit_on=sub)
                if cal is not None and ex.sum() >= 50:
                    Pe, Ge = P[ex], G[ex]; Pc = adapt_np(Pe, *cal); rr = lambda X: float(np.linalg.norm((X - X[:, [7, 8]].mean(1)[:, None]) - (Ge - Ge[:, [7, 8]].mean(1)[:, None]), axis=-1).mean()) * 1000
                    r.update(mpjpe_ex=round(float(np.linalg.norm(Pe - Ge, axis=-1).mean()) * 1000, 1), mpjpe_c=round(float(np.linalg.norm(Pc - Ge, axis=-1).mean()) * 1000, 1),
                             rootrel_ex=round(rr(Pe), 1), rootrel_c=round(rr(Pc), 1), pa_c=round(float(pa(Pc[::5], Ge[::5]).mean()) * 1000, 1),
                             pel_3d_c=round(float(np.linalg.norm(Pc[:, [7, 8]].mean(1) - Ge[:, [7, 8]].mean(1), axis=1).mean()) * 1000, 1))
            Ps = smooth_root(P, H["seq"][idx], H["t"][idx]); es = np.linalg.norm(Ps - G, axis=-1); su = H["sub"][idx]; ac = H["act"][idx]
            r.update(mpjpe_sm=round(float(es.mean()) * 1000, 1), pel_h_sm=round(float(np.linalg.norm(Ps[:, [7, 8]].mean(1)[:, :2] - pg[:, :2], axis=1).mean()) * 1000, 1),
                     per_sub={str(q): [int((su == q).sum()), round(float(e[su == q].mean()) * 1000, 1), round(float(es[su == q].mean()) * 1000, 1)] for q in np.unique(su)},
                     per_act={str(a): [int((ac == a).sum()), round(float(e[ac == a].mean()) * 1000, 1), round(float(es[ac == a].mean()) * 1000, 1)] for a in np.unique(ac)})
            log("eval", tag, sub, json.dumps({k: v for k, v in r.items() if k not in ("per_sub", "per_act")}, ensure_ascii=False))
    del TH; torch.cuda.empty_cache()
    return out


H = load([A.hold])
if inside:
    H = {k: v[np.isin(H["sub"], test_s)] for k, v in H.items()}
H = {k: v[::A.evalstride] for k, v in H.items()}
R["results"] = run_eval(H, A.hold, extra=(["rgb1", "rgb1+radar", "rgb1+radar+imu"] if A.hold == "mri" else []), calib=A.calib, dump=A.dump); del H
json.dump(R, open(f"{OUT}/eval.json", "w"), indent=1, ensure_ascii=False)
for i_, d_ in enumerate(train_ds):
    if A.noval or A.oracle != "none" or not VAL.get(d_):
        continue
    Hv = load([d_]); Hv = {k: v[np.isin(Hv["sub"].astype(str), VAL[d_])][::A.valstride] for k, v in Hv.items()}
    R["val"][d_] = run_eval(Hv, f"val:{d_}", adsi=i_, prefix=f"val_{d_}_"); del Hv
    json.dump(R, open(f"{OUT}/eval.json", "w"), indent=1, ensure_ascii=False)
log("TRAIN_DONE")
