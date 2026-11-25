"""LHR 데이터 층: 프레임 토큰(model_v4.build = 광선·점·물리 토큰 + 정답) 위에 시퀀스·시간·라벨(활동/장면/뼈 길이)을 붙이고 창(window) 단위로 표본화한다.
- 창 = 같은 시퀀스의 W 프레임, 프레임 간격 = round(dt · fps / step) (시간 기준이라 fps가 달라도 같은 길이의 창).
- 라벨: 활동·장면 = lhr_labels 공통 어휘 id (-1 없음), 뼈 길이 = 시퀀스 정답 중앙값 (12,) (라벨 20프레임 미만이면 NaN).
- 학습 표본화 = 길이 비례 무작위 창, 평가 = 겹치지 않게 미끄러뜨려 모든 프레임을 한 번 이상 덮음 (중복은 첫 예측만)."""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_v1 as m1
import model_v3 as m3
import model_v4 as m4
import physics_baseline as pb
import lhr_labels as LB

m1.PACKS["octonet"] = pb.PACKS["octonet"] = Path("/home/interns/unisense/dev/packs/octonet_v07")
pb.SPECS["octonet"] = pb.CODE / "specs" / "octonet_v07.json"
# 평가 전용 개발 데이터셋 (학습 금지): ITOP 옆 시점 (2026-11-09)
m1.PACKS["itop"] = pb.PACKS["itop"] = Path("/home/interns/unisense/dev/packs/itop_v01"); pb.SPECS["itop"] = pb.CODE / "specs" / "itop_v01.json"; m1.DSID.setdefault("itop", 4)
# 평가 전용 검증 세트 (학습 금지): MM-Fi LiDAR (같은 MM-Fi 사람·정답에 처음 보는 깊이 센서, adapt_mmfi_lidar.py, 2026-11-11)
m1.PACKS["mmfi_lidar"] = pb.PACKS["mmfi_lidar"] = Path("/home/interns/unisense/dev/packs/mmfi_lidar_v01"); pb.SPECS["mmfi_lidar"] = pb.CODE / "specs" / "mmfi_lidar_v01.json"; m1.DSID.setdefault("mmfi_lidar", 5)
EVAL_ONLY = {"itop", "mmfi_lidar"}
m3.PHYS = Path("/home/interns/unisense/dev/packs/phys_v3")
BI, BJ = np.array(m4.BI), np.array(m4.BJ)
KEYS = m4.KEYS + ("cam", "t", "act", "scene", "blgt", "seq")
DT = 0.3                                   # 창 안 프레임 간격 [s] (사전)
import os
TRANSLATOR = int(os.environ.get("LHR_TRANSLATOR", "1"))   # 2 = 점 정리 + 키 비율 기준점 (2026-11-08). 기본 1은 기존 실행과 같다.
ADAPTER_DIR = Path(os.environ.get("VS_MODELS", "/home/interns/unisense/dev/packs/sim_v2_models"))   # 라벨 어댑터 v1 (δ): vsensor2 fit 결과 (각 데이터셋 train 분할)
NO_ADAPTER = False                                         # 교차 생성 팩은 라벨이 이미 정준 몸 (vs_common.build_cross 가 켬)


def build(ds, part, step=1, dt=DT):
    assert not (ds in EVAL_ONLY and part != "test"), f"{ds} 는 평가 전용"
    D = m4.build(ds, part, step)
    files = m1.pack_files(ds, m1.split_of(ds, part))
    seq, t, act, sc, bl, starts, lens, gap = [], [], [], [], [], [], [], []
    off = 0; fidx, foff = [], 0                                                                   # ① 팔 (LHR_ARM): step 1 행 → 이 step 행
    for i, f in enumerate(files):
        P = np.load(f); meta = json.loads(P["meta"].item()); T = int(meta["T"]); fps = float(meta["fps"])
        a, s = LB.frame_labels(ds, P, meta); sl = slice(0, None, step); n = len(range(0, T, step))
        y = P["labels"].astype(float); ok = P["label_mask"].all(1) & np.isfinite(y).all((1, 2))
        g = np.median(np.linalg.norm(y[ok][:, BI] - y[ok][:, BJ], axis=-1), 0) if ok.sum() >= 20 else np.full(12, np.nan)
        seq.append(np.full(n, i)); t.append(np.arange(T)[sl] / fps); act.append(a[sl]); sc.append(np.full(n, s)); bl.append(np.broadcast_to(g, (n, 12)))
        starts.append(off); lens.append(n); gap.append(max(1, int(round(dt * fps / step)))); off += n; fidx.append(foff + np.arange(0, T, step)); foff += T
    D["seq"] = torch.from_numpy(np.concatenate(seq)); D["t"] = torch.from_numpy(np.concatenate(t)).float()
    D["act"] = torch.from_numpy(np.concatenate(act)); D["scene"] = torch.from_numpy(np.concatenate(sc)); D["blgt"] = torch.from_numpy(np.ascontiguousarray(np.concatenate(bl))).float()
    D["starts"], D["lens"], D["gap"] = torch.tensor(starts), torch.tensor(lens), torch.tensor(gap)
    if MONO and ds == "mmfi":                                                                      # ① 단안 RGB v3 (LHR_MONO): 원래 프레임 속도에서 계산한 뒤 step 으로 줄인다
        import mono_refine as MR
        X0, A0 = [], []
        for f in files:
            P = np.load(f); Q = np.load(m3.PHYS / ds / f.relative_to(m1.PACKS[ds])); xr, ar, _ = MR.refine(Q["x3"][:, 0], Q["a3"][:, 0], np.asarray(P["t_c2r"], float), float(json.loads(P["meta"].item())["fps"]), MONO)
            X0.append(xr[::step]); A0.append(ar[::step])
        D["x3"] = D["x3"].clone(); D["a3"] = D["a3"].clone(); D["x3"][:, 0] = torch.from_numpy(np.concatenate(X0)); D["a3"][:, 0] = torch.from_numpy(np.concatenate(A0))
    sanitize(D); label_valid(D)
    if TRANSLATOR >= 2:
        translator_v2(D)
    if FB:                                                                                         # ① 번역기 v2.1 (LHR_FB=1, 2026-11-11): 대체 규칙 (trans_fallback.apply) — 빈 기준점 채움 + RGB+깊이 연결 없는 관절 정면 평면
        fallback_D(D)
    if os.environ.get("LHR_CANON", "0") == "1":
        canon_labels(D, ds)
    if os.environ.get("LHR_NODEF", "0") == "1":
        drop_def(D)
    assert len(D["seq"]) == len(D["y"]) == off
    if ARM:
        load_arm(D, ds, part, np.concatenate(fidx), foff)
    return D


def build_subrig(ds, part, step=1):
    """⑤ 부분 장비 가상 센서 (2026-11-10): 스테레오 RGB 데이터셋 (mRI) 에서 카메라 2 를 뺀 '카메라 1대 RGB'. 입력은 실측 키포인트 그대로 (카메라 1),
    물리 추정은 단안 경로 (발목 광선–바닥 + 정면 평면 = MM-Fi 와 같은 번역기, phys_features_v2 의 식) 로 다시 계산한다. 정답·몸은 같으므로 D["ds"] (라벨 관례 색인) 는 그대로.
    근거: 스테레오로만 RGB 를 배운 겹 (MM-Fi 를 뺀 겹) 은 카메라 1대 RGB 를 학습에서 한 번도 못 본다 (a1 MM-Fi 겹 RGB 214 mm, 뿌리 216)."""
    assert ds == "mri", "부분 장비는 지금 mRI (스테레오) 만"
    import phys_features_v2 as pf, conv_common as cc
    from phys_features import hips
    D = build(ds, part, step); spec = json.loads(pb.SPECS[ds].read_text()); Km = np.asarray(spec["rgb"]["K"], float); fpx = 0.5 * (Km[0, 0] + Km[1, 1]) if Km.ndim == 2 else 0.5 * (Km[0] + Km[1])
    spx = np.asarray(pf.SPX["mri"], float); n = np.array([0, 0, 1.0]); fps = float(spec.get("fps", 10)); A = {k: [] for k in ("a3", "x3", "sa", "sx")}
    for f in m1.pack_files(ds, m1.split_of(ds, part)):
        P = np.load(f); Rc, tc = np.asarray(P["R_c2r"]), np.asarray(P["t_c2r"], float); o, d = pb.rays(P["kp_px"], spec["rgb"]["K"], Rc, tc); d = np.array(d, float); d[P["kp_score"].astype(float) < pb.MIN_SCORE] = np.nan
        with np.errstate(all="ignore"):
            Xm, _ = pb.mono_floor(o, d, Rc, n, 0.0)
            nd = d[:, 15:17] @ n; t = (pb.ANKLE_H - (o @ n)) / nd; okk = (nd < -0.05) & (t > 0.3) & (t < 10); t = np.where(okk, t, np.inf); k = np.argmin(t, 1)
            da = d[np.arange(len(d)), 15 + k]; nda = np.abs(da @ n); ax = Rc[:, 2]; fwd = ax - (ax @ n) * n; fwd /= np.linalg.norm(fwd)
            dj = np.maximum(d @ fwd, 0.1); s_a = (pf.SIG_H / np.maximum(nda, 0.05)) * (da @ fwd); s_par = np.sqrt(s_a[:, None] ** 2 + pf.SIG_FP ** 2) / dj
            S17 = pf.ray_cov(d, s_par, np.linalg.norm(Xm - o, axis=-1) * spx / fpx)
        x3 = cc.to_common13(Xm, "coco17"); a3 = hips(x3); sx = pf.cov13(S17); sa = pf.hip_cov(sx)
        if np.isfinite(a3).any():
            a3, sa, _ = pf.finish(a3, sa, fps)
        if MONO:
            import mono_refine as MR
            x3, a3, _ = MR.refine(x3, a3, tc, fps, MONO)
        sa = np.where(np.isfinite(sa).all((-1, -2))[..., None, None], sa, np.eye(3)); sx = np.where(np.isfinite(sx).all((-1, -2))[..., None, None], sx, np.eye(3))
        for kk, v in (("a3", a3), ("x3", x3), ("sa", sa), ("sx", sx)):
            A[kk].append(v[::step].astype(np.float32))
    for kk in A:
        v = torch.from_numpy(np.concatenate(A[kk])); assert len(v) == len(D["y"]); D[kk] = D[kk].clone(); D[kk][:, 0] = v
    D["rv"] = D["rv"].clone(); D["rvok"] = D["rvok"].clone(); D["rv"][:, 1] = 0; D["rvok"][:, 1] = False; D["bl3"] = D["bl3"].clone(); D["bl3"][:, 0] = float("nan")
    sanitize(D); return D


FB = os.environ.get("LHR_FB", "0") == "1"


def fallback_D(D, chunk=65536):
    import trans_fallback as TF
    n = len(D["y"]); st = {}
    for s in range(0, n, chunk):
        b = {k: D[k][s:s + chunk] for k in ("rv", "rvok", "a3", "x3", "sa", "sx")}; b = TF.apply(b, stats=st)
        for k in ("a3", "x3", "sa", "sx"):
            D[k][s:s + chunk] = b[k]
    print(f"[번역기 v2.1] 채움 {st}", flush=True)                                                   # D 에는 텐서만 둔다 (학습이 .to() 로 옮김)


MONO = os.environ.get("LHR_MONO", "")                                                              # ① 단안 RGB 번역기 v3 변형 (mono_refine.py; "" = v2 그대로)
ARM = os.environ.get("LHR_ARM", "0") == "1"; ARM_DIR = Path("/home/interns/unisense/runs/lhr2/arm_x")


def arm_key(ds, part):
    return f"{ds}_{part}" + (f"_{m1.MMFI_SPLIT}" if ds == "mmfi" else "")


def load_arm(D, ds, part, rows, n_full):
    """① 깊이 팔 번역기 v2.1 (arm_translator a30_tmpl, arm_precompute.py 로 미리 계산) 손목 추정을 D 에 붙인다: xarm (N,2,3) 왼·오 손목, farm (N,2) 찾음.
    파일이 없거나 행 수가 다르면 전부 '못 찾음' (= 깊이 틀 그대로)."""
    n = len(D["y"]); D["xarm"] = torch.zeros(n, 2, 3); D["farm"] = torch.zeros(n, 2, dtype=torch.bool)
    sh = sorted(ARM_DIR.glob(arm_key(ds, part) + "__*.pt"))
    if not sh:
        print(f"[arm] {arm_key(ds, part)}: 파일 없음 → 틀", flush=True); return
    parts = sorted((torch.load(f) for f in sh), key=lambda r: r["lo"]); xw = torch.cat([r["xw"] for r in parts]); fd = torch.cat([r["found"] for r in parts])
    if len(xw) != n_full or parts[0]["lo"] != 0:
        print(f"[arm] {arm_key(ds, part)}: 행 수 불일치 ({len(xw)} vs {n_full}) → 틀", flush=True); return
    D["xarm"], D["farm"] = xw[torch.from_numpy(rows)].float(), fd[torch.from_numpy(rows)]
    print(f"[arm] {arm_key(ds, part)}: 찾음 {D['farm'].float().mean(0).tolist()}", flush=True)


def adapter_of(ds):
    """라벨 어댑터 v1 (2026-11-08): 정준 몸 = 라벨 + δ·u (u = 깊이 카메라 → 관절 단위벡터; 몸 관절 δ, 머리 δ_h). δ 는 그 데이터셋 train 분할에서 몸 모형 렌더와 실측 깊이 점을 맞춘 값.
    값 (sim_v2_models): MM-Fi 0.06/0.09, OctoNet 0/0, mmBody 0.04/0.06 m. 깊이 없는 mRI 는 0 (사전: 2D 관절 삼각측량은 몸 축 위). 라벨 관례와 센서 깊이 치우침이 함께 들어간 '라벨↔센서' 관계다."""
    f = ADAPTER_DIR / f"body_{ds}.json"; bd = json.loads(f.read_text()) if f.exists() else {}
    return float(bd.get("depth_shift") or 0.0), float(bd.get("head_shift") or 0.0)


def canon_labels(D, ds):
    """D["y"] → 정준 몸, 원래 라벨은 D["y_raw"]. 교차 생성 팩 (NO_ADAPTER) 은 이미 정준이라 그대로."""
    D["y_raw"] = D["y"].clone(); dl, dh = (0.0, 0.0) if NO_ADAPTER else adapter_of(ds); D["adapter"] = torch.tensor([dl, dh])
    if dl == 0.0 and dh == 0.0:
        _adapter_v2(D, ds); return
    y = D["y"]; u = y - D["dcam"][:, None]; u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6); sh = torch.full((13,), dl); sh[0] = dh
    D["y"] = torch.where(D["ymask"][..., None], y + sh[None, :, None] * u, y)
    _adapter_v2(D, ds)


def _adapter_v2(D, ds):
    """어댑터 v2 (LHR_ADAPTER=2, 진단·평가용): 관절별 라벨 관례 오프셋 (시선 정렬 좌표: 거리·옆·높이) 을 뺀다. runs/lhr2/adapter_v2.json (adapter_v2.py)."""
    f = Path("/home/interns/unisense/runs/lhr2/adapter_v2.json")
    if os.environ.get("LHR_ADAPTER", "1") != "2" or NO_ADAPTER or not f.exists():
        return
    tab = json.loads(f.read_text())
    if ds not in tab:
        return
    off = torch.tensor(tab[ds], dtype=torch.float32); y = D["y"]; cam = (D["cam"] if ds == "mri" else D["dcam"])[:, None]
    r = y - cam; r = torch.stack([r[..., 0], r[..., 1], torch.zeros_like(r[..., 0])], -1); ur = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    ul = torch.stack([-ur[..., 1], ur[..., 0], torch.zeros_like(ur[..., 0])], -1); uz = torch.zeros_like(ur); uz[..., 2] = 1
    U = torch.stack([ur, ul, uz], -2); D["y"] = torch.where(D["ymask"][..., None], y - (U.transpose(-1, -2) @ off[None, :, :, None])[..., 0], y)


def drop_def(D, sig_def=0.05):
    """번역기 Σ 에서 라벨 관례 항 SIG_DEF² I (phys_features_v2 의 ray_cov·tri_cov 가 모든 관절·기준점 Σ 에 더함) 를 뺀다. 라벨 관례는 어댑터 몫 (계획 v3). 번역기 v3 Σ 와 같은 효과, 물리 곁 파일은 그대로."""
    I = torch.eye(3) * sig_def ** 2
    for k in ("sx", "sa"):
        S = D[k] - I; ev = torch.linalg.eigvalsh(S.double()).min(-1).values; D[k] = torch.where((ev > 1e-6)[..., None, None], S, D[k])


def clean_points(P, ok):
    """번역기 v2 점 정리 (정답 없음, 사전 상수): 바닥 점 (z < 1.5 cm)과 몸 위 잡음 (점 높이 중앙값 위로 0.5 m 넘는 곳에서 25 cm 넘는 높이 틈 위, 또는 중앙값 + 1.35 m 위)을 뺀다.
    근거 2026-11-08: OctoNet 머리 위 점의 87%가 팔을 들지 않은 프레임에서 머리보다 0.84 m 위 (배경 잡음)."""
    z = P[..., 2]; ok = ok & (z > 0.015); zm = torch.where(ok, z, torch.full_like(z, float("nan"))).nanmedian(1).values.nan_to_num(0.0)
    zs = torch.where(ok, z, torch.full_like(z, -9.0)).sort(1).values; gap = (zs[:, 1:] - zs[:, :-1]); valid_gap = (zs[:, :-1] > zm[:, None] + 0.5) & (zs[:, :-1] > -8) & (gap > 0.25)
    cut = torch.where(valid_gap.any(1), torch.gather(zs[:, :-1], 1, valid_gap.float().argmax(1, keepdim=True))[:, 0] + 1e-4, zm + 1.35)
    return ok & (z <= torch.minimum(cut, zm + 1.35)[:, None])


def translator_v2(D):
    """번역기 v2: 깊이 점 정리 후 깊이 기준점을 다시 계산한다. 수평 = 정리된 점 중앙값 + 시선 방향 10 cm, 높이 = 0.53 × 점 최고 높이 (몸 틀 비율 0.90/1.70, 사전).
    both 조건 기준점은 엉덩이 추정이 없을 때 깊이 기준점을 따른다. 정답 없음."""
    if not bool(D["pok"].any()):
        return
    P = D["pts"][..., :3]; ok = clean_points(P, D["pok"].clone()); D["pok_raw"] = D["pok"]; D["pok"] = ok
    pn = torch.where(ok[..., None], P, torch.full_like(P, float("nan"))); med = pn.nanmedian(1).values; u = med - D["dcam"]; u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    z = torch.where(ok, P[..., 2], torch.full_like(P[..., 2], -9.0)).sort(1, descending=True).values; n = ok.sum(1); k = (0.02 * n.float()).long().clamp(max=P.shape[1] - 1); H = z[torch.arange(len(z)), k]
    a = med + 0.10 * u; a[:, 2] = 0.53 * H; good = (n >= 5) & torch.isfinite(a).all(-1)
    a3 = D["a3"].clone(); old = a3[:, 1].clone(); a3[:, 1] = torch.where(good[:, None], a, old)
    hb = 0.5 * (D["x3"][:, 2, 7] + D["x3"][:, 2, 8]); hv = torch.isfinite(hb).all(-1); a3[:, 2] = torch.where(hv[:, None], a3[:, 2], a3[:, 1]); D["a3"] = a3


def label_valid(D):
    """라벨 유효 규칙 (벤치마크 v1, 모든 데이터셋 동일, 사전 상수): 관절이 정답 골반에서 1.6 m 넘게 떨어지거나 뼈 길이 비(대 ELL0)가 0.4–1.8 밖이면 그 프레임의 라벨을 무효로 한다.
    2026-11-07 감사: mRI 라벨에 골반에서 최대 893 m 떨어진 관절이 프레임의 0.3–0.5% 있었고 MM-Fi는 뼈 비 이상이 1.3–1.9%였다. 학습 손실과 시험 지표에서 모두 뺀다."""
    y = D["y"]; r = 0.5 * (y[:, 7] + y[:, 8]); ell = torch.tensor(m4.ELL0)
    far = (y - r[:, None]).norm(dim=-1).max(1).values > 1.6
    ratio = torch.stack([(y[:, i] - y[:, j]).norm(dim=-1) for i, j in m4.BONES], -1) / ell
    has = D["ymask"].all(1); bad = (far | ((ratio < 0.4) | (ratio > 1.8)).any(1)) & has
    D["ymask"] = D["ymask"] & ~bad[:, None]; D["label_bad"] = torch.tensor([int(bad.sum()), int(has.sum())])


def sanitize(D):
    """번역기 이상치 거부 (정답 없음, 사전 상수만): 기준점이 시퀀스 중앙값에서 2.5 m 넘게 벗어나면 중앙값으로 바꾸고, 관절 추정이 기준점에서 사슬 길이 + 0.5 m 넘게 벗어나면 측정 없음으로 둔다. mRI 삼각측량의 수 m–수백 m 추정을 막는다."""
    chain = torch.tensor([0.70, 0.50, 0.50, 0.80, 0.80, 1.05, 1.05, 0.11, 0.11, 0.53, 0.53, 0.93, 0.93]) + 0.5
    a3, x3 = D["a3"].clone(), D["x3"].clone(); n_a = n_x = 0
    for s0, L in zip(D["starts"].tolist(), D["lens"].tolist()):
        for c in range(a3.shape[1]):
            a = a3[s0:s0 + L, c]; ok = torch.isfinite(a).all(-1)
            if int(ok.sum()) < 5:
                continue
            med = a[ok].median(0).values; bad = ok & ((a - med).norm(dim=-1) > 2.5); n_a += int(bad.sum()); a[bad] = med
            x = x3[s0:s0 + L, c]; far = (x - a[:, None]).norm(dim=-1) > chain[None]; n_x += int(far.sum()); x[far] = float("nan")
    D["a3"], D["x3"] = a3, x3; D["sanitized"] = torch.tensor([n_a, n_x])


def sample_windows(D, W, n, gen):
    """학습용 무작위 창 (n, W) 프레임 색인 (CPU). 시퀀스는 (길이 − 창 폭)에 비례해 뽑는다."""
    lens, gap, starts = D["lens"].cpu(), D["gap"].cpu(), D["starts"].cpu(); span = (W - 1) * gap
    p = (lens - span).clamp(min=0).float()
    s = torch.multinomial(p / p.sum(), n, replacement=True, generator=gen)
    st = starts[s] + (torch.rand(n, generator=gen) * (lens[s] - span[s])).long()
    return st[:, None] + torch.arange(W)[None] * gap[s][:, None]


def slide_windows(D, W):
    """평가용 창: 시퀀스마다 겹치지 않게 미끄러뜨리고 꼬리를 덮는 창 하나를 더한다. 짧은 시퀀스는 간격을 줄인다."""
    out = []
    for s0, L, g in zip(D["starts"].tolist(), D["lens"].tolist(), D["gap"].tolist()):
        if L <= (W - 1) * g:
            g = max(1, (L - 1) // max(W - 1, 1))
        span = (W - 1) * g; st = list(range(0, L - span, W * g)); last = L - 1 - span
        if last >= 0 and last not in st:
            st.append(last)
        out += [s0 + x + np.arange(W) * g for x in st]
    return torch.from_numpy(np.stack(out)).long()


def gather(D, idx):
    """(n, W) 창 색인 → 평평한 (n·W) 배치 (점 토큰은 256개로 채움)."""
    flat = idx.reshape(-1).to(D["y"].device); b = {k: D[k][flat] for k in KEYS + tuple(k for k in ("xarm", "farm") if k in D)}
    b["pts"], b["pok"] = m1.pad_pts(b["pts"]), m1.pad_pts(b["pok"])
    return b
