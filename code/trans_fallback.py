"""① 번역기 대체 규칙 (2026-11-11, 정답 없음, 사전 상수만). 배치 b (lhr_data.gather 형식) 를 받아 기준점·관절 물리 추정이 없는 곳을 채운다.
배경 (occl_eval 11-11): 카메라 1대 RGB 는 발목이 가려지면, 스테레오는 엉덩이가 가려지면 기준점이 없어 출력이 아예 없다. 둘 다 (RGB+깊이) 는 광선–깊이 연결이 없는 관절이 단안 바닥 해로 넘어가는데 그 해도 사라진다.
규칙:
- mono (카메라 1대, 기준점 없음): 크기 단서. 보이는 뼈마다 두 끝 광선 사이 각 θ (atan2) 로 거리 Z_b = L_b / θ (L_b = 인구 평균 뼈 길이 m4.ELL0; 뼈가 시선 쪽으로 기울면 Z_b 는 참보다 큼).
  몸 깊이 D = 뼈별 정면 깊이 Z_b·(m_b·fwd) 의 통계 (STAT: min / q25 / median), fwd = 보이는 광선 평균 방향의 수평 성분. 관절 = 광선과 정면 평면 (깊이 D) 의 교차 (단안 바닥 해와 같은 정면 평면 가정).
  Σ: 광선 방향 σ = sqrt((t·SIZE_SIG)² + (SIG_FP / cos)²), 수직 σ = t·σ_각.
- 골반 (기준점): 두 엉덩이가 있으면 그 중점, 없고 두 어깨가 있으면 어깨 중점 − TORSO·ẑ (Σ 에 SIG_PEL 더함). 스테레오 (mRI) 기준점 대체에도 같은 규칙.
- both_plane (RGB+깊이): 광선은 있는데 깊이 연결이 없는 관절 = 깊이 기준점을 지나는 정면 평면과 광선의 교차 (깊이 기준점 σ + SIG_FP).
- relink (진단): 광선–깊이 연결을 반경 tau, 통계 (median / q30) 로 다시 계산 (physics_baseline.mmfi_sparse 와 같은 식, 13관절 오프셋 m4.R_J)."""
import math
import torch
import model_v4 as m4

BI, BJ = list(m4.BI), list(m4.BJ); ELL = torch.tensor(m4.ELL0); R_J = m4.R_J
SIZE_SIG = 0.20                  # 크기 단서 거리의 상대 σ (사전: 뼈 길이 개인차 ~7 % + 기울어짐)
SIG_FP = 0.10                    # 정면 평면 가정의 몸 두께 σ [m] (phys_features_v2.SIG_FP)
TORSO = 0.50                     # 어깨 중점 → 골반 수직 거리 [m] (m4.ELL0 몸통)
SIG_PEL = (0.10, 0.15)           # 어깨에서 만든 골반의 (수평, 수직) 추가 σ [m] (사전: 상체 기울기·앉음)
MIN_THETA = 0.01                 # 뼈 끝 광선 사이 최소 각 [rad] (검출 잡음 수준보다 큰 것만)
NAN = float("nan")


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def ray_cov(d, s_par, s_perp):
    P = d[..., :, None] * d[..., None, :]; I = torch.eye(3, dtype=d.dtype, device=d.device)
    return s_par[..., None, None] ** 2 * P + s_perp[..., None, None] ** 2 * (I - P)


def fwd_of(d, ok):
    m = (d * ok[..., None].to(d.dtype)).sum(1); f = torch.stack([m[:, 0], m[:, 1], torch.zeros_like(m[:, 0])], -1)
    return unit(f)


def size_depth(d, ok, fwd, stat="min"):
    """크기 단서 몸 정면 깊이 D (N,), 쓴 뼈 수 (N,)."""
    di, dj = d[:, BI], d[:, BJ]; okb = ok[:, BI] & ok[:, BJ]
    th = torch.atan2(torch.cross(di, dj, dim=-1).norm(dim=-1), (di * dj).sum(-1)); okb = okb & (th > MIN_THETA)
    mb = unit(di + dj); Db = ELL.to(d)[None] / th.clamp(min=1e-6) * (mb * fwd[:, None]).sum(-1)
    Db = torch.where(okb, Db, torch.full_like(Db, NAN)); n = okb.sum(1)
    if stat == "min":
        D = torch.where(okb, Db, torch.full_like(Db, float("inf"))).min(1).values
    else:
        D = Db.nanquantile(0.25 if stat == "q25" else 0.5, dim=1)
    return torch.where(n > 0, D, torch.full_like(D, NAN)), n


def frontal(o, d, D, fwd):
    c = (d * fwd[:, None]).sum(-1).clamp(min=0.1); t = D[:, None] / c
    return o[:, None] + t[..., None] * d, t, c


def pelvis(X, S, okj):
    """X (N,13,3), S (N,13,3,3), okj (N,13) → 골반, Σ (없으면 NaN), 출처 (0 없음, 1 엉덩이, 2 어깨)."""
    hip = okj[:, 7] & okj[:, 8]; sh = okj[:, 1] & okj[:, 2]
    a_h, S_h = 0.5 * (X[:, 7] + X[:, 8]), 0.5 * (S[:, 7] + S[:, 8])
    z = torch.zeros_like(a_h); z[:, 2] = TORSO; a_s = 0.5 * (X[:, 1] + X[:, 2]) - z
    S_s = 0.5 * (S[:, 1] + S[:, 2]) + torch.diag(torch.tensor([SIG_PEL[0] ** 2, SIG_PEL[0] ** 2, SIG_PEL[1] ** 2], dtype=X.dtype, device=X.device))
    a = torch.where(hip[:, None], a_h, torch.where(sh[:, None], a_s, torch.full_like(a_h, NAN)))
    Sa = torch.where(hip[:, None, None], S_h, torch.where(sh[:, None, None], S_s, torch.eye(3, dtype=X.dtype, device=X.device).expand_as(S_h)))
    return a, Sa, torch.where(hip, 1, torch.where(sh, 2, 0))


def mono_size(b, stat="min"):
    """카메라 1 광선만으로 크기 단서 해. 반환 a, Sa, x (N,13,3; 안 보이면 NaN), Sx, 출처."""
    rv, ok = b["rv"][:, 0], b["rvok"][:, 0]; o, d, sa = rv[:, 0, 0:3], unit(rv[..., 3:6]), rv[..., 6]
    fwd = fwd_of(d, ok); D, n = size_depth(d, ok, fwd, stat); X, t, c = frontal(o, d, D, fwd)
    S = ray_cov(d, torch.sqrt((t * SIZE_SIG) ** 2 + (SIG_FP / c) ** 2), t * sa)
    okj = ok & torch.isfinite(D)[:, None]; X = torch.where(okj[..., None], X, torch.full_like(X, NAN))
    a, Sa, src = pelvis(X, S, okj); return a, Sa, X, S, src


def apply(b, mono=True, stereo=True, both_plane=True, stat="min", stats=None):
    """빈 기준점·관절을 채운 새 배치. stats (dict) 에 채운 행 수를 더한다."""
    b = dict(b); a3, x3, sa3, sx3 = b["a3"].clone(), b["x3"].clone(), b["sa"].clone(), b["sx"].clone()
    st = stats if stats is not None else {}
    cam1, cam2 = b["rvok"][:, 0].any(1), b["rvok"][:, 1].any(1)
    if mono:                                                                                  # 카메라 1대: 발목–바닥 기준점이 없으면 크기 단서
        miss = ~torch.isfinite(a3[:, 0]).all(-1) & cam1 & ~cam2
        if bool(miss.any()):
            a, Sa, X, S, src = mono_size(b, stat); m = miss & (src > 0)
            a3[m, 0], sa3[m, 0], x3[m, 0], sx3[m, 0] = a[m], Sa[m], X[m], S[m]; st["mono"] = st.get("mono", 0) + int(m.sum()); st["mono_sh"] = st.get("mono_sh", 0) + int((m & (src == 2)).sum())
    if stereo:                                                                                # 스테레오: 엉덩이가 없으면 어깨에서 골반
        miss = ~torch.isfinite(a3[:, 0]).all(-1) & cam2
        if bool(miss.any()):
            X = x3[:, 0]; okj = torch.isfinite(X).all(-1); a, Sa, src = pelvis(torch.nan_to_num(X), sx3[:, 0], okj); m = miss & (src > 0)
            a3[m, 0], sa3[m, 0] = a[m], Sa[m]; st["stereo"] = st.get("stereo", 0) + int(m.sum())
    if both_plane:                                                                            # RGB+깊이: 연결 없는 관절 = 깊이 기준점 정면 평면
        ad = a3[:, 1]; hasd = torch.isfinite(ad).all(-1) & cam1 & ~cam2
        if bool(hasd.any()):
            rv, ok = b["rv"][:, 0], b["rvok"][:, 0]; o, d, sa = rv[:, 0, 0:3], unit(rv[..., 3:6]), rv[..., 6]; fwd = fwd_of(d, ok)
            D = ((torch.nan_to_num(ad) - o) * fwd).sum(-1); X, t, c = frontal(o, d, D, fwd)
            s_a = torch.sqrt((fwd[:, None, :] @ sa3[:, 1] @ fwd[:, :, None])[:, 0, 0].clamp(min=0))
            S = ray_cov(d, torch.sqrt(s_a[:, None] ** 2 + SIG_FP ** 2) / c, t * sa)
            m = hasd[:, None] & ok & ~torch.isfinite(x3[:, 2]).all(-1)
            x3[:, 2] = torch.where(m[..., None], X, x3[:, 2]); sx3[:, 2] = torch.where(m[..., None, None], S, sx3[:, 2]); st["both_plane"] = st.get("both_plane", 0) + int(m.sum())
            miss = ~torch.isfinite(a3[:, 2]).all(-1) & hasd; a3[:, 2] = torch.where(miss[:, None], ad, a3[:, 2]); sa3[:, 2] = torch.where(miss[:, None, None], sa3[:, 1], sa3[:, 2])
    b["a3"], b["x3"], b["sa"], b["sx"] = a3, x3, sa3, sx3; return b


def relink(b, tau=0.05, q=0.5, nmin=3):
    """광선–깊이 연결 (진단). 반환 x (N,13,3; 연결 없으면 NaN), 점 수."""
    rv, ok = b["rv"][:, 0], b["rvok"][:, 0]; o, d = rv[:, 0, 0:3], unit(rv[..., 3:6]); p, pok = b["pts"][..., 0:3], b["pok"]
    v = p - o[:, None]; t = torch.einsum("njc,npc->njp", d, v); perp2 = (v ** 2).sum(-1)[:, None] - t ** 2
    sel = pok[:, None] & (perp2 < tau ** 2) & (t > 0) & ok[..., None]; n = sel.sum(-1)
    ts = torch.where(sel, t, torch.full_like(t, NAN)).nanquantile(q, dim=-1)
    X = o[:, None] + (ts + R_J.to(d))[..., None] * d; return torch.where((n >= nmin)[..., None], X, torch.full_like(X, NAN)), n
