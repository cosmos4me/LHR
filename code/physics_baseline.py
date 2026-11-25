"""물리 단독 기준선 v2 (학습 없음): 측정 + 물리 + 사전 상수만으로 13관절 3D를 계산해 GT와 비교한다.
  MM-Fi  : RGB 키포인트 광선을 원본 깊이 영상에 투영 → 5×5 창의 사람 깊이 중앙값 = 표면 → 표면→관절 오프셋 (관절별, dense)
           (v1 비교용: 팩의 256점 중 광선 5 cm 안 점의 중앙값)
           깊이만: 사람 점 중앙값 + 몸 두께 절반 → 골반 (root만)
  mRI    : 두 카메라 광선 삼각측량 (검출 신뢰도 가중 최소제곱, 관절별)
  OctoNet: 깊이만 → root만
  시간: 프레임별 결과에 시간 중앙값 필터 (사람 움직임 연속성, 창 WIN 프레임) 적용한 값도 함께 보고.
사전 상수 (GT로 맞추지 않음): 관절별 표면→관절 깊이 = 성인 평균 분절 둘레/2π 근사, 몸 두께 절반 0.10 m.
사용: python physics_baseline.py [--part test] [--out notes/physics_baseline_v2.json]"""
import json, sys, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import conv_common as cc

CODE = Path(__file__).parent
PACKS = {"mmfi": Path("/home/interns/unisense/dev/packs/mmfi_v02_specA"), "mri": Path("/home/interns/unisense/dev/packs/mri_v03"),
         "octonet": Path("/home/interns/unisense/dev/packs/octonet_v06")}
SPECS = {k: CODE / "specs" / f"{v}.json" for k, v in (("mmfi", "mmfi_v02_specA"), ("mri", "mri_v03"), ("octonet", "octonet_v06"))}
PACKS["mmbody"] = Path("/home/interns/unisense/dev/packs/mmbody_v01"); SPECS["mmbody"] = CODE / "specs" / "mmbody_v01.json"   # 2026-11-02
MMFI_RAW = Path("/home/interns/unisense/dev/data/mmfi/extracted")      # 2026-11-02 삭제됨 (캐시 cache_v1만 남음): mmfi_dense는 원본 전체 프레임이 필요 (기각된 방법)
# COCO17 순서: 코, 눈×2, 귀×2, 어깨×2, 팔꿈치×2, 손목×2, 엉덩이×2, 무릎×2, 발목×2
OFF = np.array([0.0, 0.0, 0.0, 0.08, 0.08, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.09, 0.09, 0.05, 0.05, 0.04, 0.04])
HALF_BODY = 0.10       # 사람 점 중앙값 → 골반: 몸 두께 절반 (사전)
TAU = 0.05             # v1: 광선에서 이 거리 안의 팩 깊이 점만 표면으로 본다 [m]
BAND = 0.6             # dense: 사람 깊이 중앙값 ± BAND 안의 화소만 사람으로 본다 [m] (몸 크기 수준)
MIN_SCORE = 0.3        # mRI 검출 신뢰도 하한
WIN = 5                # 시간 중앙값 창 [프레임] (10 fps에서 0.5 s)
ANKLE_H = 0.08         # COCO 발목(복사뼈) 높이 [m] (사전)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


MMFI_SPLIT = "E1_cross_subject"     # "E04_new_env": E04 새 환경 시험 (진단)


def split_of(ds, part):
    if ds == "mmfi":
        return set(json.loads((CODE / "splits" / "mmfi.json").read_text())[MMFI_SPLIT][part])
    s = json.loads((CODE / "splits" / f"{ds}.json").read_text())[part]
    if ds == "mmbody":                                                               # 시퀀스 이름 ("train/sequence_0")
        return set(s)
    return {f"S{x:02d}" for x in s} if ds == "mri" else {f"U{x:02d}" for x in s}


def files_of(ds, subjects):
    if ds == "mmbody":
        return [f for f in sorted(PACKS[ds].glob("*/*.npz")) if f"{f.parent.name}/{f.stem}" in subjects]
    if ds == "mmfi":
        return [f for f in sorted(PACKS[ds].glob("E*/*.npz")) if f.name.split("_")[0] in subjects]
    if ds == "mri":
        return [f for f in sorted(PACKS[ds].glob("S*.npz")) if f.stem in subjects]
    out = []
    for f in sorted(PACKS[ds].glob("*.npz")):
        m = json.loads(np.load(f)["meta"].item())
        if m["subject"] in subjects and not m.get("multi_person_action"):
            out.append(f)
    return out


def kvec(K):
    K = np.asarray(K, float)
    return [K[0, 0], K[1, 1], K[0, 2], K[1, 2]] if K.ndim == 2 else list(K)


def depth_points(P, dm):
    """tokens()와 같은 방식: 팩 깊이 점 → 방 좌표."""
    fx, fy, cx, cy = kvec(dm["K"])
    uvz = P["pts_uvz"].astype(float)
    a = np.stack([(uvz[..., 0] - cx) / fx, (uvz[..., 1] - cy) / fy, np.ones_like(uvz[..., 2])], -1)
    p = (a * uvz[..., 2:3]) @ np.asarray(P["R_d2r"]).T + P["t_d2r"]
    return p, P["pts_valid"].astype(bool) & np.isfinite(p).all(-1)


def rays(kp, K, R, t):
    o, d, *_ = cc.ray_tokens(kp.astype(float), kvec(K), R, t, np.ones(kp.shape[:-1]))
    return o, d


def tmed(X, win=WIN):
    """시간 축 중앙값 필터 (NaN 무시). X (T,...)"""
    h = win // 2
    nan = np.full((h,) + X.shape[1:], np.nan)
    pad = np.concatenate([nan, X, nan])
    return np.nanmedian(np.stack([pad[i:i + len(X)] for i in range(win)]), 0)


# ---------------- MM-Fi ----------------
def mmfi_sparse(P, spec, off):
    """v1: 팩 256점 중 광선 TAU 안 점의 광선 방향 거리 중앙값 + 오프셋."""
    o, d = rays(P["kp_px"], spec["rgb"]["K"], P["R_c2r"], P["t_c2r"])
    p, ok = depth_points(P, spec["depth"])
    v = p - o
    t = np.einsum("tjc,tnc->tjn", d, v)
    sel = ok[:, None, :] & ((v ** 2).sum(-1)[:, None, :] - t ** 2 < TAU ** 2) & (t > 0)
    ts = np.nanmedian(np.where(sel, t, np.nan), -1)
    ts[sel.sum(-1) < 3] = np.nan
    return o + (ts + off)[..., None] * d


def read_depth(path):
    from PIL import Image                                   # dense 판독에서만 필요 (학습 환경에는 PIL 없음)
    d = np.array(Image.open(path)).astype(np.float32)
    d[(d == 0) | (d == 65535)] = np.nan
    return d * 1e-3


def mmfi_dense(job):
    """원본 깊이 영상에서 RGB 키포인트 광선의 표면 깊이를 직접 읽는다 (관절별). 반환: 표면점 (T,17,3) 방 좌표, 방 좌표 광선 방향."""
    f, spec = job
    P = np.load(f); m = json.loads(P["meta"].item())
    df = sorted((MMFI_RAW / m["env"] / m["subject"] / m["action"] / "depth").glob("frame*"), key=lambda p: int(p.stem[5:]))
    T = len(P["labels"])
    fxr, fyr, cxr, cyr = kvec(spec["rgb"]["K"]); fx, fy, cx, cy = kvec(spec["depth"]["K"])
    R, t = np.array(spec["depth"]["R_rgb2d"]), np.array(spec["depth"]["t_rgb2d"])
    kp = P["kp_px"].astype(float)
    a = np.stack([(kp[..., 0] - cxr) / fxr, (kp[..., 1] - cyr) / fyr, np.ones(kp.shape[:-1])], -1)   # RGB 카메라, z = 1
    dirn = a @ R.T                                                                                     # 깊이 카메라 좌표: X = t + s·dirn (s = RGB z)
    S = np.full((T, 17), np.nan)
    for i in range(min(T, len(df))):
        d = read_depth(df[i]); s = np.full(17, 3.0); zg = None
        for it in range(3):
            X = t + s[:, None] * dirn[i]
            ui = np.clip(np.rint(fx * X[:, 0] / X[:, 2] + cx).astype(int), 2, 637)
            vi = np.clip(np.rint(fy * X[:, 1] / X[:, 2] + cy).astype(int), 2, 477)
            w = np.stack([d[vi + dy, ui + dx] for dy in range(-2, 3) for dx in range(-2, 3)], -1)   # (17,25)
            if zg is None:
                zg = np.nanmedian(np.nanmin(w[5:], -1))                                               # 몸 관절 창의 가장 가까운 깊이 → 사람 깊이
                if not np.isfinite(zg):
                    break
            w = np.where(np.abs(w - zg) < BAND, w, np.nan)
            zm = np.nanmedian(w, -1); zm[np.isfinite(w).sum(-1) < 3] = np.nan
            s_new = (zm - t[2]) / dirn[i, :, 2]
            s = np.where(np.isfinite(s_new), s_new, s)
            S[i] = np.where(np.isfinite(zm), s_new, np.nan)
    Xd = t + S[..., None] * dirn                                                                       # 깊이 카메라 좌표 표면점
    Xr = Xd @ np.asarray(P["R_d2r"]).T + P["t_d2r"]
    dr = a @ np.asarray(P["R_c2r"]).T; dr /= np.linalg.norm(dr, axis=-1, keepdims=True)
    return str(f), Xr, dr


def person_root(p, ok, o):
    """사람 점 중앙값을 카메라→중앙값 방향으로 몸 두께 절반만큼 민다."""
    c = np.nanmedian(np.where(ok[..., None], p, np.nan), 1)
    u = c - o; u /= np.linalg.norm(u, axis=-1, keepdims=True)
    return c + HALF_BODY * u


def triangulate(o1, d1, w1, o2, d2, w2):
    """두 광선의 가중 수직 거리 제곱합 최소점. d: (T,J,3), w: (T,J)"""
    A1 = (np.eye(3) - d1[..., :, None] * d1[..., None, :]) * w1[..., None, None]
    A2 = (np.eye(3) - d2[..., :, None] * d2[..., None, :]) * w2[..., None, None]
    return np.linalg.solve(A1 + A2, (A1 @ o1 + A2 @ o2)[..., None])[..., 0]



# ---------------- 단안 RGB: 발목 광선–바닥 교차 + 정면 평면 가정 ----------------
def mono_floor(o, d, R_c2r, n, c):
    """바닥 n·X + c = 0 (방 좌표, n 위쪽). 두 발목 광선과 '바닥 + ANKLE_H' 평면의 교차 중 가까운 것(땅에 닿은 발) = 기준점.
    나머지 관절은 기준점을 지나고 카메라 수평 시선에 수직인 평면과 광선의 교차 (정면 평면 가정). 반환 (T,17,3), 기준점 유효 마스크."""
    nd = d[:, 15:17] @ n                                                        # (T,2)
    t = (ANKLE_H - (n @ o + c)) / nd
    ok = (nd < -0.05) & (t > 0.3) & (t < 10)
    t = np.where(ok, t, np.inf); k = np.argmin(t, 1); ta = t[np.arange(len(t)), k]
    Xa = o + ta[:, None] * d[np.arange(len(t)), 15 + k]
    ax = np.asarray(R_c2r)[:, 2]; fwd = ax - (ax @ n) * n; fwd /= np.linalg.norm(fwd)
    tj = ((Xa - o) @ fwd)[:, None] / (d @ fwd)
    X = o + tj[..., None] * d
    valid = np.isfinite(ta)
    X[~valid] = np.nan
    return X, valid


def mri_tri_all(spec, subjects):
    """mRI 두 카메라 삼각측량 (관절별) — 바닥 추정과 평가에 공용."""
    R12, T12 = np.asarray(spec["rgb"]["R_cam1_to_cam2"]), np.asarray(spec["rgb"]["t_cam1_to_cam2"])
    out = {}
    for f in files_of("mri", subjects):
        P = np.load(f)
        Rc, tc = np.asarray(P["R_c2r"]), np.asarray(P["t_c2r"], float)
        o1, d1 = rays(P["kp_px"], spec["rgb"]["K"], Rc, tc)
        R2r = Rc @ R12.T; t2r = Rc @ (-R12.T @ T12) + tc                     # cam2 → 방 ("R,T" 규약: X2 = R12 X1 + T12)
        o2, d2 = rays(P["kp_px_cam2"], spec["rgb"]["K_cam2"], R2r, t2r)
        s1, s2 = P["kp_score"].astype(float), P["kp_score_cam2"].astype(float)
        X17 = triangulate(o1, d1, s1 ** 2, o2, d2, s2 ** 2)
        X17[(s1 < MIN_SCORE) | (s2 < MIN_SCORE)] = np.nan
        out[f] = (P, X17, o1, d1)
    return out


def mri_floor_gtfree(tri, iters=3000, tol=0.03, seed=0):
    """정답 없는 mRI 바닥: 프레임마다 더 낮은 삼각측량 발목(입력만, 모든 피험자)에 RANSAC 평면 → ANKLE_H만큼 내린다.
    반환: 방 좌표 바닥 (n 위쪽, c), 내점 수. (방 좌표계 자체는 좌표 표현일 뿐, 바닥 값은 여기서 새로 구한다)"""
    A = []
    for P, X17, *_ in tri.values():
        a = X17[:, 15:17]; z = np.where(np.isfinite(a[..., 2]), a[..., 2], np.inf); k = np.argmin(z, 1)
        low = a[np.arange(len(a)), k]; A.append(low[np.isfinite(low).all(-1)])
    A = np.concatenate(A); rng = np.random.default_rng(seed); best = (0, None)
    for _ in range(iters):
        a, b, c3 = A[rng.choice(len(A), 3, replace=False)]
        n = np.cross(b - a, c3 - a); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn * np.sign(n[2] + 1e-12)
        if n[2] < 0.9:
            continue
        inl = np.abs(A @ n - n @ a) < tol
        if inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    Q = A[best[1]]; m = Q.mean(0); n = np.linalg.svd(Q - m)[2][-1]; n *= np.sign(n[2])
    return n, float(-(n @ m) + ANKLE_H), best[0], len(A)

# ---------------- 집계 ----------------
class Acc:
    def __init__(self):
        self.e, self.r, self.l, self.n = [], [], [], [0, 0]

    def add(self, X, Y, w):
        """X: (T,13,3) 관절, 또는 (T,3) root만. w: (T,13) 깨끗한 GT 마스크."""
        groot = 0.5 * (Y[:, 7] + Y[:, 8])
        if X.ndim == 3:
            ok = w & np.isfinite(X).all(-1); root = 0.5 * (X[:, 7] + X[:, 8])
            self.e.append(np.where(ok, np.linalg.norm(X - Y, axis=-1), np.nan))
            self.l.append(np.where(ok, np.linalg.norm((X - root[:, None]) - (Y - groot[:, None]), axis=-1), np.nan))
            self.n[0] += ok.sum(); self.n[1] += w.sum()
        else:
            root = X
        rw = w[:, 7] & w[:, 8] & np.isfinite(root).all(-1)
        self.r.append(np.where(rw, np.linalg.norm(root - groot, axis=-1), np.nan))
        if X.ndim == 2:
            self.n[0] += rw.sum(); self.n[1] += (w[:, 7] & w[:, 8]).sum()

    def summary(self):
        cat = lambda L: np.concatenate([x.ravel() for x in L]) if L else None
        f = lambda L: None if not L else round(float(np.nanmean(cat(L)) * 1000), 1)
        md = lambda L: None if not L else round(float(np.nanmedian(cat(L)) * 1000), 1)
        p90 = lambda L: None if not L else round(float(np.nanpercentile(cat(L), 90) * 1000), 1)
        return dict(abs_mm=f(self.e), abs_med=md(self.e), root_mm=f(self.r), root_med=md(self.r), root_p90=p90(self.r),
                    rel_mm=f(self.l), coverage=round(float(self.n[0] / max(self.n[1], 1)), 3))

    def per_joint(self):
        return (np.nanmean(np.concatenate(self.e), 0) * 1000).round(1).tolist() if self.e else None


def clean_mask(P):
    Y = P["labels"].astype(float)
    return Y, P["label_mask"] & np.isfinite(Y).all(-1) & ((P["label_quality"] & 1) == 0)


def run(part, workers=16):
    out = {}
    # ---- MM-Fi
    spec = json.loads(SPECS["mmfi"].read_text())
    files = files_of("mmfi", split_of("mmfi", part))
    acc = {k: Acc() for k in ("sparse_v1", "dense_nooff", "dense_off", "dense_off_tmed", "depth_root", "depth_root_tmed", "mono", "mono_tmed")}
    with Pool(workers) as pool:
        dense = {f: (Xr, dr) for f, Xr, dr in pool.imap_unordered(mmfi_dense, [(f, spec) for f in files])}
    for f in files:
        P = np.load(f); Y, w = clean_mask(P)
        acc["sparse_v1"].add(cc.to_common13(mmfi_sparse(P, spec, OFF), "coco17"), Y, w)
        Xr, dr = dense[str(f)]
        acc["dense_nooff"].add(cc.to_common13(Xr, "coco17"), Y, w)
        X = cc.to_common13(Xr + OFF[:, None] * dr, "coco17")
        acc["dense_off"].add(X, Y, w); acc["dense_off_tmed"].add(tmed(X), Y, w)
        p, ok = depth_points(P, spec["depth"])
        r = person_root(p, ok, np.asarray(P["t_d2r"], float))
        acc["depth_root"].add(r, Y, w); acc["depth_root_tmed"].add(tmed(r), Y, w)
        o, d = rays(P["kp_px"], spec["rgb"]["K"], P["R_c2r"], P["t_c2r"])
        Xm, _ = mono_floor(o, d, P["R_c2r"], np.array([0, 0, 1.0]), 0.0)          # 방 좌표 바닥 z = 0 (깊이 RANSAC, 정답 없음)
        Xm = cc.to_common13(Xm, "coco17"); acc["mono"].add(Xm, Y, w); acc["mono_tmed"].add(tmed(Xm), Y, w)
    out["mmfi"] = {k: a.summary() for k, a in acc.items()}
    out["mmfi"]["dense_off_tmed_per_joint"] = acc["dense_off_tmed"].per_joint()
    # ---- mRI
    spec = json.loads(SPECS["mri"].read_text())
    tri_all = mri_tri_all(spec, {f"S{x:02d}" for x in range(1, 21)})               # 바닥 추정은 입력만 쓰므로 모든 피험자
    n_f, c_f, inl, nA = mri_floor_gtfree(tri_all)
    test = set(files_of("mri", split_of("mri", part)))
    acc = {k: Acc() for k in ("tri", "tri_tmed", "mono_cam1", "mono_cam1_tmed")}
    for f, (P, X17, o1, d1) in tri_all.items():
        if f not in test:
            continue
        Y, w = clean_mask(P)
        X = cc.to_common13(X17, "coco17"); acc["tri"].add(X, Y, w); acc["tri_tmed"].add(tmed(X), Y, w)
        Xm, _ = mono_floor(o1, d1, P["R_c2r"], n_f, c_f)
        Xm = cc.to_common13(Xm, "coco17"); acc["mono_cam1"].add(Xm, Y, w); acc["mono_cam1_tmed"].add(tmed(Xm), Y, w)
    out["mri"] = {k: a.summary() for k, a in acc.items()}
    out["mri"]["tri_tmed_per_joint"] = acc["tri_tmed"].per_joint()
    out["mri"]["floor_gtfree"] = dict(n=np.round(n_f, 4).tolist(), c=round(c_f, 4), inliers=inl, n_ankles=nA,
                                      tilt_vs_label_frame_deg=round(float(np.degrees(np.arccos(min(1.0, n_f[2])))), 2),
                                      height_offset_vs_label_floor_m=round(c_f, 3))
    # ---- OctoNet (깊이만 → root)
    spec = json.loads(SPECS["octonet"].read_text())
    acc = {k: Acc() for k in ("depth_root", "depth_root_tmed")}
    per_rec = []
    for f in files_of("octonet", split_of("octonet", part)):
        P = np.load(f); m = json.loads(P["meta"].item()); Y, w = clean_mask(P)
        p, ok = depth_points(P, spec["depth"][f"{m['env']}_{m['day']}"])
        r = person_root(p, ok, np.asarray(P["t_d2r"], float))
        acc["depth_root"].add(r, Y, w); acc["depth_root_tmed"].add(tmed(r), Y, w)
        e = np.linalg.norm(tmed(r) - 0.5 * (Y[:, 7] + Y[:, 8]), axis=-1)
        per_rec.append((m["stamp"], m["env"], m["subject"], m["action"], round(float(np.nanmedian(e) * 1000))))
    out["octonet"] = {k: a.summary() for k, a in acc.items()}
    out["octonet"]["worst_recordings"] = sorted(per_rec, key=lambda x: -x[-1])[:12]
    out["octonet"]["env_median_root_mm"] = {e: round(float(np.median([x[-1] for x in per_rec if x[1] == e]))) for e in sorted({x[1] for x in per_rec})}
    return out


if __name__ == "__main__":
    part = arg("--part", "test")
    res = dict(part=part, consts=dict(OFF=OFF.tolist(), HALF_BODY=HALF_BODY, TAU=TAU, BAND=BAND, MIN_SCORE=MIN_SCORE, WIN=WIN),
               **run(part))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    o = arg("--out")
    if o:
        Path(o).write_text(json.dumps(res, indent=1, ensure_ascii=False))
