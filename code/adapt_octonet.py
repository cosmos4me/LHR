"""OctoNet 어댑터 (변환기 v0.7): 노드 1 깊이(D455, 8비트) + 모캡 정답 → 녹화별 팩 (dev/packs/octonet_v07/{stamp}.npz).
v0.7: 추적기 기반 사람 선택 (v0.6은 adapt_octonet_v06.py로 보존). 모두 정답 없음.
      v0.6은 정지한 사람(stretch 등)을 "한 번도 안 변한 화소 = 배경" 규칙으로 지웠고(움직인 화소 비율 0.15–0.19 < 0.3),
      freestyle에서는 사람과 붙은 큰 물체 때문에 성분 중심이 0.3–0.8 m 밀렸다.
      - 추적 시작(탄생): 움직인 화소 수 ≥ MOV_PX(절대 개수)이고 높이 범위 ≥ 0.4 m인 성분 중 활동 중심 R_CENTER 안에서 가장 가까운 것.
      - 추적: 수평 위치·속도 등속 칼만 필터 (가속도 σ_A, 측정 σ_C), χ² 게이트 (2자유도 99.9%).
        측정 = 예측 위치에서 수평 R_GATE 안 전경 화소의 중앙값 (연결 성분을 쓰지 않으므로 붙은 물체는 원통 밖으로 잘림).
      - 측정이 없거나 게이트 밖이면 예측만 하고 그 프레임은 빈 프레임. LOST_MAX 넘게 놓치면 다시 탄생.
      - 녹화 앞부분(첫 탄생 전)은 첫 탄생 상태에서 시간을 거꾸로 추적해 채운다.
      - 추적이 활동 중심에서 R_TRACK 밖으로 나가면 끊고, 게이트 밖에 측정보다 SWITCH배 큰 탄생 후보가 있으면 갈아탄다
        (freestyle 녹화에서 뒤쪽 작은 움직이는 물체에 붙어 버린 실패를 고침).
      - 시각 정렬: notes/octonet_align_v2.json (octonet_realign.py; 2024-09-26 env2는 +4.55 s로 v1의 +1.0 s와 크게 다름).
v0.6: 움직임 기반 사람 선택 — 녹화 안에서 깊이가 변한 적 없는 화소(가구 등 정지 물체)는 전경에서 빼고, 움직인 화소가 적은 성분은 후보에서 뺀다 (정답 없음).
      v0.5는 264개 중 81개 녹화에서 정지 물체를 추적했다 (physics_baseline.py로 발견).
- 카메라: notes/calib_octonet_days_v5.json (f = 385 공유, 날짜별 자세 = 기준일 자세 ∘ 정답 없는 배경 ICP, 깊이 배율 a).
- 시각: 녹화별 깊이–모캡 오프셋(notes/octonet_align.json)으로 정답을 정렬. 깊이 30 Hz → 10 Hz (3프레임마다).
- 방 좌표: z 위(모캡 바닥 z = 0), 원점 = 카메라 바로 아래 바닥점, x = 광축의 바닥 투영 (MM-Fi와 같은 규약).
- 사람 영역 (정답 없음): 날짜별 배경 = 여러 녹화 깊이의 90 백분위(사람이 자주 같은 자리에 있어도 배경이 남게),
  배경보다 0.12 m 넘게 가깝고 바닥 위인 화소의 가장 큰 연결 성분. 섞인 가장자리 화소는 빼고 256점 표본, 가장자리 표시.
- RGB 없음 (공개본 제한) → kp_valid = False.
사용: python adapt_octonet.py [--bg-only]"""
import collections, json, sys, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.ndimage import label as cc_label, maximum_filter, minimum_filter, binary_opening

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_common as cc
import octonet_io as oi

NOTES = Path("/home/interns/unisense/notes")
CODE = Path(__file__).resolve().parent
CACHE = Path("/home/interns/unisense/dev/data/octonet/derived/mocap13")
BGF = Path("/home/interns/unisense/dev/data/octonet/derived/backgrounds_p90.npz")
OUT = Path("/home/interns/unisense/dev/packs/octonet_v07")
SPEC_PATH = CODE / "specs" / "octonet_v07.json"
N_PTS, STEP, FPS = 256, 3, 10.0
Z_TOP = 2.4                                    # 사람 화소 높이 상한 [m] (팔 든 점프 포함 사람 키 한계)
V_MAX, LOST_MAX, R_CENTER = 2.0, 30, 1.5       # 사람 수평 이동 한계 [m/s], 추적 유지 최대 놓친 프레임(3 s), 다시 찾기 반경 [m] (활동 영역)
N_MOT, MOV_FRAC, MOV_MIN, MOV_GROW = 60, 0.05, 0.3, 9   # 움직임 지도: 표본 프레임 수, "움직인" 프레임 비율 하한, 성분의 움직인 화소 비율 하한, 전경 허용 반경 [px]
MOV_PX = 500                                   # 탄생 성분의 움직인 화소 수 하한 (정지한 사람도 팔·몸통 일부는 움직임)
SIG_A, SIG_C = 3.0, 0.10                       # 칼만: 사람 수평 가속도 σ [m/s²], 측정(점 중앙값 ↔ 몸 중심) σ [m] (사전)
R_GATE, N_MIN, CHI2_2 = 0.8, 300, 13.82        # 측정 원통 반경 [m] (팔 벌린 사람 반폭), 최소 화소 수, χ²₂ 99.9%
R_TRACK, SWITCH = 2.0, 3.0                     # 추적 허용 영역 (활동 중심에서) [m], 더 큰 사람 후보로 갈아타는 화소 수 배율
MULTI = {"kicksomeone", "punchsomeone", "pushsomeone", "handshake", "hug"}       # 상대가 있는 동작: 다른 사람이 깊이에 잡힐 수 있음
CAL = json.loads((NOTES / "calib_octonet_days_v5.json").read_text())
OFF = {x["stamp"]: x for x in json.loads((NOTES / "octonet_align_v2.json").read_text())["offsets"]}     # v0.7: 재정렬 (octonet_realign.py)
CENTER_F = Path("/home/interns/unisense/dev/data/octonet/derived/activity_centers.json")
CENTERS = {k: np.array(v) for k, v in json.loads(CENTER_F.read_text()).items()} if CENTER_F.exists() else {}


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def geometry(env, day):
    E = CAL[str(env)]; D = E["days"][day]
    R_w2c, C = np.array(D["R_w2c"]), np.array(D["C_world"])
    fwd = R_w2c[2]; psi = np.arctan2(fwd[1], fwd[0])
    R_w2r = Rz(-psi); t_w2r = -R_w2r @ np.array([C[0], C[1], 0.0])                 # 세계 → 방
    R_c2r = R_w2r @ R_w2c.T; t_c2r = R_w2r @ C + t_w2r                                # 카메라 → 방 (카메라 중심 = (0, 0, C_z))
    return dict(K=E["K"], a=E["depth_scale_a"], R_w2r=R_w2r, t_w2r=t_w2r, R_c2r=R_c2r, t_c2r=t_c2r)


def depth_model(a):
    return dict(type="stereo", sensor="RealSense D455 (8-bit storage)", baseline_m=0.095, f_px=385.0,
                stereo_constant_px_m=385.0 * 0.095, subpixel_steps=32, sigma_disp_px=0.08,
                quant_step_m=10.0 * a / 255.0, sigma_pix_px=float(1 / np.sqrt(12)), depth_scale_a=a)


def backgrounds():
    """날짜별 배경 = 녹화 최대 12개 × 30프레임 깊이의 90 백분위 (원래 8비트 미터값, 배율 a 적용 전)."""
    by = collections.defaultdict(list)
    for r in oi.recordings():
        by[(r["env"], r["stamp"][:8])].append(r)
    rng = np.random.default_rng(0); out = {}
    for (e, day), recs in sorted(by.items()):
        Z = []
        for r in [recs[i] for i in rng.choice(len(recs), min(12, len(recs)), replace=False)]:
            dz = oi.DepthZip(r["depth"])
            Z += [dz.depth(i) for i in np.linspace(0, len(dz) - 1, 30).astype(int)]
        with np.errstate(all="ignore"):
            out[f"e{e}_{day}"] = np.nanpercentile(np.stack(Z), 90, axis=0).astype(np.float32)
        print("bg", e, day, len(Z), flush=True)
    np.savez_compressed(BGF, **out)


def motion_map(Z, idx, g, dm):
    """녹화 안 움직임 지도 (정답 없음): 표본 프레임에서 시간 중앙값과 max(0.10 m, 4σ_z) 넘게 다른 프레임이 MOV_FRAC 이상인 화소."""
    ii = idx[np.linspace(0, len(idx) - 1, min(N_MOT, len(idx))).astype(int)]
    D = np.stack([Z.depth(i) * g["a"] for i in ii])
    with np.errstate(all="ignore"):
        med = np.nanmedian(D, 0)
        thr = np.maximum(0.10, 4 * cc.axial_sigma(np.nan_to_num(med, nan=3.0), dm))
        return np.nanmean(np.abs(D - med) > thr, 0) >= MOV_FRAC


def components(z, bg, g, moving=None):
    """전경(배경보다 0.12 m 넘게 가까움, 바닥 위)의 연결 성분과 방 좌표 통계 (크기, 수평 중심, 높이 범위)."""
    a = g["a"]; fx, fy, cx, cy = g["K"]
    zm, bm = z * a, bg * a
    with np.errstate(invalid="ignore"):
        fg = np.isfinite(zm) & (zm > 0.3) & (zm < 9.0) & (~np.isfinite(bm) | (zm < bm - 0.12))
    vv, uu = np.mgrid[0:480, 0:640]
    P = np.stack([(uu - cx) / fx * zm, (vv - cy) / fy * zm, zm], -1) @ g["R_c2r"].T + g["t_c2r"]
    fg &= (np.nan_to_num(P[..., 2], nan=-1) > 0.05) & (np.nan_to_num(P[..., 2], nan=99) < Z_TOP)   # 바닥 위 ~ 사람 키 한계 (천장·벽 윗부분과 붙지 않게)
    if moving is not None:                                                        # 녹화 안에서 한 번도 안 변한 화소(정지 물체)는 전경에서 뺀다 (움직인 곳 근처는 허용)
        fg &= maximum_filter(moving, size=2 * MOV_GROW + 1)
    fg = binary_opening(fg, iterations=1)
    lab, n = cc_label(fg)
    stats = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < 300:
            continue
        Q = P[m]
        lo, hi = np.percentile(Q, 5, 0), np.percentile(Q, 95, 0)
        stats.append(dict(k=k, size=int(m.sum()), xy=np.median(Q[:, :2], 0), zspan=float(hi[2] - lo[2]), top=float(hi[2]),
                          yspan=float(hi[1] - lo[1]), mov=float(moving[m].mean()) if moving is not None else 1.0))
    return lab, stats, zm


def choose(stats, center, prev, lost=0):
    """사람 후보 = 높이 범위 ≥ 0.4 m이고 움직인 화소 비율 ≥ MOV_MIN인 성분 (전경은 이미 바닥 위 Z_TOP 아래로 잘림).
    1) 추적 중이면 이전 위치에서 사람 이동 한계(0.6 m + V_MAX·놓친 시간) 안의 가장 가까운 후보.
    2) 없으면 활동 중심 R_CENTER 안의 후보 중 가장 가까운 것 (다시 찾기). 처음 프레임도 이 규칙.
    3) 둘 다 없으면 고르지 않는다 (사람 아닌 물체를 잡는 것보다 빈 프레임이 낫다)."""
    cand = [c for c in stats if c["zspan"] >= 0.4 and c["mov"] >= MOV_MIN]
    if not cand:
        return None
    if prev is not None and lost <= LOST_MAX:
        near = [c for c in cand if np.linalg.norm(c["xy"] - prev) < 0.6 + V_MAX * lost / FPS]
        if near:
            return min(near, key=lambda c: np.linalg.norm(c["xy"] - prev))
    inner = [c for c in cand if np.linalg.norm(c["xy"] - center) < R_CENTER]
    return min(inner, key=lambda c: np.linalg.norm(c["xy"] - center)) if inner else None


def segment(z, bg, g, dm, center=None, prev=None, moving=None, lost=0):
    """정답 없는 사람 점: 전경 연결 성분 중 사람다운 것을 활동 중심·추적으로 고르고 섞인 화소 제거. 반환 (점, 섞인 수, 크기, 선택 중심)."""
    lab, stats, zm = components(z, bg, g, moving)
    c = choose(stats, center if center is not None else np.array([2.8, 0.0]), prev, lost)
    if c is None:
        return np.zeros((0, 4), np.float32), 0, 0, None
    comp = lab == c["k"]
    inv = ~np.isfinite(zm)
    smax = maximum_filter(np.where(inv, -np.inf, zm), size=7); smin = minimum_filter(np.where(inv, np.inf, zm), size=7)
    sz = cc.axial_sigma(np.nan_to_num(zm, nan=3.0), dm)
    with np.errstate(invalid="ignore"):
        mixed = (smax - smin > 6 * sz) & (zm - smin > 3 * sz) & (smax - zm > 3 * sz)
    edge = maximum_filter(inv, size=7) | (smax - smin > 0.2)
    keep = comp & ~mixed
    v, u = np.nonzero(keep)
    pts = np.c_[u.astype(np.float32), v.astype(np.float32), zm[v, u].astype(np.float32), edge[v, u].astype(np.float32)]
    return pts, int((comp & mixed).sum()), c["size"], c["xy"]


def fg_points(z, bg, g):
    """전경 화소 (배경보다 0.12 m 넘게 가까움, 바닥 위 0.05 m ~ Z_TOP)와 방 좌표 점. 움직임으로 거르지 않는다."""
    a = g["a"]; fx, fy, cx, cy = g["K"]
    zm, bm = z * a, bg * a
    with np.errstate(invalid="ignore"):
        fg = np.isfinite(zm) & (zm > 0.3) & (zm < 9.0) & (~np.isfinite(bm) | (zm < bm - 0.12))
    vv, uu = np.mgrid[0:480, 0:640]
    P = np.stack([(uu - cx) / fx * zm, (vv - cy) / fy * zm, zm], -1) @ g["R_c2r"].T + g["t_c2r"]
    fg &= (np.nan_to_num(P[..., 2], nan=-1) > 0.05) & (np.nan_to_num(P[..., 2], nan=99) < Z_TOP)
    return binary_opening(fg, iterations=1), P, zm


def measure(fg, P, xy):
    """예측 위치 xy에서 수평 R_GATE 안 전경 화소: 중앙값을 구하고, 그 중앙값 기준으로 한 번 더 자른다. 반환 (중앙값, 마스크) 또는 (None, None)."""
    for _ in range(2):
        d = np.linalg.norm(np.nan_to_num(P[..., :2], nan=1e3) - xy, axis=-1)
        m = fg & (d < R_GATE)
        if m.sum() < N_MIN:
            return None, None
        xy = np.median(P[m][:, :2], 0)
    return xy, m


def birth(fg, P, moving, center):
    """탄생 후보: 연결 성분 중 높이 범위 ≥ 0.4 m, 움직인 화소 ≥ MOV_PX, 활동 중심 R_CENTER 안. 가장 가까운 것의 (수평 중심, 화소 수)."""
    lab, n = cc_label(fg); best = None
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < N_MIN or int(moving[m].sum()) < MOV_PX:
            continue
        Q = P[m]; lo, hi = np.percentile(Q[:, 2], 5), np.percentile(Q[:, 2], 95)
        xy = np.median(Q[:, :2], 0); dc = np.linalg.norm(xy - center)
        if hi - lo >= 0.4 and dc < R_CENTER and (best is None or dc < best[0]):
            best = (dc, xy, int(m.sum()))
    return None if best is None else (best[1], best[2])


def track(get, T, moving, center):
    """get(k) → (fg, P, zm). 등속 칼만 + χ² 게이트 + 다시 탄생. 프레임을 저장하지 않는다 (메모리).
    반환: 프레임별 측정된 사람 수평 중심(또는 None), 첫 탄생 프레임."""
    dt = STEP / 30.0
    Fm = np.eye(4); Fm[0, 2] = Fm[1, 3] = dt
    q = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * SIG_A ** 2
    Qm = np.zeros((4, 4)); Qm[np.ix_([0, 2], [0, 2])] = q; Qm[np.ix_([1, 3], [1, 3])] = q
    Hm = np.eye(2, 4); Rm = SIG_C ** 2 * np.eye(2); P0 = np.diag([SIG_C ** 2] * 2 + [1.0, 1.0])

    def run(order, x0):
        out = {}; x, Pc, lost = x0, (None if x0 is None else P0.copy()), 0
        for k in order:
            fg, P, _ = get(k)
            cand = birth(fg, P, moving, center)
            if x is not None and np.linalg.norm(x[:2] - center) > R_TRACK:                 # 활동 영역 밖으로 나간 추적은 끊는다
                x, lost = None, 0
            if x is None or lost > LOST_MAX:
                if cand is None:
                    out[k] = None; continue
                x, Pc, lost = np.r_[cand[0], 0.0, 0.0], P0.copy(), 0
            else:
                x, Pc = Fm @ x, Fm @ Pc @ Fm.T + Qm
            xy, m = measure(fg, P, x[:2])
            if cand is not None and np.linalg.norm(cand[0] - x[:2]) > R_GATE and cand[1] > SWITCH * (0 if m is None else m.sum()):
                x, Pc, lost = np.r_[cand[0], 0.0, 0.0], P0.copy(), 0                         # 훨씬 큰 사람 후보가 따로 있으면 갈아탄다
                xy, m = measure(fg, P, x[:2])
            if xy is not None:
                nu = xy - Hm @ x; S = Hm @ Pc @ Hm.T + Rm
                if nu @ np.linalg.solve(S, nu) <= CHI2_2:
                    Kg = Pc @ Hm.T @ np.linalg.inv(S); x = x + Kg @ nu; Pc = (np.eye(4) - Kg @ Hm) @ Pc; lost = 0
                    out[k] = xy; continue
            lost += 1; out[k] = None
        return out

    fwd = run(range(T), None)
    first = next((k for k in range(T) if fwd[k] is not None), None)
    if first:                                                                   # 첫 탄생 전 구간은 거꾸로 추적
        bwd = run(range(first - 1, -1, -1), np.r_[fwd[first], 0.0, 0.0])
        fwd.update(bwd)
    return [fwd[k] for k in range(T)], first


def activity_centers():
    """환경별 활동 중심 = 학습 사용자(1–5) 모캡 골반(엉덩이 중점)의 방 좌표 수평 위치 중앙값 (보정과 같은 학습 데이터 규칙).
    정답 없는 추정(사람 높이 전경 성분 중앙값)은 정적 물체 때문에 치우쳐서(환경 1에서 4.67 m) 쓰지 않는다."""
    train = set(json.loads((CODE / "splits" / "octonet.json").read_text())["train"])
    out = {}
    for e in (1, 2, 3):
        XY = []
        for r in oi.recordings():
            if r["env"] != e or r["user"] not in train:
                continue
            g = geometry(e, r["stamp"][:8]); J = np.load(CACHE / f"{r['stamp']}.npz")["joints13"].astype(float)[::30]
            XY.append((0.5 * (J[:, 7] + J[:, 8]) @ g["R_w2r"].T + g["t_w2r"])[:, :2])
        out[str(e)] = np.median(np.concatenate(XY), 0).tolist()
        print("activity centre env", e, np.round(out[str(e)], 2), flush=True)
    return out


def _unused_activity_centers_gtfree():
    """(사용 안 함) 정답 없는 활동 중심."""
    out = {}
    rng = np.random.default_rng(1)
    for e in (1, 2, 3):
        recs = [r for r in oi.recordings() if r["env"] == e]
        XY = []
        for r in [recs[i] for i in rng.choice(len(recs), min(30, len(recs)), replace=False)]:
            g = geometry(e, r["stamp"][:8]); bg = np.load(BGF)[f"e{e}_{r['stamp'][:8]}"]; Z = oi.DepthZip(r["depth"])
            for i in np.linspace(0, len(Z) - 1, 6).astype(int):
                _, stats, _ = components(Z.depth(i), bg, g)
                XY += [c["xy"] for c in stats if 0.8 <= c["zspan"] <= 2.2]
        out[str(e)] = np.median(np.array(XY), 0).tolist() if XY else [2.8, 0.0]
        print("activity centre env", e, np.round(out[str(e)], 2), "from", len(XY), "components", flush=True)
    return out


def _unused_segment(z, bg, g, dm):
    """(이전 방식, 사용 안 함) 가장 큰 연결 성분."""
    a = g["a"]; fx, fy, cx, cy = g["K"]
    zm, bm = z * a, bg * a
    with np.errstate(invalid="ignore"):
        fg = np.isfinite(zm) & (zm > 0.3) & (zm < 9.0) & (~np.isfinite(bm) | (zm < bm - 0.12))
    vv, uu = np.mgrid[0:480, 0:640]
    P = np.stack([(uu - cx) / fx * zm, (vv - cy) / fy * zm, zm], -1) @ g["R_c2r"].T + g["t_c2r"]
    fg &= np.nan_to_num(P[..., 2], nan=-1) > 0.05
    fg = binary_opening(fg, iterations=1)
    lab, n = cc_label(fg)
    if n == 0:
        return np.zeros((0, 4), np.float32), 0, 0
    sizes = np.bincount(lab.ravel())[1:]; k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < 300:
        return np.zeros((0, 4), np.float32), 0, int(sizes[k - 1])
    comp = lab == k
    inv = ~np.isfinite(zm)
    smax = maximum_filter(np.where(inv, -np.inf, zm), size=7); smin = minimum_filter(np.where(inv, np.inf, zm), size=7)
    sz = cc.axial_sigma(np.nan_to_num(zm, nan=3.0), dm)
    with np.errstate(invalid="ignore"):
        mixed = (smax - smin > 6 * sz) & (zm - smin > 3 * sz) & (smax - zm > 3 * sz)
    edge = maximum_filter(inv, size=7) | (smax - smin > 0.2)
    keep = comp & ~mixed
    v, u = np.nonzero(keep)
    pts = np.c_[u.astype(np.float32), v.astype(np.float32), zm[v, u].astype(np.float32), edge[v, u].astype(np.float32)]
    return pts, int((comp & mixed).sum()), int(sizes[k - 1])


def process(r):
    out = OUT / f"{r['stamp']}.npz"
    if out.exists():
        return r["stamp"], "skip"
    day = r["stamp"][:8]; g = geometry(r["env"], day); dm = depth_model(g["a"])
    bg = np.load(BGF)[f"e{r['env']}_{day}"]
    M = np.load(CACHE / f"{r['stamp']}.npz"); t, J = M["t"], M["joints13"].astype(float)
    dt = OFF[r["stamp"]]["dt"]; Z = oi.DepthZip(r["depth"])
    idx = np.where((Z.t + dt > t[0]) & (Z.t + dt < t[-1]))[0][::STEP]
    T = len(idx)
    rng = np.random.default_rng(zlib.crc32(f"octonet/{r['stamp']}".encode()))
    pts = np.zeros((T, N_PTS, 4), np.float32); valid = np.zeros((T, N_PTS), bool)
    n_raw = np.zeros(T, np.int32); n_mix = np.zeros(T, np.int32); comp = np.zeros(T, np.int32)
    center = np.array(json.loads(SPEC_PATH.read_text())["activity_center_room_xy"][str(r["env"])]) if False else None
    moving = motion_map(Z, idx, g, dm) if T else None
    get = lambda k: fg_points(Z.depth(idx[k]), bg, g)
    cen, first = track(get, T, moving, CENTERS.get(str(r["env"]), np.array([2.8, 0.0]))) if T else ([], None)
    for k in range(T):
        if cen[k] is None:
            continue
        fg, P, zm = get(k)
        sel = fg & (np.linalg.norm(np.nan_to_num(P[..., :2], nan=1e3) - cen[k], axis=-1) < R_GATE)
        inv = ~np.isfinite(zm)
        smax = maximum_filter(np.where(inv, -np.inf, zm), size=7); smin = minimum_filter(np.where(inv, np.inf, zm), size=7)
        sz = cc.axial_sigma(np.nan_to_num(zm, nan=3.0), dm)
        with np.errstate(invalid="ignore"):
            mixed = (smax - smin > 6 * sz) & (zm - smin > 3 * sz) & (smax - zm > 3 * sz)
        edge = maximum_filter(inv, size=7) | (smax - smin > 0.2)
        v, u = np.nonzero(sel & ~mixed)
        p = np.c_[u.astype(np.float32), v.astype(np.float32), zm[v, u].astype(np.float32), edge[v, u].astype(np.float32)]
        n_raw[k], n_mix[k], comp[k] = len(p), int((sel & mixed).sum()), int(sel.sum())
        if len(p):
            pick = rng.choice(len(p), min(N_PTS, len(p)), replace=False)
            pts[k, :len(pick)] = p[pick]; valid[k, :len(pick)] = True
    j, err = oi.nearest(t, Z.t[idx] + dt)
    Jw = J[j]                                                                        # (T,13,3) 세계
    lab = Jw @ g["R_w2r"].T + g["t_w2r"]
    q = np.zeros((T, 13), np.uint8)
    q[1:][np.all(Jw[1:] == Jw[:-1], -1)] |= 1
    if OFF[r["stamp"]]["dt_source"] != "own":
        q |= 8                                                                       # 시각 정렬이 날 중앙값으로 대체됨
    sp = np.linalg.norm(np.gradient(Jw, axis=0), axis=-1) * FPS
    meta = dict(dataset="octonet", env=r["env"], user=r["user"], subject=f"U{r['user']:02d}", action=r["action"], trial=r["trial"],
                stamp=r["stamp"], day=day, T=int(T), fps=FPS, label_family="optitrack_motive13", label_source="optitrack_mocap",
                kp_family=None, has_rgb=False, multi_person_action=r["action"] in MULTI, time_offset_s=dt, time_offset_source=OFF[r["stamp"]]["dt_source"],
                spec=str(SPEC_PATH), version="v0.7", first_birth=first, seed=int(zlib.crc32(f"octonet/{r['stamp']}".encode())))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, kp_px=np.zeros((T, 17, 2), np.float32), kp_valid=np.zeros((T, 17), bool),
                        pts_uvz=pts[..., :3], pts_edge=pts[..., 3] > 0.5, pts_valid=valid, pts_n_raw=n_raw, pts_n_mixed=n_mix,
                        comp_size=comp, labels=lab.astype(np.float32), labels_world=Jw.astype(np.float32),
                        label_mask=np.ones((T, 13), bool), label_quality=q, joint_speed=sp.astype(np.float32),
                        R_c2r=g["R_c2r"], t_c2r=g["t_c2r"], R_d2r=g["R_c2r"], t_d2r=g["t_c2r"], meta=json.dumps(meta))
    return r["stamp"], "ok"


def write_spec():
    S_ = dict(dataset="octonet", version="v0.7", spec_path=str(SPEC_PATH), fps=FPS, label_family="optitrack_motive13",
              label_source="optitrack_mocap", rgb=None, depth={}, provenance=dict(
                  calibration="notes/calib_octonet_days_v5.json", time_alignment="notes/octonet_align.json",
                  background="90th percentile of depth over up to 12 recordings x 30 frames per day (GT-free)",
                  person_selection="v0.7 tracker: birth = connected component with >= 500 moved pixels and z-span >= 0.4 m nearest to activity centre (1.5 m); then constant-velocity Kalman (accel 3 m/s^2, meas 0.10 m), chi2 gate 99.9%, measurement = median of foreground pixels within 0.8 m horizontal of prediction (re-centred once); lost > 3 s -> re-birth; frames before first birth tracked backwards. Foreground not motion-gated. (v0.6: motion-gated foreground (pixels that never changed within the recording are background); foreground clipped to 0.05-2.4 m above floor (human height limit); z-span >= 0.4 m; >= 30% moved pixels; tracking with 2 m/s speed limit, else re-acquire nearest to activity centre within 1.5 m, else empty frame"))
    for e in ("1", "2", "3"):
        E = CAL[e]
        for day in E["days"]:
            g = geometry(int(e), day)
            S_["depth"][f"{e}_{day}"] = dict(depth_model(g["a"]), K=E["K"], R_c2r=g["R_c2r"].tolist(), t_c2r=g["t_c2r"].tolist(),
                                              camera_height_m=float(g["t_c2r"][2]))
    SPEC_PATH.write_text(json.dumps(S_, indent=1))


if __name__ == "__main__":
    if not BGF.exists() or "--bg" in sys.argv:
        backgrounds()
    if not CENTER_F.exists() or "--centers" in sys.argv:
        CENTER_F.write_text(json.dumps(activity_centers()))
        CENTERS.update({k: np.array(v) for k, v in json.loads(CENTER_F.read_text()).items()})
    if "--bg-only" in sys.argv:
        sys.exit(0)
    write_spec()
    R = oi.recordings()
    with Pool(8) as p:                                                              # CPU 공유 (다른 작업과 함께)
        for i, (s, st) in enumerate(p.imap_unordered(process, R), 1):
            if i % 25 == 0:
                print(i, "/", len(R), flush=True)
    print("packs done", len(R))
