"""11-24 캐시 v3 = v2 + 정답 없는 정합·바닥 (보고서 notes/reports/정답 없는 레이더 정합 방법.md). v3 산출물이 있으면 그것을, 없으면 v2 를 쓰고 출처를 meta 에 기록.
- 바닥: dev/packs/v3/floors.json (팩 경로별 옛 방 → 새 방 강체 변환) 을 모든 방 좌표량 (광선·깊이·LiDAR·레이더·org·라벨) 에 적용
- 레이더: v3/mmfi_radar, v3/octonet_radar, v3/mri_radar, v3/mmbody_radar;  LiDAR: v3/mmfi_lidar
출력: dev/packs/ms_cache_v3/*.npz + ms_cache_v3/sources.json

11-23 LHR 다중 센서 캐시 v2: 데이터셋마다 통째로 (피험자 분할 없음), 디스크에 있는 모든 모달리티 → 프레임별 고정 크기 토큰.
평가 = 데이터셋 하나를 통째로 빼고 나머지로 학습, 뺀 데이터셋의 모든 입력 → 관절 (ms_train2.py --hold).
모달리티 (있는 데이터셋):
- rgb   광선 (2, 13, 7) 원점·방향·유효 — MM-Fi (카메라 1, 2D 키포인트), mRI (카메라 2대, ViTPose 2D; pose_2d 는 정답 누출이라 안 씀)
- depth 점 (128, 6) 방 x y z + 색 r g b (mmBody 만 색, 나머지 0) — MM-Fi, mmBody (원본 점군, 정답 상자로 잘린 신탁), OctoNet, ITOP
- lidar 점 (128, 3) — MM-Fi LiDAR (mmfi_lidar_v02, 모든 피험자)
- radar 점 (96, 6) x y z · 도플러 · dt · 정지 표시 — MM-Fi, mRI, mmBody (신탁 선택), OctoNet (환경 1·2)
- wifi  (513,) — MM-Fi CSI: log 진폭 평균·표준편차 + 위상 (부반송파 방향 unwrap·선형 제거) 평균, 부반송파 2 개씩 평균 (3 안테나 × 57 × 3)
- imu   (6, 13) 가속 3 · 각 3 · 자기 3 · 쿼터니언 4 — mRI (행 i ↔ 레이더 CSV i 번째 정렬 프레임)
- org   (3,) 주 카메라 중심 (RGB 없으면 깊이 카메라) — 가상 RGB (⑤) 광선 원점
출력: dev/packs/ms_cache_v2/{mmfi,mri,mmbody,octonet,itop}.npz"""
import sys, os, json, glob, csv
from multiprocessing import Pool
sys.path.insert(0, "/home/interns/unisense/code")
import numpy as np
import scipy.io as sio
import conv_common as cc

U = "/home/interns/unisense"; OUT = os.environ.get("CACHE_OUT", f"{U}/dev/packs/ms_cache_v3b"); V3 = f"{U}/dev/packs/v3"
FL = json.load(open(f"{V3}/floors.json")) if os.path.exists(f"{V3}/floors.json") else {}


def v3(path_v3, path_v2):
    """v3 산출물이 있으면 그 경로, 없으면 v2."""
    return (path_v3, "v3") if os.path.exists(path_v3) else (path_v2, "v2")


DB = {}
if os.path.exists(f"{V3}/mmfi_manifest.json"):                                # MM-Fi 세션별 D435 시차 편향 Δ(1/Z) (정답 없이 LiDAR·레이더 일치로 추정; 결정 C: 깊이·라벨 함께 보정)
    for e_, v_ in json.load(open(f"{V3}/mmfi_manifest.json"))["validation"]["depth_session_bias"].items():
        for s_, w_ in v_["per_subject"].items():
            DB[(e_, s_)] = float(w_["dinv"])


def disparity_fix(P, pts, valid, dinv):
    """깊이 카메라 좌표에서 1/Z ← 1/Z − Δ(1/Z), 점은 시선을 따라 다시 놓음. pts 는 방 좌표 (…,3)."""
    Rd, td = np.asarray(P["R_d2r"], np.float64), np.asarray(P["t_d2r"], np.float64); X = (pts.astype(np.float64) - td) @ Rd    # 방 → 깊이 카메라
    Z = X[..., 2]; ok = valid & (Z > 0.2); Zc = 1.0 / np.maximum(1.0 / np.where(ok, Z, 1.0) - dinv, 1e-3)
    Xc = X * np.where(ok, Zc / np.where(ok, Z, 1.0), 1.0)[..., None]
    return (Xc @ Rd.T + td).astype(np.float32)


PUSH, ZC = 0.10, 0.035   # 관례 (모든 데이터셋 같음): 레이더 중심 = 깊이 보이는 표면 + 0.10 m (몸 중심, HALF_BODY); 깊이 사람 중심 높이 ≈ 골반 + 0.035 m (학습 데이터셋 진단 평균)


def _tmean(x, lo=10, hi=90):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    a, b = np.percentile(x, [lo, hi]); x = x[(x >= a) & (x <= b)]
    return float(x.mean()) if len(x) else np.nan


def unify_radar(C, ref=None):
    """11-24 관례 통일 (정답 없음): 녹화마다 레이더 점을 (시선 방향 + 높이) 평행이동해
    시선 성분 = 기준 (깊이 표면 + PUSH, 또는 ref 몸 중심), 높이 = 기준 높이. ref: (T,3) 정답 없는 몸 중심 (mRI 스테레오 엉덩이) — 없으면 깊이 사용."""
    rv = C["rv"]
    if ref is None:
        dv = C["dv"]; nd = dv.sum(1); wd = dv.astype(np.float32); cd = (C["dep"][..., :3] * wd[..., None]).sum(1) / np.maximum(nd, 1)[:, None]; base = nd >= 10; push, zt = PUSH, cd[:, 2]
    else:
        cd = ref; base = np.isfinite(ref).all(1); push, zt = 0.0, ref[:, 2] + ZC
    # 11-24 감사 C2: 사람 문 (기준 수평 0.8 m 안) 을 통과한 점만으로 중심 (클러터·유령점이 녹화 이동을 끌던 문제)
    gate = C["rv"] & (np.linalg.norm(C["rad"][..., :2] - np.nan_to_num(cd)[:, None, :2], axis=-1) < 0.8)
    nr = gate.sum(1); w = gate.astype(np.float32); cr = (C["rad"][..., :3] * w[..., None]).sum(1) / np.maximum(nr, 1)[:, None]; ok = base & (nr >= 3)
    if ok.sum() < 10:
        return C, "unify:skip"
    u = cd[:, :2] - C["org"][:, :2]; u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-3); um = _tmean(u[ok, 0]), _tmean(u[ok, 1]); um = np.array(um) / max(np.hypot(*um), 1e-3)
    dlos = _tmean((((cd[:, :2] + push * u) - cr[:, :2]) * u).sum(1)[ok]); dz = _tmean((zt - cr[:, 2])[ok])
    if not (np.isfinite(dlos) and np.isfinite(dz)):
        return C, "unify:skip"
    sh = np.array([dlos * um[0], dlos * um[1], dz], np.float32); C["rad"][..., :3] = np.where(rv[..., None], C["rad"][..., :3] + sh, C["rad"][..., :3])
    return C, f"unify:dlos={dlos:+.3f},dz={dz:+.3f}"


def stereo_pelvis(ray):
    """mRI: 두 카메라 엉덩이 광선 최근접점 (정답 없음). ray (T,2,13,7)."""
    o1, d1, o2, d2 = ray[:, 0, [7, 8], :3], ray[:, 0, [7, 8], 3:6], ray[:, 1, [7, 8], :3], ray[:, 1, [7, 8], 3:6]
    v = (ray[:, 0, [7, 8], 6] > 0) & (ray[:, 1, [7, 8], 6] > 0); w = o1 - o2; a = (d1 * d1).sum(-1); b = (d1 * d2).sum(-1); c = (d2 * d2).sum(-1); dd = (d1 * w).sum(-1); e = (d2 * w).sum(-1)
    den = np.where(np.abs(a * c - b * b) < 1e-9, np.nan, a * c - b * b); s_ = (b * e - c * dd) / den; t_ = (a * e - b * dd) / den; X = 0.5 * ((o1 + s_[..., None] * d1) + (o2 + t_[..., None] * d2))
    X = np.where(v[..., None], X, np.nan); P = np.nanmean(X, 1); dist = np.linalg.norm(P - o1[:, 0], axis=1); P[(dist > 8) | (dist < 0.5)] = np.nan
    return P


def _mmfi_env_floor():
    """감사 M1: MM-Fi 바닥을 방 (E01–E04) 마다 하나로 — 시차 편향 없는 세션 (dinv = 0) 의 높이 변화 중앙값 세션 변환을 그 방 모든 녹화에 씀."""
    out = {}
    for e in ("E01", "E02", "E03", "E04"):
        ks = [k for k in FL if "/mmfi_v02_specA/" in k and k.split("/")[-2] == e and DB.get((e, os.path.basename(k).split("_")[0]), 1.0) == 0.0]
        if ks:
            hs = np.array([FL[k].get("height_change_m", 0.0) for k in ks]); out[e] = ks[int(np.argsort(hs)[len(hs) // 2])]
    return out


def floor_fix(C, y, f):
    """바닥 보정: 새 방 = R · 옛 방 + t (유효한 항목만). 반환 (C, y, 출처)."""
    e = FL.get(f)
    if f and "/mmfi_v02_specA/" in f and MMFI_FLOOR.get(f.split("/")[-2]):
        e = FL[MMFI_FLOOR[f.split("/")[-2]]]
    if e is None:
        return C, y, "none"
    Rm, t = np.asarray(e["R"], np.float32), np.asarray(e["t"], np.float32); pt = lambda X: X @ Rm.T + t
    C["dep"][..., :3] = np.where(C["dv"][..., None], pt(C["dep"][..., :3]), C["dep"][..., :3]); C["lid"] = np.where(C["lv"][..., None], pt(C["lid"]), C["lid"])
    C["rad"][..., :3] = np.where(C["rv"][..., None], pt(C["rad"][..., :3]), C["rad"][..., :3]); C["org"] = pt(C["org"])
    rv = (C["ray"][..., 6] > 0)[..., None]; C["ray"][..., :3] = np.where(rv, pt(C["ray"][..., :3]), C["ray"][..., :3]); C["ray"][..., 3:6] = np.where(rv, C["ray"][..., 3:6] @ Rm.T, C["ray"][..., 3:6])
    return C, pt(y), e.get("source", "floor"); os.makedirs(OUT, exist_ok=True)
H13_KP = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
ND, NL, NR, NRF, NW = 128, 128, 96, 32, 513
IMU_COLS = ["axg", "ayg", "azg", "AngleXdeg", "AngleYdeg", "AngleZdeg", "hx", "hy", "hz", "q0", "q1", "q2", "q3"]
MRI_RAW = f"{U}/dev/data/mri/extracted/dataset_release/aligned_data"
SP = {}


def spec(p):
    if p not in SP:
        SP[p] = json.load(open(p))
    return SP[p]


def k4(K):
    K = np.asarray(K, float)
    return K if K.ndim == 1 else np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])


def rays(kp, valid, K, Rc, tc):
    o, d, *_ = cc.ray_tokens(kp[:, H13_KP].astype(float), k4(K), Rc, tc, 1.0)
    out = np.zeros((len(kp), 13, 7), np.float32); out[..., :3] = o; out[..., 3:6] = d; out[..., 6] = valid[:, H13_KP]
    out[~np.isfinite(out).all(-1)] = 0
    return out


R_CYL, Z_LO, Z_HI, VOX = 0.6, -0.15, 2.3, 0.02   # 6번 (11-24): 모든 깊이 데이터셋 같은 사람 점 규칙 — 점 수평 중앙 기준 반경 0.6 m 원통, 2 cm 복셀, 최원점 표본


def select_person(X, n):
    """X (m,3) 방 좌표 사람 후보 점 → 같은 규칙으로 고른 인덱스 (≤ n). 정답 안 씀."""
    if len(X) == 0:
        return np.zeros(0, int)
    c = np.median(X[:, :2], 0); keep = np.flatnonzero((np.linalg.norm(X[:, :2] - c, axis=1) < R_CYL) & (X[:, 2] > Z_LO) & (X[:, 2] < Z_HI))
    if len(keep) == 0:
        return keep
    _, first = np.unique(np.floor(X[keep] / VOX).astype(np.int64), axis=0, return_index=True); keep = keep[np.sort(first)]
    if len(keep) <= n:
        return keep
    Y = X[keep]; sel = [int(np.argmin(np.linalg.norm(Y[:, :2] - c, axis=1)))]; d = np.linalg.norm(Y - Y[sel[0]], axis=1)
    for _ in range(n - 1):
        j = int(np.argmax(d)); sel.append(j); d = np.minimum(d, np.linalg.norm(Y - Y[j], axis=1))
    return keep[np.array(sel)]


def uvz_points(P, ks, K, Rd, td, n, rng):
    fx, fy, cx, cy = k4(K); out = np.zeros((len(ks), n, 3), np.float32); v = np.zeros((len(ks), n), bool)
    for i, k in enumerate(ks):
        ii = np.flatnonzero(P["pts_valid"][k])
        if len(ii) == 0:
            continue
        u_, w_, z = P["pts_uvz"][k, ii].astype(float).T; X = np.c_[(u_ - cx) / fx * z, (w_ - cy) / fy * z, z] @ np.asarray(Rd, float).T + np.asarray(td, float)
        sel = select_person(X, n); out[i, :len(sel)] = X[sel]; v[i, :len(sel)] = True
    return out, v


def radar_tokens(rf, ks, fps, rng):
    rad = np.zeros((len(ks), NR, 6), np.float32); rv = np.zeros((len(ks), NR), bool)
    Rz = np.load(rf, allow_pickle=False); rp = Rz["radar_pts"]; ro = Rz["radar_off"]; st = Rz["radar_static"] if "radar_static" in Rz.files else np.zeros(len(rp), bool)
    nb = max(1, int(round(0.25 * fps)))
    for i, k in enumerate(ks):
        rows = []
        for j in range(max(0, k - nb), k + 1):
            a, b = ro[j], ro[j + 1]
            if b > a:
                sel = np.arange(a, b) if b - a <= NRF else rng.choice(np.arange(a, b), NRF, replace=False)
                rows.append(np.c_[rp[sel, :3], np.clip(rp[sel, 3], -3, 3), np.full(len(sel), (j - k) / fps), st[sel]])
        if rows:
            R_ = np.concatenate(rows)[-NR:]; rad[i, :len(R_)] = R_; rv[i, :len(R_)] = True
    return rad, rv


def wifi_feat(path):
    m = sio.loadmat(path); a = np.nan_to_num(np.asarray(m["CSIamp"], float), nan=0, posinf=0, neginf=0); ph = np.nan_to_num(np.asarray(m["CSIphase"], float), nan=0, posinf=0, neginf=0)
    la = np.log1p(np.abs(a)); pu = np.unwrap(ph, axis=1); x = np.arange(pu.shape[1]) - (pu.shape[1] - 1) / 2
    slope = (pu * x[None, :, None]).sum(1, keepdims=True) / (x ** 2).sum(); pu = pu - pu.mean(1, keepdims=True) - slope * x[None, :, None]
    f = np.stack([la.mean(2), la.std(2), pu.mean(2)], 0)                                   # (3 종류, 3 안테나, 114)
    f = f.reshape(3, 3, 57, 2).mean(-1)
    return f.reshape(-1).astype(np.float32)


def empty(n):
    return dict(ray=np.zeros((n, 2, 13, 7), np.float32), dep=np.zeros((n, ND, 6), np.float32), dv=np.zeros((n, ND), bool), lid=np.zeros((n, NL, 3), np.float32), lv=np.zeros((n, NL), bool),
                rad=np.zeros((n, NR, 6), np.float32), rv=np.zeros((n, NR), bool), wifi=np.zeros((n, NW), np.float32), wv=np.zeros(n, bool), imu=np.zeros((n, 6, 13), np.float32), iv=np.zeros(n, bool),
                org=np.zeros((n, 3), np.float32))


def finish(C, P, ks, ds, sid, m, fps, act=None, f=None, src=None):
    n = len(ks); y = P["labels"][ks].astype(np.float32); ym = P["label_mask"][ks] & np.isfinite(y).all(-1); y[~np.isfinite(y)] = 0
    ctr = np.where((ym[:, 7] & ym[:, 8])[:, None], y[:, [7, 8]].mean(1), np.nan_to_num(np.nanmedian(np.where(ym[..., None], y, np.nan), axis=1)))   # 골반 (없으면 관절 중앙)
    bad = ~(np.linalg.norm(y - ctr[:, None], axis=-1) < 1.2)   # 11-24 감사 C1: 망가진 정답 관절 (골반에서 1.2 m 밖) → 정답 없음 (모든 데이터셋 같은 규칙)
    ym = ym & ~bad; C["_gtbad"] = int((bad & P["label_mask"][ks]).sum())
    dsrc = "none"
    if ds == "mmfi" and DB:                                                        # 결정 C (11-24): 세션 시차 보정을 깊이 점과 라벨에 함께
        key = (os.path.basename(os.path.dirname(f)), str(m.get("subject", "")).split(":")[-1]); dv_ = DB.get(key)
        if dv_ is not None and dv_ != 0.0:
            C["dep"][..., :3] = np.where(C["dv"][..., None], disparity_fix(P, C["dep"][..., :3], C["dv"], dv_), C["dep"][..., :3]); y = disparity_fix(P, y, ym, dv_); dsrc = f"dinv={dv_:+.5f}"
        elif dv_ is not None:
            dsrc = "dinv=0"
    usrc = "none"
    C["rv"] = C["rv"] & (C["rad"][..., 2] > -0.1) & (C["rad"][..., 2] < 2.4)    # 11-24 감사 C3: 높이 유령점 (바닥 아래·2.4 m 위) 제거, 모든 레이더 같은 규칙
    if C["rv"].any() and ds in ("mmfi", "octonet", "mmbody"):
        C, usrc = unify_radar(C)
    elif C["rv"].any() and ds == "mri":
        C, usrc = unify_radar(C, ref=stereo_pelvis(C["ray"].astype(np.float64)))
    if usrc == "unify:skip":                                                       # 감사 M4: 관례를 못 맞춘 녹화는 레이더를 쓰지 않음
        C["rv"][:] = False
    C, y, fsrc = floor_fix(C, y, f); y[~ym] = 0; C["_src"] = dict(src or {}, floor=fsrc, disparity=dsrc, unify=usrc, gtbad=C.pop("_gtbad", 0), pack=f)
    C.update(y=y, ym=ym, seq=np.full(n, sid, np.int32), t=(ks / fps).astype(np.float32), act=np.asarray(act if act is not None else np.full(n, str(m.get("action", ""))), dtype="U24"),
             sub=np.full(n, f"{ds}:{m.get('subject', '')}", dtype="U24"), ds=np.full(n, ds, dtype="U8"))
    for k in ("ray", "dep", "lid", "rad", "wifi", "imu"):
        C[k] = C[k].astype(np.float16)
    return C


def do_mmfi(task):
    f, sid, stride = task; env, nm = f.split("/")[-2], os.path.basename(f)[:-4]; P = np.load(f, allow_pickle=False); m = json.loads(str(P["meta"])); T = int(P["labels"].shape[0]); fps = float(m["fps"])
    rng = np.random.default_rng(sid); ks = np.arange(0, T, stride); C = empty(len(ks)); S = spec(m["spec"])
    C["ray"][:, 0] = rays(P["kp_px"][ks], np.ones((len(ks), 17), bool), S["rgb"]["K"], P["R_c2r"], P["t_c2r"]); C["org"][:] = P["t_c2r"]
    C["dep"][..., :3], C["dv"] = uvz_points(P, ks, S["depth"]["K"], P["R_d2r"], P["t_d2r"], ND, rng)
    lf, ls_ = v3(f"{V3}/mmfi_lidar/{env}_{nm}.npz", f"{U}/dev/packs/mmfi_lidar_v02/{env}_{nm}.npz")
    if os.path.exists(lf):
        L = np.load(lf, allow_pickle=False); Ls = spec(json.loads(str(L["meta"]))["spec"]); C["lid"], C["lv"] = uvz_points(L, ks, Ls["depth"]["K"], L["R_d2r"], L["t_d2r"], NL, rng)
    rf, rs_ = v3(f"{V3}/mmfi_radar/{env}/{nm}.npz", f"{U}/dev/packs/mmfi_v03_radar/{env}/{nm}.npz")
    if os.path.exists(rf):
        C["rad"], C["rv"] = radar_tokens(rf, ks, fps, rng)
    wd = f"{U}/dev/data/mmfi/extracted_rf/{env}/{nm[:3]}/{nm[4:]}/wifi-csi"
    for i, k in enumerate(ks):
        p = f"{wd}/frame{k + 1:03d}.mat"
        if os.path.exists(p):
            try:
                C["wifi"][i] = wifi_feat(p); C["wv"][i] = True
            except Exception:
                pass
    return finish(C, P, ks, "mmfi", sid, m, fps, f=f, src=dict(lidar=ls_, radar=rs_))


def mri_imu(s, f0, T):
    cs = set()
    with open(f"{MRI_RAW}/radar/singleframe/subject{s}.csv") as fh:
        for r in csv.DictReader(fh):
            cs.add(int(float(r["Camera Frame"])))
    fr = sorted(cs); imu = np.zeros((T, 6, 13), np.float32); iv = np.zeros(T, bool)
    for i in range(6):
        with open(f"{MRI_RAW}/imu/subject{s}/IMU{i}.csv") as fh:
            rows = [[float(r[c]) for c in IMU_COLS] for r in csv.DictReader(fh)]
        assert len(rows) == len(fr)
        for c, row in zip(fr, rows):
            if 0 <= c - f0 < T:
                imu[c - f0, i] = row; iv[c - f0] = True
    return imu, iv


def do_mri(task):
    f, sid, stride = task; s = int(os.path.basename(f)[1:3]); P = np.load(f, allow_pickle=False); m = json.loads(str(P["meta"])); T = int(m["T"]); fps = float(m["fps"])
    rng = np.random.default_rng(sid); ks = np.arange(0, T, stride); C = empty(len(ks)); S = spec(f"{U}/code/specs/mri_v03.json"); rm = S["room"]
    C["ray"][:, 0] = rays(P["kp_px"][ks], P["kp_score"][ks] > 0.3, S["rgb"]["K"], np.array(rm["R_c2r"]), np.array(rm["t_c2r"]))
    C["ray"][:, 1] = rays(P["kp_px_cam2"][ks], P["kp_score_cam2"][ks] > 0.3, S["rgb"]["K_cam2"], np.array(rm["R_c2r_cam2"]), np.array(rm["t_c2r_cam2"])); C["org"][:] = np.array(rm["t_c2r"])
    rf, rs_ = v3(f"{V3}/mri_radar/S{s:02d}.npz", f"{U}/dev/packs/mri_v03_radar/S{s:02d}.npz"); assert json.loads(str(np.load(rf, allow_pickle=False)["meta"])).get("room_frame") == "mri_v03"
    C["rad"], C["rv"] = radar_tokens(rf, ks, fps, rng)
    imu, iv = mri_imu(s, int(m["frame0"]), T); C["imu"], C["iv"] = imu[ks], iv[ks]
    act = P["action"][ks].astype(str) if "action" in P.files else None
    return finish(C, P, ks, "mri", sid, m, fps, act, f=f, src=dict(radar=rs_))


def do_mmbody(task):
    f, sid, stride = task; split, nm = f.split("/")[-2], os.path.basename(f)[:-4]; P = np.load(f, allow_pickle=False); m = json.loads(str(P["meta"])); T = int(P["labels"].shape[0]); fps = float(m["fps"])
    rng = np.random.default_rng(sid); ks = np.arange(0, T, stride); C = empty(len(ks)); C["org"][:] = P["t_d2r"]
    Z = np.load(f"{U}/dev/data/mmbody/cache_v1/{split}/{nm}.npz", allow_pickle=False); c = json.loads(str(Z["calib"]))
    Rm, tm = np.array(c["kinect_master"]["R"], float), np.array(c["kinect_master"]["t"], float); Rc, tc = P["R_c2r"].astype(float), P["t_c2r"].astype(float)
    o, X, RGB = Z["dpcl_off"], Z["dpcl_xyz"], Z["dpcl_rgb"]; cmax = 255.0 if len(RGB) and RGB[:100000].max() > 1.5 else 1.0
    for i, k in enumerate(ks):
        a, b = o[k], o[k + 1]
        if b <= a:
            continue
        ii = np.arange(a, b); cam = (X[ii].astype(float) - tm) @ Rm; ok = cam[:, 2] > 0.3; ii, cam = ii[ok], cam[ok]
        sel = select_person(cam @ Rc.T + tc, ND); ii, cam = ii[sel], cam[sel]
        C["dep"][i, :len(ii), :3] = cam @ Rc.T + tc; C["dep"][i, :len(ii), 3:6] = RGB[ii].astype(float) / cmax; C["dv"][i, :len(ii)] = True
    rf, rs_ = v3(f"{V3}/mmbody_radar/{split}/{nm}.npz", f"{U}/dev/packs/mmbody_v02_radar/{split}/{nm}.npz")
    if os.path.exists(rf):
        C["rad"], C["rv"] = radar_tokens(rf, ks, fps, rng)
    return finish(C, P, ks, "mmbody", sid, m, fps, f=f, src=dict(radar=rs_))


def do_octonet(task):
    f, sid, stride = task; st = os.path.basename(f)[:-4]; P = np.load(f, allow_pickle=False); m = json.loads(str(P["meta"])); T = int(P["labels"].shape[0]); fps = float(m["fps"])
    rng = np.random.default_rng(sid); ks = np.arange(0, T, stride); C = empty(len(ks)); C["org"][:] = P["t_d2r"]; S = spec(m["spec"])
    C["dep"][..., :3], C["dv"] = uvz_points(P, ks, S["depth"][f"{m['env']}_{m['day']}"]["K"], P["R_d2r"], P["t_d2r"], ND, rng)
    q = json.load(open(f"{U}/dev/packs/octonet_v08_radar/quality.json"))["status"]; rv3 = f"{V3}/octonet_radar/{st}.npz"; rs_ = "v2"
    if os.path.exists(rv3):                                                     # v3 이 있으면 v3 (품질 판정은 v3 manifest 를 따름)
        mf = json.load(open(f"{V3}/octonet_radar/manifest.json")) if os.path.exists(f"{V3}/octonet_radar/manifest.json") else {}
        ent = (mf.get("recordings", mf) or {}).get(st, {}) if isinstance(mf, dict) else {}
        q = {st: "ok" if (not isinstance(ent, dict) or ent.get("status", "ok") == "ok") else "bad"}; rs_ = "v3"
    if q.get(st) == "ok":
        C["rad"], C["rv"] = radar_tokens(rv3 if rs_ == "v3" else f"{U}/dev/packs/octonet_v08_radar/{st}.npz", ks, fps, rng)
    return finish(C, P, ks, "octonet", sid, m, fps, f=f, src=dict(radar=rs_))


def do_itop(task):
    f, sid, stride = task; P = np.load(f, allow_pickle=False); m = json.loads(str(P["meta"])); T = int(P["labels"].shape[0]); fps = float(m["fps"])
    rng = np.random.default_rng(sid); ks = np.arange(0, T, stride); C = empty(len(ks)); C["org"][:] = P["t_d2r"]; S = spec(m["spec"])
    C["dep"][..., :3], C["dv"] = uvz_points(P, ks, S["depth"]["K"], P["R_d2r"], P["t_d2r"], ND, rng)
    return finish(C, P, ks, "itop", sid, m, fps, f=f)


MMFI_FLOOR = _mmfi_env_floor() if FL else {}


def tasks():
    T = {"mmfi": [], "mri": [], "mmbody": [], "octonet": [], "itop": []}; sid = 0
    for f in sorted(glob.glob(f"{U}/dev/packs/mmfi_v02_specA/E0[1-4]/S*_A*.npz")):
        T["mmfi"].append((do_mmfi, (f, sid, 2))); sid += 1
    for f in sorted(glob.glob(f"{U}/dev/packs/mri_v03/S*.npz")):
        T["mri"].append((do_mri, (f, sid, 2))); sid += 1
    for f in sorted(glob.glob(f"{U}/dev/packs/mmbody_v01/*/*.npz")):
        T["mmbody"].append((do_mmbody, (f, sid, 2))); sid += 1
    for f in sorted(glob.glob(f"{U}/dev/packs/octonet_v07/*.npz")):
        if json.loads(str(np.load(f, allow_pickle=False)["meta"])).get("action") == "kicksomeone":      # 감사 M2: 두 사람 동작 (라벨은 한 사람) 제외
            continue
        T["octonet"].append((do_octonet, (f, sid, 2))); sid += 1
    for f in sorted(glob.glob(f"{U}/dev/packs/itop_v01/*.npz")):
        T["itop"].append((do_itop, (f, sid, 1))); sid += 1
    return T


def run(x):
    fn, a = x
    return fn(a)


if __name__ == "__main__":
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    T = tasks(); SRC = {}; os.makedirs(OUT, exist_ok=True)
    with Pool(32) as pool:
        for name, tl in T.items():
            if only and name not in only:
                continue
            R = pool.map(run, tl, chunksize=1); srcs = [r.pop("_src") for r in R]; C = {k: np.concatenate([r[k] for r in R]) for k in R[0]}
            SRC[name] = srcs; json.dump(SRC, open(f"{OUT}/sources.json", "w"), indent=1)
            print(name, "출처 집계", {k: {v: sum(1 for x in srcs if x.get(k) == v) for v in set(x.get(k) for x in srcs)} for k in set().union(*[set(x) - {"pack", "gtbad"} for x in srcs])}, flush=True)
            print(name, "망가진 정답 관절 (정답 없음 처리)", sum(x.get("gtbad", 0) for x in srcs), flush=True)
            np.savez(f"{OUT}/{name}.npz", **C)
            av = dict(rgb=(C["ray"][..., 6] > 0).any((1, 2)).mean(), depth=C["dv"].any(1).mean(), lidar=C["lv"].any(1).mean(), radar=C["rv"].any(1).mean(), wifi=C["wv"].mean(), imu=C["iv"].mean())
            print(name, "시퀀스", len(tl), "프레임", len(C["y"]), {k: round(float(v), 3) for k, v in av.items()}, flush=True)
    print("CACHE3_DONE")
