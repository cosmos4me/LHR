"""v3 OctoNet 노드 1 TI 레이더: 정답 없는 정리 + 재정합 + 팩 쓰기 (캐시 v3 단계 3–4, 항목 1a–1e).
정답(관절·라벨·모캡 몸)은 읽지 않는다. 읽는 것: 레이더 pickle (허용 목록), 깊이 zip 항목 이름(시각), 모캡 CSV 헤더 첫 줄(시각 범위)과
'Rigid Body' NodeK 열(센서 노드 자세 = dataset_calib_nonbody), 팩 octonet_v07 의 pts_uvz/pts_valid/R_d2r/t_d2r/meta (labels 안 엶),
날짜별 깊이 배경 (derived/backgrounds_p90.npz, 정답 없음).
단계
  prep     녹화별: 레이더 읽기, 녹화 안 정지 표시, 레이더만의 사람 추적, 움직임 에너지 상호상관 곡선, 노드 강체 요약 → work/prep/{stamp}.npz
  register 장면 정지 지도 → 점 분류 → 시간 (무리 합의 + 공간 다듬기) → BEV 3자유도 정합 (녹화별 + 무리 합침) → 검증 → work/register.json
  write    v3 팩 + manifest.json
실행: CUDA_VISIBLE_DEVICES="" python -I code/v3/build_octonet_radar.py {prep|register|write} (tmux v3_octo)"""
import sys
sys.path.insert(0, "/home/interns/unisense/code/v3")
import json, time, zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np

import octo_io as io_
import octo_radar_lib as L

U = Path("/home/interns/unisense")
OUT = U / "dev/packs/v3/octonet_radar"; WORK = OUT / "work"; PREP = WORK / "prep"
PACKS = U / "dev/packs/octonet_v07"


def recs():
    return [r for r in io_.recordings() if (PACKS / f"{r['stamp']}.npz").exists()]


# ------------------------------------------------------------------ prep
def prep_one(r):
    f = PREP / f"{r['stamp']}.npz"
    if f.exists():
        return r["stamp"], "skip"
    t0 = time.time(); d = L.load_recording(r)
    stat, scodes = L.static_mask(d["pts"])
    C, seen, centre = L.radar_track(d["ts"], d["pts"], stat)
    cls0 = L.classify(d["pts"], stat, C, np.zeros(0, np.int64))           # 장면 지도 전 (시간 추정용)
    er = L.radar_motion(d["pts"], cls0); ed = L.depth_motion(d["D"])
    T = len(d["tk"]); h = T // 2
    xc = L.xcorr_curve(d["ts"], er, d["tk"], ed)
    m1 = d["tk"] < d["tk"][h]; xc1 = L.xcorr_curve(d["ts"], er, d["tk"][m1], ed[m1]); xc2 = L.xcorr_curve(d["ts"], er, d["tk"][~m1], ed[~m1])
    rb = io_.node_rigid(r["mocap"])                                        # 센서 노드 강체만 (환경 1 에만 있음)
    nodes = {k: np.nanmedian(v, 0) for k, v in rb.items()}
    nstd = {k: float(np.nanstd(v[:, :3], 0).max()) for k, v in rb.items()}
    off = np.r_[0, np.cumsum([len(p) for p in d["pts"]])]
    tmp = PREP / f"{r['stamp']}.tmp.npz"
    np.savez_compressed(tmp, ts=d["ts"], tk=d["tk"], pts=np.concatenate(d["pts"]) if off[-1] else np.zeros((0, 4)), off=off,
                        stat=np.concatenate(stat) if off[-1] else np.zeros(0, bool), scodes=scodes, track=C, seen=seen,
                        centre=np.asarray(centre if centre is not None else [np.nan, np.nan]), er=er, ed=ed, xc=xc, xc1=xc1, xc2=xc2,
                        nodes=json.dumps({k: v.tolist() for k, v in nodes.items()}), nodes_std=json.dumps(nstd),
                        meta=json.dumps(dict(stamp=r["stamp"], env=r["env"], day=r["day"], user=r["user"], action=r["action"], T=T, F=len(d["ts"]))))
    tmp.replace(f)                                                        # 원자적 이름 바꾸기 (중단 시 깨진 파일 방지)
    return r["stamp"], f"ok {time.time() - t0:.1f}s"


def prep():
    PREP.mkdir(parents=True, exist_ok=True)
    R = recs(); print("녹화", len(R), flush=True)
    with Pool(24) as p:
        for i, (s, msg) in enumerate(p.imap_unordered(prep_one, R), 1):
            if i % 20 == 0 or not msg.startswith(("ok", "skip")):
                print(i, s, msg, flush=True)
    print("PREP_DONE", flush=True)


# ------------------------------------------------------------------ register
V08 = json.loads((U / "dev/packs/octonet_v08_radar/attach_report.json").read_text())       # 비교용 (v08 변환·시차); 정답 아님
WEAK_PRIOR = (np.zeros(3), np.array([1.0, 1.0, np.radians(15)]))          # 퇴화 방향만 묶는 약한 사전 (같은 노드: 카메라 ±1 m, ±15°)
FLOOR_COV = np.diag([0.03 ** 2, 0.03 ** 2, np.radians(0.5) ** 2])          # 녹화별 공분산 바닥값 (모형 밖 녹화 차이)
INJ = [(0.05, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, np.radians(2.0)), (0.15, -0.10, np.radians(3.0))]
NBOOT = {"pass1": 12, "pass2": 30}


def load_prep(st):
    Z = np.load(PREP / f"{st}.npz", allow_pickle=False)
    d = {k: Z[k] for k in Z.files}; d["meta"] = json.loads(str(d["meta"])); d["nodes"] = json.loads(str(d["nodes"]))
    d["pts_l"] = L.split_pts(d["pts"], d["off"]); d["stat_l"] = L.split_pts(d["stat"], d["off"])
    return d


def group_tau():
    S = L.P["shifts"]; G = defaultdict(list)
    for f in sorted(PREP.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        Z = np.load(f); m = json.loads(str(Z["meta"])); G[(m["env"], m["day"])].append(Z["xc"])
    return {f"{e}_{d}": float(S[np.nanargmax(np.nanmean(np.array(v), 0))]) for (e, d), v in G.items()}


def choose_tau(d, tg, R0, c, x, bev, cls):
    """시간: 무리 합의 → 녹화 움직임 에너지 → 속도 상호상관 (위치 치우침 무관) → 공간 비용 검사."""
    S = L.P["shifts"]; xc = d["xc"]; w = np.abs(S - tg) <= 0.4
    if np.isfinite(xc[w]).any():
        i = int(np.nanargmax(np.where(w, xc, np.nan))); t_xc, c_xc = float(S[i]), float(xc[i])
    else:
        t_xc, c_xc = tg, np.nan
    t_v, c_v, sp, _ = L.velocity_xcorr(d["ts"], d["track"], R0, d["tk"], bev.D, t_xc if np.isfinite(c_xc) and c_xc >= 0.3 else tg)
    vel_ok = np.isfinite(t_v) and c_v >= 0.4 and sp >= 0.15
    centre = t_v if vel_ok else (t_xc if np.isfinite(c_xc) and c_xc >= 0.3 else tg)
    J = []
    for sh in np.round(np.arange(centre - 0.3, centre + 0.3001, 0.02), 3):
        B, K = L.gather(d["ts"], d["tk"], d["pts_l"], cls, bev, sh)
        if len(B) >= 200:
            J.append((sh, L.cost(x, R0, c, B, K, bev)[0]))
    J = np.array(J) if J else np.zeros((0, 2))
    if len(J) >= 5:
        jmin = J[np.argmin(J[:, 1])]; depth = (J[:, 1].max() - jmin[1]) / max(jmin[1], 1e-9); t_j = float(jmin[0])
    else:
        depth, t_j = 0.0, np.nan
    j_ok = np.isfinite(t_j) and depth >= 0.3
    if vel_ok:
        tau, src = t_v, "velocity_xcorr"
    elif j_ok:
        tau, src = t_j, "depth_person_spatial"
    elif np.isfinite(c_xc) and c_xc >= 0.3:
        tau, src = t_xc, "motion_energy_xcorr"
    else:
        tau, src = tg, "group_motion_energy"
    # 앞·뒤 절반 (drift 검사): 움직임 에너지 곡선 반쪽
    h = []
    for k in ("xc1", "xc2"):
        v = d[k]
        h.append(float(S[int(np.nanargmax(np.where(w, v, np.nan)))]) if np.isfinite(v[w]).any() else np.nan)
    return dict(tau=tau, src=src, t_group=tg, t_xc=t_xc, c_xc=c_xc, t_vel=t_v, c_vel=c_v, speed_sd=sp, t_spatial=t_j, j_depth=float(depth),
                half_xc=h, half_drift=h[1] - h[0] if np.isfinite(h).all() else np.nan)


class BEVD(L.PersonBEV):
    def __init__(self, D):
        super().__init__(D); self.D = D


def boot_cov(R0, c, B, K, bev, x, n, rng, block=20):
    if n <= 0 or len(B) < 50:
        return np.full((3, 3), np.nan), []
    blocks = K // block; ub = np.unique(blocks); idx_of = {b: np.flatnonzero(blocks == b) for b in ub}; X = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub), replace=True); ii = np.concatenate([idx_of[b] for b in pick])
        xb, inf = L.solve_bev(R0, c, B[ii], K[ii], bev, x0=x, iters=10, prior=WEAK_PRIOR)
        if inf.get("ok"):
            X.append(xb)
    X = np.array(X)
    return (np.cov(X.T) if len(X) >= 5 else np.full((3, 3), np.nan)), X


def signed_radial(x, R0, c, B, K, bev, cam):
    """깊이 표면 대비 레이더 점의 시선 방향 부호 잔차 (+ = 레이더 점이 표면보다 멀리)."""
    q, nn, dd = L.residuals(x, R0, c, B, K, bev); u = nn - cam[:2]; u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    s = ((q - nn) * u).sum(1); m = dd <= L.P["trunc"]
    return float(np.median(s[m])) if m.any() else np.nan, float(np.median(dd)) if len(dd) else np.nan


def v08_eval(d, B_idx_frames, bev, half_mask_k, cls, Rd, td):
    """v08 변환·시차로 같은 몸 점(같은 분류)을 방에 놓았을 때 뒤 절반 BEV 최근접 거리."""
    st = d["meta"]["stamp"]; e = str(d["meta"]["env"]); Rv = np.array(V08["ext"][e]["R"]); tv = np.array(V08["ext"][e]["t"]); sh = V08["shift"][st]
    fs, ks = L.pair_frames(d["ts"], d["tk"], sh); dd, sr = [], []
    for f, k in zip(fs, ks):
        if not half_mask_k[k] or bev.trees[k] is None:
            continue
        b = d["pts_l"][f][np.isin(cls[f], L.KEEP)]
        if len(b):
            q = ((b[:, :3] @ Rv.T + tv) @ Rd.T + td)[:, :2]; d_, i_ = bev.trees[k].query(q); nn = bev.trees[k].data[i_]
            u = nn - td[:2]; u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9); dd.append(d_); sr.append(((q - nn) * u).sum(1)[d_ <= L.P["trunc"]])
    return (float(np.median(np.concatenate(dd))), float(np.median(np.concatenate(sr)))) if dd else (np.nan, np.nan)


def reg_one(args):
    st, tg, x_init, mode = args
    t0 = time.time(); rng = np.random.default_rng(zlib.crc32(st.encode()))
    d = load_prep(st); m = d["meta"]; env, day = m["env"], m["day"]
    D, Rd, td, _ = L.depth_person(st); bev = BEVD(D); R0, c = L.base_pose(Rd, td); scene = L.Scene(env, day)
    x_init = np.asarray(x_init, float)
    cls = L.classify_full(d["pts_l"], d["stat_l"], d["track"], x_init, R0, c, scene)
    tinfo = choose_tau(d, tg, R0, c, x_init, bev, cls); tau = tinfo["tau"]
    B, K = L.gather(d["ts"], d["tk"], d["pts_l"], cls, bev, tau)
    allc = np.concatenate(cls) if len(cls) else np.zeros(0, np.int8)
    res = dict(stamp=st, env=env, day=day, user=m["user"], action=m["action"], T=m["T"], F=m["F"], time=tinfo,
               cls_counts={int(k): int((allc == k).sum()) for k in L.CLS}, n_pts=int(len(allc)), n_pair_pts=int(len(B)),
               frames_with_body=float(np.mean([np.isin(cc, L.KEEP).any() for cc in cls])) if cls else 0.0, radar_seen=float(d["seen"].mean()))
    if len(B) < 100:
        res.update(ok=False, why="too few body points", secs=time.time() - t0); return res
    x, info = L.solve_bev(R0, c, B, K, bev, x0=x_init, prior=WEAK_PRIOR)
    cov, Xb = boot_cov(R0, c, B, K, bev, x, NBOOT[mode], rng)
    rmean = float(np.nanmedian([np.linalg.norm(bev.cen[k] - c[:2]) for k in np.unique(K)]))
    sr, md = signed_radial(x, R0, c, B, K, bev, c)
    res.update(ok=bool(info.get("ok")), x=x.tolist(), cov=cov.tolist(), info=info, person_range=rmean, signed_radial=sr, med_nn=md,
               lateral_at_person=float(x[1] + rmean * x[2]))
    if mode == "pass2":
        # 앞 절반으로 맞추고 뒤 절반에서 잼 (held-out)
        T = m["T"]; first = K < T // 2; second_k = np.arange(T) >= T // 2
        if first.sum() >= 100 and (~first).sum() >= 100:
            xh, ih = L.solve_bev(R0, c, B[first], K[first], bev, x0=x_init, prior=WEAK_PRIOR)
            ev = lambda xx: (L.cost(xx, R0, c, B[~first], K[~first], bev)[1], signed_radial(xx, R0, c, B[~first], K[~first], bev, c)[0])
            res["heldout"] = dict(x_first=xh.tolist(), med_first_fit=ev(xh), med_init=ev(x_init), med_full=ev(x), med_base=ev(np.zeros(3)),
                                  med_v08=v08_eval(d, None, bev, second_k, cls, Rd, td))
        # 주입–복원: 기준 자세를 알려진 만큼 흔들고 다시 풂
        inj = []
        for dl in INJ:
            dl = np.asarray(dl); R0p = L.Rz(-dl[2]) @ R0; cp = c - np.array([dl[0], dl[1], 0.0])
            xp, ip = L.solve_bev(R0p, cp, B, K, bev, x0=x_init, prior=WEAK_PRIOR)
            inj.append(dict(inj=dl.tolist(), rec=(xp - x).tolist()))
        res["injection"] = inj
        # 노드 4 (환경 1, 정지 클러터 중 모캡 노드 4 근처) — 레이더 좌표 중앙값만 넘김 (최종 변환 비교는 main)
        if d["nodes"]:
            g = io_.room_geometry(env, day); N = {k: np.array(v[:3]) @ g["R_w2r"].T + g["t_w2r"] for k, v in d["nodes"].items()}
            res["nodes_room"] = {k: v.tolist() for k, v in N.items()}
            S_ = np.concatenate([p[cc == 1, :3] for p, cc in zip(d["pts_l"], cls)]) if len(cls) else np.zeros((0, 3))
            if len(S_):
                q = L.apply(x_init, R0, c, S_)
                for nd in ("Node2", "Node3", "Node4", "Node5"):
                    mm = np.linalg.norm(q[:, :2] - N[nd][:2], axis=1) < 0.5
                    if mm.sum() >= 20:
                        res.setdefault("node_clusters", {})[nd] = dict(n=int(mm.sum()), radar_xyz=np.median(S_[mm], 0).tolist())
        # 정지 클러터 표본 (날짜별 정지 ICP 용, 레이더 좌표)
        S_ = np.concatenate([p[cc == 1, :3] for p, cc in zip(d["pts_l"], cls)]) if len(cls) else np.zeros((0, 3))
        if len(S_) > 4000:
            S_ = S_[rng.choice(len(S_), 4000, replace=False)]
        np.save(WORK / "static" / f"{st}.npy", S_.astype(np.float32))
        # 합친 ICP 검사용 표본
        ii = rng.choice(len(B), min(1500, len(B)), replace=False); ks = np.unique(K[ii])
        np.savez_compressed(WORK / "pool" / f"{st}.npz", B=B[ii].astype(np.float32), K=K[ii], ks=ks,
                            Dk=np.concatenate([D[k][:, :2] for k in ks]).astype(np.float32), Dn=np.array([len(D[k]) for k in ks]))
    res["secs"] = time.time() - t0
    return res


def fuse(rows, floor=FLOOR_COV, it=5, thr=16.0):
    """정보 융합 x̂ = (Σ Λ_r)⁻¹ Σ Λ_r x_r, Λ_r = (Σ_r + 바닥)⁻¹; 마할라노비스² > thr 녹화는 빼고 반복."""
    X = np.array([r["x"] for r in rows]); C = np.array([np.nan_to_num(np.array(r["cov"]), nan=1.0) + floor for r in rows])
    Lm = np.linalg.inv(C); use = np.ones(len(X), bool)
    for _ in range(it):
        A = Lm[use].sum(0); b = np.einsum("nij,nj->i", Lm[use], X[use]); xh = np.linalg.solve(A, b)
        d2 = np.einsum("ni,nij,nj->n", X - xh, Lm, X - xh); new = d2 <= thr
        if (new == use).all():
            break
        use = new
    A = Lm[use].sum(0); xh = np.linalg.solve(A, np.einsum("nij,nj->i", Lm[use], X[use])); cov = np.linalg.inv(A)
    d2 = np.einsum("ni,nij,nj->n", X - xh, Lm, X - xh)
    return xh, cov, use, d2


def summarize_x(x, cov=None):
    o = dict(dx=round(float(x[0]), 4), dy=round(float(x[1]), 4), dyaw_deg=round(float(np.degrees(x[2])), 3))
    if cov is not None:
        s = np.sqrt(np.clip(np.diag(cov), 0, None)); o.update(sd_x=round(float(s[0]), 4), sd_y=round(float(s[1]), 4), sd_yaw_deg=round(float(np.degrees(s[2])), 3))
    return o


def static_icp(day_rows, x_env, env, day):
    """날짜별: 레이더 정지 클러터 (그날 녹화 합침) ↔ 깊이 배경 수직 구조, 점–평면 BEV ICP (몸과 무관)."""
    S_ = [np.load(WORK / "static" / f"{r['stamp']}.npy") for r in day_rows if (WORK / "static" / f"{r['stamp']}.npy").exists()]
    S_ = np.concatenate(S_) if S_ else np.zeros((0, 3))
    if len(S_) < 50:
        return dict(ok=False, n=int(len(S_)))
    rng = np.random.default_rng(0)
    if len(S_) > 30000:
        S_ = S_[rng.choice(len(S_), 30000, replace=False)]
    g = io_.room_geometry(env, day); R0, c = L.base_pose(g["R_c2r"], g["t_c2r"]); sc = L.depth_scene(env, day, g); pre = L.static_pre(sc)
    # 최적화 없이 점수만: 레이더 정지 점이 깊이 배경 수직 구조 BEV 0.15 m 안에 드는 비율 (사람 기준 해 / 기준 자세 / v08)
    def score(q):
        dd = pre["tree"].query(q[:, :2])[0]; m5 = dd < 0.5
        return dict(frac_within_0p15=float((dd < 0.15).mean()), frac_within_0p5=float(m5.mean()), med_within_0p5=float(np.median(dd[m5])) if m5.any() else None)
    Rv = np.array(V08["ext"][str(env)]["R"]); tv = np.array(V08["ext"][str(env)]["t"])
    sc_scores = dict(person_fused=score(L.apply(np.asarray(x_env), R0, c, S_.astype(float))), base_camera=score(L.apply(np.zeros(3), R0, c, S_.astype(float))),
                     v08=score((S_.astype(float) @ Rv.T + tv) @ g["R_c2r"].T + g["t_c2r"]))
    xs, inf = L.solve_static(R0, c, S_.astype(float), sc, x0=x_env, pre=pre)
    out = dict(inf, n_static=int(len(S_)), scores=sc_scores)
    if inf.get("ok"):
        out.update(x=xs.tolist(), diff_person_minus_static=(np.asarray(x_env) - xs).tolist())
        # 부트스트랩 (점 재표본 15회)
        Xb = []
        for _ in range(15):
            ii = rng.choice(len(S_), len(S_), replace=True); xb, ib = L.solve_static(R0, c, S_[ii].astype(float), sc, x0=xs, iters=15, pre=pre)
            if ib.get("ok"):
                Xb.append(xb)
        if len(Xb) >= 5:
            out["cov"] = np.cov(np.array(Xb).T).tolist()
    return out


def pooled_icp(rows, x0):
    """환경 안 모든 녹화 표본을 한 번에 맞춘 ICP (정보 융합 검사용, 사전 없음). 모든 프레임을 (x, y, 전역 프레임·1000 m) 한 KD-tree 로."""
    from scipy.spatial import cKDTree
    Bs, Ks, allD = [], [], []; base = 0
    for r in rows:
        f = WORK / "pool" / f"{r['stamp']}.npz"
        if not f.exists():
            continue
        with np.load(f) as Zf:
            Z = {k: Zf[k] for k in Zf.files}                                     # 한 번만 풀기 (npz 항목 접근마다 다시 읽지 않게)
        with np.load(PACKS / f"{r['stamp']}.npz", allow_pickle=False) as P_:
            Rd, td = P_["R_d2r"].astype(float), P_["t_d2r"].astype(float)
        R0, c = L.base_pose(Rd, td); kmap = {k: i for i, k in enumerate(Z["ks"])}
        rs = np.random.default_rng(zlib.crc32(r["stamp"].encode())); sub = rs.choice(len(Z["B"]), min(600, len(Z["B"])), replace=False)   # 녹화당 레이더 600점
        Bs.append((Z["B"][sub].astype(float), R0, c)); Ks.append(np.array([kmap[k] for k in Z["K"][sub]]) + base)
        o = np.r_[0, np.cumsum(Z["Dn"])]
        for i in range(len(Z["ks"])):
            dk = Z["Dk"][o[i]:o[i + 1]].astype(float)
            dk = dk[rs.choice(len(dk), min(64, len(dk)), replace=False)]                                       # 프레임당 깊이 64점
            allD.append(np.c_[dk, np.full(len(dk), (base + i) * 1000.0)])
        base += len(Z["ks"])
    allD = np.concatenate(allD); tree = cKDTree(allD)
    x = np.asarray(x0, float).copy()
    for itn in range(40):
        A = np.zeros((3, 3)); bb = np.zeros(3); dall = []
        for (Bq, R0, c), K in zip(Bs, Ks):
            q = L.apply(x, R0, c, Bq)[:, :2]; dd, ii = tree.query(np.c_[q, K * 1000.0]); nn = allD[ii, :2]
            w = L.weights(dd); o = c[:2] + x[:2]; rx, ry = q[:, 0] - o[0], q[:, 1] - o[1]
            J = np.zeros((2 * len(q), 3)); J[0::2, 0] = 1; J[1::2, 1] = 1; J[0::2, 2] = -ry; J[1::2, 2] = rx
            W = np.repeat(w, 2); A += J.T @ (J * W[:, None]); bb += J.T @ (W * (nn - q).reshape(-1)); dall.append(dd)
        dx = np.linalg.solve(A, bb); x += dx
        if np.abs(dx[:2]).max() < 1e-4 and abs(dx[2]) < 1e-5:
            break
    return x, dict(n=int(sum(len(k) for k in Ks)), med=float(np.median(np.concatenate(dall))), iters=itn + 1, recs=len(Bs))


def cKDTree_(a):
    from scipy.spatial import cKDTree
    return cKDTree(a)


def register():
    for sub in ("static", "pool"):
        (WORK / sub).mkdir(parents=True, exist_ok=True)
    R = recs(); TG = group_tau(); print("녹화", len(R), "무리 시차", json.dumps(TG), flush=True)
    out = dict(params={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in L.P.items()}, group_tau=TG)
    xinit = {e: np.zeros(3) for e in (1, 2, 3)}
    for mode in ("pass1", "pass2"):
        tasks = [(r["stamp"], TG[f"{r['env']}_{r['day']}"], xinit[r["env"]].tolist(), mode) for r in R]
        rows = []
        with Pool(24) as p:
            for i, res in enumerate(p.imap_unordered(reg_one, tasks), 1):
                rows.append(res)
                if i % 25 == 0:
                    print(mode, i, "/", len(tasks), res["stamp"], f"{res.get('secs', 0):.0f}s", flush=True)
        good = [r for r in rows if r.get("ok") and np.isfinite(np.array(r["cov"])).all()]
        fusedE, fusedD = {}, {}
        for e in (1, 2, 3):
            re_ = [r for r in good if r["env"] == e]
            xh, cov, use, d2 = fuse(re_); fusedE[e] = dict(x=xh.tolist(), cov=cov.tolist(), n=len(re_), used=int(use.sum()), chi2_mean=float(np.mean(d2[use])))
            for day in sorted({r["day"] for r in re_}):
                rd = [r for r in re_ if r["day"] == day]
                xd, cd, ud, d2d = fuse(rd); fusedD[f"{e}_{day}"] = dict(x=xd.tolist(), cov=cd.tolist(), n=len(rd), used=int(ud.sum()), chi2_mean=float(np.mean(d2d[ud])))
            print(mode, "env", e, "fused", summarize_x(xh, cov), "n", len(re_), "used", int(use.sum()), "chi2/3", round(float(np.mean(d2[use])) / 3, 2), flush=True)
            xinit[e] = xh
        out[mode] = dict(rows=rows, fused_env=fusedE, fused_day=fusedD)
        (WORK / "register.json").write_text(json.dumps(out, default=float))
    # 정지 클러터 ICP (날짜별) + 합친 ICP 검사
    rows = out["pass2"]["rows"]; st_res = {}; pool_res = {}
    for e in (1, 2, 3):
        xe = np.array(out["pass2"]["fused_env"][e]["x"])
        for day in sorted({r["day"] for r in rows if r["env"] == e}):
            st_res[f"{e}_{day}"] = static_icp([r for r in rows if r["env"] == e and r["day"] == day], xe, e, day)
            print("static", e, day, {k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in st_res[f'{e}_{day}'].items() if k != "cov"}, flush=True)
        xp, ip = pooled_icp([r for r in rows if r["env"] == e and r.get("ok")], xe)
        pool_res[e] = dict(x=xp.tolist(), **ip); print("pooled ICP env", e, summarize_x(xp), ip, flush=True)
    out["static"] = st_res; out["pooled_icp"] = pool_res
    (WORK / "register.json").write_text(json.dumps(out, default=float))
    print("REGISTER_DONE", flush=True)


# ------------------------------------------------------------------ write
CARD = dict(device="TI IWR1843 (OctoNet node 1)", doppler="연속값 m/s (대부분 0.137 m/s 칸)", frames_per_s=9.1, has_intensity=False,
            range_filter_m=[0.3, 10.0], columns="x y z (방 m), 도플러 m/s, 세기 = 0 (원본에 없음)",
            static_flag="v3: radar_pts 에는 몸 점만 남김 (정지 클러터·유령점 제거) → radar_static 은 모두 False. 전체 점과 분류는 all_pts/all_cls")


def decide_level(fe, fd):
    """환경 단위 융합값을 쓸지, 날짜 단위를 쓸지: 날짜 융합값이 환경 값과 (마할라노비스² ≤ 11.34 (3자유도 99 %) 또는 |Δxy| ≤ 0.05 m 이고 |Δψ| ≤ 1°) 이면 환경 값."""
    out = {}
    xe, ce = np.array(fe["x"]), np.array(fe["cov"])
    for key, v in fd.items():
        xd, cd = np.array(v["x"]), np.array(v["cov"]); dlt = xd - xe
        m2 = float(dlt @ np.linalg.solve(cd + ce, dlt)); small = np.abs(dlt[:2]).max() <= 0.05 and abs(np.degrees(dlt[2])) <= 1.0
        out[key] = dict(m2=m2, small=bool(small), use_env=bool(m2 <= 11.34 or small or v["n"] < 3), delta=summarize_x(dlt))
    return out


def write_one(args):
    st, x, tau, info = args
    d = load_prep(st); m = d["meta"]; env, day = m["env"], m["day"]
    with np.load(PACKS / f"{st}.npz", allow_pickle=False) as Z:
        Rd, td = Z["R_d2r"].astype(float), Z["t_d2r"].astype(float); T = int(Z["pts_valid"].shape[0])
    R0, c = L.base_pose(Rd, td); scene = L.Scene(env, day); x = np.asarray(x, float)
    cls = L.classify_full(d["pts_l"], d["stat_l"], d["track"], x, R0, c, scene)
    R, t = L.full_RT(x, R0, c)
    # 레이더 프레임 → 가장 가까운 팩 프레임 하나 (|Δt| ≤ tol, 레이더 프레임 중복 사용 없음)
    tq = d["ts"] + tau; k = np.clip(np.searchsorted(d["tk"], tq), 1, len(d["tk"]) - 1)
    k = np.where(np.abs(d["tk"][k - 1] - tq) < np.abs(d["tk"][k] - tq), k - 1, k); ok = np.abs(d["tk"][k] - tq) <= L.P["tol_s"]
    rows = [[] for _ in range(T)]; allr = [[] for _ in range(T)]; allc = [[] for _ in range(T)]
    for f in np.flatnonzero(ok):
        p = d["pts_l"][f]
        if not len(p):
            continue
        q = np.c_[p[:, :3] @ R.T + t, p[:, 3], np.zeros(len(p))].astype(np.float32); keep = np.isin(cls[f], L.KEEP)
        rows[k[f]].append(q[keep]); allr[k[f]].append(q); allc[k[f]].append(cls[f].astype(np.uint8))
    cat = lambda L_, w, dt: [np.concatenate(x_) if x_ else np.zeros((0, w), dt) if w else np.zeros(0, dt) for x_ in L_]
    rows = cat(rows, 5, np.float32); allr = cat(allr, 5, np.float32); allc = cat(allc, 0, np.uint8)
    off = np.r_[0, np.cumsum([len(x_) for x_ in rows])]; aoff = np.r_[0, np.cumsum([len(x_) for x_ in allr])]
    allcls = np.concatenate(allc) if aoff[-1] else np.zeros(0, np.uint8); every = np.concatenate(cls) if len(cls) else np.zeros(0)
    kept_frac = float(np.isin(every, L.KEEP).mean()) if len(every) else 0.0
    meta = dict(dataset="octonet", stamp=st, env=env, user=m["user"], T=T, card=CARD, version="v3-radar",
                time=dict(radar_shift_s=float(tau), source=info["time_src"], tol_s=L.P["tol_s"], assign="레이더 프레임마다 가장 가까운 팩 프레임 하나"),
                extrinsic=dict(frame="radar (TI) → room (팩 octonet_v07 방 좌표)", R=R.round(6).tolist(), t=t.round(6).tolist(),
                               delta_vs_camera=summarize_x(x), source=info["source"], level=info["level"]),
                cleaning=dict(classes={str(k_): v for k_, v in L.CLS.items()}, kept=list(L.KEEP), kept_frac=kept_frac,
                              class_frac={str(k_): float((every == k_).mean()) if len(every) else 0.0 for k_ in L.CLS}),
                quality=info["quality"])
    np.savez_compressed(OUT / f"{st}.npz", radar_pts=np.concatenate(rows) if off[-1] else np.zeros((0, 5), np.float32), radar_static=np.zeros(off[-1], bool),
                        radar_off=off, all_pts=np.concatenate(allr) if aoff[-1] else np.zeros((0, 5), np.float32), all_cls=allcls, all_off=aoff,
                        meta=json.dumps(meta, ensure_ascii=False))
    return st, kept_frac, float(np.mean(np.diff(off) > 0)), {str(k_): float((every == k_).mean()) if len(every) else 0.0 for k_ in L.CLS}


def validation(reg, final):
    """정답 없는 검증 요약: 반복성, held-out, 주입–복원, 시간, 정지 ICP, 노드 (환경 1)."""
    rows = [r for r in reg["pass2"]["rows"] if r.get("ok")]; V = {}
    for e in (1, 2, 3):
        re_ = [r for r in rows if r["env"] == e]; X = np.array([r["x"] for r in re_])
        lat = np.array([r["lateral_at_person"] for r in re_])
        fd = {k: v for k, v in reg["pass2"]["fused_day"].items() if k.startswith(f"{e}_")}
        FD = np.array([v["x"] for v in fd.values() if v["n"] >= 3])
        Vt = dict(n=len(re_),
                  per_rec_sd=dict(dx=float(X[:, 0].std()), dy=float(X[:, 1].std()), dyaw_deg=float(np.degrees(X[:, 2].std())), lateral_at_person=float(lat.std())),
                  per_rec_iqr_dx=[float(v) for v in np.percentile(X[:, 0], [25, 75])],
                  per_day_fused_sd=dict(dx=float(FD[:, 0].std()), dy=float(FD[:, 1].std()), dyaw_deg=float(np.degrees(FD[:, 2].std()))) if len(FD) > 1 else None,
                  per_day_fused={k: summarize_x(np.array(v["x"]), np.array(v["cov"])) | dict(n=v["n"]) for k, v in fd.items()},
                  median_boot_sd=dict(dx=float(np.median([np.sqrt(r["cov"][0][0]) for r in re_])), dy=float(np.median([np.sqrt(r["cov"][1][1]) for r in re_])),
                                      dyaw_deg=float(np.degrees(np.median([np.sqrt(r["cov"][2][2]) for r in re_])))))
        H = [r["heldout"] for r in re_ if "heldout" in r]
        if H:
            g = lambda k, i: float(np.nanmedian([h[k][i] if isinstance(h[k], (list, tuple)) else np.nan for h in H]))
            Vt["heldout_second_half"] = {k: dict(med_nn_m=g(k, 0), med_signed_radial_m=g(k, 1)) for k in ("med_first_fit", "med_full", "med_init", "med_base", "med_v08")}
            Vt["heldout_second_half"]["first_vs_full_shift"] = dict(
                dx=float(np.median(np.abs([h["x_first"][0] - r["x"][0] for h, r in zip(H, re_)]))))
        I = [r["injection"] for r in re_ if "injection" in r]
        if I:
            inj = {}
            for j in range(len(INJ)):
                err = np.array([np.array(ii[j]["rec"]) - np.array(ii[j]["inj"]) for ii in I])
                inj[str(np.round([INJ[j][0], INJ[j][1], np.degrees(INJ[j][2])], 3).tolist())] = dict(
                    per_rec_median_abs_err=dict(dx=float(np.median(np.abs(err[:, 0]))), dy=float(np.median(np.abs(err[:, 1]))), dyaw_deg=float(np.degrees(np.median(np.abs(err[:, 2]))))))
                # 융합 단계의 복원 (같은 공분산으로 주입된 해들을 융합)
                rr = [dict(r, x=(np.array(r["x"]) + np.array(ii[j]["rec"])).tolist()) for r, ii in zip(re_, I)]
                x0_, _, _, _ = fuse(re_); x1_, _, _, _ = fuse(rr)
                inj[str(np.round([INJ[j][0], INJ[j][1], np.degrees(INJ[j][2])], 3).tolist())]["fused_recovered"] = summarize_x(x1_ - x0_)
            Vt["injection"] = inj
        tm = [r["time"] for r in re_]
        Vt["time"] = dict(src_counts={s: sum(t_["src"] == s for t_ in tm) for s in sorted({t_["src"] for t_ in tm})},
                          tau_median=float(np.median([t_["tau"] for t_ in tm])), tau_iqr=[float(v) for v in np.percentile([t_["tau"] for t_ in tm], [25, 75])],
                          vel_minus_spatial_med_abs=float(np.nanmedian([abs(t_["t_vel"] - t_["t_spatial"]) for t_ in tm if t_["src"] == "velocity_xcorr"])) if any(t_["src"] == "velocity_xcorr" for t_ in tm) else None,
                          half_drift_med_abs=float(np.nanmedian([abs(t_["half_drift"]) for t_ in tm])),
                          minus_v08_median=float(np.median([t_["tau"] - V08["shift"][r["stamp"]] for t_, r in zip(tm, re_)])))
        V[e] = Vt
    # 노드 (환경 1): 최종 변환으로 레이더 노드 4 군집 → 방, 모캡 노드 4 와 비교; 레이더 원점 − 노드 1
    nod = []
    for r in rows:
        if r["env"] != 1 or "nodes_room" not in r:
            continue
        with np.load(PACKS / f"{r['stamp']}.npz", allow_pickle=False) as Z:
            Rd, td = Z["R_d2r"].astype(float), Z["t_d2r"].astype(float)
        R0, c = L.base_pose(Rd, td); x = np.array(final[r["stamp"]]["x"]); org = c + np.array([x[0], x[1], 0.0])
        N1 = np.array(r["nodes_room"]["Node1"]); row = dict(stamp=r["stamp"], day=r["day"], origin_minus_node1=(org - N1).tolist())
        cl = r.get("node_clusters", {}).get("Node4")
        if cl:
            q = L.apply(x, R0, c, np.array(cl["radar_xyz"])[None])[0]; N4 = np.array(r["nodes_room"]["Node4"])
            u = (N4[:2] - org[:2]) / np.linalg.norm(N4[:2] - org[:2])
            row.update(node4_bev_err=float(np.linalg.norm(q[:2] - N4[:2])), node4_range_err=float((q[:2] - N4[:2]) @ u),
                       node4_bearing_err_deg=float(np.degrees(np.arctan2(u[0] * (q[1] - org[1]) - u[1] * (q[0] - org[0]), u @ (q[:2] - org[:2])))))
        nod.append(row)
    if nod:
        O = np.array([n_["origin_minus_node1"] for n_ in nod]); n4 = [n_ for n_ in nod if "node4_bev_err" in n_]
        V["node_nonbody_env1"] = dict(n=len(nod), origin_minus_node1_median=np.median(O, 0).tolist(), origin_minus_node1_sd=O.std(0).tolist(),
                                      node4_n=len(n4), node4_bev_err_median=float(np.median([n_["node4_bev_err"] for n_ in n4])) if n4 else None,
                                      node4_range_err_median=float(np.median([n_["node4_range_err"] for n_ in n4])) if n4 else None,
                                      node4_range_err_iqr=[float(v) for v in np.percentile([n_["node4_range_err"] for n_ in n4], [25, 75])] if n4 else None,
                                      node4_bearing_err_median_deg=float(np.median([n_["node4_bearing_err_deg"] for n_ in n4])) if n4 else None,
                                      per_rec=nod)
    return V


def write():
    reg = json.loads((WORK / "register.json").read_text()); p2 = reg["pass2"]
    fe = {int(k): v for k, v in p2["fused_env"].items()}; fd = p2["fused_day"]
    lev = {}
    for e in (1, 2, 3):
        lev.update(decide_level(fe[e], {k: v for k, v in fd.items() if k.startswith(f"{e}_")}))
    final, tasks = {}, []
    for r in p2["rows"]:
        key = f"{r['env']}_{r['day']}"; use_env = lev.get(key, {}).get("use_env", True)
        src = fe[r["env"]] if use_env else fd[key]; x = np.array(src["x"]); cov = np.array(src["cov"])
        ok = bool(r.get("ok")) and r.get("frames_with_body", 0) >= 0.3
        q = "ok" if ok else ("low_radar_person" if r.get("frames_with_body", 0) < 0.3 else
                             "unverified_no_depth_person" if r.get("n_pair_pts", 0) == 0 else "per_recording_fit_failed")
        info = dict(time_src=r["time"]["src"], level="env" if use_env else "env_day", quality=q,
                    source=dict(xy_yaw="depth_person (깊이 사람 보이는 표면 BEV ICP, 녹화별 부트스트랩 공분산 정보 융합)",
                                z_roll_pitch="floor (깊이 카메라 자세의 정답 없는 바닥 높이·기울기 공유, 레이더 고도로 추정 안 함)",
                                time=r["time"]["src"]))
        final[r["stamp"]] = dict(x=x.tolist(), cov=cov.tolist(), tau=r["time"]["tau"], info=info, row=r)
        tasks.append((r["stamp"], x.tolist(), r["time"]["tau"], info))
    kept = {}
    with Pool(24) as p:
        for st, kf, cover, cf in p.imap_unordered(write_one, tasks):
            kept[st] = (kf, cover, cf)
    V = validation(reg, final)
    man = dict(version="v3-radar", created="2026-11-24", code="code/v3/build_octonet_radar.py", gt_used=False,
               rule="빌드는 라벨·관절·몸 정답을 읽지 않음. 노드 강체(센서 자세)는 검사에만. 감사는 code/v3/audit_octonet_radar.py (출력은 변환에 안 들어감)",
               params=reg["params"], group_tau=reg["group_tau"], fused_env={str(k): summarize_x(np.array(v["x"]), np.array(v["cov"])) | dict(n=v["n"], used=v["used"], chi2_per_dof=v["chi2_mean"] / 3) for k, v in fe.items()},
               level_decision=lev, static_icp=reg["static"], pooled_icp=reg["pooled_icp"], validation=V, recordings={})
    for st, f in final.items():
        r = f["row"]; x, cov = np.array(f["x"]), np.array(f["cov"])
        with np.load(PACKS / f"{st}.npz", allow_pickle=False) as Z:
            Rd, td = Z["R_d2r"].astype(float), Z["t_d2r"].astype(float)
        R0, c = L.base_pose(Rd, td); R, t = L.full_RT(x, R0, c); kf, cover, cf = kept[st]
        man["recordings"][st] = dict(env=r["env"], day=r["day"], user=r["user"], action=r["action"], quality=f["info"]["quality"],
                                     R=R.round(6).tolist(), t=t.round(6).tolist(), delta_vs_camera=summarize_x(x, cov), source=f["info"]["source"], level=f["info"]["level"],
                                     bootstrap_sd_used=summarize_x(x, cov), per_recording_fit=summarize_x(np.array(r["x"]), np.array(r["cov"])) if r.get("ok") else None,
                                     tau_s=f["tau"], time=r["time"], kept_frac=kf, frames_with_kept=cover, class_frac=cf,
                                     residual=dict(med_nn_m=r.get("med_nn"), signed_radial_m=r.get("signed_radial"), person_range_m=r.get("person_range")))
    (OUT / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=float))
    (OUT / "quality.json").write_text(json.dumps(dict(rule="status != ok 인 녹화의 레이더는 학습·평가 입력에서 뺌", status={st: v["quality"] for st, v in man["recordings"].items()}), indent=1))
    kf = np.array([v[0] for v in kept.values()])
    print("WRITE_DONE 녹화", len(kept), "남긴 비율 중앙", round(float(np.median(kf)), 3), "평균", round(float(kf.mean()), 3), flush=True)


def static_only(do_pool=True):
    """정지 ICP 단계만 다시 (발산 안전장치 추가 후). register.json 의 static 을 바꿔 씀."""
    out = json.loads((WORK / "register.json").read_text()); rows = out["pass2"]["rows"]; st_res = {}
    for e in (1, 2, 3):
        xe = np.array(out["pass2"]["fused_env"][str(e)]["x"])
        for day in sorted({r["day"] for r in rows if r["env"] == e}):
            st_res[f"{e}_{day}"] = static_icp([r for r in rows if r["env"] == e and r["day"] == day], xe, e, day)
            print("static", e, day, {k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in st_res[f'{e}_{day}'].items() if k != "cov"}, flush=True)
    out["static"] = st_res; pool_res = out.get("pooled_icp", {})
    for e in ((1, 2, 3) if do_pool else ()):
        xe = np.array(out["pass2"]["fused_env"][str(e)]["x"])
        xp, ip = pooled_icp([r for r in rows if r["env"] == e and r.get("ok")], xe)
        pool_res[e] = dict(x=xp.tolist(), **ip); print("pooled ICP env", e, summarize_x(xp), ip, flush=True)
    out["pooled_icp"] = pool_res
    (WORK / "register.json").write_text(json.dumps(out, default=float)); print("STATIC_DONE", flush=True)


def pooled_only():
    out = json.loads((WORK / "register.json").read_text()); rows = out["pass2"]["rows"]; pool_res = {}
    for e in (1, 2, 3):
        xe = np.array(out["pass2"]["fused_env"][str(e)]["x"])
        xp, ip = pooled_icp([r for r in rows if r["env"] == e and r.get("ok")], xe)
        pool_res[e] = dict(x=xp.tolist(), **ip); print("pooled ICP env", e, summarize_x(xp), ip, flush=True)
    out["pooled_icp"] = pool_res; (WORK / "register.json").write_text(json.dumps(out, default=float)); print("POOLED_DONE", flush=True)


if __name__ == "__main__":
    step = sys.argv[1]
    {"prep": prep, "register": register, "write": write, "static": static_only, "static_score": lambda: static_only(False), "pooled": pooled_only}[step]()
