"""11-23 다중 센서 LHR — 데이터 1단계: OctoNet 노드 1 TI 레이더를 LHR 팩 (octonet_v07, 방 좌표) 에 붙임. v2 = 정답 없는 레이더 ↔ 깊이 정합.
v1 (정답 ICP, attach_radar_octonet_v1_label.py) 실패 원인 (11-23 진단, run_log): 점의 55–87 % 가 정지 반사 (벽·가구) → 점–뼈 ICP 가 사람을 못 잡음 (중앙 1.04 m),
녹화마다 레이더 시계가 −0.5 ~ −1.75 s 어긋남, 좌표 반사는 아님. 정지 칸 제거 + 녹화별 시차 → 사람이 움직이는 녹화에서 중심 잔차 0.24–0.32 m.
방법 (정답 안 씀):
 1) 레이더 점: 유한 · 거리 0.3–10 m (|좌표| ~1e29 이상값, 원점 근처 점 제거). 녹화 안 0.2 m 칸 중 레이더 프레임 20 % 넘게 차 있는 칸 = 정지 (radar_static 표시, 지우지는 않음).
 2) 동적 점 중앙 c_r(t) (레이더 좌표) ↔ 깊이 사람 점 중앙 c_d(k) (깊이 카메라 좌표, 팩 pts_uvz 역투영; 깊이 사람 분할은 이미 정답 없음).
 3) 반복: 환경별 레이더 → 카메라 강체 (잘라낸 Kabsch, 깊이 중심이 0.2 m 넘게 움직인 녹화만) ↔ 녹화별 시차 (−3 … +1 s 격자; 깊이 중심이 거의 안 움직이면 환경 중앙값).
 4) 검사에만 정답: 몸 0.3 m 안 점 비율 (전체 / 동적), 동적 중심 − 골반 거리.
출력: dev/packs/octonet_v08_radar/<stamp>.npz (radar_pts (N,5) = 방 x y z, 도플러 m/s, 세기 0 (없음); radar_static (N,) bool; radar_off (T+1); meta) + attach_report.json.
읽기는 허용 목록 Unpickler (octonet_io._Safe). 실행: unicorrn 환경 python -I (PIL 필요)."""
import sys, json
from pathlib import Path
sys.path.insert(0, "/home/interns/unisense/code")
import numpy as np
import octonet_io as oi

U = Path("/home/interns/unisense"); PK = U / "dev/packs/octonet_v07"; OUT = U / "dev/packs/octonet_v08_radar"; CACHE = U / "dev/data/octonet/derived/mocap13"
OFF = {x["stamp"]: x for x in json.loads((U / "notes/octonet_align_v2.json").read_text())["offsets"]}
SPEC = json.loads((U / "code/specs/octonet_v07.json").read_text())["depth"]
SPL = json.loads((U / "code/splits/octonet.json").read_text()); TRAIN = set(int(u) for u in SPL["train"])
BN = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12), (1, 2)]
STEP, TOL, RMIN, RMAX, VOX, OCC = 3, 0.06, 0.3, 10.0, 0.2, 0.2
SHIFTS = np.round(np.arange(-3.0, 1.01, 0.1), 2); MOVE_MIN = 0.2
CARD = dict(device="TI mmWave (OctoNet node 1)", doppler="연속값 m/s", frames_per_s=8.9, has_intensity=False, range_filter_m=[RMIN, RMAX],
            columns="x y z (방 m), 도플러 m/s, 세기 = 0 (원본에 없음)", static_flag=f"녹화 안 {VOX} m 칸이 레이더 프레임 {OCC:.0%} 넘게 차 있으면 정지 (정답 없음)")
OUT.mkdir(parents=True, exist_ok=True)


def seg_dist(P, J):
    A = J[[a for a, _ in BN]]; d = J[[b for _, b in BN]] - A; L2 = (d * d).sum(-1).clip(1e-9)
    t = (((P[:, None] - A[None]) * d[None]).sum(-1) / L2[None]).clip(0, 1); return np.linalg.norm(P[:, None] - (A[None] + t[..., None] * d[None]), axis=-1).min(1)


def kabsch(X, Y):
    mx, my = X.mean(0), Y.mean(0); U_, S, Vt = np.linalg.svd((X - mx).T @ (Y - my)); Dg = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U_.T))]); R = Vt.T @ Dg @ U_.T
    return R, my - R @ mx


def robust(X, Y, it=6):
    R, t = kabsch(X, Y)
    for _ in range(it):
        d = np.linalg.norm(X @ R.T + t - Y, axis=1); k = d <= np.percentile(d, 70); R, t = kabsch(X[k], Y[k])
    return R, t, float(np.median(np.linalg.norm(X @ R.T + t - Y, axis=1)))


def read_radar(path):
    fr = []
    with open(path, "rb") as fh:
        while True:
            try:
                fr.append(oi._Safe(fh).load())
            except EOFError:
                break
    ts = np.array([f["timestamp"].timestamp() for f in fr]); pts = []
    for f in fr:
        d = f["data"]; p = np.c_[d["x"], d["y"], d["z"], d["velocity"]].astype(float)[:int(d["numObj"])]; rg = np.linalg.norm(np.nan_to_num(p[:, :3], nan=99.0), axis=1)
        pts.append(p[np.isfinite(p).all(1) & (rg >= RMIN) & (rg <= RMAX)])
    return ts, pts


def static_flags(pts):
    vox = [np.floor(p[:, :3] / VOX).astype(np.int64) for p in pts]; nz = [v for v in vox if len(v)]
    if not nz:
        return [np.zeros(0, bool) for _ in pts]
    keys, cnt = np.unique(np.concatenate([np.unique(v, axis=0) for v in nz]), axis=0, return_counts=True); st = keys[cnt > OCC * len(pts)]
    code = lambda v: (v[:, 0] * 1000003 + v[:, 1]) * 1000003 + v[:, 2]; sc = code(st) if len(st) else np.zeros(0, np.int64)
    return [np.isin(code(v), sc) if len(v) else np.zeros(0, bool) for v in vox]


# 1) 읽기
D = {}
for r in oi.recordings():
    st = r["stamp"]; f = PK / f"{st}.npz"
    if not f.exists():
        continue
    P = np.load(f, allow_pickle=False); m = json.loads(str(P["meta"])); T = int(P["labels"].shape[0])
    t = np.load(CACHE / f"{st}.npz", allow_pickle=False)["t"]; Z = oi.DepthZip(r["depth"]); dt = OFF[st]["dt"]
    idx = np.where((Z.t + dt > t[0]) & (Z.t + dt < t[-1]))[0][::STEP]; assert len(idx) == T, st
    fx, fy, cx, cy = SPEC[f"{m['env']}_{m['day']}"]["K"]; uvz = P["pts_uvz"].astype(float); val = P["pts_valid"]
    cd = np.full((T, 3), np.nan)
    for k in np.flatnonzero(val.sum(1) >= 20):
        u, v, z = uvz[k, val[k]].T; cd[k] = np.median(np.c_[(u - cx) / fx * z, (v - cy) / fy * z, z], 0)
    ts, pts = read_radar(r["radar"]); sf = static_flags(pts)
    cr = np.array([np.median(p[~s, :3], 0) if (~s).sum() >= 2 else [np.nan] * 3 for p, s in zip(pts, sf)]).reshape(-1, 3)
    okd = np.isfinite(cd).all(1); mv = float(np.nanstd(cd[okd], 0)[[0, 2]].max()) if okd.sum() > 20 else 0.0
    D[st] = dict(env=int(m["env"]), user=int(m["user"]), action=m["action"], tk=Z.t[idx], ts=ts, pts=pts, sf=sf, cr=cr, cd=cd, move=mv,
                 L=P["labels"].astype(float), Rc=P["R_c2r"].astype(float), tc=P["t_c2r"].astype(float))
print("녹화", len(D), "깊이 중심이 0.2 m 넘게 움직인 녹화", sum(d["move"] > MOVE_MIN for d in D.values()), flush=True)


def pairs(d, sh):
    if len(d["ts"]) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    j, e = oi.nearest(d["ts"] + sh, d["tk"]); ok = (np.abs(e) <= TOL) & np.isfinite(d["cd"]).all(1) & np.isfinite(d["cr"][j]).all(1)
    return d["cr"][j[ok]], d["cd"][ok]


# 2) 환경별 외부 보정 ↔ 녹화별 시차 (반복)
SH = {st: -1.2 for st in D}; EXT = {}
for rnd in range(4):
    for e in (1, 2, 3):
        X, Y = [], []
        for st, d in D.items():
            if d["env"] == e and d["move"] > MOVE_MIN:
                x, y = pairs(d, SH[st]); X.append(x); Y.append(y)
        EXT[e] = robust(np.concatenate(X), np.concatenate(Y))
    for st, d in D.items():
        if d["move"] <= MOVE_MIN:
            continue
        R, t, _ = EXT[d["env"]]; best = (9.0, SH[st])
        for sh in SHIFTS:
            x, y = pairs(d, sh)
            if len(x) >= 30:
                md = float(np.median(np.linalg.norm(x @ R.T + t - y, axis=1)))
                if md < best[0]:
                    best = (md, float(sh))
        SH[st] = best[1]
    for e in (1, 2, 3):
        sv = [SH[st] for st, d in D.items() if d["env"] == e and d["move"] > MOVE_MIN]; med = float(np.median(sv))
        for st, d in D.items():
            if d["env"] == e and d["move"] <= MOVE_MIN:
                SH[st] = med
        print(f"반복 {rnd} 환경 {e}: 중심 잔차 중앙 {EXT[e][2]:.3f} m, 시차 중앙 {med:+.2f} s (사분위 {np.percentile(sv, 25):+.2f}…{np.percentile(sv, 75):+.2f})", flush=True)
for a in (1, 2, 3):
    for b in (1, 2, 3):
        if a < b:
            print(f"환경 {a}↔{b} 레이더→카메라 회전 차 {np.degrees(np.arccos(np.clip((np.trace(EXT[a][0].T @ EXT[b][0]) - 1) / 2, -1, 1))):.1f}°, 위치 차 {np.linalg.norm(EXT[a][1] - EXT[b][1]):.2f} m", flush=True)

# 3) 쓰기 + 검사 (정답은 검사에만)
rep = {}; rng = np.random.default_rng(0)
for st, d in D.items():
    R, t, _ = EXT[d["env"]]; T = len(d["tk"]); j, e_ = oi.nearest(d["ts"] + SH[st], d["tk"]) if len(d["ts"]) else (np.zeros(T, int), np.full(T, 9.0))
    rows, flags = [], []; nb_all, nb_dyn, cen = [], [], []
    for k in range(T):
        if abs(e_[k]) > TOL or len(d["pts"][j[k]]) == 0:
            rows.append(np.zeros((0, 5), np.float32)); flags.append(np.zeros(0, bool)); continue
        p = d["pts"][j[k]]; s = d["sf"][j[k]]; q = (p[:, :3] @ R.T + t) @ d["Rc"].T + d["tc"]
        rows.append(np.c_[q, p[:, 3], np.zeros(len(p))].astype(np.float32)); flags.append(s)
        J = d["L"][k]
        if rng.random() < 0.1 and np.isfinite(J).all():
            dd = seg_dist(q, J); nb_all.append(float((dd < 0.3).mean()))
            if (~s).sum():
                nb_dyn.append(float((dd[~s] < 0.3).mean())); cen.append(float(np.linalg.norm(np.median(q[~s], 0) - J[[7, 8]].mean(0))))
    off = np.r_[0, np.cumsum([len(x) for x in rows])]
    meta = dict(dataset="octonet", stamp=st, env=d["env"], user=d["user"], T=T, card=CARD, version="v0.8-radar",
                time=dict(radar_shift_s=SH[st], source="GT-free: dynamic radar centroid ↔ depth person centroid" if d["move"] > MOVE_MIN else "GT-free: env median (depth centroid moved < 0.2 m)", tol_s=TOL),
                extrinsic=dict(source="GT-free radar↔depth centroid registration (per env, trimmed Kabsch)", frame="radar → depth camera (node 1), then pack R_c2r/t_c2r", R=R.round(6).tolist(), t=t.round(6).tolist()))
    np.savez_compressed(OUT / f"{st}.npz", radar_pts=np.concatenate(rows), radar_static=np.concatenate(flags), radar_off=off, meta=json.dumps(meta, ensure_ascii=False))
    rep[st] = dict(env=d["env"], user=d["user"], action=d["action"], train=d["user"] in TRAIN, T=T, move=round(d["move"], 3), shift=SH[st], frames_with_radar=int(sum(len(x) > 0 for x in rows)),
                   pts_med=float(np.median([len(x) for x in rows if len(x)])) if any(len(x) for x in rows) else 0.0, static_frac=round(float(np.concatenate(flags).mean()), 3) if off[-1] else None,
                   near_all=round(float(np.mean(nb_all)), 3) if nb_all else None, near_dyn=round(float(np.mean(nb_dyn)), 3) if nb_dyn else None, cen_pelvis_m=round(float(np.median(cen)), 3) if cen else None)
summ = {}
for e in (1, 2, 3):
    for tr in (True, False):
        v = [x for x in rep.values() if x["env"] == e and x["train"] == tr]
        if not v:
            continue
        g = lambda k: [x[k] for x in v if x[k] is not None]
        summ[f"E{e}_{'train' if tr else 'other'}"] = dict(recs=len(v), radar_cover=round(sum(x["frames_with_radar"] for x in v) / sum(x["T"] for x in v), 3), static_frac=round(float(np.mean(g("static_frac"))), 3),
                                                          near_all=round(float(np.mean(g("near_all"))), 3), near_dyn=round(float(np.mean(g("near_dyn"))), 3), cen_pelvis_m=round(float(np.median(g("cen_pelvis_m"))), 3))
for k_, v in summ.items():
    print(k_, json.dumps(v, ensure_ascii=False), flush=True)
mv = [x for x in rep.values() if x["move"] > MOVE_MIN]; st_ = [x for x in rep.values() if x["move"] <= MOVE_MIN]
for nm, v in (("움직인 녹화", mv), ("제자리 녹화", st_)):
    print(nm, len(v), "동적 몸 근처", round(float(np.mean([x["near_dyn"] for x in v if x["near_dyn"] is not None])), 3), "동적 중심−골반 중앙", round(float(np.median([x["cen_pelvis_m"] for x in v if x["cen_pelvis_m"] is not None])), 3), flush=True)
json.dump(dict(shift=SH, ext={e: dict(R=EXT[e][0].tolist(), t=EXT[e][1].tolist(), resid_m=EXT[e][2]) for e in EXT}, summary=summ, recordings=rep), open(OUT / "attach_report.json", "w"), indent=1, ensure_ascii=False)
print("ATTACH_DONE")
