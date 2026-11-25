"""v3 MM-Fi LiDAR → 방(깊이 카메라 방 좌표) 정답 없는 재정합 (2026-11-24). 정답·라벨·IR 을 읽지 않는다 (mmfi_common.SafeNpz).
단계 (녹화마다; 방·세션 결합은 build_lidar_mmfi.py):
 A. 바닥: LiDAR 원 구름의 낮은 점 (z < −0.9) RANSAC 평면 → roll·pitch·높이 h_L. 목표 바닥 = 녹화별 깊이 바닥 (cache_v1 bg_med, 방 좌표).
    방 좌표를 깊이 바닥 정렬 좌표 F (z = 깊이 바닥 위 높이) 로 바꾸고, LiDAR 는 자기 바닥으로 수평·높이를 맞춘 좌표 L' 로 둔다.
 B. 거친 수평 이동 (yaw 0): 깊이 사람 중심 (프레임별) − LiDAR 점 (몸 높이 0.3–1.9 m) 의 2D 히스토그램 (방 단위로 합침).
    방 안 녹화마다 사람 위치가 달라 정적 물체 봉우리는 흐려지고 사람 봉우리만 남는다.
 C. 사람 재추출: 깊이 사람 중심을 LiDAR 좌표로 옮긴 앵커 둘레 0.7 m 원통, 바닥 위 0.05–2.3 m, 0.2 m 연결 성분 중 앵커 0.35 m 안 성분.
 D. 미세 정합 (x, y, yaw): 같은 프레임 (시차 τ) 깊이 사람 점과 LiDAR 사람 점의 3D 최근접 쌍 → 가중 2D Procrustes 반복 (Huber 3 cm, 15 cm 절삭).
    τ ∈ {−2..2} 는 절삭 잔차 최소. 블록 부트스트랩 (15 프레임 블록) 으로 σ.
출력: dev/packs/v3/mmfi_lidar/_reg/{env}_{nm}.npz (사람 점, 추정치) — 팩은 build_lidar_mmfi.py 가 씀.
사용: python -I reg_lidar_mmfi.py stageA|stageC [env ...]"""
import sys, json, os, zlib
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import mmfi_common as mc

HB, HSTEP = 1.5, 0.02                     # 거친 탐색 히스토그램 반폭·칸 (m)
NB = int(round(2 * HB / HSTEP))
H_LO, H_HI = 0.05, 2.3                    # 사람 점 높이 (바닥 위)
R_CYL, R_IN, LINK = 0.7, 0.35, 0.2       # 앵커 원통 반경, 성분 선택 반경, 연결 거리
TAUS = (-2, -1, 0, 1, 2)


def floor_lidar(F):
    P = np.concatenate([f[f[:, 2] < -0.9] for f in F[::3]])
    n, c, inl, sig = mc.ransac_plane(P, up=np.array([0, 0, 1.0]), min_cos=np.cos(np.radians(15)), tol=0.015, iters=600, seed=1)
    # 바닥 점의 수평 범위 (조건수 진단)
    Q = P[np.abs(P @ n + c) < 0.015]
    ext = (np.percentile(Q[:, 0], [5, 95]).tolist(), np.percentile(Q[:, 1], [5, 95]).tolist())
    return n, c, inl, sig, ext


def frame_F(fl_d):
    """방 → F (깊이 바닥 정렬) 변환: X_F = Rt^T (X − o)."""
    if fl_d is None:
        return np.eye(3), np.zeros(3)
    n = np.array(fl_d["n"]); c = fl_d["c"]
    return mc.rot_a_to_b(np.array([0, 0, 1.0]), n), -c * n


def stageA(job):
    env, nm = job; out = mc.WORK_L / f"A_{env}_{nm}.npz"
    if out.exists():
        return env, nm, "skip"
    F = mc.lidar_frames(env, nm)
    D, Rd, td, zc = mc.depth_person(env, nm)
    T = min(len(F), len(D))
    if T == 0:
        return env, nm, "empty"
    fl_d = mc.depth_floor_rec(env, nm, Rd, td); Rt, o = frame_F(fl_d)
    nL, cL, inl, sig, ext = floor_lidar(F)
    Rl = mc.rot_a_to_b(nL, np.array([0, 0, 1.0]))
    cd = np.full((T, 2), np.nan)
    for k in range(T):
        if len(D[k]) >= 30:
            cd[k] = np.median((D[k] - o) @ Rt, 0)[:2]
    Hh = np.zeros((NB, NB), np.float32); nf = 0
    for k in range(0, T, 2):
        if not np.isfinite(cd[k]).all():
            continue
        q = F[k] @ Rl.T; h = q[:, 2] + cL; m = (h > 0.3) & (h < 1.9)
        r = cd[k] - q[m, :2]; i = np.floor((r + HB) / HSTEP).astype(int); ok = (i >= 0).all(1) & (i < NB).all(1)
        np.add.at(Hh, (i[ok, 0], i[ok, 1]), 1.0); nf += 1
    np.savez_compressed(out, H=Hh / max(nf, 1), nL=nL, cL=cL, Rl=Rl, Rt=Rt, o=o, cd=cd, T=T, floor_d=json.dumps(fl_d), floor_l=json.dumps(dict(inliers=inl, sigma=sig, ext=ext)))
    return env, nm, f"ok hL {cL:.3f} tiltL {np.degrees(np.arccos(nL[2])):.2f} floorD {None if fl_d is None else round(fl_d['tilt_deg'], 2)}"


def disk_smooth(H, r=0.3):
    from scipy.signal import fftconvolve
    g = np.arange(-int(r / HSTEP), int(r / HSTEP) + 1) * HSTEP; X, Y = np.meshgrid(g, g, indexing="ij"); K = (X ** 2 + Y ** 2 <= r * r).astype(float)
    return fftconvolve(H, K, mode="same")


def peak(H, center=None, rad=None, rmax=1.0):
    S = disk_smooth(H); g = -HB + (np.arange(NB) + 0.5) * HSTEP; X, Y = np.meshgrid(g, g, indexing="ij")
    m = X ** 2 + Y ** 2 <= rmax ** 2
    if center is not None:
        m &= (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= rad ** 2
    S2 = np.where(m, S, -1); i = np.unravel_index(np.argmax(S2), S2.shape)
    # 두 번째 봉우리 (0.6 m 밖) 대비 비
    far = (X - X[i]) ** 2 + (Y - Y[i]) ** 2 > 0.6 ** 2
    s2 = S[far & (X ** 2 + Y ** 2 <= 1.4 ** 2)].max() if (far & (X ** 2 + Y ** 2 <= 1.4 ** 2)).any() else 0.0
    return np.array([X[i], Y[i]]), float(S[i]), float(s2)


def extract(q, h, anchor):
    """LiDAR 수평 좌표 q (n,2), 높이 h → 앵커 둘레 사람 점 인덱스."""
    m = (h > H_LO) & (h < H_HI) & (np.linalg.norm(q - anchor, axis=1) < R_CYL)
    idx = np.flatnonzero(m)
    if len(idx) < 5:
        return idx[:0]
    X = np.c_[q[idx], h[idx]]; tr = cKDTree(X); pr = tr.query_pairs(LINK, output_type="ndarray")
    G = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(idx), len(idx)))
    nc, lab = connected_components(G, directed=False)
    d = np.linalg.norm(q[idx] - anchor, axis=1)
    keep = np.zeros(len(idx), bool)
    for c in range(nc):
        mm = lab == c
        if d[mm].min() < R_IN and mm.sum() >= 3:
            keep |= mm
    return idx[keep]


def icp(P, Q, ks_p, ks_q, x0, iters=25, huber=0.03, trim=0.15):
    """P: LiDAR 점 (n,3) [L' x, y, 높이], Q: 깊이 점 (m,3) [F x, y, z], ks = 프레임 번호 (같은 프레임끼리만 짝).
    반환 ψ, t, 절삭 잔차 중앙값, 사용 쌍 수."""
    SC = 1000.0
    tr = cKDTree(np.c_[Q, ks_q * SC]); psi, t = x0[0], np.array(x0[1:3], float)
    for _ in range(iters):
        c, s = np.cos(psi), np.sin(psi); Pt = np.c_[P[:, :2] @ np.array([[c, s], [-s, c]]) + t, P[:, 2]]
        d, j = tr.query(np.c_[Pt, ks_p * SC], distance_upper_bound=trim)
        ok = np.isfinite(d)
        if ok.sum() < 50:
            return psi, t, np.nan, int(ok.sum())
        w = mc.huber_w(d[ok], huber); p2, tt = mc.procrustes2d(P[ok, :2], Q[j[ok], :2], w)
        conv = abs(p2 - psi) < 1e-5 and np.linalg.norm(tt - t) < 1e-4; psi, t = p2, tt
        if conv:
            break
    c, s = np.cos(psi), np.sin(psi); Pt = np.c_[P[:, :2] @ np.array([[c, s], [-s, c]]) + t, P[:, 2]]
    d, j = tr.query(np.c_[Pt, ks_p * SC], distance_upper_bound=trim); ok = np.isfinite(d)
    return psi, t, float(np.median(d[ok])) if ok.any() else np.nan, int(ok.sum())


def gather(F, D, A, t0, tau, frames, Rt, o):
    """프레임 목록에서 (LiDAR 사람 점, 깊이 점) 모으기. LiDAR 점은 앵커 = 깊이 중심 − t0 (yaw 0) 로 추출."""
    Rl, cL, cd = A["Rl"], float(A["cL"]), A["cd"]; P, kp, Qs, kq, per = [], [], [], [], {}
    for k in frames:
        kd = k + tau
        if kd < 0 or kd >= len(D) or not np.isfinite(cd[kd]).all() or len(D[kd]) < 30:
            continue
        q = F[k] @ Rl.T; h = q[:, 2] + cL; idx = extract(q[:, :2], h, cd[kd] - t0)
        if len(idx) < 15:
            continue
        P.append(np.c_[q[idx, :2], h[idx]]); kp.append(np.full(len(idx), k)); Qs.append((D[kd] - o) @ Rt); kq.append(np.full(len(D[kd]), k)); per[k] = (q[idx, :2], h[idx])
    if not P:
        return None
    return np.concatenate(P), np.concatenate(kp), np.concatenate(Qs), np.concatenate(kq), per


def stageC(job):
    env, nm, t0 = job; out = mc.WORK_L / f"C_{env}_{nm}.npz"
    if out.exists():
        return env, nm, "skip"
    A = dict(np.load(mc.WORK_L / f"A_{env}_{nm}.npz", allow_pickle=False)); F = mc.lidar_frames(env, nm)
    D, Rd, td, zc = mc.depth_person(env, nm, drop_edge=True); T = int(A["T"]); Rt, o = A["Rt"], A["o"]
    t0 = np.array(t0, float)
    # τ 선택: 프레임 2 개마다, 각 τ 에서 ICP 후 절삭 잔차
    res = {}
    for tau in TAUS:
        g = gather(F, D, A, t0, tau, range(2, T - 2, 2), Rt, o)
        if g is None:
            continue
        P, kp, Q, kq, _ = g; psi, t, med, n = icp(P, Q, kp, kq, [0.0, *t0])
        res[tau] = (psi, t, med, n)
    if 0 not in res:
        np.savez_compressed(out, ok=False); return env, nm, "fail"
    taus = np.array([k for k in res]); meds = np.array([res[k][2] for k in res])
    tau = int(taus[np.nanargmin(meds)])
    # 최종: 모든 프레임, 선택 τ, 앵커를 추정 변환으로 다시 잡아 재추출 (yaw 반영: 앵커 = R^T (cd − t))
    psi, t = res[tau][0], res[tau][1]
    for _ in range(2):
        g = gather_tr(F, D, A, psi, t, tau, range(T), Rt, o)
        P, kp, Q, kq, per = g; psi, t, med, n = icp(P, Q, kp, kq, [psi, *t])
    # 블록 부트스트랩 (15 프레임 블록)
    rng = np.random.default_rng(zlib.crc32(f"{env}/{nm}".encode())); ks = np.array(sorted(per)); blocks = [ks[i:i + 15] for i in range(0, len(ks), 15)]; bs = []
    for b in range(30):
        pick = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        cnt = np.bincount(np.searchsorted(ks, pick), minlength=len(ks))
        wP = cnt[np.searchsorted(ks, kp)]; wQ = cnt[np.searchsorted(ks, kq)]
        selP = np.repeat(np.arange(len(P)), wP); selQ = np.repeat(np.arange(len(Q)), wQ)
        # 같은 프레임이 여러 번 뽑히면 복제 (프레임 번호에 복제 번호를 더해 구분하지 않아도 같은 쌍이 반복될 뿐)
        p_, t_, m_, n_ = icp(P[selP], Q[selQ], kp[selP], kq[selQ], [psi, *t], iters=10); bs.append([p_, *t_])
    bs = np.array(bs)
    # 프레임별 사람 점 저장 (L' 수평 + 높이), 점 수
    nper = np.zeros(T, np.int32); off = [0]; pts = []
    for k in range(T):
        if k in per:
            pts.append(np.c_[per[k][0], per[k][1]].astype(np.float32)); nper[k] = len(per[k][1])
        off.append(off[-1] + nper[k])
    # 진단: 프레임별 LiDAR 사람 중심 (방 F) vs 깊이 중심, 녹화 안 옆 상관
    c, s = np.cos(psi), np.sin(psi); R2 = np.array([[c, -s], [s, c]]); cl = np.full((T, 2), np.nan)
    for k in per:
        cl[k] = np.median(per[k][0] @ R2.T + t, 0)
    cd = A["cd"]; cdt = np.full((T, 2), np.nan)
    for k in range(T):
        if 0 <= k + tau < len(cd):
            cdt[k] = cd[k + tau]
    ok = np.isfinite(cl).all(1) & np.isfinite(cdt).all(1)
    corr = [float(np.corrcoef(cl[ok, i], cdt[ok, i])[0, 1]) if ok.sum() > 20 and np.std(cdt[ok, i]) > 1e-4 else np.nan for i in (0, 1)]
    diff = cl[ok] - cdt[ok]
    np.savez_compressed(out, ok=True, psi=psi, t=t, tau=tau, med=med, npair=n, boot=bs, tau_curve=np.array([[k, *res[k][0:1], *res[k][1], res[k][2]] for k in sorted(res)]),
                        pts=np.concatenate(pts) if pts else np.zeros((0, 3), np.float32), off=np.array(off, np.int64), nper=nper, cl=cl, cdt=cdt,
                        corr=np.array(corr), diff_med=np.median(diff, 0) if len(diff) else np.full(2, np.nan), diff_std=np.std(diff, 0) if len(diff) else np.full(2, np.nan))
    return env, nm, f"ok psi {np.degrees(psi):.2f} t {t.round(3)} tau {tau} med {med:.3f} σψ {np.degrees(bs[:, 0].std()):.2f} σt {bs[:, 1:].std(0).round(3)} corr {np.round(corr, 2)} pts {nper[nper > 0].mean() if (nper > 0).any() else 0:.0f}"


def gather_tr(F, D, A, psi, t, tau, frames, Rt, o):
    """추정 변환 (ψ, t) 로 앵커를 LiDAR 좌표로 옮겨 추출: 앵커 = R(ψ)^T (cd − t)."""
    Rl, cL, cd = A["Rl"], float(A["cL"]), A["cd"]; c, s = np.cos(psi), np.sin(psi); R2 = np.array([[c, -s], [s, c]])
    P, kp, Qs, kq, per = [], [], [], [], {}
    for k in frames:
        kd = k + tau
        if kd < 0 or kd >= len(D) or not np.isfinite(cd[kd]).all() or len(D[kd]) < 30:
            continue
        q = F[k] @ Rl.T; h = q[:, 2] + cL; anchor = (cd[kd] - t) @ R2; idx = extract(q[:, :2], h, anchor)
        if len(idx) < 15:
            continue
        P.append(np.c_[q[idx, :2], h[idx]]); kp.append(np.full(len(idx), k)); Qs.append((D[kd] - o) @ Rt); kq.append(np.full(len(D[kd]), k)); per[k] = (q[idx, :2], h[idx])
    return np.concatenate(P), np.concatenate(kp), np.concatenate(Qs), np.concatenate(kq), per


def coarse_room(envs):
    """방마다 히스토그램 합 → t0_room; 녹화마다 자기 히스토그램의 봉우리 (t0_room 0.3 m 안)."""
    out = {}
    for env in envs:
        recs = [r for r in mc.recordings((env,))]; Hs = {}
        for e, nm in recs:
            f = mc.WORK_L / f"A_{e}_{nm}.npz"
            if f.exists():
                Hs[nm] = np.load(f)["H"]
        Hsum = sum(Hs.values()) / len(Hs); t_room, s1, s2 = peak(Hsum)
        per = {}
        for nm, H in Hs.items():
            t_r, a1, a2 = peak(H, center=t_room, rad=0.3); per[nm] = dict(t0=t_r.tolist(), peak=a1, peak2_far=a2)
        out[env] = dict(t0_room=t_room.tolist(), peak=s1, peak2_far=s2, rec=per)
        print(env, "coarse t0_room", t_room.round(3), "peak", round(s1, 1), "2nd (>0.6 m)", round(s2, 1), flush=True)
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mc.WORK_L.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1]; envs = tuple(sys.argv[2:]) or mc.ENVS
    lim = int(os.environ.get("V3_LIMIT", "0"))
    recs = mc.recordings(envs)
    if lim:
        recs = [r for i, r in enumerate(recs) if i % max(1, len(recs) // lim) == 0]
    if stage == "stageA":
        with Pool(int(os.environ.get("V3_NP", "24"))) as p:
            for env, nm, st in p.imap_unordered(stageA, recs):
                print("A", env, nm, st, flush=True)
    elif stage == "coarse":
        co = coarse_room(envs); json.dump(co, open(mc.WORK_L / f"coarse_{'_'.join(envs)}.json", "w"), indent=1)
    elif stage == "stageC":
        co = {}
        for f in mc.WORK_L.glob("coarse_*.json"):
            co.update(json.load(open(f)))
        jobs = [(e, nm, co[e]["rec"][nm]["t0"]) for e, nm in recs if e in co and nm in co[e]["rec"]]
        with Pool(int(os.environ.get("V3_NP", "24"))) as p:
            for env, nm, st in p.imap_unordered(stageC, jobs):
                print("C", env, nm, st, flush=True)
    print("LIDAR", stage, "DONE", mc.assert_no_gt_access(), flush=True)
