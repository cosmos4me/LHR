"""v3 OctoNet 레이더 감사 (정답은 여기서만 읽는다; 출력은 변환·팩에 다시 들어가지 않는다).
before = dev/packs/octonet_v08_radar (v2 캐시가 쓴 팩), after = dev/packs/v3/octonet_radar.
지표 (ms_cache2.radar_tokens 와 같은 토큰 창: 팩 프레임 2개마다, 0.25 s (현재 + 앞 2 프레임), 프레임당 ≤ 32점, 마지막 96점, 정지 표시 아닌 점):
- 레이더 토큰 중심 − 정답 골반 (엉덩이 두 관절 중점): 방 x, y, z, 시선 (깊이 카메라 → 골반 수평) 성분
  · 녹화 평균의 녹화 사이 표준편차, 녹화 안 표준편차 (녹화별 std 의 중앙값)
  · 정지 (골반 속력 < 0.1 m/s) / 움직임 (> 0.4 m/s) 구간의 녹화별 중앙값 → 녹화 사이 중앙값 (reg_timediag.py 와 같은 방식)
  · τ (속도 회귀) 중앙값
- 몸 근처 비율: 토큰 점 중 뼈대 선분 0.3 m 안 비율. v3 는 정리 분류별 (all_cls) 몸 근처 비율도 냄 (정리 정밀도 감사).
출력: dev/packs/v3/octonet_radar/audit.json + 표준 출력.
실행: CUDA_VISIBLE_DEVICES="" python -I code/v3/audit_octonet_radar.py (tmux v3_octo_audit)"""
import sys
sys.path.insert(0, "/home/interns/unisense/code/v3")
import json, glob, os, zlib
from multiprocessing import Pool
import numpy as np

U = "/home/interns/unisense"; PK = f"{U}/dev/packs/octonet_v07"; V08 = f"{U}/dev/packs/octonet_v08_radar"; V3 = f"{U}/dev/packs/v3/octonet_radar"
BN = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12), (1, 2)]
NR, NRF, STRIDE, FPS = 96, 32, 2, 10.0
Q08 = json.load(open(f"{V08}/quality.json"))["status"]


def seg_dist(P, J):
    A = J[[a for a, _ in BN]]; d = J[[b for _, b in BN]] - A; L2 = (d * d).sum(-1).clip(1e-9)
    t = (((P[:, None] - A[None]) * d[None]).sum(-1) / L2[None]).clip(0, 1)
    return np.linalg.norm(P[:, None] - (A[None] + t[..., None] * d[None]), axis=-1).min(1)


def tokens(rp, ro, st, ks, rng):
    nb = max(1, int(round(0.25 * FPS))); out = []
    for k in ks:
        rows = []
        for j in range(max(0, k - nb), k + 1):
            a, b = ro[j], ro[j + 1]
            if b > a:
                sel = np.arange(a, b) if b - a <= NRF else rng.choice(np.arange(a, b), NRF, replace=False)
                rows.append(sel)
        sel = np.concatenate(rows)[-NR:] if rows else np.zeros(0, int)
        sel = sel[~st[sel]] if len(sel) else sel
        out.append(rp[sel, :3])
    return out


def one(stamp):
    P = np.load(f"{PK}/{stamp}.npz", allow_pickle=False); m = json.loads(str(P["meta"])); T = int(P["labels"].shape[0])
    J = P["labels"].astype(float); lm = P["label_mask"]; org = P["t_d2r"].astype(float)
    ks = np.arange(0, T, STRIDE); pel = J[ks][:, [7, 8]].mean(1); okp = lm[ks][:, [7, 8]].all(1) & np.isfinite(pel).all(1)
    out = dict(stamp=stamp, env=int(m["env"]), q08=Q08.get(stamp))
    for ver, path in (("v08", f"{V08}/{stamp}.npz"), ("v3", f"{V3}/{stamp}.npz")):
        if not os.path.exists(path):
            continue
        R = np.load(path, allow_pickle=False); rp = R["radar_pts"]; ro = R["radar_off"]; st = R["radar_static"] if "radar_static" in R.files else np.zeros(len(rp), bool)
        rng = np.random.default_rng(zlib.crc32(stamp.encode()))
        tk = tokens(rp, ro, st, ks, rng); n = np.array([len(x) for x in tk]); c = np.array([x.mean(0) if len(x) else [np.nan] * 3 for x in tk])
        near = []
        for i in np.flatnonzero((n >= 3) & okp)[::3]:
            near.append(float((seg_dist(tk[i], J[ks[i]]) < 0.3).mean()))
        ok = okp & (n >= 3)
        o = c - pel; u = pel[:, :2] - org[:2]; u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
        los = (o[:, :2] * u).sum(1)
        t = ks / FPS; v = np.gradient(pel, t, axis=0); sp = np.linalg.norm(v[:, :2], axis=1)
        r = dict(n_frames=int(ok.sum()), frac_frames=float(ok.mean()), pts_med=float(np.median(n[n > 0])) if (n > 0).any() else 0.0,
                 near_body=float(np.mean(near)) if near else None)
        if ok.sum() >= 20:
            r.update(mean=o[ok].mean(0).tolist(), los_mean=float(los[ok].mean()), std=o[ok].std(0).tolist(), los_std=float(los[ok].std()),
                     still=np.median(o[ok & (sp < 0.1)][:, :2], 0).tolist() if (ok & (sp < 0.1)).sum() > 5 else None,
                     moving=np.median(o[ok & (sp > 0.4)][:, :2], 0).tolist() if (ok & (sp > 0.4)).sum() > 5 else None,
                     still_los=float(np.median(los[ok & (sp < 0.1)])) if (ok & (sp < 0.1)).sum() > 5 else None,
                     moving_los=float(np.median(los[ok & (sp > 0.4)])) if (ok & (sp > 0.4)).sum() > 5 else None)
            tau = []
            for ax in range(2):
                tau.append(np.linalg.lstsq(np.c_[np.ones(ok.sum()), v[ok, ax]], o[ok, ax], rcond=None)[0][1])
            r["tau"] = float(np.mean(tau))
        if ver == "v3" and "all_cls" in R.files:
            ac = R["all_cls"]; ap = R["all_pts"]; ao = R["all_off"]; cnt = {}; nearc = {}
            for i in np.flatnonzero(okp)[::5]:
                k = ks[i]; a, b = ao[k], ao[k + 1]
                if b > a:
                    dd = seg_dist(ap[a:b, :3].astype(float), J[k]) < 0.3
                    for cl in np.unique(ac[a:b]):
                        mm = ac[a:b] == cl; cnt[int(cl)] = cnt.get(int(cl), 0) + int(mm.sum()); nearc[int(cl)] = nearc.get(int(cl), 0) + int(dd[mm].sum())
            r["class_near_body"] = {str(k_): [cnt[k_], nearc[k_] / max(cnt[k_], 1)] for k_ in cnt}
        out[ver] = r
    return out


def summarize(rows, ver, sel):
    R = [r[ver] for r in rows if sel(r) and ver in r and "mean" in r[ver]]
    if not R:
        return None
    M = np.array([r["mean"] for r in R]); LO = np.array([r["los_mean"] for r in R]); S = np.array([r["std"] for r in R]); LS = np.array([r["los_std"] for r in R])
    g = lambda k: np.array([r[k] for r in R if r.get(k) is not None])
    st, mv, stl, mvl = g("still"), g("moving"), g("still_los"), g("moving_los")
    return dict(recs=len(R), mean_xyz_mm=(M.mean(0) * 1000).round(0).tolist(), los_mean_mm=round(float(LO.mean() * 1000), 0),
                between_sd_xyz_mm=(M.std(0) * 1000).round(0).tolist(), between_sd_los_mm=round(float(LO.std() * 1000), 0),
                within_sd_xyz_mm=(np.median(S, 0) * 1000).round(0).tolist(), within_sd_los_mm=round(float(np.median(LS) * 1000), 0),
                still_xy_m=np.median(st, 0).round(3).tolist() if len(st) else None, moving_xy_m=np.median(mv, 0).round(3).tolist() if len(mv) else None,
                still_los_m=round(float(np.median(stl)), 3) if len(stl) else None, moving_los_m=round(float(np.median(mvl)), 3) if len(mvl) else None,
                n_still=int(len(st)), n_moving=int(len(mv)), tau_med_s=round(float(np.median(g("tau"))), 3) if len(g("tau")) else None,
                near_body=round(float(np.mean(g("near_body"))), 3) if len(g("near_body")) else None,
                frac_frames=round(float(np.mean(g("frac_frames"))), 3), pts_med=float(np.median(g("pts_med"))))


if __name__ == "__main__":
    st = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{PK}/*.npz"))
    with Pool(24) as p:
        rows = p.map(one, st)
    sets = {"env1": lambda r: r["env"] == 1, "env2": lambda r: r["env"] == 2, "env3": lambda r: r["env"] == 3,
            "env12_v08ok (v2 캐시 집합)": lambda r: r["q08"] == "ok", "all": lambda r: True}
    S = {}
    for nm, f in sets.items():
        for ver in ("v08", "v3"):
            S[f"{nm}|{ver}"] = summarize(rows, ver, f)
            print(nm, ver, json.dumps(S[f"{nm}|{ver}"], ensure_ascii=False), flush=True)
    # v3 정리 분류별 몸 근처 비율
    agg = {}
    for r in rows:
        for k, (n, fr) in (r.get("v3", {}).get("class_near_body") or {}).items():
            a = agg.setdefault(k, [0, 0.0]); a[0] += n; a[1] += n * fr
    cls_near = {k: dict(n=v[0], near_body=round(v[1] / max(v[0], 1), 3)) for k, v in sorted(agg.items())}
    print("v3 분류별 몸 0.3 m 안 비율", json.dumps(cls_near, ensure_ascii=False), flush=True)
    json.dump(dict(summary=S, class_near_body=cls_near, rows=rows), open(f"{V3}/audit.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("AUDIT_DONE")
