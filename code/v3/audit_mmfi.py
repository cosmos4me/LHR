"""v3 MM-Fi 정합 감사 (2026-11-24) — 라벨 (정답 골반) 을 읽는 유일한 스크립트. 출력은 로그·노트로만 가고 변환에 되먹이지 않는다.
비교: 이전 (LiDAR mmfi_lidar_v02, 레이더 mmfi_v03_radar = 정답 Kabsch) vs 이후 (v3 정답 없는 정합).
 - 방마다: 깊이 / LiDAR / 레이더 중심 − 정답 골반 (시선 LOS, 옆 lat, 높이 z) 의 평균, 녹화 사이 std, 녹화 안 std.
   중심 = 점 중앙값. 레이더 점은 깊이 사람 중심 (정답 없음) 수평 1.0 m 안, 높이 −0.3–2.5 m 만 (이전·이후 같은 게이트).
   LOS = 방 원점 (카메라 바닥 투영) → 골반 수평 단위 벡터.
 - 피험자마다: LiDAR − 깊이 중심 수평 거리, 녹화 안 옆 상관.
 - 세션마다: 깊이 − 골반 LOS 와 LiDAR − 골반 LOS (깊이 시차 오프셋 가설의 정답 감사).
사용: python -I audit_mmfi.py  → dev/logs/v3_mmfi_audit.log (마지막 줄 JSON)"""
import sys, os, json
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
import mmfi_common as mc

U = mc.U; PK = mc.DPK
SRC = dict(lidar_v02=lambda e, nm: U / f"dev/packs/mmfi_lidar_v02/{e}_{nm}.npz", lidar_v3=lambda e, nm: mc.OUT_L / f"{e}_{nm}.npz",
           radar_v03=lambda e, nm: U / f"dev/packs/mmfi_v03_radar/{e}/{nm}.npz", radar_v3=lambda e, nm: mc.OUT_R / e / f"{nm}.npz")
STEP = 3


def uvz_room(P, K, k):
    m = P["pts_valid"][k]; u, w, z = P["pts_uvz"][k, m].astype(float).T; fx, fy, cx, cy = K
    return np.c_[(u - cx) / fx * z, (w - cy) / fy * z, z] @ np.asarray(P["R_d2r"], float).T + np.asarray(P["t_d2r"], float)


def one(job):
    e, nm = job
    D = np.load(PK / e / f"{nm}.npz", allow_pickle=False); y = D["labels"].astype(float); ym = D["label_mask"]; T = len(y)
    Kd = mc.spec_depth_K(e); out = {}
    L = {k: np.load(f(e, nm), allow_pickle=False) for k, f in SRC.items() if f(e, nm).exists()}
    KL = np.array([400.0, 400.0, 320.0, 240.0])
    rows = {k: [] for k in ("depth", *L.keys())}
    for k in range(0, T, STEP):
        if not (ym[k, 7] and ym[k, 8]) or not np.isfinite(y[k, 7:9]).all():
            continue
        pel = 0.5 * (y[k, 7] + y[k, 8]); u = pel[:2] / np.linalg.norm(pel[:2]); lat = np.array([-u[1], u[0]])
        Dp = uvz_room(D, Kd, k)
        if len(Dp) < 30:
            continue
        cd = np.median(Dp, 0)
        cents = {"depth": cd}
        for s, Z in L.items():
            if s.startswith("lidar"):
                if k < len(Z["pts_valid"]) and Z["pts_valid"][k].sum() >= 10:
                    cents[s] = np.median(uvz_room(Z, KL, k), 0)
            else:
                off = Z["radar_off"]
                if k + 1 < len(off) and off[k + 1] > off[k]:
                    p = Z["radar_pts"][off[k]:off[k + 1], :3].astype(float)
                    g = (np.linalg.norm(p[:, :2] - cd[:2], axis=1) < 1.0) & (p[:, 2] > -0.3) & (p[:, 2] < 2.5)
                    if g.sum() >= 3:
                        cents[s] = np.median(p[g], 0)
        for s, c in cents.items():
            d = c - pel; rows[s].append((k, d @ np.r_[u, 0], d @ np.r_[lat, 0], d[2], c[0], c[1], c[2], cd[0], cd[1]))
    for s, r in rows.items():
        r = np.array(r)
        if len(r) < 10:
            continue
        o = dict(n=len(r), los=float(np.median(r[:, 1])), lat=float(np.median(r[:, 2])), z=float(np.median(r[:, 3])),
                 los_sd=float(np.std(r[:, 1])), lat_sd=float(np.std(r[:, 2])), z_sd=float(np.std(r[:, 3])))
        if s.startswith("lidar"):
            dd = r[:, 4:6] - r[:, 7:9]; o["dist_to_depth"] = float(np.linalg.norm(np.median(dd, 0)))
            # 녹화 안 옆 상관 (LiDAR y vs 깊이 y)
            o["corr_lat_depth"] = float(np.corrcoef(r[:, 5], r[:, 8])[0, 1]) if np.std(r[:, 8]) > 0.02 and np.std(r[:, 5]) > 1e-6 else np.nan
            o["within_sd_lidar_xy"] = float(np.linalg.norm(np.std(r[:, 4:6], 0))); o["within_sd_depth_xy"] = float(np.linalg.norm(np.std(r[:, 7:9], 0)))
        out[s] = o
    return e, nm, out


def summarize(R):
    S = {}
    for e in mc.ENVS:
        S[e] = {}
        for s in ("depth", *SRC.keys()):
            v = [R[(e, nm)][s] for (ee, nm) in R if ee == e and s in R[(e, nm)]]
            if not v:
                continue
            a = {k: np.array([x[k] for x in v]) for k in ("los", "lat", "z", "los_sd", "lat_sd", "z_sd")}
            S[e][s] = dict(n_rec=len(v), **{f"{k}_mean": float(np.mean(a[k])) for k in ("los", "lat", "z")}, **{f"{k}_between_sd": float(np.std(a[k])) for k in ("los", "lat", "z")},
                           **{f"{k}_within_sd": float(np.median(a[k + "_sd"])) for k in ("los", "lat", "z")})
            if s.startswith("lidar"):
                subs = sorted({nm[:3] for (ee, nm) in R if ee == e})
                S[e][s]["per_subject_dist_to_depth"] = {sj: float(np.median([R[(e, nm)][s]["dist_to_depth"] for (ee, nm) in R if ee == e and nm[:3] == sj and s in R[(e, nm)]])) for sj in subs}
                S[e][s]["corr_lat_depth_med"] = float(np.nanmedian([x["corr_lat_depth"] for x in v]))
                S[e][s]["within_sd_lidar_xy_med"] = float(np.median([x["within_sd_lidar_xy"] for x in v])); S[e][s]["within_sd_depth_xy_med"] = float(np.median([x["within_sd_depth_xy"] for x in v]))
        # 세션별 LOS (깊이 시차 오프셋 감사)
        subs = sorted({nm[:3] for (ee, nm) in R if ee == e})
        S[e]["per_subject_los"] = {sj: {s: float(np.median([R[(e, nm)][s]["los"] for (ee, nm) in R if ee == e and nm[:3] == sj and s in R[(e, nm)]]))
                                        for s in ("depth", "lidar_v3", "radar_v3", "lidar_v02", "radar_v03")} for sj in subs}
    return S


def table(S):
    lines = []
    for e in mc.ENVS:
        for s in ("depth", "lidar_v02", "lidar_v3", "radar_v03", "radar_v3"):
            if s not in S[e]:
                continue
            x = S[e][s]
            lines.append(f"{e} {s:9s} n{x['n_rec']:4d} | 평균 LOS {x['los_mean']*1000:+6.0f} lat {x['lat_mean']*1000:+6.0f} z {x['z_mean']*1000:+6.0f} mm | 녹화 사이 sd {x['los_between_sd']*1000:4.0f} {x['lat_between_sd']*1000:4.0f} {x['z_between_sd']*1000:4.0f} | 녹화 안 sd {x['los_within_sd']*1000:4.0f} {x['lat_within_sd']*1000:4.0f} {x['z_within_sd']*1000:4.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    jobs = mc.recordings()
    with Pool(int(os.environ.get("V3_NP", "24"))) as p:
        R = {(e, nm): o for e, nm, o in p.imap_unordered(one, jobs)}
    S = summarize(R)
    print(table(S), flush=True)
    for e in mc.ENVS:
        for s in ("lidar_v02", "lidar_v3"):
            if s in S[e]:
                print(e, s, "LiDAR−깊이 수평 거리 (피험자)", {k: round(v, 3) for k, v in S[e][s]["per_subject_dist_to_depth"].items()}, "옆 상관 중앙값", round(S[e][s]["corr_lat_depth_med"], 2),
                      "녹화 안 xy sd LiDAR/깊이", round(S[e][s]["within_sd_lidar_xy_med"], 3), round(S[e][s]["within_sd_depth_xy_med"], 3), flush=True)
        print(e, "세션별 LOS − 골반 (깊이, LiDAR v3, 레이더 v3)", {k: [round(v[s] * 1000) if np.isfinite(v[s]) else None for s in ("depth", "lidar_v3", "radar_v3")] for k, v in S[e]["per_subject_los"].items()}, flush=True)
    print("AUDIT_JSON " + json.dumps(S, ensure_ascii=False, default=lambda o: None), flush=True)
