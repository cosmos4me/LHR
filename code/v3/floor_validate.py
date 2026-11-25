"""v3 바닥 정답 없는 검증 (floors.json 이후). 정답 안 읽음.
1. 반복성: 장면(같은 rig) 안 녹화별 평면의 흩어짐 (깊이 카메라 좌표), 같은 환경의 장면(피험자·날짜) 사이 차.
2. MM-Fi 깊이 vs LiDAR: (a) rig 높이 차 Δ = h_LiDAR(원시 LiDAR RANSAC) − h_깊이카메라(v3 바닥) 가 강체 플랫폼이면 일정해야 함 →
   환경 안·피험자 안 표준편차, 환경 평균 차. (b) LiDAR 자체 바닥 법선의 녹화 사이 흩어짐. (c) 사람 높이 분포 차 (LiDAR 시야·빔 밀도 때문에 오프셋 검사로는 부적합 — 참고).
3. mmBody 문서값 vs 발바닥 (floor_mmbody.py 결과 요약).
4. 서 있는 축: 깊이 사람 점 (MM-Fi 팩 점, OctoNet, ITOP, mmBody 점군) 의 서 있는 프레임에서 아래 층 [0.25, 0.55] m 와 위 층 [1.05, 1.35] m 수평 중심을 잇는 축 →
   바닥 법선과 각 (옆 = 센서 시선에 수직 성분, 시선 성분), 옛 바닥·새 바닥 각각. 서 있음 = 높이 범위(p98−p2) ≥ 1.3 m, 수평 반경 p90 < 0.35 m, 점 ≥ 120.
   시선 성분은 앞면만 보이는 깊이 표면 때문에 치우침 (다리 앞면과 몸통 앞면 깊이 차) → 옆 성분이 바닥 기울기 검사의 주 지표.
   mRI 는 floor_mri.py 의 키포인트 축 (발목 중점 → 어깨 중점).
출력: dev/packs/v3/floor/validation.json. 실행: python -I floor_validate.py"""
import sys, glob, json, collections, os, zlib, ast
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
import floor_common as fc

U = "/home/interns/unisense"; F = f"{U}/dev/packs/v3/floor"
FL = json.load(open(f"{U}/dev/packs/v3/floors.json"))
OSPEC = json.load(open(f"{U}/code/specs/octonet_v07.json"))
ITOP_K = [1 / 0.0035, 1 / 0.0035, 160.0, 120.0]


def axis_frame(Q, cam):
    """Q (N,3) 방 좌표 (z = 바닥 위 높이). 반환 (옆 각, 시선 각, 전체 각) [deg] 또는 None."""
    if len(Q) < 120:
        return None
    z = Q[:, 2]; lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    if hi - lo < 1.3:
        return None
    c = np.median(Q[:, :2], 0)
    if np.percentile(np.linalg.norm(Q[:, :2] - c, axis=1), 90) > 0.35:
        return None
    a = Q[(z > 0.25) & (z < 0.55)]; b = Q[(z > 1.05) & (z < 1.35)]
    if len(a) < 15 or len(b) < 15:
        return None
    ca, cb = np.median(a[:, :2], 0), np.median(b[:, :2], 0); dz = b[:, 2].mean() - a[:, 2].mean()
    v = c - cam[:2]; v /= np.linalg.norm(v); l = np.array([-v[1], v[0]])
    dh = cb - ca
    return np.degrees(np.arctan2(dh @ l, dz)), np.degrees(np.arctan2(dh @ v, dz)), np.degrees(np.arctan2(np.linalg.norm(dh), dz))


def stand_pack(job):
    pk, ds = job; e = FL[pk]; Rn, tn = np.array(e["R"]), np.array(e["t"])
    P = fc.safe_npz(pk); m = P.meta(); R, t = P["R_d2r"].astype(float), P["t_d2r"].astype(float)
    frames = []
    if ds == "mmbody":
        split, nm = pk.split("/")[-2], os.path.basename(pk)[:-4]
        Z = fc.safe_npz(f"{U}/dev/data/mmbody/cache_v1/{split}/{nm}.npz"); c = json.loads(str(Z["calib"]))
        Rm, tm = np.array(c["kinect_master"]["R"], float), np.array(c["kinect_master"]["t"], float); o, X = Z["dpcl_off"], Z["dpcl_xyz"]
        rng = np.random.default_rng(0)
        for k in range(0, len(o) - 1, 5):
            a_, b_ = o[k], o[k + 1]
            if b_ - a_ < 500:
                continue
            ii = rng.choice(np.arange(a_, b_), 3000, replace=False) if b_ - a_ > 3000 else np.arange(a_, b_)
            frames.append(((X[ii].astype(float) - tm) @ Rm) @ R.T + t)
    else:
        if ds == "mmfi":
            K = json.load(open(m["spec"] if "E04" not in pk else f"{U}/code/specs/mmfi_v02_specA_e04.json"))["depth"]["K"]
        elif ds == "octonet":
            K = OSPEC["depth"][f"{m['env']}_{m['day']}"]["K"]
        else:
            K = ITOP_K
        fx, fy, cx, cy = K; uvz, val = P["pts_uvz"], P["pts_valid"]
        for k in range(0, len(uvz), 3):
            v = val[k]
            if v.sum() < 120:
                continue
            u_, w_, z = uvz[k, v].astype(float).T
            frames.append(np.c_[(u_ - cx) / fx * z, (w_ - cy) / fy * z, z] @ R.T + t)
    old, new = [], []
    cam_new = Rn @ t + tn
    for Q in frames:
        r = axis_frame(Q, t)
        if r is not None:
            old.append(r)
        r = axis_frame(Q @ Rn.T + tn, cam_new)
        if r is not None:
            new.append(r)
    med = lambda L: np.median(np.array(L), 0).tolist() if len(L) >= 10 else None
    return pk, dict(ds=ds, n_old=len(old), n_new=len(new), old=med(old), new=med(new))


def rig_lidar():
    L = json.load(open(f"{F}/mmfi_lidar.json")); rows = []
    for pk, v in L.items():
        if not v.get("ok"):
            continue
        P = fc.safe_npz(pk); t = P["t_d2r"].astype(float); e = FL[pk]
        hc = float(np.array(e["plane_old_room"]["n"]) @ t + e["plane_old_room"]["d"])                 # 깊이 카메라 높이 (v3 바닥)
        hc_old = float(t[2])
        rows.append((v["env"], v["subject"], v["lidar_height_m"], hc, hc_old, v["lidar_tilt_deg"], v["clip_frac"], v["lidar_top"], v["depth_top_new"], v["lidar_mid"], v["depth_mid_new"]))
    out = {}
    for env in ("E01", "E02", "E03", "E04"):
        R_ = [r for r in rows if r[0] == env]; d = np.array([r[2] - r[3] for r in R_]); d_old = np.array([r[2] - r[4] for r in R_])
        subj = collections.defaultdict(list)
        for r in R_:
            subj[r[1]].append(r[2] - r[3])
        out[env] = dict(n=len(R_), rig_dz_mean_m=float(d.mean()), rig_dz_sd_m=float(d.std()), rig_dz_sd_old_m=float(d_old.std()),
                        rig_dz_within_subject_sd_m=float(np.median([np.std(x) for x in subj.values()])),
                        rig_dz_between_subject_sd_m=float(np.std([np.mean(x) for x in subj.values()])),
                        lidar_tilt_med_deg=float(np.median([r[5] for r in R_])), lidar_tilt_sd_deg=float(np.std([r[5] for r in R_])),
                        lidar_clip_frac_med=float(np.median([r[6] for r in R_])),
                        person_top_depth_minus_lidar_m=float(np.nanmedian([r[8] - r[7] for r in R_])), person_mid_depth_minus_lidar_m=float(np.nanmedian([r[10] - r[9] for r in R_])))
    allm = [out[e]["rig_dz_mean_m"] for e in out]
    out["between_env_rig_dz_range_m"] = float(max(allm) - min(allm))
    return out


def repeatability():
    S = collections.defaultdict(list)
    for pk, e in FL.items():
        if pk.startswith("_") or "scene_spread" not in e:
            continue
        S[(pk.split("/dev/packs/")[1].split("/")[0], e["scene"])].append(e)
    out = collections.defaultdict(dict)
    for (ds, sc), L in S.items():
        out[ds][sc] = dict(n=len(L), spread_angle_deg=L[0]["scene_spread"]["angle_deg"], spread_height_m=L[0]["scene_spread"]["height_m"],
                           own_vs_scene_angle_med_deg=float(np.median([x["own_vs_scene"]["angle_deg"] for x in L])),
                           own_vs_scene_abs_height_med_m=float(np.median([abs(x["own_vs_scene"]["height_m"]) for x in L])))
    summ = {}
    for ds, D in out.items():
        a = np.array([v["spread_angle_deg"] for v in D.values()]); h = np.array([v["spread_height_m"] for v in D.values()])
        summ[ds] = dict(n_scenes=len(D), spread_angle_med_deg=float(np.median(a)), spread_angle_max_deg=float(a.max()), spread_height_med_m=float(np.median(h)), spread_height_max_m=float(h.max()))
    # 같은 환경 안 장면 사이 (깊이 카메라 좌표 합의 평면)
    between = {}
    for ds, key in (("mmfi_v02_specA", 1), ("octonet_v07", 1)):
        G = collections.defaultdict(list)
        for pk, e in FL.items():
            if pk.startswith("_") or not pk.split("/dev/packs/")[1].startswith(ds) or "scene_spread" not in e:
                continue
            P = fc.safe_npz(pk); R, t = P["R_d2r"].astype(float), P["t_d2r"].astype(float)
            n = R.T @ np.array(e["plane_old_room"]["n"]); h = float(np.array(e["plane_old_room"]["n"]) @ t + e["plane_old_room"]["d"])
            sc = ast.literal_eval(e["scene"]); G[sc[key]].append((sc, n, h))
        for env, L in G.items():
            byscene = collections.defaultdict(list)
            for sc, n, h in L:
                byscene[sc].append((n, h))
            ns = np.array([np.mean([x[0] for x in v], 0) for v in byscene.values()]); hs = np.array([np.mean([x[1] for x in v]) for v in byscene.values()])
            nm = ns.mean(0); nm /= np.linalg.norm(nm)
            between[f"{ds}:{env}"] = dict(n_scenes=len(byscene), angle_to_env_mean_sd_deg=float(np.sqrt(np.mean(np.degrees(np.arccos(np.clip(ns @ nm / np.linalg.norm(ns, axis=1), -1, 1))) ** 2))),
                                          cam_height_sd_m=float(hs.std()), cam_height_range_m=float(hs.max() - hs.min()))
    return dict(summary=summ, between_scenes_same_env=between, per_scene=out)


def main():
    out = dict(repeatability=repeatability(), mmfi_depth_vs_lidar=rig_lidar())
    mb = json.load(open(f"{F}/mmbody.json"))
    x = [(v["doc_minus_feet_at_person_m"], v["doc_vs_feet_angle_deg"], v["feet_minus_old_at_person_m"], v["feet"]["angle_change_deg"], v["radar_height_old_m"]) for v in mb.values() if "doc_minus_feet_at_person_m" in v]
    x = np.array(x)
    out["mmbody_doc_vs_feet"] = dict(n=len(x), doc_minus_feet_at_person_med_m=float(np.median(x[:, 0])), doc_minus_feet_range_m=[float(x[:, 0].min()), float(x[:, 0].max())],
                                     doc_vs_feet_angle_med_deg=float(np.median(x[:, 1])), doc_vs_feet_angle_range_deg=[float(x[:, 1].min()), float(x[:, 1].max())],
                                     feet_minus_current_at_person_med_m=float(np.median(x[:, 2])), feet_minus_current_range_m=[float(x[:, 2].min()), float(x[:, 2].max())],
                                     feet_vs_current_angle_med_deg=float(np.median(x[:, 3])), feet_vs_current_angle_p90_deg=float(np.percentile(x[:, 3], 90)),
                                     radar_height_current_frame_range_m=[float(x[:, 4].min()), float(x[:, 4].max())], doc_rejected_all=bool(all(v["doc_rejected"] for v in mb.values() if "doc_minus_feet_at_person_m" in v)))
    jobs = []
    for pk in FL:
        if pk.startswith("_"):
            continue
        ds = pk.split("/dev/packs/")[1].split("_")[0]
        if ds == "mmfi" and zlib.crc32(pk.encode()) % 4:
            continue
        if ds in ("mmfi", "octonet", "itop", "mmbody"):
            jobs.append((pk, ds))
    jobs = sorted(jobs)
    S = {}
    with Pool(24) as pool:
        for pk, r in pool.imap_unordered(stand_pack, jobs, chunksize=2):
            S[pk] = r
    st = {}
    for ds in ("mmfi", "octonet", "itop", "mmbody"):
        L = [r for r in S.values() if r["ds"] == ds and r["old"] is not None and r["new"] is not None]
        if not L:
            continue
        o = np.array([r["old"] for r in L]); n = np.array([r["new"] for r in L])
        st[ds] = dict(n_rec=len(L), n_frames=int(sum(r["n_new"] for r in L)),
                      old=dict(lateral_med=float(np.median(o[:, 0])), lateral_abs_p90=float(np.percentile(np.abs(o[:, 0]), 90)), along_med=float(np.median(o[:, 1])), total_med=float(np.median(o[:, 2]))),
                      new=dict(lateral_med=float(np.median(n[:, 0])), lateral_abs_p90=float(np.percentile(np.abs(n[:, 0]), 90)), along_med=float(np.median(n[:, 1])), total_med=float(np.median(n[:, 2]))),
                      frac_lateral_within_2deg_new=float(np.mean(np.abs(n[:, 0]) <= 2)))
        if ds == "octonet":
            for env in (1, 2, 3):
                Le = [r for pk, r in S.items() if r["ds"] == "octonet" and r["new"] is not None and r["old"] is not None and fc.safe_npz(pk).meta()["env"] == env]
                if Le:
                    o = np.array([r["old"] for r in Le]); n = np.array([r["new"] for r in Le])
                    st[f"octonet_env{env}"] = dict(n_rec=len(Le), old_lateral_med=float(np.median(o[:, 0])), new_lateral_med=float(np.median(n[:, 0])),
                                                   old_along_med=float(np.median(o[:, 1])), new_along_med=float(np.median(n[:, 1])))
        if ds == "mmfi":
            for env in ("E01", "E02", "E03", "E04"):
                Le = [r for pk, r in S.items() if r["ds"] == "mmfi" and f"/{env}/" in pk and r["new"] is not None and r["old"] is not None]
                if Le:
                    o = np.array([r["old"] for r in Le]); n = np.array([r["new"] for r in Le])
                    st[f"mmfi_{env}"] = dict(n_rec=len(Le), old_lateral_med=float(np.median(o[:, 0])), new_lateral_med=float(np.median(n[:, 0])),
                                             old_along_med=float(np.median(o[:, 1])), new_along_med=float(np.median(n[:, 1])))
    mri = json.load(open(f"{F}/mri.json"))["final"]
    a = np.array([[v["standing_new"]["lateral_med_deg"], v["standing_new"]["along_view_med_deg"], v["standing_new"]["angle_med_deg"]] for v in mri.values() if "lateral_med_deg" in v["standing_new"]])
    st["mri_keypoints"] = dict(n_rec=len(a), new=dict(lateral_med=float(np.median(a[:, 0])), lateral_abs_max=float(np.abs(a[:, 0]).max()), along_med=float(np.median(a[:, 1])),
                                                      along_range=[float(a[:, 1].min()), float(a[:, 1].max())], total_med=float(np.median(a[:, 2]))))
    out["standing_axis"] = st; out["standing_axis_per_pack"] = S
    open(f"{F}/validation.json", "w").write(fc.jdump(out))
    print(fc.jdump({k: v for k, v in out.items() if k not in ("standing_axis_per_pack",)} | {"repeatability": {k: v for k, v in out["repeatability"].items() if k != "per_scene"}}), flush=True)
    print("VALIDATE_DONE", flush=True)


if __name__ == "__main__":
    main()
