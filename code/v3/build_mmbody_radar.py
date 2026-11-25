"""캐시 v3 — mmBody Arbe 레이더 팩 (정답 없는 사람 점 고르기). 노트: notes/v3_mmbody_radar.md

변환 (허용, 몸 무관): 레이더 → 카메라 X_c = Rmᵀ(X_r − tm) (데이터셋 calib kinect_master), 카메라 → 방 = 팩 R_c2r, t_c2r
(방 = OptiTrack 보정의 바닥, 데이터셋 calib). 사람 점 고르기는 radar_person_select.select_sequence (정답 없음).
읽는 키 (허용 목록, 그 밖의 키를 읽으면 즉시 오류):
  cache_v1/{split}/{name}.npz : frame, radar_data, radar_off, calib
  mmbody_v01/{split}/{name}.npz : frame, R_c2r, t_c2r
joints / pose / shape / labels / labels_world / dpcl_* / bbox_* 는 읽지 않는다.
출력: dev/packs/v3/mmbody_radar/{split}/{name}.npz (radar_pts (N,5) = 방 x y z, 도플러 m/s, 세기 dB; radar_off (T+1,); meta JSON;
       + 추가 키 trk_xy (T,2) float32 트랙 수평 위치, trk_status (T,) int8 (2 이동 군집, 1 정지 측정, 0 coast, −1 없음)),
      dev/packs/v3/mmbody_radar/manifest.json (파라미터, 녹화별 정답 없는 통계).
실행: CUDA_VISIBLE_DEVICES="" python -I build_mmbody_radar.py [--only split/name ...] [--workers 8] [--dry]
"""
import os, sys, json, glob, time, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from radar_person_select import PARAMS, select_sequence, background_mask
from scipy.spatial import cKDTree

U = "/home/interns/unisense"; SRC = f"{U}/dev/data/mmbody/cache_v1"; PK = f"{U}/dev/packs/mmbody_v01"; OUT = f"{U}/dev/packs/v3/mmbody_radar"
CARD = dict(device="Arbe Phoenix 4D (mmBody)", doppler_step_mps=0.071, frames_per_s=14.0,
            columns="x y z (방 m, 바닥 z=0), 도플러 m/s (원본 열 4), 세기 dB (원본 열 5); 원본 열 3 은 전부 0")
ALLOWED = {"cache": {"frame", "radar_data", "radar_off", "calib"}, "pack": {"frame", "R_c2r", "t_c2r"}}
SOURCE_TAG = "v3_gtfree_radar_bg_doppler_dbscan_kf"


class Guard:
    """npz 를 허용 키만 읽게 감싼다 (관문 a: 빌드가 라벨을 열지 않음)."""
    def __init__(self, path, kind):
        self.z = np.load(path, allow_pickle=False); self.kind = kind; self.read = set()

    def __getitem__(self, k):
        if k not in ALLOWED[self.kind]:
            raise PermissionError(f"build may not read '{k}' from {self.kind}")
        self.read.add(k); return self.z[k]


def load(split, name):
    Z = Guard(f"{SRC}/{split}/{name}.npz", "cache"); P = Guard(f"{PK}/{split}/{name}.npz", "pack")
    c = json.loads(str(Z["calib"])); Rm, tm = np.array(c["kinect_master"]["R"], float), np.array(c["kinect_master"]["t"], float)
    Rc, tc = P["R_c2r"].astype(float), P["t_c2r"].astype(float)
    to_room = lambda X: ((X - tm) @ Rm) @ Rc.T + tc
    zfr = Z["frame"]; pfr = P["frame"]; ro = Z["radar_off"]; RD = Z["radar_data"]
    idx = {int(fr): i for i, fr in enumerate(zfr)}
    # 팩 프레임 순서로 다시 묶음 (cache 와 팩 프레임이 같으면 그대로)
    seg = []; off = [0]
    for fr in pfr:
        i = idx.get(int(fr))
        if i is None:
            off.append(off[-1]); continue
        seg.append((ro[i], ro[i + 1])); off.append(off[-1] + ro[i + 1] - ro[i])
    rows = np.concatenate([RD[a:b] for a, b in seg]) if seg else np.zeros((0, 6), np.float32)
    q = to_room(rows[:, :3].astype(float)); s = to_room(np.zeros((1, 3)))[0]
    keys = dict(cache=sorted(Z.read), pack=sorted(P.read))
    return dict(q=q, v=rows[:, 4].astype(float), inten=rows[:, 5].astype(float), off=np.array(off, np.int64), s=s, T=len(pfr), keys=keys)


def gtfree_stats(r, d, fps):
    """정답 없는 검증 통계 (트랙 연속성, 사람 군집 프레임 비율, 정적 장면 대비 중심 안정성)."""
    st = r["status"]; xy = r["trk_xy"]; T = len(st); sel = r["sel"]; n = np.array([len(s) for s in sel])
    q = d["q"]; has = np.isfinite(xy[:, 0])
    # 트랙 연속성: 앞 필터 기준 시작 횟수, 끊김, coast 구간
    tid = r["trk_id"]; segs = int(((tid >= 0) & np.r_[True, tid[:-1] != tid[1:]]).sum())   # 고른 트랙의 연속 구간 수 (1 = 끊김 없음)
    runs0 = []; k = 0
    for s_ in st:
        if s_ == 0:
            k += 1
        elif k:
            runs0.append(k); k = 0
    if k:
        runs0.append(k)
    # 선택 점 중심 (프레임별) 과 그 흔들림
    C = np.full((T, 3), np.nan)
    for t in range(T):
        if n[t]:
            C[t] = q[sel[t]].mean(0)
    ok = np.isfinite(C[:, 0]); dC = np.linalg.norm(np.diff(C, axis=0), axis=1); dC = dC[np.isfinite(dC)]
    dT = np.linalg.norm(np.diff(xy, axis=0), axis=1); dT = dT[np.isfinite(dT)]
    # 고역 성분: 1 s 이동 중앙값 대비 잔차 (사람 이동을 뺀 프레임 흔들림)
    w = int(round(fps)); hp = []
    for t in np.where(ok)[0]:
        lo, hi = max(0, t - w // 2), min(T, t + w // 2 + 1); m = np.nanmedian(C[lo:hi], 0); hp.append(C[t] - m)
    hp = np.array(hp) if hp else np.zeros((0, 3))
    # 정적 장면 대비: 중심에서 가장 가까운 배경 복셀 중심까지 수평 거리, 선택 점 중 배경 인접(0.2 m 안) 비율
    bc = r["diag"]["bg"]["bg_centres"]
    tr = cKDTree(bc) if len(bc) else None                                                      # 3D
    dist_bg = tr.query(C[ok])[0] if tr is not None and ok.any() else np.array([np.nan])
    # 정적 장면 안정성: 녹화 앞·뒤 절반의 배경 복셀 일치 (레이더가 움직이지 않았는지)
    half = d["off"][T // 2]
    def half_bg(sl, f0, Th):
        _, b = background_mask(q[sl], None, r["fi"][sl] - f0, Th, r["roi"][sl], r["mov"][sl], PARAMS); return b["bg_centres"]
    A, B = half_bg(slice(0, half), 0, T // 2), half_bg(slice(half, len(q)), T // 2, T - T // 2)
    ka = set(map(tuple, np.round(A / PARAMS["VOX"]).astype(int))); kb = set(map(tuple, np.round(B / PARAMS["VOX"]).astype(int)))
    jac = len(ka & kb) / max(1, len(ka | kb))
    nn_ab = float(np.median(cKDTree(B).query(A)[0])) if len(A) and len(B) else float("nan")
    # 정적 지도 이동: 앞 절반 → 뒤 절반 최근접 차 벡터의 중앙값 (0.2 m 안 쌍만)
    if len(A) and len(B):
        dd, ii = cKDTree(B).query(A); m_ = dd < PARAMS["VOX"] * 1.01; shift = np.median(B[ii[m_]] - A[m_], 0) if m_.any() else np.full(3, np.nan)
    else:
        shift = np.full(3, np.nan)
    sel_all = np.concatenate(sel) if len(sel) else np.zeros(0, int)
    q_sel = q[sel_all] if len(sel_all) else np.zeros((0, 3))
    nbg_sel = float((tr.query(q_sel)[0] < PARAMS["VOX"]).mean()) if tr is not None and len(q_sel) else float("nan")
    rnd = lambda x, k=3: None if x is None or not np.isfinite(x) else round(float(x), k)
    return dict(
        T=T, frames_with_track=rnd(has.mean()), frames_with_person_cluster=rnd((st == 2).mean()),
        frames_with_person_meas=rnd((st >= 1).mean()), frames_stationary_meas=rnd((st == 1).mean()), frames_coast=rnd((st == 0).mean()),
        frames_no_track=rnd((st < 0).mean()), chosen_track_segments=segs, track_switches=r["diag"]["n_switch"], n_tracks=r["diag"]["n_tracks"], coast_runs=len(runs0), coast_run_max_s=rnd(max(runs0) / fps if runs0 else 0.0, 2),
        kept_pts_per_frame_median=float(np.median(n)), kept_pts_per_frame_p10=float(np.percentile(n, 10)), kept_pts_per_frame_p90=float(np.percentile(n, 90)),
        frames_capped=rnd((n >= PARAMS["CAP"]).mean()), frames_empty=rnd((n == 0).mean()),
        track_step_m_median=rnd(np.median(dT) if len(dT) else np.nan), track_step_m_p99=rnd(np.percentile(dT, 99) if len(dT) else np.nan),
        centroid_step_m_median=rnd(np.median(dC) if len(dC) else np.nan), centroid_step_m_p95=rnd(np.percentile(dC, 95) if len(dC) else np.nan),
        centroid_jump_gt_0p3m_frac=rnd((dC > 0.3).mean() if len(dC) else np.nan),
        centroid_hp_std_m=[rnd(x) for x in (hp.std(0) if len(hp) else [np.nan] * 3)],
        centroid_seq_std_m=[rnd(x) for x in np.nanstd(C, 0)], centroid_seq_median_m=[rnd(x) for x in np.nanmedian(C, 0)],
        centroid_z_p05_p95=[rnd(np.nanpercentile(C[:, 2], 5)), rnd(np.nanpercentile(C[:, 2], 95))],
        centroid_to_static_m_median=rnd(np.median(dist_bg)), centroid_to_static_m_p05=rnd(np.percentile(dist_bg, 5)),
        sel_pts_near_static_frac=rnd(nbg_sel), static_map_half_jaccard=rnd(jac), static_map_half_nn_m=rnd(nn_ab), static_map_half_shift_m=[rnd(x) for x in shift],
        roi_pts_per_frame=r["diag"]["roi_pts_per_frame"], cand_pts_per_frame=r["diag"]["cand_pts_per_frame"], seed_pts_per_frame=r["diag"]["seed_pts_per_frame"],
        n_clusters_median=r["diag"]["n_clusters_median"], tracks_top=r["diag"]["tracks"], bg_vox=r["diag"]["bg"]["n_bg_vox"], roi_pts_bg_frac=rnd(r["diag"]["bg"]["roi_pts_bg_frac"]),
        raw_pts_per_frame_median=float(np.median(np.diff(d["off"]))))


def process(job):
    split, name, dry = job; t0 = time.time()
    d = load(split, name)
    r = select_sequence(d["q"], d["v"], d["inten"], d["off"], d["s"][:2], CARD["doppler_step_mps"], CARD["frames_per_s"], PARAMS)
    stats = gtfree_stats(r, d, CARD["frames_per_s"])
    sel = r["sel"]; allidx = np.concatenate(sel) if sel else np.zeros(0, np.int64)
    pts = np.c_[d["q"][allidx], d["v"][allidx], d["inten"][allidx]].astype(np.float32)
    off = np.r_[0, np.cumsum([len(s) for s in sel])].astype(np.int64)
    meta = dict(dataset="mmbody", part=split, seq=name, T=d["T"], card=CARD,
                extrinsic=dict(source="dataset calib (kinect_master) + pack R_c2r", tag="dataset_calib_nonbody"),
                person_select=dict(method="GT-free: per-sequence static voxel background (occupancy + dynamic ratio), Doppler seeds, BEV DBSCAN, "
                                          "damped-CV Kalman fwd+bwd fused, all non-background points (incl. zero Doppler) within R_SEL of track, top-CAP by power",
                                   params=PARAMS, oracle=False, gt_free=True, source=SOURCE_TAG, keys_read=d["keys"]),
                version="v3-radar")
    if not dry:
        os.makedirs(f"{OUT}/{split}", exist_ok=True)
        np.savez_compressed(f"{OUT}/{split}/{name}.npz", radar_pts=pts, radar_off=off, meta=json.dumps(meta, ensure_ascii=False),
                            trk_xy=r["trk_xy"].astype(np.float32), trk_status=r["status"].astype(np.int8))
    stats["keys_read"] = d["keys"]; stats["sec"] = round(time.time() - t0, 1)
    return f"{split}/{name}", stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*"); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    jobs = []
    for split in ("train", "test"):
        for f in sorted(glob.glob(f"{PK}/{split}/*.npz")):
            nm = os.path.basename(f)[:-4]
            if a.only and f"{split}/{nm}" not in a.only:
                continue
            jobs.append((split, nm, a.dry))
    res = {}
    if a.workers > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(min(a.workers, len(jobs))) as pool:
            for k, s in pool.imap_unordered(process, jobs):
                res[k] = s; print(k, json.dumps(s, ensure_ascii=False), flush=True)
    else:
        for j in jobs:
            k, s = process(j); res[k] = s; print(k, json.dumps(s, ensure_ascii=False), flush=True)
    if a.dry or a.only:
        print("DRY/partial run: manifest not written"); return
    code_hash = {os.path.basename(p): hashlib.sha1(open(p, "rb").read()).hexdigest()[:12] for p in (__file__, os.path.join(os.path.dirname(os.path.abspath(__file__)), "radar_person_select.py"))}
    keys = ["frames_with_track", "frames_with_person_cluster", "frames_with_person_meas", "chosen_track_segments", "track_switches", "kept_pts_per_frame_median",
            "centroid_jump_gt_0p3m_frac", "centroid_to_static_m_median", "sel_pts_near_static_frac", "static_map_half_jaccard"]
    summ = {k: dict(median=float(np.median([res[s][k] for s in res if res[s][k] is not None])), min=float(np.min([res[s][k] for s in res if res[s][k] is not None])),
                    max=float(np.max([res[s][k] for s in res if res[s][k] is not None]))) for k in keys}
    man = dict(dataset="mmbody", sensor="Arbe Phoenix", version="v3-radar", created=time.strftime("%Y-%m-%d %H:%M"), source_tag=SOURCE_TAG, oracle=False, gt_free=True,
               extrinsic_tag="dataset_calib_nonbody", params=PARAMS, card=CARD, allowed_keys={k: sorted(v) for k, v in ALLOWED.items()}, code_sha1=code_hash,
               format="radar_pts (N,5) room x y z, Doppler m/s, power dB; radar_off (T+1,); meta; extra trk_xy (T,2), trk_status (T,) [2 moving cluster, 1 stationary meas, 0 coast, -1 none]",
               summary=summ, sequences=dict(sorted(res.items())), license="mmBody: non-commercial academic research only; no redistribution")
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1, ensure_ascii=False); print("BUILD_DONE")


if __name__ == "__main__":
    main()
