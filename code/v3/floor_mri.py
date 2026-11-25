"""v3 바닥 (mRI, 스테레오 발 접촉). 정답(라벨) 안 읽음 — 팩에서 kp_px, kp_score, kp_px_cam2, kp_score_cam2, R_c2r, t_c2r, meta 만 연다
(labels, labels_cam, label_mask, label_quality, action(라벨 파일에서 온 동작 구간) 은 열지 않는다).
절차·상수는 floor_feet.py 에 MM-Fi 검증과 함께 mRI 실행 전에 고정했다. 이 스크립트는 한 번만 돌린다.
- 카메라: 사양 mri_v03.json 의 공개 K, K_cam2, R_cam1_to_cam2, t_cam1_to_cam2 (기선 0.250 m, 공개 camera_matrix).
- 발목 삼각측량 (두 시점 점수 ≥ 0.5) → 팩 방 좌표 → 접촉 구간 (2D 속도, 큰 쪽 시점) → 피험자별 평면 (진단) + 전체 묶음 평면 (최종).
- 최종: 법선 = 묶음, 높이 = 묶음 (피험자 자기 평면이 사람 위치에서 5 cm 넘게 다르면 그 피험자만 자기 높이).
- 불확도: 피험자 블록 부트스트랩 + 방법 오차 (MM-Fi 검증: 기울기 1.9°, 높이 2 cm) + 발목 사전 2 cm.
- 서 있는 축 검사 (정답 없음): 두 발목 모두 접촉, 어깨 중점 − 발목 중점 ≥ 1.0 m, 엉덩이 중점이 발목 중점에서 수평 0.15 m 안 →
  축 = 어깨 중점 − 발목 중점, 새 바닥 법선과 각도 (옆 성분 = cam1 시선에 수직, 시선 성분), 프레임 중앙값.
출력: dev/packs/v3/floor/mri.json. 실행: python -I floor_mri.py"""
import sys, os, glob, json
sys.path.insert(0, "/home/interns/unisense/code/v3")
import numpy as np
import floor_common as fc
import floor_feet as ff

U = "/home/interns/unisense"
SPEC = json.load(open(f"{U}/code/specs/mri_v03.json"))
ALLOWED = ("kp_px", "kp_score", "kp_px_cam2", "kp_score_cam2", "R_c2r", "t_c2r", "meta")


def load(pk):
    z = np.load(pk, allow_pickle=False); out = {}
    for k in ALLOWED:
        assert k not in fc.FORBIDDEN
        out[k] = z[k]
    out["meta"] = json.loads(str(out["meta"]))
    return out


def subject(pk):
    P = load(pk); m = P["meta"]; fps = float(m["fps"])
    g = SPEC["rgb"]; K1, K2 = np.array(g["K"]), np.array(g["K_cam2"]); R12, t12 = np.array(g["R_cam1_to_cam2"]), np.array(g["t_cam1_to_cam2"])
    Rc, tc = P["R_c2r"].astype(float), P["t_c2r"].astype(float)
    k1, s1, k2, s2 = P["kp_px"].astype(float), P["kp_score"].astype(float), P["kp_px_cam2"].astype(float), P["kp_score_cam2"].astype(float)
    J = [5, 6, 11, 12, 15, 16]
    Xc = ff.triangulate(k1[:, J], k2[:, J], s1[:, J], s2[:, J], K1, K2, R12, t12)          # (T,6,3) cam1
    X = Xc @ Rc.T + tc
    A = X[:, 4:6]; valid = np.isfinite(A).all(-1)
    Z = np.nanmedian(Xc[:, 4:6, 2], 1)
    Zs = np.array([np.nanmedian(Z[max(0, t - 2):t + 3]) if np.isfinite(Z[max(0, t - 2):t + 3]).any() else np.nan for t in range(len(Z))])
    sp = np.stack([ff.metric_speed([k1[:, j], k2[:, j]], Zs, [K1[0, 0], K2[0, 0]], fps) for j in ff.ANK], 1)
    segs = ff.contact_segments(A, sp, valid)
    cont = np.zeros(valid.shape, bool)
    for s in segs:
        cont[s[3]:s[3] + s[1], s[2]] = True
    return dict(name=os.path.basename(pk)[:-4], T=len(X), fps=fps, valid_frac=float(valid.mean()), contact_frac=float(cont.mean()),
                segs=segs, X=X, both_contact=cont.all(1), Rc=Rc, tc=tc, Zmed=float(np.nanmedian(Z)))


def standing_axis(S, n, d):
    """서 있는 프레임 축 vs 법선 n (방 좌표). 반환 dict(각 중앙값, 옆·시선 성분 중앙값 [deg], 프레임 수)."""
    X = S["X"]; sh = X[:, 0:2].mean(1); hip = X[:, 2:4].mean(1); an = X[:, 4:6].mean(1)
    ok = S["both_contact"] & np.isfinite(sh).all(-1) & np.isfinite(hip).all(-1) & np.isfinite(an).all(-1)
    if ok.sum() < 20:
        return dict(n_frames=int(ok.sum()))
    ax = sh[ok] - an[ok]; hv = ax @ n
    hor = lambda v: v - np.outer(v @ n, n)
    ok2 = (hv >= 1.0) & (np.linalg.norm(hor(hip[ok] - an[ok]), axis=1) < 0.15)
    if ok2.sum() < 20:
        return dict(n_frames=int(ok2.sum()))
    ax = ax[ok2]; cam = S["tc"]
    view = hor(an[ok][ok2] - cam); view /= np.linalg.norm(view, axis=1, keepdims=True); lat = np.cross(n, view)
    hz = ax @ n; along = np.degrees(np.arctan2(np.sum(ax * view, 1), hz)); side = np.degrees(np.arctan2(np.sum(ax * lat, 1), hz))
    tot = np.degrees(np.arccos(np.clip(hz / np.linalg.norm(ax, axis=1), -1, 1)))
    mvec = np.median(ax / np.linalg.norm(ax, axis=1, keepdims=True), 0); mvec /= np.linalg.norm(mvec)
    return dict(n_frames=int(ok2.sum()), angle_med_deg=float(np.median(tot)), along_view_med_deg=float(np.median(along)), lateral_med_deg=float(np.median(side)),
                mean_axis_angle_deg=float(np.degrees(np.arccos(np.clip(mvec @ n, -1, 1)))), mean_axis=mvec)


def person_h(fl, pc):
    n = np.asarray(fl["n"]); return float(pc @ n + fl["d"])


def main():
    pks = sorted(glob.glob(f"{U}/dev/packs/mri_v03/S*.npz"))
    Ss = [subject(pk) for pk in pks]
    out = dict(rule="floor_feet.py (pre-registered before this run)", ankle_to_sole=ff.ANKLE_TO_SOLE, subjects={})
    sub_fl = {}
    for pk, S in zip(pks, Ss):
        fl = ff.floor_from_ankles(S["segs"], sensor=S["tc"], B=200)
        sub_fl[pk] = fl
        out["subjects"][pk] = dict(name=S["name"], T=S["T"], valid_frac=S["valid_frac"], contact_frac=S["contact_frac"], n_segments=len(S["segs"]), cam_dist_med=S["Zmed"],
                                   own=None if fl is None else {k: v for k, v in fl.items()})
        print(S["name"], "segs", len(S["segs"]), "valid %.2f contact %.2f Z %.2f" % (S["valid_frac"], S["contact_frac"], S["Zmed"]),
              "" if fl is None else "own: tilt %.2f° d %.3f n_pts %d minor %.2f σt %.2f σh %.3f" % (fc.tilt_deg(fl["n"]), fl["d"], fl["n_points"], fl["xy_minor_sd_m"], fl["sigma_tilt_deg"], fl["sigma_h0_fit_m"]), flush=True)
    # 묶음 평면 (모든 피험자 같은 옛 방 좌표 = 사양 room)
    allsegs = [s for S in Ss for s in S["segs"]]
    pooled = ff.floor_from_ankles(allsegs, sensor=Ss[0]["tc"], B=200)
    rng = np.random.default_rng(0); ang, hh = [], []
    for _ in range(100):
        pick = rng.integers(0, len(Ss), len(Ss)); ss = [s for i in pick for s in Ss[i]["segs"]]
        f2 = ff.floor_from_ankles(ss, sensor=Ss[0]["tc"], B=1)
        if f2 is not None:
            ang.append(fc.deviation(f2["n"], f2["d"], pooled["n"], pooled["d"])[0]); hh.append(person_h(f2, pooled["person_center"]) - person_h(pooled, pooled["person_center"]))
    sb_t = float(np.sqrt(np.mean(np.square(ang)))); sb_h = float(np.std(hh))
    sig_t = max(sb_t, pooled["sigma_tilt_deg"], ff.METHOD_TILT_SIGMA); sig_h = float(np.sqrt(sb_h ** 2 + pooled["sigma_h_person_fit_m"] ** 2 + ff.ANKLE_PRIOR_SIGMA ** 2 + ff.METHOD_H_SIGMA ** 2))
    out["pooled"] = dict(pooled, subject_block_sigma_tilt_deg=sb_t, subject_block_sigma_h_person_m=sb_h, sigma_tilt_total_deg=sig_t, sigma_h_total_m=sig_h, **fc.plane_summary(pooled["n"], pooled["d"]))
    print("POOLED tilt %.2f° d %.4f segs %d inl %d minor %.2f constrained %s | σ fit %.2f° sb %.2f° %.3f m | total σ %.2f° %.3f m" % (
        fc.tilt_deg(pooled["n"]), pooled["d"], pooled["n_segments"], pooled["n_points"], pooled["xy_minor_sd_m"], pooled["slope_constrained"], pooled["sigma_tilt_deg"], sb_t, sb_h, sig_t, sig_h), flush=True)
    # 최종 + 서 있는 축
    final = {}
    for pk, S in zip(pks, Ss):
        n, d = np.asarray(pooled["n"]), float(pooled["d"]); choice = "pooled_all_subjects"; own = sub_fl[pk]
        dev = None
        if own is not None:
            pc = np.asarray(own["person_center"]); dh = person_h(own, pc) - person_h(pooled, pc); dev = dict(angle_deg=fc.deviation(own["n"], own["d"], n, d)[0], dh_person_m=dh)
            if abs(dh) > ff.SUBJECT_H_SWITCH:
                d = d + dh; choice = "pooled_normal+own_height(rig_height_moved?)"
        st_new = standing_axis(S, n, d); st_old = standing_axis(S, np.array([0, 0, 1.0]), 0.0)
        final[pk] = dict(n=n, d=d, choice=choice, own_vs_pooled=dev, standing_new=st_new, standing_old=st_old)
        print(S["name"], choice, "" if dev is None else "own−pooled: %.2f° %.3f m" % (dev["angle_deg"], dev["dh_person_m"]),
              "| stand new %.2f° (along %.2f, lat %.2f) old %.2f° n=%d" % (st_new.get("angle_med_deg", np.nan), st_new.get("along_view_med_deg", np.nan), st_new.get("lateral_med_deg", np.nan),
                                                                         st_old.get("angle_med_deg", np.nan), st_new.get("n_frames", 0)), flush=True)
    out["final"] = final
    # 피험자 사이 반복성 (자기 평면)
    own = [f for f in sub_fl.values() if f is not None]
    if own:
        pc = np.asarray(pooled["person_center"])
        out["repeatability"] = dict(n=len(own), tilt_sd_deg=float(np.std([fc.tilt_deg(f["n"]) for f in own])),
                                    angle_to_pooled_med_deg=float(np.median([fc.deviation(f["n"], f["d"], pooled["n"], pooled["d"])[0] for f in own])),
                                    h_at_pooled_center_sd_m=float(np.std([person_h(f, pc) for f in own])))
        print("REPEAT", out["repeatability"], flush=True)
    open(f"{U}/dev/packs/v3/floor/mri.json", "w").write(fc.jdump(out))
    print("MRI_DONE", flush=True)


if __name__ == "__main__":
    main()
