"""v3 바닥 (mmBody). 정답 안 읽음 (cache_v1 의 joints/pose/shape/bbox 키는 열지 않는다).
mmBody 에는 잘리지 않은 깊이가 없다 (점군이 사람 상자로 잘려 있음). 그래서 바닥 후보 셋을 비교한다.
 (a) 문서값 [dataset_calib_nonbody]: README "레이더 = 세계 좌표, Z = 높이" + 논문 "삼각대 위 약 0.8 m"
     → 레이더 좌표에서 평면 z_r = −0.8 (법선 = 레이더 Z). calib.txt kinect_master (카메라 → 레이더, 홀더 설계값) 로 팩 방 좌표에 옮김.
 (b) 발바닥 [person_feet; 주의: 점군 아래 경계가 정답 상자(가장 낮은 발 관절 + 약 1 cm)로 잘렸으므로 GT 오염]:
     프레임마다 사람 점의 가장 낮은 2 cm 층(p1 기준) 중심 → 녹화 전체에서 RANSAC 평면 (허용 2 cm, 20° 안). 점프·공중 프레임은 바깥점.
 (c) 현재 방 좌표 = calib.txt optitrack 의 모캡 바닥 (y = 0) [dataset_calib_nonbody; 레이더 하우징 마커로 맞춘 모캡 → 레이더, 몸 정답 아님].
판정 (mRI 실행 전에 고정): (a) 의 높이가 사람 점의 가장 낮은 높이보다 5 cm 넘게 위면 (사람이 바닥 아래) 물리적으로 불가능 → (a) 높이 기각.
최종: (a) 가 통과하면 (a), 아니면 (c) (보정 = 항등). (b) 는 교차 확인 전용 (GT 오염 때문에 출처로 쓰지 않음).
출력: dev/packs/v3/floor/mmbody.json. 실행: python -I floor_mmbody.py"""
import sys, os, glob, json
sys.path.insert(0, "/home/interns/unisense/code/v3")
from multiprocessing import Pool
import numpy as np
import floor_common as fc

U = "/home/interns/unisense"; CACHE = f"{U}/dev/data/mmbody/cache_v1"
RADAR_H_DOC = 0.8
MINOR_MIN = 0.3            # 발바닥 점의 수평 짧은 축 표준편차가 이보다 작으면 (거의 한 줄) 그 축 방향 기울기는 관측 불가 → 0 으로 둔다


def feet_plane(Q):
    """z = a x + b y + c 를 Huber IRLS 로 (수직 잔차). xy 퍼짐이 한 줄에 가까우면 짧은 축 방향 기울기를 0 으로 고정."""
    xy = Q[:, :2] - Q[:, :2].mean(0); ev, V = np.linalg.eigh(np.cov(xy.T)); minor = float(np.sqrt(max(ev[0], 0)))
    A = np.c_[xy, np.ones(len(Q))] if minor >= MINOR_MIN else np.c_[xy @ V[:, 1], np.ones(len(Q))]
    w = np.ones(len(Q))
    for _ in range(10):
        coef = np.linalg.lstsq(A * w[:, None], Q[:, 2] * w, rcond=None)[0]; r = Q[:, 2] - A @ coef
        s_ = 1.4826 * np.median(np.abs(r)) + 1e-6; w = np.sqrt(np.minimum(1.0, 1.345 * s_ / np.maximum(np.abs(r), 1e-9)))
    g = coef[:2] if minor >= MINOR_MIN else coef[0] * V[:, 1]
    n = np.array([-g[0], -g[1], 1.0]); n /= np.linalg.norm(n)
    c0 = coef[-1] - g @ Q[:, :2].mean(0)                                        # z = g·xy + c0
    return n, float(-c0 * n[2]), minor


def one(pk):
    split, nm = pk.split("/")[-2], os.path.basename(pk)[:-4]
    P = fc.safe_npz(pk); m = P.meta(); R, t = P["R_c2r"].astype(float), P["t_c2r"].astype(float)
    Z = fc.safe_npz(f"{CACHE}/{split}/{nm}.npz"); c = json.loads(str(Z["calib"]))
    Rm, tm = np.array(c["kinect_master"]["R"], float), np.array(c["kinect_master"]["t"], float)
    # 방 → 레이더: X_r = Rm (R^T (X − t)) + tm = A X + b
    A = Rm @ R.T; b = tm - A @ t
    n_a = A.T @ np.array([0, 0, 1.0]); d_a = float(np.array([0, 0, 1.0]) @ b + RADAR_H_DOC)
    radar_room = R @ (Rm.T @ (-tm)) + t
    # (b) 발바닥
    o, X = Z["dpcl_off"], Z["dpcl_xyz"]; T = len(o) - 1
    soles, lows = [], []
    for k in range(0, T, 2):
        a_, b_ = o[k], o[k + 1]
        if b_ - a_ < 200:
            continue
        q = ((X[a_:b_].astype(float) - tm) @ Rm) @ R.T + t
        z1 = np.percentile(q[:, 2], 1); s = q[q[:, 2] < z1 + 0.02]
        soles.append(np.r_[np.median(s[:, :2], 0), np.median(s[:, 2])]); lows.append(np.percentile(q[:, 2], 0.5))
    out = dict(split=split, name=nm, subject=m.get("subject"), radar_room=radar_room, radar_height_old_m=float(radar_room[2]),
               doc=dict(n=n_a, d=d_a, **fc.plane_summary(n_a, d_a)), n_frames_pts=len(soles))
    if len(soles) >= 30:
        S = np.array(soles); lows = np.array(lows)
        res = fc.ransac_plane(S, tol=0.02, iters=2000, max_tilt_deg=20.0, sensor=None, seed=0, max_below_frac=0.05)
        if res is not None:
            Q = S[res["inl"]]; n_f, d_f, minor = feet_plane(Q)
            bs = fc.block_bootstrap(Q, B=100, cell=0.5, seed=1)
            pc = np.median(Q, 0)                                                    # 사람 활동 중심 (발바닥)
            out["feet"] = dict(n=n_f, d=d_f, n_points=res["n_inl"], n_frames=len(S), xy_extent=np.percentile(Q[:, :2], 95, 0) - np.percentile(Q[:, :2], 5, 0),
                               xy_minor_sd_m=minor, slope_constrained=bool(minor < MINOR_MIN), person_center=pc,
                               resid_mad_m=float(1.4826 * np.median(np.abs(Q @ n_f + d_f))), **bs, **fc.plane_summary(n_f, d_f))
            # 사람 위치에서 높이 차 (원점 높이 차는 3–5 m 떨어진 기울기에 지배되므로)
            out["feet_minus_old_at_person_m"] = float(-(pc[:2] @ n_f[:2] + d_f) / n_f[2])                   # 사람 xy 에서 발 평면의 옛 z
            out["doc_minus_old_at_person_m"] = float(-(pc[:2] @ n_a[:2] + d_a) / n_a[2])
            out["doc_minus_feet_at_person_m"] = out["doc_minus_old_at_person_m"] - out["feet_minus_old_at_person_m"]
            out["doc_vs_feet_angle_deg"] = fc.deviation(n_a, d_a, n_f, d_f)[0]
        out["person_low_p05_old_m"] = float(np.percentile(lows, 5)); out["person_low_med_old_m"] = float(np.median(lows))
        # (a) 물리 검사: 문서 바닥 기준 사람 최저점 높이
        out["doc_person_low_m"] = float(np.median([np.array([s_[0], s_[1], l_]) @ n_a + d_a for s_, l_ in zip(S, lows)]))
    out["doc_rejected"] = bool(out.get("doc_person_low_m", 0.0) < -0.05)
    out["depth_empty"] = len(soles) < 30
    return pk, out


if __name__ == "__main__":
    pks = sorted(glob.glob(f"{U}/dev/packs/mmbody_v01/*/*.npz"))
    res = {}
    with Pool(17) as pool:
        for pk, r in pool.imap_unordered(one, pks):
            res[pk] = r
            f = r.get("feet", {})
            print(r["split"], r["name"], "radar_h_old %.3f" % r["radar_height_old_m"], "doc: ang %.2f dh %.3f" % (r["doc"]["angle_change_deg"], r["doc"]["height_change_m"]),
                  "| feet: ang %.2f n %s minor %.2f | at person: feet−old %.3f doc−old %.3f doc−feet %.3f ang %.2f" % (f.get("angle_change_deg", np.nan), f.get("n_points"), f.get("xy_minor_sd_m", np.nan), r.get("feet_minus_old_at_person_m", np.nan), r.get("doc_minus_old_at_person_m", np.nan), r.get("doc_minus_feet_at_person_m", np.nan), r.get("doc_vs_feet_angle_deg", np.nan)),
                  "| doc_person_low %.3f rej %s" % (r.get("doc_person_low_m", np.nan), r["doc_rejected"]), flush=True)
    open(f"{U}/dev/packs/v3/floor/mmbody.json", "w").write(fc.jdump(res))
    print("MMBODY_DONE", flush=True)
