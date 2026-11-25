"""v3 바닥 → 녹화별 보정 변환 floors.json (정답 없음). 입력은 데이터셋별 결과 (dev/packs/v3/floor/*.json) 와 팩의 센서→방 변환뿐.
- 장면 합의는 깊이 카메라 좌표에서 한다 (평면 n_c = R_d2rᵀ n, d_c = n·t_d2r + d): 옛 방 좌표가 녹화마다 달라도 같은 카메라면 비교 가능.
  장면: MM-Fi (환경, 피험자), OctoNet (환경, 날짜), ITOP 전체 4명 (같은 카메라 높이 1.30 m), mRI 는 floor_feet 사전 등록 규칙 (floor_mri.py 결과 그대로),
  mmBody 는 모캡 바닥 유지 (문서 레이더 높이 기각; floor_mmbody.py).
- 선택 규칙: floor_common.choose_final (허용 안 → 합의, 실패 한계 밖 → 합의 + 표시, 그 사이 → 녹화 자신).
  단 '녹화 자신' 은 같은 장면의 다른 이탈 녹화가 같은 평면(1°, 2 cm 안)을 보일 때만 (rig 이동 확인); 혼자 이탈하면 맞춤 오류로 보고 합의.
- 보정: floor_common.correction (새 z = 바닥 법선, 바닥 z = 0, 새 x = 옛 x 의 바닥 투영, 원점 = 옛 원점의 바닥 투영).
출력: dev/packs/v3/floors.json (키 = 팩 절대 경로). 실행: python -I floor_build.py"""
import sys, glob, json, collections
sys.path.insert(0, "/home/interns/unisense/code/v3")
import numpy as np
import floor_common as fc

U = "/home/interns/unisense"; F = f"{U}/dev/packs/v3/floor"
OUT = f"{U}/dev/packs/v3/floors.json"


def cam_plane(n, d, R, t):
    n = np.asarray(n, float); return R.T @ n, float(n @ t + d)


def room_plane(nc, dc, R, t):
    n = R @ nc; return n, float(dc - n @ t)


def entry(n, d, source, unc_t, unc_h, n_points, **kw):
    R, t = fc.correction(n, d); s = fc.plane_summary(n, d)
    e = dict(R=R, t=t, source=source, angle_change_deg=s["angle_change_deg"], height_change_m=s["height_change_m"],
             tilt_x_deg=s["tilt_x_deg"], tilt_y_deg=s["tilt_y_deg"], uncertainty=dict(tilt_deg=float(unc_t), height_m=float(unc_h)),
             n_points=None if n_points is None else int(n_points), plane_old_room=dict(n=np.asarray(n), d=float(d)))
    e.update(kw)
    return e


def depth_scenes(res, scene_key, source, extra_unc_h=0.0, stats=None):
    """res: {pk: per-recording result (n, d in pack room)}. 반환 {pk: entry}."""
    pks_all = sorted(res); out = {}
    T = {pk: (fc.safe_npz(pk)["R_d2r"].astype(float), fc.safe_npz(pk)["t_d2r"].astype(float)) for pk in pks_all}
    groups = collections.defaultdict(list)
    for pk in pks_all:
        groups[scene_key(pk, res[pk])].append(pk)
    for g, pl in groups.items():
        ok = [pk for pk in pl if res[pk].get("ok")]
        if not ok:
            for pk in pl:
                out[pk] = dict(R=np.eye(3), t=np.zeros(3), source="none(identity; no floor estimate in scene)", angle_change_deg=0.0, height_change_m=0.0,
                               uncertainty=None, n_points=0, scene=str(g), choice="identity")
            continue
        C = {pk: dict(zip(("n", "d"), cam_plane(res[pk]["n"], res[pk]["d"], *T[pk])), sigma_tilt_deg=res[pk]["sigma_tilt_deg"], sigma_h0_m=res[pk]["sigma_h0_m"]) for pk in ok}
        nc, dc = fc.consensus([(C[pk]["n"], C[pk]["d"]) for pk in ok]); cons = dict(n=nc, d=dc)
        sp_a, sp_h = fc.scene_spread(list(C.values()), cons)
        med_st = float(np.median([C[pk]["sigma_tilt_deg"] for pk in ok])); med_sh = float(np.median([C[pk]["sigma_h0_m"] for pk in ok]))
        # 1차 선택 → 장면 안 '녹화 자신' 후보가 혼자면 (다른 이탈 녹화 중 평면이 허용 안에 드는 것이 없으면) 맞춤 오류로 보고 합의로 되돌린다.
        # (고정 rig 가 한 녹화만 움직였다 돌아올 가능성보다 가림·다인 동작·바닥 위 물체로 맞춤이 틀릴 가능성이 크다; MM-Fi E02 S12 A19–A27 처럼 연속 이탈은 유지)
        first = {pk: fc.choose_final(C[pk], cons, sp_a, sp_h) for pk in ok}
        devs = [pk for pk in ok if first[pk][2].startswith("own")]
        confirmed = {pk for pk in devs if any(q != pk and fc.deviation(C[pk]["n"], C[pk]["d"], C[q]["n"], C[q]["d"])[0] <= fc.ANG_TOL
                                              and abs(fc.deviation(C[pk]["n"], C[pk]["d"], C[q]["n"], C[q]["d"])[1]) <= fc.H_TOL for q in devs)}
        for pk in pl:
            R, t = T[pk]
            if pk not in C:
                n, d = room_plane(nc, dc, R, t)
                out[pk] = entry(n, d, source, max(sp_a, med_st, 0.1), np.hypot(max(sp_h, med_sh, 0.005), extra_unc_h), None, scene=str(g),
                                choice="scene_consensus(no_own_estimate: " + res[pk].get("why", "?") + ")", n_scene=len(ok))
                continue
            nn, dd, ch, a, h = first[pk]
            if ch.startswith("own") and pk not in confirmed:
                nn, dd, ch = np.asarray(nc), float(dc), "scene_consensus(single_recording_deviation_treated_as_fit_error)"
            elif ch.startswith("own"):
                ch = "own_recording(deviates_from_scene; confirmed by other deviating recordings → rig moved)"
            n, d = room_plane(nn, dd, R, t)
            if ch.startswith("scene"):
                ut, uh = max(sp_a, med_st, 0.1), max(sp_h, med_sh, 0.005)
            else:
                ut, uh = max(C[pk]["sigma_tilt_deg"], 0.1), max(C[pk]["sigma_h0_m"], 0.005)
            out[pk] = entry(n, d, source, ut, np.hypot(uh, extra_unc_h), res[pk]["n_points"], scene=str(g), choice=ch, n_scene=len(ok),
                            own_vs_scene=dict(angle_deg=a, height_m=h), scene_spread=dict(angle_deg=sp_a, height_m=sp_h),
                            own_estimate=dict(angle_change_deg=fc.tilt_deg(res[pk]["n"]), height_change_m=res[pk]["d"]))
    return out


def main():
    floors = {}
    # MM-Fi
    r = json.load(open(f"{F}/mmfi_depth.json"))
    floors.update(depth_scenes(r, lambda pk, v: ("mmfi", pk.split("/")[-2], pk.split("/")[-1].split("_")[0]), "floor_depth_ransac(D435 background median + 3 full frames)"))
    # OctoNet
    r = json.load(open(f"{F}/octonet_depth.json"))
    meta = {pk: fc.safe_npz(pk).meta() for pk in r}
    floors.update(depth_scenes(r, lambda pk, v: ("octonet", meta[pk]["env"], meta[pk]["day"]), "floor_depth_ransac(D455 node1 8-bit, 40-frame median)", extra_unc_h=0.0))
    # ITOP
    r = json.load(open(f"{F}/itop_depth.json"))
    floors.update(depth_scenes(r, lambda pk, v: ("itop", "side_cam"), "floor_depth_ransac(Xtion background median + 6 frames)"))
    # mmBody
    r = json.load(open(f"{F}/mmbody.json"))
    ang = [v["feet"]["angle_change_deg"] for v in r.values() if "feet" in v]; hp = [v["feet_minus_old_at_person_m"] for v in r.values() if "feet_minus_old_at_person_m" in v]
    for pk, v in r.items():
        f = v.get("feet", {})
        floors[pk] = entry(np.array([0, 0, 1.0]), 0.0, "dataset_calib_nonbody(OptiTrack ground plane via calib.txt; documented radar height 0.8 m rejected)",
                           float(np.median(ang)), float(np.std(hp)), f.get("n_points"), scene=f"mmbody:{v['split']}", choice="keep_mocap_floor(identity)",
                           doc_check=dict(rejected=v["doc_rejected"], doc_minus_feet_at_person_m=v.get("doc_minus_feet_at_person_m"), doc_vs_feet_angle_deg=v.get("doc_vs_feet_angle_deg"),
                                          radar_height_in_current_frame_m=v["radar_height_old_m"]),
                           feet_check=None if not f else dict(angle_vs_current_deg=f["angle_change_deg"], feet_minus_current_at_person_m=v.get("feet_minus_old_at_person_m"),
                                                              note="person cloud bottom was cut by the dataset's GT box (lowest foot joint + 1 cm) → cross-check only"),
                           depth_empty=v["depth_empty"])
    # mRI
    r = json.load(open(f"{F}/mri.json")); P = r["pooled"]
    for pk, v in r["final"].items():
        floors[pk] = entry(np.array(v["n"]), v["d"], "stereo_person(ViTPose ankles, 2 views, contact frames, ankle→sole 0.09 m)", P["sigma_tilt_total_deg"], P["sigma_h_total_m"],
                           P["n_points"], scene="mri:all_subjects", choice=v["choice"], own_vs_pooled=v["own_vs_pooled"],
                           n_segments_pooled=P["n_segments"], slope_constrained=P["slope_constrained"])
    # 모든 팩이 있는지
    expect = (sorted(glob.glob(f"{U}/dev/packs/mmfi_v02_specA/E0[1-4]/S*_A*.npz")) + sorted(glob.glob(f"{U}/dev/packs/mri_v03/S*.npz")) + sorted(glob.glob(f"{U}/dev/packs/mmbody_v01/*/*.npz"))
              + sorted(glob.glob(f"{U}/dev/packs/octonet_v07/*.npz")) + sorted(glob.glob(f"{U}/dev/packs/itop_v01/*.npz")))
    miss = [p for p in expect if p not in floors]; extra = [p for p in floors if p not in set(expect)]
    assert not miss and not extra, (len(miss), len(extra))
    hdr = dict(_about=dict(version="v3_floor_20261124", convention="X_new = R @ X_old + t (old = pack room frame). New z = floor normal, floor at z = 0, new x = old x projected on floor, origin = old origin projected on floor.",
                           height_change_m="= d of plane n·X_old + d = 0; z_new(old origin) = d", gt_used=False, code="code/v3/floor_*.py", note="notes/v3_floor.md"))
    open(OUT, "w").write(fc.jdump({**hdr, **floors}))
    # 요약
    S = collections.defaultdict(list)
    for pk, e in floors.items():
        ds = pk.split("/dev/packs/")[1].split("/")[0]; S[ds].append(e)
    for ds, L in S.items():
        a = np.array([e["angle_change_deg"] for e in L]); h = np.array([e["height_change_m"] for e in L]); ch = collections.Counter(e["choice"].split("(")[0] for e in L)
        print(ds, len(L), "angle med %.2f p95 %.2f max %.2f | dh med %.3f |·| p95 %.3f max %.3f |" % (np.median(a), np.percentile(a, 95), a.max(), np.median(h), np.percentile(np.abs(h), 95), np.abs(h).max()), dict(ch), flush=True)
    print("FLOORS_DONE", len(floors), flush=True)


if __name__ == "__main__":
    main()
