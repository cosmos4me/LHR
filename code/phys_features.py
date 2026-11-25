"""번역기 물리 추정 (학습 없음): 팩마다 조건별 기준점과 관절별 물리 추정값을 곁 파일로 만든다 → dev/packs/phys_v1/{ds}/...npz
조건 순서 [rgb, depth, both]:
  a3 (T,3,3) 기준점 = 골반(두 엉덩이 중점) 물리 추정, x3 (T,3,13,3) 관절별 물리 추정 (없으면 NaN).
  - MM-Fi rgb  : 발목 광선–바닥 교차 + 정면 평면 (physics_baseline.mono_floor)
          depth: 사람 점 중앙값 + 몸 두께 절반 (기준점만)
          both : RGB 광선 × 팩 깊이 점 (광선 5 cm 안 점의 중앙값 + 표면→관절 오프셋), 기준점은 엉덩이 둘 다 있으면 그 중점, 아니면 depth 기준점
  - mRI   rgb  : 두 시점 삼각측량
  - OctoNet depth: 사람 점 중앙값 + 몸 두께 절반
기준점이 빈 프레임은 같은 녹화의 가장 가까운 유효 프레임으로 채운다 (움직임 연속성). 녹화 전체가 비면 NaN으로 두고 개수를 보고한다.
사용: python phys_features.py [mmfi mri octonet]"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import conv_common as cc
import physics_baseline as pb

OUT = Path("/home/interns/unisense/dev/packs/phys_v1")


def fill_nearest(a):
    """(T,3) NaN 행을 시간상 가장 가까운 유효 행으로 채운다."""
    ok = np.isfinite(a).all(-1)
    if ok.all() or not ok.any():
        return a, int((~ok).sum())
    idx = np.where(ok)[0]; j = idx[np.abs(np.arange(len(a))[:, None] - idx[None]).argmin(1)]
    return a[j], int((~ok).sum())


def hips(X):
    r = 0.5 * (X[:, 7] + X[:, 8]); r[~np.isfinite(r).all(-1)] = np.nan
    return r


def one(job):
    ds, f = job
    P = np.load(f); T = len(P["labels"]); m = json.loads(P["meta"].item())
    a3 = np.full((T, 3, 3), np.nan); x3 = np.full((T, 3, 13, 3), np.nan)
    if ds == "mmfi":
        spec = json.loads(pb.SPECS["mmfi"].read_text())
        o, d = pb.rays(P["kp_px"], spec["rgb"]["K"], P["R_c2r"], P["t_c2r"])
        Xm, _ = pb.mono_floor(o, d, P["R_c2r"], np.array([0, 0, 1.0]), 0.0)
        x3[:, 0] = cc.to_common13(Xm, "coco17"); a3[:, 0] = hips(x3[:, 0])
        p, ok = pb.depth_points(P, spec["depth"])
        a3[:, 1] = pb.person_root(p, ok, np.asarray(P["t_d2r"], float))
        x3[:, 2] = cc.to_common13(pb.mmfi_sparse(P, spec, pb.OFF), "coco17")
        hb = hips(x3[:, 2]); a3[:, 2] = np.where(np.isfinite(hb).all(-1)[:, None], hb, a3[:, 1])
    elif ds == "mri":
        spec = json.loads(pb.SPECS["mri"].read_text())
        R12, T12 = np.asarray(spec["rgb"]["R_cam1_to_cam2"]), np.asarray(spec["rgb"]["t_cam1_to_cam2"])
        Rc, tc = np.asarray(P["R_c2r"]), np.asarray(P["t_c2r"], float)
        o1, d1 = pb.rays(P["kp_px"], spec["rgb"]["K"], Rc, tc)
        o2, d2 = pb.rays(P["kp_px_cam2"], spec["rgb"]["K_cam2"], Rc @ R12.T, Rc @ (-R12.T @ T12) + tc)
        s1, s2 = P["kp_score"].astype(float), P["kp_score_cam2"].astype(float)
        X17 = pb.triangulate(o1, d1, s1 ** 2, o2, d2, s2 ** 2); X17[(s1 < pb.MIN_SCORE) | (s2 < pb.MIN_SCORE)] = np.nan
        x3[:, 0] = cc.to_common13(X17, "coco17"); a3[:, 0] = hips(x3[:, 0])
    else:
        spec = json.loads(pb.SPECS["octonet"].read_text())
        p, ok = pb.depth_points(P, spec["depth"][f"{m['env']}_{m['day']}"])
        a3[:, 1] = pb.person_root(p, ok, np.asarray(P["t_d2r"], float))
    filled = {}
    for c in range(3):
        if np.isfinite(a3[:, c]).any():
            a3[:, c], filled[c] = fill_nearest(a3[:, c])
    rel = f.relative_to(pb.PACKS[ds])
    out = OUT / ds / rel; out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, a3=a3.astype(np.float32), x3=x3.astype(np.float32))
    return ds, str(rel), filled, bool(np.isnan(a3).all(-1).all(0).sum() < 3)


if __name__ == "__main__":
    dss = sys.argv[1:] or ["mmfi", "mri", "octonet"]
    jobs = []
    for ds in dss:
        files = sorted(pb.PACKS[ds].glob("E*/*.npz" if ds == "mmfi" else "*.npz"))
        jobs += [(ds, f) for f in files]
    stats = {}
    with Pool(16) as pool:
        for ds, rel, filled, _ in pool.imap_unordered(one, jobs):
            s = stats.setdefault(ds, dict(files=0, filled={0: 0, 1: 0, 2: 0}))
            s["files"] += 1
            for c, n in filled.items():
                s["filled"][c] += n
    print(json.dumps(stats, indent=1))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
