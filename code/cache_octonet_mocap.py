"""OctoNet 모캡 CSV(녹화당 약 50 MB)를 한 번만 파싱해 공통 13관절·강체·시각을 npz로 캐시하고, 깊이와의 시각 정렬을 점검한다.
출력: dev/data/octonet/derived/mocap13/{stamp}.npz, notes/octonet_sync_check.json. 사용: python cache_octonet_mocap.py"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import octonet_io as oi

OUT = Path("/home/interns/unisense/dev/data/octonet/derived/mocap13")


def one(r):
    out = OUT / f"{r['stamp']}.npz"
    try:
        if not out.exists():
            t, J, rig, hdr = oi.read_mocap(r["mocap"])
            np.savez_compressed(out, t=t, joints13=J.astype(np.float32), rigid_names=np.array(list(rig)),
                                rigid=np.stack([rig[k] for k in rig], 1).astype(np.float64) if rig else np.zeros((len(t), 0, 7)),
                                header=json.dumps(hdr))
        M = np.load(out)
        t, J = M["t"], M["joints13"]
        D = oi.DepthZip(r["depth"])
        _, dt = oi.nearest(t, D.t)
        inside = (D.t >= t[0]) & (D.t <= t[-1])
        return dict(stamp=r["stamp"], env=r["env"], user=r["user"], action=r["action"], n_mocap=int(len(t)), n_depth=len(D),
                    depth_inside_mocap=float(inside.mean()), dt_max_ms=float(np.abs(dt[inside]).max() * 1000) if inside.any() else None,
                    joint_nan=float(np.isnan(J).mean()), rigid=[str(x) for x in M["rigid_names"]], ok=True)
    except Exception as e:
        return dict(stamp=r["stamp"], env=r["env"], user=r["user"], action=r["action"], ok=False, err=str(e)[:200])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    R = oi.recordings()
    with Pool(8) as p:
        res = p.map(one, R)
    bad = [x for x in res if not x["ok"] or x["depth_inside_mocap"] < 0.95 or (x["dt_max_ms"] or 1e9) > 20 or x["joint_nan"] > 0.2]
    rig = {}
    for x in res:
        if x["ok"]:
            rig.setdefault(x["env"], set()).update(x["rigid"])
    summary = dict(n=len(res), n_bad=len(bad), bad=bad[:30], rigid_by_env={k: sorted(v) for k, v in rig.items()},
                   joint_nan_median=float(np.median([x["joint_nan"] for x in res if x["ok"]])))
    Path("/home/interns/unisense/notes/octonet_sync_check.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("n", "n_bad", "rigid_by_env", "joint_nan_median")}, ensure_ascii=False))
    for b in bad[:10]:
        print(b)
