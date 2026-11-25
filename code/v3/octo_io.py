"""v3 OctoNet 레이더 정합용 읽기 도구 (PIL 없이 mmfi 환경에서 돈다). 정답(몸 관절·라벨)은 읽지 않는다.
- recordings(): 노드 1 깊이 zip 이름에서 녹화 목록 (octonet_io.recordings 와 같은 규칙)
- depth_times(zip): zip 항목 이름의 시각만 읽음 (압축 해제·PNG 해독 없음, 경로 탈출 이름은 버림)
- read_radar(pickle): 허용 목록 Unpickler 로 연속 프레임 읽기 (numpy 재구성/ndarray/dtype/scalar, datetime 만 허용)
- mocap_span(csv): 모캡 CSV 의 첫 줄(헤더)만 읽어 시각 범위 (팩 프레임 격자 재현용; 관절은 읽지 않음)
- node_rigid(csv): 'Rigid Body' 형식 열 중 NodeK 위치·회전만 읽음 (센서 노드 자세 = 몸 정답 아님)
- room_geometry(env, day): 팩의 방 좌표 정의 (notes/calib_octonet_days_v5.json; adapt_octonet.geometry 와 같은 식)
내려받은 파일은 신뢰하지 않는다: 이 모듈을 쓰는 스크립트는 python -I 로 실행하고 sys.path 에 code/v3 만 넣는다."""
import csv, datetime, io, json, pickle, re, zipfile
from pathlib import Path
import numpy as np

U = Path("/home/interns/unisense")
RAW = U / "dev/data/octonet/raw"
NAME = re.compile(r"^(\d|x)_([A-Za-z]+)_(\d+)_(\d+)_(.+)_(\d+)_(\d{14})\.")
CAL = json.loads((U / "notes/calib_octonet_days_v5.json").read_text())


def recordings():
    out = []
    for f in sorted(RAW.glob("node_1/1_depthcam_*.zip")):
        m = NAME.match(f.name)
        env, user, act, trial, stamp = m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)
        tail = f"{env}_{user}_{act}_{trial}_{stamp}"
        out.append(dict(env=int(env), user=int(user), action=act, trial=int(trial), stamp=stamp, day=stamp[:8], depth=f,
                        mocap=RAW / "node_x" / f"x_mocapcsv_{tail}.csv", radar=RAW / "node_1" / f"1_mmWave_{tail}.pickle"))
    return out


def _safe_name(n):
    return not (n.startswith("/") or ".." in n.split("/") or "\\" in n or ":" in n.split("/")[0])


def depth_times(path):
    """깊이 PNG 시각 (epoch 초, 지역 시각 규칙은 octonet_io.DepthZip 과 같음), 정렬됨."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if _safe_name(n) and n.endswith(".png") and "/depth_" in n]
    ts = [datetime.datetime.strptime(n.rsplit("depth_", 1)[1][:-4], "%Y-%m-%d %H:%M:%S.%f").timestamp() for n in names]
    return np.sort(np.asarray(ts))


ALLOW = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"),
         ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"), ("datetime", "datetime")}


class _Safe(pickle.Unpickler):
    def find_class(self, m, n):
        if (m, n) in ALLOW:
            return super().find_class(m, n)
        raise pickle.UnpicklingError(f"blocked {m}.{n}")


def read_radar(path, rmin=0.3, rmax=10.0):
    """반환: ts (F,) epoch 초, pts list of (n,4) 레이더 좌표 x y z [m], 도플러 [m/s]. 유한·거리 [rmin, rmax] 만 (|좌표| ~1e29 이상값, 원점 근처 점 제거)."""
    fr = []
    with open(path, "rb") as fh:
        buf = io.BytesIO(fh.read())
    while True:
        try:
            fr.append(_Safe(buf).load())
        except EOFError:
            break
    ts, pts = [], []
    for f in fr:
        d = f["data"]
        p = np.c_[d["x"], d["y"], d["z"], d["velocity"]].astype(float)[:int(d["numObj"])]
        with np.errstate(all="ignore"):
            rg = np.linalg.norm(np.nan_to_num(p[:, :3], nan=99.0, posinf=99.0, neginf=99.0), axis=1)
        ts.append(f["timestamp"].timestamp()); pts.append(p[np.isfinite(p).all(1) & (rg >= rmin) & (rg <= rmax)])
    o = np.argsort(ts)
    return np.asarray(ts)[o], [pts[i] for i in o]


def _motive_time(s):
    s = s.replace("上午", "AM").replace("下午", "PM")
    return datetime.datetime.strptime(s, "%Y-%m-%d %I.%M.%S.%f %p").timestamp()


def mocap_span(path):
    """헤더 첫 줄 + 첫·마지막 데이터 행의 프레임 번호(첫 칸)만: (t_first, t_last, fps, 행 수 추정). octonet_io.read_mocap 의 t = t0 + frame/fps 와 같은 격자.
    일부 CSV 는 'Total Exported Frames' 보다 일찍 끝나므로 마지막 행의 프레임 번호를 쓴다. 관절 값은 해석하지 않는다."""
    import os
    with open(path) as fh:
        row = next(csv.reader(fh))
        for _ in range(6):
            next(fh)
        first = next(fh).split(",", 1)[0]
    hdr = dict(zip(row[0::2], row[1::2]))
    t0 = _motive_time(hdr["Capture Start Time"]); fps = float(hdr["Export Frame Rate"])
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END); size = fh.tell(); fh.seek(max(0, size - 200000))
        lines = [x for x in fh.read().decode("ascii", errors="replace").splitlines() if x.strip()]
    last = lines[-1].split(",", 1)[0]
    f0, f1 = int(first), int(last)
    return t0 + f0 / fps, t0 + f1 / fps, fps, f1 - f0 + 1


def node_rigid(path, names=("Node1", "Node2", "Node3", "Node4", "Node5"), stride=60):
    """'Rigid Body' 열 (NodeK 위치 mm → m, 회전 쿼터니언 xyzw) 만 읽는다. 없으면 {}. 반환 {NodeK: (N,7)} 세계(모캡) 좌표.
    노드는 녹화 안에서 움직이지 않으므로 stride 행마다 하나만 해석 (120 Hz → 2 Hz)."""
    with open(path) as fh:
        rows = [r for _, r in zip(range(7), csv.reader(fh))]
    hdr = dict(zip(rows[0][0::2], rows[0][1::2])); scale = 1e-3 if hdr.get("Length Units", "Millimeters") == "Millimeters" else 1.0
    types, nms, kinds, axes = rows[2], rows[3], rows[5], rows[6]
    want = {}
    for i, (t, n, k, a) in enumerate(zip(types, nms, kinds, axes)):
        if t == "Rigid Body" and n in names and k in ("Position", "Rotation"):
            want[(n, k, a)] = i
    if not want:
        return {}
    cols = sorted(want.values()); pos = {c: j for j, c in enumerate(cols)}
    raw = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i < 7 or (i - 7) % stride:
                continue
            row = next(csv.reader([line]))
            if row:
                raw.append([(row[c] if c < len(row) else "") or "nan" for c in cols])
    A = np.asarray(raw, dtype=np.float64)
    out = {}
    for n in names:
        try:
            P = [A[:, pos[want[(n, "Position", ax)]]] * scale for ax in "XYZ"]; Q = [A[:, pos[want[(n, "Rotation", ax)]]] for ax in "XYZW"]
        except KeyError:
            continue
        out[n] = np.stack(P + Q, 1)
    return out


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def room_geometry(env, day):
    """팩 방 좌표 (adapt_octonet.geometry 와 같은 식): 세계→방, 깊이 카메라→방, K, 깊이 배율 a."""
    E = CAL[str(env)]; D = E["days"][day]
    R_w2c, C = np.array(D["R_w2c"]), np.array(D["C_world"])
    fwd = R_w2c[2]; psi = np.arctan2(fwd[1], fwd[0])
    R_w2r = Rz(-psi); t_w2r = -R_w2r @ np.array([C[0], C[1], 0.0])
    R_c2r = R_w2r @ R_w2c.T; t_c2r = R_w2r @ C + t_w2r
    return dict(K=E["K"], a=E["depth_scale_a"], R_w2r=R_w2r, t_w2r=t_w2r, R_c2r=R_c2r, t_c2r=t_c2r)


def nearest(t_src, t_query):
    i = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    j = np.where(np.abs(t_src[i - 1] - t_query) < np.abs(t_src[i] - t_query), i - 1, i)
    return j, t_src[j] - t_query


def quat_R(q):
    """쿼터니언 (x, y, z, w) → 회전행렬."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
