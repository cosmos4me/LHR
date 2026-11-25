"""OctoNet 읽기 도구: 녹화 목록, 모캡 CSV(OptiTrack Motive, 120 Hz, mm, z 위쪽), 깊이 zip(8비트 PNG, 0–10 m 선형), 시각 정렬.
내려받은 파일은 신뢰하지 않는 데이터로 다룬다: zip은 압축을 풀지 않고 메모리에서 읽고, 레이더 pickle은 허용 목록으로만 연다."""
import csv, datetime, io, pickle, re, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

RAW = Path("/home/interns/unisense/dev/data/octonet/raw")
NAME = re.compile(r"^(\d|x)_([A-Za-z]+)_(\d+)_(\d+)_(.+)_(\d+)_(\d{14})\.")
DEPTH_MAX_M = 10.0
# Motive 뼈 이름(접미사) → 공통 13관절. 뼈 위치 = 그 뼈의 근위 관절
BONE13 = ["Head", "LUArm", "RUArm", "LFArm", "RFArm", "LHand", "RHand", "LThigh", "RThigh", "LShin", "RShin", "LFoot", "RFoot"]


def recordings():
    out = []
    for f in sorted(RAW.glob("node_1/1_depthcam_*.zip")):
        m = NAME.match(f.name); env, user, act, trial, stamp = m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)
        tail = f"{env}_{user}_{act}_{trial}_{stamp}"
        out.append(dict(env=int(env), user=int(user), action=act, trial=int(trial), stamp=stamp, depth=f,
                        mocap=RAW / "node_x" / f"x_mocapcsv_{tail}.csv", radar=RAW / "node_1" / f"1_mmWave_{tail}.pickle"))
    return out


def _motive_time(s):
    """Motive 'Capture Start Time' (예: 2024-05-30 05.41.03.232 PM). 일부 파일은 오전/오후를 上午/下午로 적는다.
    깊이 파일 이름 시각과 같은 방식(시간대 없는 지역 시각)으로 epoch 초로 바꾼다."""
    s = s.replace("上午", "AM").replace("下午", "PM")
    return datetime.datetime.strptime(s, "%Y-%m-%d %I.%M.%S.%f %p").timestamp()


def read_mocap(path):
    """반환: t (N,) epoch 초, joints13 (N,13,3) m, rigid {NodeK: (N,7) = x,y,z [m], qx,qy,qz,qw}, header."""
    with open(path) as fh:
        rows = [r for _, r in zip(range(7), csv.reader(fh))]
    hdr = dict(zip(rows[0][0::2], rows[0][1::2]))
    t0 = _motive_time(hdr["Capture Start Time"]); fps = float(hdr["Export Frame Rate"])
    types, names, kinds, axes = rows[2], rows[3], rows[5], rows[6]
    scale = 1e-3 if hdr.get("Length Units", "Millimeters") == "Millimeters" else 1.0

    def cols(type_, suffix, kind):
        return [i for i, (t, n, k) in enumerate(zip(types, names, kinds)) if t == type_ and n.split(":")[-1] == suffix and k == kind]
    jc = [cols("Bone", b, "Position") for b in BONE13]
    rig_names = sorted({n for t, n in zip(types, names) if t == "Rigid Body"})
    rc = {n: ([i for i, (t, nn, k) in enumerate(zip(types, names, kinds)) if t == "Rigid Body" and nn == n and k == "Position"],
              [i for i, (t, nn, k) in enumerate(zip(types, names, kinds)) if t == "Rigid Body" and nn == n and k == "Rotation"])
          for n in rig_names}
    need = sorted({0} | {i for c in jc for i in c} | {i for p, r in rc.values() for i in p + r})
    pos = {c: k for k, c in enumerate(need)}
    with open(path) as fh:                                            # 필요한 열만 문자열로 모은 뒤 한 번에 변환 (빈칸 → NaN)
        it = csv.reader(fh)
        for _ in range(7):
            next(it)
        raw = [[row[c] if c < len(row) else "" for c in need] for row in it if row]
    A = np.array(raw, dtype=object)
    A[A == ""] = "nan"
    data = A.astype(np.float64)
    J = [data[:, [pos[i] for i in c]] * scale if len(c) == 3 else np.full((len(data), 3), np.nan) for c in jc]
    rigid = {n: np.c_[data[:, [pos[i] for i in p]] * scale, data[:, [pos[i] for i in r]]] for n, (p, r) in rc.items()}
    t = t0 + data[:, pos[0]] / fps
    return t, np.stack(J, 1), rigid, hdr


class DepthZip:
    """깊이 zip을 메모리에서 읽는다. frames: 시각 순으로 정렬된 (epoch 초, 항목 이름)."""
    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        names = [n for n in self.z.namelist() if n.endswith(".png") and "/depth_" in n]
        ts = [datetime.datetime.strptime(n.rsplit("depth_", 1)[1][:-4], "%Y-%m-%d %H:%M:%S.%f").timestamp() for n in names]
        order = np.argsort(ts)
        self.t = np.asarray(ts)[order]; self.names = [names[i] for i in order]

    def __len__(self):
        return len(self.names)

    def depth(self, i):
        """미터 단위 깊이 (H,W) float32, 0 = 무효 → NaN. 8비트 값 v → v/255·10 m (OctoNet 공식 로더와 같음)."""
        try:
            v = np.array(Image.open(io.BytesIO(self.z.read(self.names[i])))).astype(np.float32)
        except Exception:                                             # 손상된 PNG (일부 zip): 전부 무효로 둔다
            return np.full((480, 640), np.nan, np.float32)
        z = v / 255.0 * DEPTH_MAX_M
        z[v == 0] = np.nan
        return z


def nearest(t_src, t_query):
    """t_query 각 시각에 가장 가까운 t_src 인덱스와 시차 [s]."""
    i = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    j = np.where(np.abs(t_src[i - 1] - t_query) < np.abs(t_src[i] - t_query), i - 1, i)
    return j, t_src[j] - t_query


ALLOW = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"),
         ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"), ("datetime", "datetime")}


class _Safe(pickle.Unpickler):
    def find_class(self, m, n):
        if (m, n) in ALLOW:
            return super().find_class(m, n)
        raise pickle.UnpicklingError(f"blocked {m}.{n}")


def read_radar(path):
    return _Safe(io.BytesIO(Path(path).read_bytes())).load()
