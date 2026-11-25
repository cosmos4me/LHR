"""v3 바닥용 원시 깊이 배경 추출 (unicorrn 환경 python -I: PIL·h5py 가 필요해서. mmfi 환경에는 둘 다 없음).
- OctoNet: 팩(octonet_v07/{stamp}.npz)마다 node_1 깊이 zip 에서 40프레임 (고르게) → 화소별 90 백분위·중앙값·시간 MAD [m, 8비트 v/255·10, 배율 a 적용 전]
  + 유효 비율. (바닥 맞춤은 중앙값을 쓴다: 잡음 큰 바닥 화소에서 90 백분위는 먼 쪽으로 치우쳐 바닥이 낮게 보인다 — env 2 에서 확인.) zip 은 풀지 않고 메모리에서 읽는다 (항목 이름 검사: 절대 경로·'..' 거부).
- ITOP: 사람(U00–U03)마다 깊이 프레임 5장 간격 → 화소별 90 백분위·중앙값·시간 MAD [m] + 유효 비율.
출력: dev/packs/v3/floor/raw_bg/{octonet_<stamp>|itop_U0x}.npz (bg float16, valid float16).
정답 파일(라벨 h5, 모캡 CSV)은 열지 않는다.
실행: /home/interns/.conda/envs/unicorrn/bin/python -I floor_extract_raw.py [octonet|itop]"""
import sys, os, io, re, glob, json, zipfile, datetime
from multiprocessing import Pool
import numpy as np

U = "/home/interns/unisense"; RAW_O = f"{U}/dev/data/octonet/raw/node_1"; RAW_I = f"{U}/dev/data/itop/raw"
OUT = f"{U}/dev/packs/v3/floor/raw_bg"
NAME = re.compile(r"^(\d|x)_([A-Za-z]+)_(\d+)_(\d+)_(.+)_(\d+)_(\d{14})\.zip$")


def ok_member(n):
    return not (n.startswith("/") or ".." in n.split("/") or "\\" in n)


def octonet_one(stamp):
    from PIL import Image
    out = f"{OUT}/octonet_{stamp}.npz"
    if os.path.exists(out):
        return stamp, "skip"
    zs = [f for f in os.listdir(RAW_O) if NAME.match(f) and f.startswith("1_depthcam_") and NAME.match(f).group(7) == stamp]
    if len(zs) != 1:
        return stamp, f"zip count {len(zs)}"
    p = os.path.realpath(os.path.join(RAW_O, zs[0]))
    assert p.startswith(os.path.realpath(RAW_O) + os.sep)
    with zipfile.ZipFile(p) as z:
        names = [n for n in z.namelist() if n.endswith(".png") and "/depth_" in n and ok_member(n)]
        ts = [datetime.datetime.strptime(n.rsplit("depth_", 1)[1][:-4], "%Y-%m-%d %H:%M:%S.%f").timestamp() for n in names]
        names = [names[i] for i in np.argsort(ts)]
        D = []
        for i in np.linspace(0, len(names) - 1, min(40, len(names))).astype(int):
            info = z.getinfo(names[i])
            if info.file_size > 20_000_000:
                continue
            try:
                v = np.array(Image.open(io.BytesIO(z.read(names[i])))).astype(np.float32)
            except Exception:
                continue
            if v.shape != (480, 640):
                continue
            d = v / 255.0 * 10.0; d[v == 0] = np.nan; D.append(d)
    if not D:
        return stamp, "no frames"
    D = np.stack(D)
    with np.errstate(all="ignore"):
        bg = np.nanpercentile(D, 90, axis=0); va = np.mean(np.isfinite(D), 0)
        med = np.nanmedian(D, 0); mad = np.nanmedian(np.abs(D - med), 0)
    np.savez_compressed(out, bg=bg.astype(np.float16), med=med.astype(np.float16), mad=mad.astype(np.float16), valid=va.astype(np.float16), n_frames=len(D), src=zs[0])
    return stamp, "ok"


def itop():
    import h5py
    fd = h5py.File(f"{RAW_I}/ITOP_side_test_depth_map.h5", "r")
    ids = np.array([x.decode() for x in fd["id"][:]]); depth = fd["data"]
    for pid in sorted(set(i.split("_")[0] for i in ids)):
        sel = np.where(np.char.startswith(ids, pid + "_"))[0]
        Z = np.asarray(depth[sel[0]:sel[-1] + 1], np.float32)
        Zn = np.where(Z > 0.3, Z, np.nan)
        with np.errstate(all="ignore"):
            bg = np.nanpercentile(Zn[::5], 90, axis=0); va = np.mean(np.isfinite(Zn[::5]), 0)
            med = np.nanmedian(Zn[::5], 0); mad = np.nanmedian(np.abs(Zn[::5] - med), 0)
        full = Zn[np.linspace(0, len(Zn) - 1, 6).astype(int)]
        np.savez_compressed(f"{OUT}/itop_U{int(pid):02d}.npz", bg=bg.astype(np.float16), med=med.astype(np.float16), mad=mad.astype(np.float16), valid=va.astype(np.float16), full=full.astype(np.float16), n_frames=len(Zn[::5]))
        print("itop", pid, len(sel), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "octonet"
    if what == "itop":
        itop()
    else:
        stamps = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{U}/dev/packs/octonet_v07/*.npz"))
        with Pool(16) as pool:
            for i, (s, st) in enumerate(pool.imap_unordered(octonet_one, stamps), 1):
                if st != "ok" or i % 20 == 0:
                    print(i, "/", len(stamps), s, st, flush=True)
    print("EXTRACT_DONE", flush=True)
