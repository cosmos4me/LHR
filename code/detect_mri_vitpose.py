"""mRI 영상에서 독립 2D 키포인트를 뽑는다 (정답을 만든 HRNet이 아닌 검출기: RT-DETR 사람 상자 → ViTPose-B, COCO-17).
영상 프레임 i = 정답 정렬 프레임 i (프레임 수가 같음). 피험자·카메라별로 kp (N,17,2) px, score (N,17), box (N,4 xyxy), box_score 저장.
GPU 0만 사용 (CUDA_VISIBLE_DEVICES=0). 사용: python detect_mri_vitpose.py --cams 0 1 [--subjects 1 2 ...] [--max-frames N]"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, RTDetrForObjectDetection, RTDetrImageProcessor, VitPoseForPoseEstimation

VID = Path("/home/interns/unisense/dev/data/mri/extracted/blurred_videos")
OUT = Path("/home/interns/unisense/dev/data/mri/derived/vitpose_b")
DET_ID, POSE_ID = "PekingU/rtdetr_r50vd_coco_o365", "usyd-community/vitpose-base-simple"
BATCH = 16


def arg_list(name, cast=int):
    a = sys.argv
    if name not in a:
        return None
    i = a.index(name) + 1; out = []
    while i < len(a) and not a[i].startswith("--"):
        out.append(cast(a[i])); i += 1
    return out


def main():
    cams = arg_list("--cams") or [0, 1]
    subjects = arg_list("--subjects")
    max_frames = (arg_list("--max-frames") or [None])[0]
    dev = torch.device("cuda")
    det_proc = RTDetrImageProcessor.from_pretrained(DET_ID)
    det = RTDetrForObjectDetection.from_pretrained(DET_ID, torch_dtype=torch.float16).to(dev).eval()
    pose_proc = AutoProcessor.from_pretrained(POSE_ID)
    pose = VitPoseForPoseEstimation.from_pretrained(POSE_ID, torch_dtype=torch.float16).to(dev).eval()
    vids = sorted(VID.glob("subject*/subject*_color*.mp4"), key=lambda p: (int(p.parent.name.split("_")[0][7:]), p.name))
    for vp in vids:
        subj = int(vp.parent.name.split("_")[0][7:]); cam = int(vp.stem[-1])
        if cam not in cams or (subjects and subj not in subjects):
            continue
        out = OUT / f"subject{subj}_color{cam}.npz"
        if out.exists() and max_frames is None:
            continue
        cap = cv2.VideoCapture(str(vp)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            n = min(n, max_frames)
        KP = np.full((n, 17, 2), np.nan, np.float32); SC = np.zeros((n, 17), np.float32)
        BX = np.full((n, 4), np.nan, np.float32); BS = np.zeros(n, np.float32)
        t0 = time.time(); i = 0
        while i < n:
            imgs = []
            for _ in range(min(BATCH, n - i)):
                ok, fr = cap.read()
                if not ok:
                    break
                imgs.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if not imgs:
                break
            with torch.no_grad():
                di = det_proc(images=imgs, return_tensors="pt").to(dev)
                do = det(pixel_values=di["pixel_values"].half())
                dets = det_proc.post_process_object_detection(do, target_sizes=[(im.shape[0], im.shape[1]) for im in imgs],
                                                             threshold=0.3)
                boxes, keep = [], []
                for k, d in enumerate(dets):
                    m = d["labels"] == 0                                       # 사람
                    if m.any():
                        b = d["boxes"][m].float().cpu().numpy(); s = d["scores"][m].float().cpu().numpy()
                        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                        j = int(np.argmax(area * s))                           # 피험자 한 명: 크고 확실한 상자
                        BX[i + k], BS[i + k] = b[j], s[j]
                        x0, y0, x1, y1 = b[j]
                        boxes.append([[x0, y0, x1 - x0, y1 - y0]]); keep.append(k)
                if keep:
                    pi = pose_proc([imgs[k] for k in keep], boxes=boxes, return_tensors="pt").to(dev)
                    po = pose(pixel_values=pi["pixel_values"].half())
                    po["heatmaps"] = po.heatmaps.float()                       # 후처리(scipy)는 float32만 받음
                    res = pose_proc.post_process_pose_estimation(po, boxes=boxes)
                    for k, r in zip(keep, res):
                        KP[i + k] = r[0]["keypoints"].float().cpu().numpy(); SC[i + k] = r[0]["scores"].float().cpu().numpy()
            i += len(imgs)
            if (i // BATCH) % 100 == 0:
                print(f"{vp.name} {i}/{n} {i / (time.time() - t0):.1f} fps", flush=True)
        cap.release()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, kp_px=KP, kp_score=SC, box_xyxy=BX, box_score=BS, n_frames=n,
                            detector=f"{DET_ID} + {POSE_ID}", video=str(vp))
        print(f"done {out.name} n={n} {n / (time.time() - t0):.1f} fps missing={int(np.isnan(KP[:, 0, 0]).sum())}", flush=True)


main()
