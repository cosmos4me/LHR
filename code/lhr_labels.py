"""LHR 라벨 사전: 데이터셋별 활동·장면 이름을 공통 텍스트 어휘로 바꾼다 (데이터셋별 분류 헤드 대신 텍스트 정렬 라벨).
- 같은 문자열은 데이터셋이 달라도 같은 어휘 항목 (예: squat = MM-Fi A12 = OctoNet squat). 처음 보는 데이터셋의 이름은 학습 없이 어휘에 추가된다 (제로샷).
- 활동: MM-Fi A01–A27 (MM-Fi 표), mRI 라벨 열(T pose, pose_1–10, free_form, walk, T pose 2), OctoNet 20 동작, mmBody 없음(-1).
- 장면: 데이터셋 환경 이름 → 짧은 영어 묘사. 피험자: 신원이 아니라 체형 속성(뼈 길이 12)만 LHR 출력으로 쓴다.
- 텍스트 임베딩은 lhr_text.py가 만든다 (dev/packs/lhr_text_v01.npz)."""
import json
import numpy as np

MMFI_ACT = {1: "stretching and relaxing", 2: "chest expansion horizontal", 3: "chest expansion vertical", 4: "twist to the left", 5: "twist to the right",
            6: "marking time", 7: "limb extension left", 8: "limb extension right", 9: "lunge toward left front", 10: "lunge toward right front",
            11: "limb extension both", 12: "squat", 13: "raising left hand", 14: "raising right hand", 15: "lunge toward left side",
            16: "lunge toward right side", 17: "waving left hand", 18: "waving right hand", 19: "picking up things", 20: "throwing toward left side",
            21: "throwing toward right side", 22: "kicking toward left side", 23: "kicking toward right side", 24: "body extension left",
            25: "body extension right", 26: "jumping up", 27: "bowing"}
MRI_COLS = ["T pose", "pose_1", "pose_2", "pose_3", "pose_4", "pose_5", "pose_6", "pose_7", "pose_8", "pose_9", "pose_10", "free_form", "walk", "T pose 2"]
MRI_ACT = {"T pose": "T pose", "free_form": "free form movement", "walk": "walking", "T pose 2": "T pose"}
MRI_ACT.update({f"pose_{k}": f"rehabilitation exercise {k}" for k in range(1, 11)})       # 공개 문서에 동작 이름 없음 → 번호 (확인되면 바꾼다)
OCTO_ACT = {"kicksomeone": "kicking someone", "squat": "squat", "jumpingjack": "jumping jack", "dance": "dancing", "stretchoneself": "stretching oneself",
            "walk": "walking", "legraise": "leg raise", "jog": "jogging", "sit": "sitting down", "gym": "gym exercise", "pushup": "push up",
            "pickup": "picking up things", "jump": "jumping up", "freestyle": "free form movement", "bowling": "bowling", "bow": "bowing",
            "dodge": "dodging", "falldown": "falling down", "boxing": "boxing", "lunge": "lunge"}
SCENE = {("mmfi", "E01"): "indoor room 1", ("mmfi", "E02"): "indoor room 2", ("mmfi", "E03"): "indoor room 3", ("mmfi", "E04"): "indoor room 4",
         ("mri", None): "indoor room", ("octonet", 1): "indoor environment 1", ("octonet", 2): "indoor environment 2", ("octonet", 3): "indoor environment 3",
         ("mmbody", "train"): "laboratory", ("mmbody", "lab1"): "laboratory 1", ("mmbody", "lab2"): "laboratory 2", ("mmbody", "furnished"): "furnished room",
         ("mmbody", "poor_lighting"): "poorly lit room", ("mmbody", "rain"): "rain", ("mmbody", "smoke"): "smoke filled room", ("mmbody", "occlusion"): "occluded by furniture"}

ACT_VOCAB = sorted(set(MMFI_ACT.values()) | set(MRI_ACT.values()) | set(OCTO_ACT.values()))
SCENE_VOCAB = sorted(set(SCENE.values()))
ACT_ID = {s: i for i, s in enumerate(ACT_VOCAB)}; SCENE_ID = {s: i for i, s in enumerate(SCENE_VOCAB)}
DS_ACT = {"mmfi": sorted({ACT_ID[s] for s in MMFI_ACT.values()}), "mri": sorted({ACT_ID[s] for s in MRI_ACT.values()}),
          "octonet": sorted({ACT_ID[s] for s in OCTO_ACT.values()}), "mmbody": []}
DS_SCENE = {ds: sorted({SCENE_ID[s] for (d, _), s in SCENE.items() if d == ds}) for ds in ("mmfi", "mri", "octonet", "mmbody")}


def frame_labels(ds, P, meta):
    """팩 하나 → (활동 id (T,), 장면 id) ; 없으면 -1."""
    T = int(meta["T"])
    if ds == "mmfi":
        act = np.full(T, ACT_ID[MMFI_ACT[int(meta["action"][1:])]], np.int64); sc = SCENE_ID[SCENE[("mmfi", meta["env"])]]
    elif ds == "mri":
        a = np.asarray(P["action"]); cols = meta.get("actions", MRI_COLS)
        lut = np.array([ACT_ID[MRI_ACT[c]] for c in cols]); act = np.where(a >= 0, lut[np.clip(a, 0, len(lut) - 1)], -1); sc = SCENE_ID[SCENE[("mri", None)]]
    elif ds == "octonet":
        act = np.full(T, ACT_ID[OCTO_ACT[meta["action"]]], np.int64); sc = SCENE_ID[SCENE[("octonet", int(meta["env"]))]]
    elif ds in ("itop", "mmfi_lidar"):                                                      # 평가 전용 (활동·장면 라벨 없음)
        act = np.full(T, -1, np.int64); sc = -1
    else:
        act = np.full(T, -1, np.int64); sc = SCENE_ID[SCENE[("mmbody", meta["env"])]]
    return act, sc


if __name__ == "__main__":
    print(len(ACT_VOCAB), "activities:", ACT_VOCAB); print(len(SCENE_VOCAB), "scenes:", SCENE_VOCAB)
    print({k: len(v) for k, v in DS_ACT.items()}, {k: len(v) for k, v in DS_SCENE.items()})
    shared = [s for s in ACT_VOCAB if sum(s in V for V in (set(MMFI_ACT.values()), set(MRI_ACT.values()), set(OCTO_ACT.values()))) > 1]
    print("shared across datasets:", shared)
