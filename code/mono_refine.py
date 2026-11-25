"""① 단안 RGB 번역기 v3 후보 (2026-11-10): 발목 광선–바닥 + 정면 평면 해 (v2) 의 프레임별 거리 오차를 줄인다. 정답 없음, 사전 상수만.
근거 (mRI 카메라 1대, train): 골반 거리 방향 오차의 표준편차 127 mm 중 시퀀스 평균의 편차는 27 mm, 시퀀스 안 (프레임별) 이 126 mm → 사람은 그렇게 빨리 움직이지 않으므로 시간·크기 일관성으로 줄일 수 있다.
방법: 카메라 중심 o 에서 한 프레임의 모든 관절을 같은 배율 g_t 로 민다 (X' = o + g_t (X − o)). 2D 투영은 그대로이고 거리만 바뀐다.
변형 (이름을 + 로 이으면 차례로 적용):
- t<N>: 기준점 거리의 시간 중앙값, 창 N/10 초 (t5 = 0.5 s, t15 = 1.5 s, t31 = 3.1 s)
- szmed / szmin2: 몸 크기는 한 시퀀스에서 일정 → 몸통 6선분 (어깨·엉덩이 너비, 양 옆, 대각선) 의 겉보기 길이를 시퀀스 중앙값에 맞춘다. 선분별 비 (기준 / 겉보기) 의 중앙값 또는 둘째로 작은 값 (원근 단축은 겉보기를 줄이기만 하므로 작은 쪽이 덜 단축된 선분)
계산은 원래 프레임 속도 (step 1) 에서 하고 그 뒤에 줄인다 (lhr_data.build / build_subrig, 환경 LHR_MONO)."""
import warnings
import numpy as np
TORSO = [(1, 2), (7, 8), (1, 7), (2, 8), (1, 8), (2, 7)]; G_LO, G_HI = 0.7, 1.4


def tmed(v, win):
    """(T,) NaN 무시 가운데 정렬 중앙값, 창 win (홀수)."""
    h = win // 2; W = np.lib.stride_tricks.sliding_window_view(np.pad(v, (h, h), constant_values=np.nan), win)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); return np.nanmedian(W, 1)


def refine(x, a, o, fps, variant):
    """x (T,13,3), a (T,3) (NaN 가능), o (3,) 카메라 중심, fps → 새 x, a, 배율 g (T,)."""
    x, a, o = np.asarray(x, float), np.asarray(a, float), np.asarray(o, float); g = np.ones(len(x))
    for step in [s for s in variant.split("+") if s]:
        xs, as_ = o + g[:, None, None] * (x - o), o + g[:, None] * (a - o)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if step.startswith("sz"):
                L = np.stack([np.linalg.norm(xs[:, i] - xs[:, j], axis=-1) for i, j in TORSO], 1); L = np.where(L > 0.05, L, np.nan); r = np.nanmedian(L, 0)[None] / L
                if step == "szmed":
                    f = np.nanmedian(r, 1)
                else:
                    rs = np.sort(r, 1); f = np.where(np.isfinite(rs[:, 1]), rs[:, 1], rs[:, 0])           # NaN 은 정렬에서 뒤로 간다
            else:
                win = max(3, int(round(int(step[1:]) / 10 * fps)) | 1); rho = np.linalg.norm(as_ - o, axis=-1); f = tmed(rho, win) / np.where(rho > 0.3, rho, np.nan)
        g = g * np.where(np.isfinite(f), np.clip(f, G_LO, G_HI), 1.0)
    return (o + g[:, None, None] * (x - o)).astype(np.float32), (o + g[:, None] * (a - o)).astype(np.float32), g
