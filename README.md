# LHR — 범용 다중 센서 3D 몸 추정 (v9 코드)

어떤 센서 조합이 들어와도 (RGB, 깊이, LiDAR, mmWave 레이더) 방 좌표 13 관절 + 불확실도를 내는 모델.
이 저장소는 **v9 까지 실제로 쓰는 코드만** 담습니다 (데이터·체크포인트·결과 없음).

## 흐름

```
원본 데이터 ─① 어댑터─▶ 팩 (센서 → 방 좌표, 정답 13 관절)
            ─② v3 정합─▶ 바닥 · 레이더/LiDAR 방 변환 · 시차 보정 (정답 안 씀)
            ─③ 캐시─▶ ms_cache_v3b (모든 데이터셋 같은 형식)
            ─④ 학습/평가─▶ ms_train9.py (번역기 pel_phys4 → 토크나이저 → Transformer → ④ 라벨 어댑터(학습만) → ③ 풀이기(평가))
```

| 단계 | 파일 |
|---|---|
| ① 어댑터 (원본 → 팩) | `adapt_mmfi_v02_raw.py`, `calib_depth.py`, `est_gt_scale.py`, `adapt_mmfi_lidar.py`, `adapt_mri.py`, `detect_mri_vitpose.py`, `adapt_mmbody.py`, `adapt_octonet.py`, `calib_octonet.py`, `cache_octonet_mocap.py`, `adapt_itop.py`, `attach_radar_*.py`, `conv_common.py` |
| ② v3 정합 (정답 없이) | `v3/floor_*.py` (바닥), `v3/reg_*`, `v3/fit_radar_mmfi.py`, `v3/build_*` (레이더·LiDAR 방 변환), `v3/mri_radar_*`, `v3/radar_person_select.py`, 검증 `v3/validate_mmfi.py`, `v3/audit_*` |
| ③ 캐시 | `ms_cache3.py` → `ms_cache_v3b/{mmfi,mri,mmbody,octonet,itop}.npz` |
| ④ 학습·평가 | `ms_train9.py` (모델·증강·학습·평가), `pel_phys4.py` (번역기: 물리 단서 → 골반 기준점 + Σ), `subset_eval.py` |
| 설정 | `code/specs/*.json` (센서 사양), `code/splits/*.json` (피험자 분할, mRI dev/final), `notes_config/*.json` (MM-Fi 정답 배율·깊이 보정) |

나머지 파일 (`model_v*.py`, `lhr_*.py`, `phys_features*.py`, `physics_baseline.py`, `xproj_train.py` 등) 은 위 파일들이 함수를 가져다 쓰는 의존 파일입니다.

## 실행 예

```bash
python code/ms_cache3.py                                        # 캐시 v3b
python code/ms_train9.py --hold mri --drop imu,wifi --dump 1 --densfix 0 --out v9a_mri_s0
python code/subset_eval.py runs/ms9/v9a_mri_s0                  # mRI dev 묶음별
```

## 규칙

- mRI 는 봉인된 시험셋: 학습 안 함, 상수 맞춤 안 함, dev 10 / final 10 (`splits/mri_devfinal.json`).
- 정합·캐시 코드는 변환에 정답을 쓰지 않음 (정답은 진단에만).
- 데이터셋 ID·기기 ID 는 망 입력으로 쓰지 않음.

## 주의

- 경로가 작업 환경 기준 절대 경로 (`/home/interns/unisense/...`) 로 적혀 있습니다. 다른 환경에서는 각 파일 위쪽의 `U = ...` 등을 바꿔야 합니다.
- 데이터셋 (MM-Fi, mRI, mmBody, OctoNet, ITOP) 은 각 배포처에서 받아야 합니다.
