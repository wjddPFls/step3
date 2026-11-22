# STEP 3 - 건설현장 구조물 결함 탐지 프로젝트

YOLOv8s-Seg 모델을 활용하여 건설현장 구조물 이미지에서 균열, 백태, 박리 및 박락을 탐지하고 세그멘테이션 성능을 평가하는 프로젝트입니다.

## 1. 프로젝트 목표

건설현장 구조물의 표면 결함을 자동으로 탐지하고, 결함 영역을 분할하는 모델을 구현합니다.

탐지 대상은 다음과 같이 총 3개 클래스입니다.

| 클래스 | 설명 |
|---|---|
| crack | 균열 |
| efflorescence | 백태 |
| spall_detach | 박리 및 박락 |

## 2. 개발 환경

| 항목 | 환경 |
|---|---|
| OS | Windows |
| Language | Python 3.13.14 |
| Model | YOLOv8s-Seg |
| Framework | PyTorch 2.15.0+cpu |
| Ultralytics | 8.4.203 |
| Device | CPU |

## 3. 데이터셋

데이터셋은 학습, 검증, 테스트 세트로 구성하였습니다.

| 구분 | 이미지 수 |
|---|---:|
| Train | 4,036 |
| Validation | 40 |
| Test | 39 |
| Total | 4,115 |

데이터 분할 과정에서 원본 이미지 ID 중복 여부를 검사하였으며, 중복 데이터 1건을 정리한 후 최종 중복 검사 결과 0건을 확인하였습니다.

※ 데이터셋 원본은 용량 문제로 GitHub 저장소에 포함하지 않았습니다.

## 4. 프로젝트 진행 과정

1. 데이터셋 구조 및 이미지 확인
2. 라벨 형식과 무결성 검사
3. 빈 라벨 및 중복 데이터 확인
4. 데이터 분할 및 증강 상태 확인
5. YOLOv8s-Seg 학습 환경 확인
6. 소규모 테스트 학습 진행
7. 테스트 데이터 성능 평가
8. 클래스별 Confidence Threshold 탐색
9. 테스트 데이터 ON/OFF 평가
10. 오탐 및 미탐 사례 추출과 분석

## 5. 학습 설정

CPU 환경에서 전체 데이터 학습에 많은 시간이 소요되어 파이프라인 확인을 위한 소규모 테스트 학습을 진행하였습니다.

| 항목 | 설정 |
|---|---|
| Model | YOLOv8s-Seg |
| Training Data | 20장 (0.5%) |
| Epochs | 3 |
| Image Size | 640 |
| Batch Size | 2 |
| Device | CPU |

※ 전체 데이터셋을 이용한 최종 학습이 아닌 소규모 테스트 학습입니다.

## 6. 테스트 평가 결과

테스트 데이터 39장, 총 46개 결함 인스턴스를 대상으로 평가하였습니다.

| 평가 항목 | 결과 |
|---|---:|
| Mask mAP50 | 0.000 |
| Mask mAP50-95 | 0.000 |
| Macro F1 | 0.0476 |

### 클래스별 ON/OFF 평가

| 클래스 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| crack | 0 | 0 | 13 | 0.0000 |
| efflorescence | 0 | 1 | 13 | 0.0000 |
| spall_detach | 1 | 0 | 12 | 0.1429 |

검증 데이터에서 클래스별 Threshold를 탐색한 후, 해당 값을 테스트 데이터에 적용하여 평가하였습니다.

## 7. 오류 사례 분석

테스트 데이터에서 오탐(False Positive)과 미탐(False Negative) 사례를 추출하였습니다.

| 클래스 | 오탐(FP) | 미탐(FN) |
|---|---:|---:|
| crack | 0 | 13 |
| efflorescence | 1 | 13 |
| spall_detach | 0 | 12 |

오류 사례 분석을 통해 다음과 같은 개선 필요성을 확인하였습니다.

- 균열의 굵기와 대비가 다양한 이미지 보강
- 백태와 박리·박락의 시각적 특징 구분
- 결함 경계가 모호한 이미지의 라벨링 기준 검토
- 오탐 및 미탐 사례를 활용한 데이터 보강

## 8. 주요 파일

| 파일 | 설명 |
|---|---|
| `stage1_inspect.py` ~ `stage9_remove_overlap.py` | 데이터셋 확인 및 정리 |
| `stage10_check_augmentation.py` | 데이터 증강 상태 확인 |
| `stage11_check_training_setup.py` | 학습 환경 확인 |
| `stage12_train_test.py` | 모델 학습 테스트 |
| `stage13_eval.py` | 테스트 데이터 성능 평가 |
| `stage14_onoff.py` | ON/OFF 평가 |
| `stage15_threshold_search.py` | Threshold 탐색 |
| `stage16_test_onoff.py` | 테스트 데이터 ON/OFF 평가 |
| `stage17_error_cases.py` | 오탐 및 미탐 사례 추출 |

### 주요 결과 파일

- `stage15_best_thresholds.csv`
- `stage15_threshold_results.csv`
- `stage15_val_threshold_f1.png`
- `stage16_test_onoff_results.csv`
- `stage17_error_cases/`

## 9. 결론

YOLOv8s-Seg를 활용하여 건설현장 구조물의 결함 탐지 및 세그멘테이션 실습을 진행하였습니다.

데이터셋 검사, 중복 데이터 정리, 소규모 학습, 성능 평가, Threshold 탐색 및 오류 사례 분석을 수행하였습니다.

소규모 학습으로 인해 테스트 성능은 낮게 나타났으며, 이를 통해 학습 데이터 보강과 라벨링 기준 개선의 필요성을 확인하였습니다.