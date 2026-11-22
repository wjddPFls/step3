
from pathlib import Path
from ultralytics import YOLO

# 프로젝트 폴더 기준 경로
BASE_DIR = Path.cwd()

DATA_YAML = BASE_DIR / "사용한 데이터" / "data.yaml"
BEST_PT = (
    BASE_DIR
    / "runs"
    / "segment"
    / "runs"
    / "step3_small_test"
    / "weights"
    / "best.pt"
)

# 파일 확인
if not DATA_YAML.exists():
    raise FileNotFoundError(f"data.yaml을 찾을 수 없습니다: {DATA_YAML}")

if not BEST_PT.exists():
    raise FileNotFoundError(f"best.pt를 찾을 수 없습니다: {BEST_PT}")

print("===== STEP 3 테스트셋 성능 평가 =====")
print("데이터 설정:", DATA_YAML)
print("학습 모델:", BEST_PT)

# YOLOv8s-Seg 모델 불러오기
model = YOLO(str(BEST_PT))

# 테스트셋 평가
results = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    plots=True,
    project="runs/segment",
    name="step3_week3_test_eval",
    exist_ok=True
)

# 전체 세그멘테이션 성능
print("\n===== 전체 Mask 성능 =====")
print("mask mAP50:", round(results.seg.map50, 4))
print("mask mAP50-95:", round(results.seg.map, 4))

# 클래스별 세그멘테이션 성능
class_names = [
    "crack",
    "efflorescence",
    "spall_detach"
]

print("\n===== 클래스별 Mask mAP50-95 =====")

for i, name in enumerate(class_names):
    print(name, ":", round(results.seg.maps[i], 4))

print("\n===== 테스트셋 평가 완료 =====")
