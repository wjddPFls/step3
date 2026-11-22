
from pathlib import Path
import yaml
from ultralytics import YOLO

print("===== 학습 환경 및 설정 확인 =====")

# data.yaml 확인
yaml_path = Path("사용한 데이터/data.yaml")

if not yaml_path.exists():
    raise FileNotFoundError("data.yaml 파일이 없습니다.")

with yaml_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("\n[데이터 설정]")
print("경로:", config["path"])
print("Train:", config["train"])
print("Val:", config["val"])
print("Test:", config["test"])
print("클래스:", config["names"])

# 모델 확인
print("\n[모델 확인]")
model_path = Path("yolov8s-seg.pt")

if not model_path.exists():
    raise FileNotFoundError("yolov8s-seg.pt 파일이 없습니다.")

model = YOLO(str(model_path))
print("YOLOv8s-Seg 모델 로드 성공")

# 학습 환경 확인
import torch

print("\n[실행 환경]")
print("PyTorch:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())
print("학습 장치:", "GPU" if torch.cuda.is_available() else "CPU")

print("\n학습 설정 사전 검사 완료")
