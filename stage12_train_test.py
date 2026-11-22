
from ultralytics import YOLO

model = YOLO("yolov8s-seg.pt")

results = model.train(
    data="사용한 데이터/data.yaml",
    epochs=3,
    imgsz=640,
    batch=2,
    optimizer="AdamW",
    lr0=0.00015,
    mosaic=0.0,
    mixup=0.0,
    patience=20,
    seed=42,
    device="cpu",
    workers=0,
    fraction=0.005,
    project="runs",
    name="step3_small_test",
    exist_ok=True
)

print("===== 소규모 테스트 학습 완료 =====")
