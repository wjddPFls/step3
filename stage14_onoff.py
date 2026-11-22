
from pathlib import Path
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "사용한 데이터"

IMAGE_DIR = DATA_DIR / "images" / "val"
LABEL_DIR = DATA_DIR / "labels" / "val"

BEST_PT = (
    BASE_DIR
    / "runs"
    / "segment"
    / "runs"
    / "step3_small_test"
    / "weights"
    / "best.pt"
)

IMG_SIZE = 640
THRESHOLD = 0.5

class_names = [
    "crack",
    "efflorescence",
    "spall_detach"
]

# 파일 확인
if not BEST_PT.exists():
    raise FileNotFoundError(f"best.pt를 찾을 수 없습니다: {BEST_PT}")

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {IMAGE_DIR}")

if not LABEL_DIR.exists():
    raise FileNotFoundError(f"라벨 폴더를 찾을 수 없습니다: {LABEL_DIR}")

# 테스트할 이미지 목록 준비
image_extensions = {".jpg", ".jpeg", ".png"}

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
)

# 이미지별 정답 클래스 목록 생성
gts = []

for image_path in image_paths:
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    gt_classes = []

    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()

            if parts:
                gt_classes.append(int(parts[0]))

    gts.append(set(gt_classes))

print("검증 이미지 수:", len(image_paths))

# 모델 불러오기
model = YOLO(str(BEST_PT))

def onoff(model, paths, gts, thr, cid_target):
    tp = fp = fn = tn = 0

    for p, gt in zip(paths, gts):
        result = model.predict(
            str(p),
            imgsz=IMG_SIZE,
            verbose=False
        )[0]

        if result.boxes is None:
            conf = np.array([])
            cid = np.array([], dtype=int)
        else:
            conf = result.boxes.conf.cpu().numpy()
            cid = result.boxes.cls.cpu().numpy().astype(int)

        # 해당 클래스 예측 중 threshold 이상이 있으면 ON
        pred_on = bool(
            ((cid == cid_target) & (conf >= thr)).any()
        )

        # 정답 라벨에 해당 클래스가 있으면 ON
        gt_on = cid_target in gt

        if gt_on and pred_on:
            tp += 1
        elif not gt_on and pred_on:
            fp += 1
        elif gt_on and not pred_on:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

print("\n===== ON/OFF 평가: threshold 0.5 =====")

for cid, name in enumerate(class_names):
    result = onoff(
        model,
        image_paths,
        gts,
        THRESHOLD,
        cid
    )

    print(f"\n[{name}]")
    print("TP:", result["tp"])
    print("FP:", result["fp"])
    print("FN:", result["fn"])
    print("TN:", result["tn"])
    print("Precision:", round(result["precision"], 4))
    print("Recall:", round(result["recall"], 4))
    print("F1:", round(result["f1"], 4))

print("\n===== ON/OFF 평가 함수 확인 완료 =====")