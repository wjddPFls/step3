
from pathlib import Path
import shutil
import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "사용한 데이터"
IMAGE_DIR = DATA_DIR / "images" / "test"
LABEL_DIR = DATA_DIR / "labels" / "test"

BEST_PT = (
    BASE_DIR
    / "runs"
    / "segment"
    / "runs"
    / "step3_small_test"
    / "weights"
    / "best.pt"
)

THRESHOLDS = {
    0: 0.001,
    1: 0.001,
    2: 0.01,
}

CLASS_NAMES = {
    0: "crack",
    1: "efflorescence",
    2: "spall_detach",
}

IMG_SIZE = 640
OUTPUT_DIR = BASE_DIR / "stage17_error_cases"


def read_gt_classes(label_path):
    classes = set()

    if not label_path.exists():
        return classes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            try:
                classes.add(int(float(parts[0])))
            except ValueError:
                continue

    return classes


def save_error_image(image_path, error_type, class_name,
                     gt_classes, pred_classes):
    image = cv2.imread(str(image_path))

    if image is None:
        print("이미지 읽기 실패:", image_path)
        return

    gt_text = ", ".join(
        CLASS_NAMES[c] for c in sorted(gt_classes)
    ) or "None"

    pred_text = ", ".join(
        CLASS_NAMES[c] for c in sorted(pred_classes)
    ) or "None"

    text_lines = [
        f"{error_type}: {class_name}",
        f"GT: {gt_text}",
        f"Pred: {pred_text}",
    ]

    y = 30

    for text in text_lines:
        cv2.putText(
            image,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    save_dir = OUTPUT_DIR / class_name / error_type
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / image_path.name
    cv2.imwrite(str(save_path), image)


def main():
    print("===== 테스트셋 FP/FN 사례 추출 =====")

    image_paths = sorted(
        list(IMAGE_DIR.glob("*.jpg"))
        + list(IMAGE_DIR.glob("*.jpeg"))
        + list(IMAGE_DIR.glob("*.png"))
    )

    print("테스트 이미지 수:", len(image_paths))
    print("모델 로드 중...")

    model = YOLO(str(BEST_PT))

    error_counts = {
        class_id: {"FP": 0, "FN": 0}
        for class_id in CLASS_NAMES
    }

    for index, image_path in enumerate(image_paths, start=1):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        gt_classes = read_gt_classes(label_path)

        result = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            verbose=False,
        )[0]

        pred_classes = set()

        if result.boxes is not None and len(result.boxes) > 0:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for class_id, confidence in zip(
                class_ids, confidences
            ):
                if (
                    class_id in THRESHOLDS
                    and confidence >= THRESHOLDS[class_id]
                ):
                    pred_classes.add(class_id)

        for class_id, class_name in CLASS_NAMES.items():
            gt_on = class_id in gt_classes
            pred_on = class_id in pred_classes

            if pred_on and not gt_on:
                error_counts[class_id]["FP"] += 1

                save_error_image(
                    image_path,
                    "FP",
                    class_name,
                    gt_classes,
                    pred_classes,
                )

            elif gt_on and not pred_on:
                error_counts[class_id]["FN"] += 1

                save_error_image(
                    image_path,
                    "FN",
                    class_name,
                    gt_classes,
                    pred_classes,
                )

        print(f"진행: {index}/{len(image_paths)}")

    print("\n===== 오류 사례 개수 =====")

    for class_id, class_name in CLASS_NAMES.items():
        print(f"\n[{class_name}]")
        print("FP:", error_counts[class_id]["FP"])
        print("FN:", error_counts[class_id]["FN"])

    print("\n저장 위치:", OUTPUT_DIR)
    print("===== FP/FN 사례 추출 완료 =====")


if __name__ == "__main__":
    main()
