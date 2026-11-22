
from pathlib import Path
import csv
import numpy as np
from ultralytics import YOLO

# 경로 설정
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

# 검증셋에서 선택한 threshold를 그대로 사용
THRESHOLDS = {
    0: 0.001,  # crack
    1: 0.001,  # efflorescence
    2: 0.01,   # spall_detach
}

CLASS_NAMES = {
    0: "crack",
    1: "efflorescence",
    2: "spall_detach",
}

IMG_SIZE = 640


def read_gt_classes(label_path):
    """라벨 파일에서 이미지에 존재하는 클래스 목록을 읽음."""
    classes = set()

    if not label_path.exists():
        return classes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            try:
                class_id = int(float(parts[0]))
                classes.add(class_id)
            except ValueError:
                continue

    return classes


def calculate_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1


def main():
    print("===== 테스트셋 ON/OFF 평가 =====")
    print("테스트 이미지 경로:", IMAGE_DIR)
    print("학습 모델:", BEST_PT)

    image_paths = sorted(
        list(IMAGE_DIR.glob("*.jpg"))
        + list(IMAGE_DIR.glob("*.jpeg"))
        + list(IMAGE_DIR.glob("*.png"))
    )

    if not image_paths:
        print("테스트 이미지를 찾을 수 없습니다.")
        return

    print("테스트 이미지 수:", len(image_paths))
    print("모델 로드 중...")

    model = YOLO(str(BEST_PT))

    # 클래스별 TP, FP, FN, TN 저장
    counts = {
        class_id: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for class_id in CLASS_NAMES
    }

    print("\n테스트 이미지 예측 중...")

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
            pred_class_ids = result.boxes.cls.cpu().numpy().astype(int)
            pred_confidences = result.boxes.conf.cpu().numpy()

            for class_id, confidence in zip(
                pred_class_ids, pred_confidences
            ):
                if class_id in THRESHOLDS:
                    if confidence >= THRESHOLDS[class_id]:
                        pred_classes.add(class_id)

        for class_id in CLASS_NAMES:
            gt_on = class_id in gt_classes
            pred_on = class_id in pred_classes

            if gt_on and pred_on:
                counts[class_id]["TP"] += 1
            elif not gt_on and pred_on:
                counts[class_id]["FP"] += 1
            elif gt_on and not pred_on:
                counts[class_id]["FN"] += 1
            else:
                counts[class_id]["TN"] += 1

        print(f"진행: {index}/{len(image_paths)}")

    # 결과 출력 및 CSV 저장
    output_path = BASE_DIR / "stage16_test_onoff_results.csv"

    f1_scores = []

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Class",
            "Threshold",
            "TP",
            "FP",
            "FN",
            "TN",
            "Precision",
            "Recall",
            "F1",
        ])

        print("\n===== 테스트셋 최종 ON/OFF 결과 =====")

        for class_id, class_name in CLASS_NAMES.items():
            tp = counts[class_id]["TP"]
            fp = counts[class_id]["FP"]
            fn = counts[class_id]["FN"]
            tn = counts[class_id]["TN"]

            precision, recall, f1 = calculate_metrics(
                tp, fp, fn, tn
            )

            f1_scores.append(f1)

            print(f"\n[{class_name}]")
            print("Threshold:", THRESHOLDS[class_id])
            print(f"TP: {tp} FP: {fp} FN: {fn} TN: {tn}")
            print("Precision:", round(precision, 4))
            print("Recall:", round(recall, 4))
            print("F1:", round(f1, 4))

            writer.writerow([
                class_name,
                THRESHOLDS[class_id],
                tp,
                fp,
                fn,
                tn,
                round(precision, 6),
                round(recall, 6),
                round(f1, 6),
            ])

        macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        writer.writerow([])
        writer.writerow(["Macro F1", round(macro_f1, 6)])

    print("\nMacro F1:", round(macro_f1, 4))
    print("결과 CSV 저장:", output_path)
    print("\n===== 테스트셋 평가 완료 =====")


if __name__ == "__main__":
    main()
