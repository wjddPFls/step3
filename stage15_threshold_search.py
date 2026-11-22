
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "사용한 데이터" / "images" / "val"
LABEL_DIR = BASE_DIR / "사용한 데이터" / "labels" / "val"

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
CLASSES = ["crack", "efflorescence", "spall_detach"]

THRESHOLDS = [0.001, 0.004, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3]


def read_gt_classes(label_path):
    classes = set()

    if not label_path.exists():
        return classes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if parts:
                try:
                    classes.add(int(parts[0]))
                except ValueError:
                    pass

    return classes


def calculate_metrics(gt_on_list, pred_on_list):
    tp = fp = fn = tn = 0

    for gt_on, pred_on in zip(gt_on_list, pred_on_list):
        if gt_on and pred_on:
            tp += 1
        elif not gt_on and pred_on:
            fp += 1
        elif gt_on and not pred_on:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return tp, fp, fn, tn, precision, recall, f1


def main():
    print("===== 검증셋 Threshold 탐색 =====")

    if not BEST_PT.exists():
        print("모델 파일을 찾을 수 없습니다:")
        print(BEST_PT)
        return

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    if not image_paths:
        print("검증 이미지가 없습니다.")
        return

    print("검증 이미지 수:", len(image_paths))
    print("모델 로드 중...")

    model = YOLO(str(BEST_PT))

    # 이미지별 실제 클래스 ON/OFF 정답
    gt_by_image = []

    # 이미지별 클래스 최대 confidence
    max_conf_by_image = []

    print("검증 이미지 예측 중...")

    for i, image_path in enumerate(image_paths, start=1):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        gt_classes = read_gt_classes(label_path)
        gt_by_image.append(gt_classes)

        result = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=0.001,
            verbose=False
        )[0]

        image_max_conf = {0: 0.0, 1: 0.0, 2: 0.0}

        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()

            for confidence, class_id in zip(confidences, class_ids):
                class_id = int(class_id)

                if class_id in image_max_conf:
                    image_max_conf[class_id] = max(
                        image_max_conf[class_id],
                        float(confidence)
                    )

        max_conf_by_image.append(image_max_conf)

        print(f"진행: {i}/{len(image_paths)}")

    # threshold별 클래스 성능 계산
    all_results = []
    best_thresholds = {}

    for class_id, class_name in enumerate(CLASSES):
        gt_on_list = [
            class_id in gt_classes
            for gt_classes in gt_by_image
        ]

        class_results = []

        for threshold in THRESHOLDS:
            pred_on_list = [
                image_conf[class_id] >= threshold
                for image_conf in max_conf_by_image
            ]

            tp, fp, fn, tn, precision, recall, f1 = calculate_metrics(
                gt_on_list,
                pred_on_list
            )

            row = {
                "class": class_name,
                "threshold": threshold,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "precision": precision,
                "recall": recall,
                "F1": f1
            }

            class_results.append(row)
            all_results.append(row)

        # F1이 가장 높은 threshold 선택
        # F1이 같으면 더 낮은 threshold 선택
        best_row = sorted(
            class_results,
            key=lambda row: (-row["F1"], row["threshold"])
        )[0]

        best_thresholds[class_name] = best_row["threshold"]

        print()
        print(f"[{class_name}]")
        print("선택 threshold:", best_row["threshold"])
        print("F1:", round(best_row["F1"], 4))
        print(
            "TP:", best_row["TP"],
            "FP:", best_row["FP"],
            "FN:", best_row["FN"],
            "TN:", best_row["TN"]
        )

    # 전체 threshold 결과 CSV 저장
    csv_path = BASE_DIR / "stage15_threshold_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class", "threshold", "TP", "FP", "FN", "TN",
                "precision", "recall", "F1"
            ]
        )
        writer.writeheader()
        writer.writerows(all_results)

    # 선택된 threshold 저장
    threshold_path = BASE_DIR / "stage15_best_thresholds.csv"

    with open(threshold_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "selected_threshold"])

        for class_name, threshold in best_thresholds.items():
            writer.writerow([class_name, threshold])

    # 클래스별 F1 그래프 저장
    plt.figure(figsize=(9, 6))

    for class_id, class_name in enumerate(CLASSES):
        class_rows = [
            row for row in all_results
            if row["class"] == class_name
        ]

        x = [row["threshold"] for row in class_rows]
        y = [row["F1"] for row in class_rows]

        plt.plot(x, y, marker="o", label=class_name)

    plt.xscale("log")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 by Confidence Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = BASE_DIR / "stage15_val_threshold_f1.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print()
    print("===== Threshold 탐색 완료 =====")
    print("전체 결과:", csv_path)
    print("선택 threshold:", threshold_path)
    print("F1 그래프:", plot_path)


if __name__ == "__main__":
    main()
