
from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("사용한 데이터")
SPLITS = ["train", "val", "test"]

empty_images = []
missing_labels = []

# 라벨이 비어 있는 이미지 찾기
for split in SPLITS:
    image_dir = DATA_DIR / "images" / split
    label_dir = DATA_DIR / "labels" / split

    images = sorted(image_dir.rglob("*.jpg"))
    empty_count = 0
    missing_count = 0

    for image_path in images:
        label_path = label_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            missing_count += 1
            missing_labels.append(image_path)
            continue

        content = label_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).strip()

        if content == "":
            empty_count += 1

            if split == "train":
                empty_images.append(image_path)

    print(f"\n[{split}]")
    print(f"전체 이미지: {len(images)}개")
    print(f"빈 라벨: {empty_count}개")
    print(f"라벨 파일 없음: {missing_count}개")

# 후보 목록 저장
with open("hard_negative_candidates.txt", "w", encoding="utf-8") as f:
    for path in empty_images:
        f.write(str(path) + "\n")

print("\n===== Hard Negative 후보 =====")
print(f"학습 데이터의 빈 라벨 이미지: {len(empty_images)}개")

# 빈 라벨 이미지가 있는 경우 최대 9장 출력
if empty_images:
    random.seed(42)
    selected = random.sample(
        empty_images,
        min(9, len(empty_images))
    )

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for ax in axes.flat:
        ax.axis("off")

    for ax, image_path in zip(axes.flat, selected):
        image_array = np.fromfile(
            str(image_path),
            dtype=np.uint8
        )
        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is not None:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
            ax.imshow(image)
            ax.set_xlabel(image_path.name, fontsize=7)

    plt.tight_layout()
    plt.savefig(
        "hard_negative_candidates.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.show()

    print("후보 이미지 저장 완료")
else:
    print("빈 라벨 이미지가 없습니다.")

print("후보 목록 저장: hard_negative_candidates.txt")
