
from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("사용한 데이터")
IMAGE_DIR = DATA_DIR / "images" / "train"
LABEL_DIR = DATA_DIR / "labels" / "train"

CLASS_NAMES = {
    0: "crack",
    1: "efflorescence",
    2: "spall_detach"
}

CLASS_COLORS = {
    0: (255, 0, 0),
    1: (0, 180, 0),
    2: (0, 100, 255)
}

image_paths = sorted(IMAGE_DIR.rglob("*.jpg"))

random.seed(42)
selected_images = random.sample(image_paths, min(30, len(image_paths)))

fig, axes = plt.subplots(5, 6, figsize=(20, 15))

for ax in axes.flat:
    ax.axis("off")

for ax, image_path in zip(axes.flat, selected_images):

    image_array = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        continue

    height, width = image.shape[:2]
    label_path = LABEL_DIR / (image_path.stem + ".txt")

    # 라벨 표시
    if label_path.exists():
        lines = label_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()

        for line in lines:
            values = list(map(float, line.split()))

            if len(values) < 7:
                continue

            class_id = int(values[0])
            points = np.array(values[1:]).reshape(-1, 2)

            points[:, 0] *= width
            points[:, 1] *= height
            points = points.astype(np.int32)

            polygon = points.reshape((-1, 1, 2))
            color = CLASS_COLORS.get(class_id, (255, 255, 0))

            cv2.polylines(
                image,
                [polygon],
                True,
                color,
                2
            )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.imshow(image)
    ax.set_title(image_path.name, fontsize=6)
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "hard_negative_review.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("확인 이미지 수:", len(selected_images))
print("저장 완료: hard_negative_review.png")
print("\n색상 안내")
print("빨강: crack")
print("초록: efflorescence")
print("파랑: spall_detach")
