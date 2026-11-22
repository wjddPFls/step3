
from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 데이터 경로
DATA_DIR = Path("사용한 데이터")
IMAGE_DIR = DATA_DIR / "images" / "train"
LABEL_DIR = DATA_DIR / "labels" / "train"

# 클래스 이름
CLASS_NAMES = {
    0: "crack",
    1: "efflorescence",
    2: "spall_detach"
}

# 이미지 목록
image_paths = sorted(IMAGE_DIR.rglob("*.jpg"))

# 이전 실습과 동일한 이미지 9장 선택
random.seed(42)
selected_images = random.sample(image_paths, 9)

fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for ax, image_path in zip(axes.flat, selected_images):

    image_array = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        print(f"이미지 읽기 실패: {image_path.name}")
        continue

    height, width = image.shape[:2]
    label_path = LABEL_DIR / (image_path.stem + ".txt")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("\n" + "=" * 60)
    print("이미지:", image_path.name)

    if not label_path.exists():
        print("라벨 파일 없음")
        ax.imshow(image)
        ax.set_title("Label missing", fontsize=8)
        ax.axis("off")
        continue

    lines = label_path.read_text(encoding="utf-8").splitlines()

    if not lines:
        print("라벨 내용 없음: 주석된 결함이 없는 이미지")
        ax.imshow(image)
        ax.set_title("No annotation", fontsize=8)
        ax.axis("off")
        continue

    for line in lines:
        values = list(map(float, line.split()))

        class_id = int(values[0])
        points = np.array(values[1:]).reshape(-1, 2)

        # 정규화 좌표를 이미지 픽셀 좌표로 변환
        points[:, 0] *= width
        points[:, 1] *= height
        points = points.astype(np.int32)

        polygon = points.reshape((-1, 1, 2))

        cv2.polylines(
            image,
            [polygon],
            isClosed=True,
            color=(255, 0, 0),
            thickness=3
        )

        class_name = CLASS_NAMES.get(class_id, "unknown")

        print(f"클래스: {class_name} (ID: {class_id})")

    ax.imshow(image)
    ax.set_title(image_path.name, fontsize=7)
    ax.axis("off")

plt.tight_layout()

plt.savefig("label_check9.png", dpi=200, bbox_inches="tight")
plt.show()

print("\n라벨 확인 이미지 저장 완료: label_check9.png")
