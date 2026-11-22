
from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 데이터 경로
DATA_DIR = Path("사용한 데이터")
IMAGE_DIR = DATA_DIR / "images" / "train"

# 이미지 파일 목록
image_paths = sorted(IMAGE_DIR.rglob("*.jpg"))

print(f"전체 학습 이미지 수: {len(image_paths)}개")

if len(image_paths) < 9:
    raise ValueError("이미지가 9장보다 적습니다.")

# 이미지 9장 무작위 선택
random.seed(42)
selected_images = random.sample(image_paths, 9)

# 3행 3열 이미지 출력
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

success_count = 0

for ax, image_path in zip(axes.flat, selected_images):
    try:
        # 한글 경로에서도 이미지 읽기
        image_array = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("이미지 디코딩 실패")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax.imshow(image)
        ax.set_title(image_path.name, fontsize=8)
        success_count += 1

    except Exception as e:
        ax.set_title("Image load failed")
        print(f"오류 파일: {image_path}")
        print(f"오류 내용: {e}")

    ax.axis("off")

plt.tight_layout()

# 결과 저장
output_path = Path("samples9.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"이미지 로딩 성공: {success_count}/9")
print(f"저장 위치: {output_path.resolve()}")
