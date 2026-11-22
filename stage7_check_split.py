
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("사용한 데이터")
IMAGE_DIR = DATA_DIR / "images"

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_origin_id(stem):
    # Roboflow 파일명의 해시 부분 제거
    origin_id = stem.split("_jpg.rf.")[0]

    # 증강 이미지의 원본 ID 추출
    origin_id = origin_id.split("_aug")[0]

    return origin_id


groups = defaultdict(set)
split_counts = {}

for split in SPLITS:
    image_dir = IMAGE_DIR / split

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    split_counts[split] = len(images)

    for image in images:
        origin_id = get_origin_id(image.stem)
        groups[origin_id].add(split)

    print(f"{split.upper()} 이미지 수: {len(images)}")

# 여러 split에 동시에 포함된 원본 ID 확인
overlap = {
    origin_id: splits
    for origin_id, splits in groups.items()
    if len(splits) > 1
}

print("\n===== 원본 ID 기준 분할 검사 =====")
print("전체 원본 ID 수:", len(groups))
print("여러 split에 중복된 원본 ID:", len(overlap))

if overlap:
    print("\n중복된 원본 ID 예시:")
    for origin_id, splits in list(overlap.items())[:10]:
        print(origin_id, "->", sorted(splits))
else:
    print("\n원본 ID 분할 검사 통과")
