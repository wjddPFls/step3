
from pathlib import Path

DATA_DIR = Path("사용한 데이터")
IMAGE_DIR = DATA_DIR / "images"
LABEL_DIR = DATA_DIR / "labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "val", "test"]

total_images = 0
total_labels = 0
total_instances = 0

missing_labels = []
orphan_labels = []
empty_labels = []
bad_format = []
bad_class = []
bad_coordinates = []

for split in SPLITS:
    image_dir = IMAGE_DIR / split
    label_dir = LABEL_DIR / split

    images = {
        p.stem: p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    labels = {
        p.stem: p
        for p in label_dir.rglob("*.txt")
    }

    total_images += len(images)
    total_labels += len(labels)

    # 이미지와 라벨 파일 대응 확인
    for stem in images:
        if stem not in labels:
            missing_labels.append((split, stem))

    for stem in labels:
        if stem not in images:
            orphan_labels.append((split, stem))

    # 라벨 내용 검사
    for path in labels.values():
        lines = path.read_text(encoding="utf-8").splitlines()
        valid_lines = [line.split() for line in lines if line.strip()]

        if not valid_lines:
            empty_labels.append(path)
            continue

        for line_number, values in enumerate(valid_lines, start=1):
            total_instances += 1

            if len(values) < 7 or (len(values) - 1) % 2 != 0:
                bad_format.append((path, line_number))
                continue

            try:
                class_id = int(values[0])
                if class_id not in [0, 1, 2]:
                    bad_class.append((path, line_number, values[0]))
            except ValueError:
                bad_class.append((path, line_number, values[0]))
                continue

            try:
                coordinates = [float(v) for v in values[1:]]
                if not all(0 <= v <= 1 for v in coordinates):
                    bad_coordinates.append((path, line_number))
            except ValueError:
                bad_coordinates.append((path, line_number))

    print(f"\n===== {split.upper()} 검사 =====")
    print("이미지:", len(images))
    print("라벨:", len(labels))
    print("라벨 누락:", sum(1 for s, _ in missing_labels if s == split))
    print("이미지 없는 라벨:", sum(1 for s, _ in orphan_labels if s == split))

print("\n===== 전체 무결성 검사 결과 =====")
print("전체 이미지:", total_images)
print("전체 라벨:", total_labels)
print("전체 인스턴스:", total_instances)
print("라벨 누락:", len(missing_labels))
print("이미지 없는 라벨:", len(orphan_labels))
print("빈 라벨:", len(empty_labels))
print("잘못된 Segmentation 형식:", len(bad_format))
print("잘못된 클래스 ID:", len(bad_class))
print("잘못된 좌표:", len(bad_coordinates))

if not any([
    missing_labels,
    orphan_labels,
    empty_labels,
    bad_format,
    bad_class,
    bad_coordinates
]):
    print("\n전체 데이터 무결성 검사 통과")
else:
    print("\n이상 항목 발견 - 상세 확인 필요")
