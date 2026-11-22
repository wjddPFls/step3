
from pathlib import Path

LABEL_DIR = Path("사용한 데이터/labels")

label_files = sorted(LABEL_DIR.rglob("*.txt"))

empty_files = []
bad_format = []
bad_class = []
bad_coordinates = []
total_instances = 0

for path in label_files:
    lines = path.read_text(encoding="utf-8").splitlines()

    valid_lines = [line.split() for line in lines if line.strip()]

    if not valid_lines:
        empty_files.append(path)
        continue

    for line_number, values in enumerate(valid_lines, start=1):
        total_instances += 1

        # Segmentation 형식 확인
        if len(values) < 7 or (len(values) - 1) % 2 != 0:
            bad_format.append((path, line_number, len(values)))
            continue

        # 클래스 ID 확인
        try:
            class_id = int(values[0])
        except ValueError:
            bad_class.append((path, line_number, values[0]))
            continue

        if class_id not in [0, 1, 2]:
            bad_class.append((path, line_number, class_id))

        # 좌표 범위 확인
        try:
            coordinates = [float(v) for v in values[1:]]
            if not all(0 <= v <= 1 for v in coordinates):
                bad_coordinates.append((path, line_number))
        except ValueError:
            bad_coordinates.append((path, line_number))

print("===== 전체 라벨 포맷 검사 결과 =====")
print("전체 라벨 파일:", len(label_files))
print("전체 인스턴스:", total_instances)
print("빈 라벨 파일:", len(empty_files))
print("잘못된 Segmentation 형식:", len(bad_format))
print("잘못된 클래스 ID:", len(bad_class))
print("잘못된 좌표:", len(bad_coordinates))

if not (bad_format or bad_class or bad_coordinates):
    print("\n포맷 검사 완료")
else:
    print("\n이상 항목이 있습니다. 해당 파일을 확인해야 합니다.")
