
from pathlib import Path
import shutil

DATA_DIR = Path("사용한 데이터")

# 중복으로 확인된 TEST 이미지
image_name = "URT017_2020_02_08_001_24512_jpg.rf.c2fd9d2931d850bd20aef4922bbe65b0.jpg"

image_path = DATA_DIR / "images" / "test" / image_name
label_path = DATA_DIR / "labels" / "test" / f"{Path(image_name).stem}.txt"

# 원본 보관 폴더
backup_dir = DATA_DIR / "overlap_backup"
backup_image_dir = backup_dir / "images"
backup_label_dir = backup_dir / "labels"

backup_image_dir.mkdir(parents=True, exist_ok=True)
backup_label_dir.mkdir(parents=True, exist_ok=True)

print("===== 중복 데이터 정리 =====")

if not image_path.exists():
    raise FileNotFoundError(f"TEST 이미지가 없습니다: {image_path}")

if not label_path.exists():
    raise FileNotFoundError(f"TEST 라벨이 없습니다: {label_path}")

backup_image_path = backup_image_dir / image_name
backup_label_path = backup_label_dir / label_path.name

if backup_image_path.exists() or backup_label_path.exists():
    raise FileExistsError("백업 폴더에 같은 파일이 이미 있습니다.")

shutil.move(str(image_path), str(backup_image_path))
shutil.move(str(label_path), str(backup_label_path))

print("중복 이미지 백업 완료:", backup_image_path)
print("중복 라벨 백업 완료:", backup_label_path)
print("TEST 이미지와 라벨을 함께 분리했습니다.")
