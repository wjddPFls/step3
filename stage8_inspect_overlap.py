
from pathlib import Path

DATA_DIR = Path("사용한 데이터/images")
TARGET_ID = "URT017_2020_02_08_001_24512"

for split in ["train", "val", "test"]:
    folder = DATA_DIR / split

    print(f"\n===== {split.upper()} =====")

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        origin_id = path.stem.split("_jpg.rf.")[0]
        origin_id = origin_id.split("_aug")[0]

        if origin_id == TARGET_ID:
            print("파일명:", path.name)
            print("경로:", path)
            print("파일 크기:", path.stat().st_size, "bytes")
