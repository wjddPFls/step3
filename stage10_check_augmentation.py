
from pathlib import Path

root = Path("사용한 데이터/images")

print("===== 증강 이미지 분포 확인 =====")

for split in ["train", "val", "test"]:
    folder = root / split

    files = [
        p for p in folder.rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    augmented = [
        p for p in files
        if "_aug" in p.stem.lower()
    ]

    print(f"\n[{split.upper()}]")
    print(f"전체 이미지: {len(files)}장")
    print(f"증강 이미지: {len(augmented)}장")
    print(f"일반 이미지: {len(files) - len(augmented)}장")

    if split != "train" and augmented:
        print("주의: 검증 또는 테스트에 증강 이미지가 있습니다.")
        for p in augmented:
            print(" ", p.name)

print("\n===== 증강 이미지 분포 검사 완료 =====")
