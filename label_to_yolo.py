import os

# 수정할 라벨 파일들이 있는 디렉토리 경로
# Roboflow에서 다운로드한 'train/labels' 폴더 경로를 지정하세요.
labels_dir = "/Users/moss/Downloads/sag.v1i.yolov8/valid/labels"

# 클래스 ID 매핑 딕셔너리
# {새로운 ID: 기존 ID}
# 기존 YOLOv8_train.py의 클래스 순서를 따릅니다.
id_mapping = {
    0: 1
}


def update_labels(labels_dir, mapping):
    """
    라벨 파일의 클래스 ID를 매핑에 따라 업데이트합니다.

    Args:
        labels_dir (str): 라벨 파일들이 있는 디렉토리 경로
        mapping (dict): 새로운 ID와 기존 ID의 매핑 딕셔너리
    """
    if not os.path.exists(labels_dir):
        print(f"오류: {labels_dir} 경로를 찾을 수 없습니다.")
        return

    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(labels_dir, filename)

            with open(filepath, 'r') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                old_id = int(parts[0])
                if old_id in mapping:
                    new_id = mapping[old_id]
                    updated_line = f"{new_id} {' '.join(parts[1:])}\n"
                    updated_lines.append(updated_line)
                else:
                    # 매핑에 없는 클래스는 그대로 둡니다 (오류 방지)
                    updated_lines.append(line)

            with open(filepath, 'w') as f:
                f.writelines(updated_lines)

    print("모든 라벨 파일의 클래스 ID 수정이 완료되었습니다.")


if __name__ == "__main__":
    update_labels(labels_dir, id_mapping)