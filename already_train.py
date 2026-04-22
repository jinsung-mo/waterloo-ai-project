import os
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import json
from pycocotools.coco import COCO
import shutil


class YOLOv8TrafficSignTrainer:
    def __init__(self, data_dir='/Users/moss/PycharmProjects/PythonProject/training/yolo_data', model_size='n'):
        """
        YOLOv8 트래픽 사인 트레이너 초기화

        Args:
            data_dir: 데이터 디렉토리 경로
            model_size: 모델 크기 ('n', 's', 'm', 'l', 'x')
        """
        self.data_dir = Path(data_dir)
        self.model_size = model_size
        self.categories = {
            70: 0,  # Turn Left -> class 0
            69: 1,  # Turn Right -> class 1
            68: 2,  # Go Straight -> class 2
            43: 3  # Stop -> class 3
        }
        self.class_names = ['turn_left', 'turn_right', 'go_straight', 'stop']

        # 디렉토리 생성
        self.setup_directories()

    def setup_directories(self):
        """필요한 디렉토리 구조 생성"""
        dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
        for d in dirs:
            (self.data_dir / d).mkdir(parents=True, exist_ok=True)

    def coco_to_yolo(self, coco_json_path, images_dir, output_dir, split='train'):
        """
        COCO 형식을 YOLO 형식으로 변환

        Args:
            coco_json_path: COCO JSON 파일 경로
            images_dir: 이미지 디렉토리 경로
            output_dir: 출력 디렉토리 경로
            split: 'train' 또는 'val'
        """
        print(f"Converting COCO to YOLO format for {split} set...")

        coco = COCO(coco_json_path)

        # 관심있는 카테고리의 어노테이션 가져오기
        ann_ids = coco.getAnnIds(catIds=list(self.categories.keys()))
        anns = coco.loadAnns(ann_ids)

        # 이미지 ID 추출
        img_ids = list(set([ann['image_id'] for ann in anns]))

        images_output = self.data_dir / split / 'images'
        labels_output = self.data_dir / split / 'labels'

        converted_count = 0

        for img_id in img_ids:
            # 이미지 정보 로드
            img_info = coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']
            img_path = Path(images_dir) / img_filename

            if not img_path.exists():
                continue

            # 이미지 복사
            img_output_path = images_output / img_filename
            shutil.copy2(img_path, img_output_path)

            # 해당 이미지의 어노테이션들 가져오기
            img_ann_ids = coco.getAnnIds(imgIds=img_id, catIds=list(self.categories.keys()))
            img_anns = coco.loadAnns(img_ann_ids)

            # YOLO 형식으로 변환
            img_width = img_info['width']
            img_height = img_info['height']

            yolo_labels = []
            for ann in img_anns:
                category_id = ann['category_id']
                if category_id not in self.categories:
                    continue

                class_id = self.categories[category_id]

                # COCO bbox: [x, y, width, height] (절대 좌표)
                x, y, width, height = ann['bbox']

                # YOLO 형식으로 변환: [class_id, x_center, y_center, width, height] (정규화)
                x_center = (x + width / 2) / img_width
                y_center = (y + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height

                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

            # 레이블 파일 저장
            if yolo_labels:
                label_filename = img_filename.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = labels_output / label_filename

                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))

                converted_count += 1

        print(f"Converted {converted_count} images for {split} set")
        return converted_count

    def create_yaml_config(self):
        """YOLO 학습을 위한 YAML 설정 파일 생성"""
        config = {
            'path': str(self.data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }

        yaml_path = self.data_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)

        print(f"Created YAML config: {yaml_path}")
        return yaml_path

    def prepare_dataset(self, train_json, train_images, val_json, val_images):
        """
        데이터셋 준비 (COCO -> YOLO 변환)

        Args:
            train_json: 학습용 COCO JSON 파일 경로
            train_images: 학습용 이미지 디렉토리 경로
            val_json: 검증용 COCO JSON 파일 경로
            val_images: 검증용 이미지 디렉토리 경로
        """
        print("Preparing dataset...")

        # 학습 데이터 변환
        train_count = self.coco_to_yolo(train_json, train_images, self.data_dir, 'train')

        # 검증 데이터 변환
        val_count = self.coco_to_yolo(val_json, val_images, self.data_dir, 'val')

        # YAML 설정 파일 생성
        yaml_path = self.create_yaml_config()

        print(f"Dataset prepared successfully!")
        print(f"Training images: {train_count}")
        print(f"Validation images: {val_count}")

        return yaml_path

    def train_model(self, yaml_config, epochs=50, imgsz=640, batch=32, device='auto', weights=None, classes=[1]):
        """
        YOLOv8 모델 학습

        Args:
            yaml_config: YAML 설정 파일 경로
            epochs: 학습 에포크 수
            imgsz: 입력 이미지 크기
            batch: 배치 크기
            device: 디바이스 ('auto', 'cpu', 'cuda', 'mps')
            weights: 사전 학습된 모델 가중치 파일 경로
            classes: 학습에 사용할 클래스 ID (예: [0, 1])
        """
        print(f"Starting YOLOv8{self.model_size} training...")

        if weights:
            model = YOLO(weights)
        else:
            model = YOLO(f'yolov8{self.model_size}.pt')

        # 학습 시작
        results = model.train(
            data=yaml_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project='traffic_sign_detection',
            name='yolov8_training_upgrade',
            save=True,
            cache=True,
            workers=4,
            patience=5
        )

        print("Training completed!")
        return results

    def validate_model(self, model_path, yaml_config):
        """모델 검증"""
        print("Validating model...")

        model = YOLO(model_path)
        results = model.val(
            data=yaml_config,
            save_json=True,
            save_hybrid=True
        )

        print(f"Validation mAP50: {results.box.map50:.4f}")
        print(f"Validation mAP50-95: {results.box.map:.4f}")

        return results

    def predict_and_visualize(self, model_path, test_image_path, conf=0.25):
        """
        예측 및 시각화

        Args:
            model_path: 학습된 모델 경로
            test_image_path: 테스트 이미지 경로
            conf: 신뢰도 임계값
        """
        model = YOLO(model_path)

        # 예측 수행
        results = model(test_image_path, conf=conf)

        # 결과 시각화
        for r in results:
            im_array = r.plot()  # BGR numpy array
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.show()

            # 예측 결과 출력
            if r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = self.class_names[class_id]
                    print(f"Detected: {class_name} (confidence: {confidence:.2f})")

        return results

    def export_model(self, model_path, format='onnx'):
        """
        모델을 다른 형식으로 내보내기 (Raspberry Pi 배포용)

        Args:
            model_path: 학습된 모델 경로
            format: 내보낼 형식 ('onnx', 'torchscript', 'tflite', etc.)
        """
        print(f"Exporting model to {format}...")

        model = YOLO(model_path)
        model.export(format=format, imgsz=320)  # Raspberry Pi에 최적화된 크기

        print(f"Model exported successfully!")


def main():
    """메인 실행 함수"""
    # 트레이너 초기화
    trainer = YOLOv8TrafficSignTrainer(data_dir='/Users/moss/PycharmProjects/PythonProject/training/yolo_data', model_size='s')

    # 데이터셋 준비 (전체 클래스가 포함된 YAML 파일 사용)
    yaml_config = '/Users/moss/PycharmProjects/PythonProject/training/yolo_data/dataset.yaml'

    # 기존 best.pt 모델 가중치 경로
    best_model_path = '/Users/moss/PycharmProjects/PythonProject/training/traffic_sign_detection/yolov8_training13/weights/best.pt'

    # 모델 학습 (turn_right 클래스만 추가 학습)
    results = trainer.train_model(
        yaml_config=yaml_config,
        epochs=10,
        imgsz=320,
        batch=32,
        device='mps',
        weights=best_model_path
    )

    # 최적 모델로 검증
    validation_results = trainer.validate_model(best_model_path, yaml_config)

    # 모델 내보내기 (Raspberry Pi 배포용)
    trainer.export_model(best_model_path, format='onnx')

    print("YOLOv8 training pipeline completed!")


if __name__ == "__main__":
    main()