#!/usr/bin/env python3
"""
YOLOv8 모델 변환 스크립트
PyTorch (.pt) → ONNX (.onnx)
"""

import os
import sys
from pathlib import Path
import argparse


def convert_yolov8_to_onnx():
    """YOLOv8 .pt 파일을 .onnx로 변환"""

    try:
        from ultralytics import YOLO
        print("✅ ultralytics 라이브러리 로드 완료")
    except ImportError:
        print("❌ ultralytics 라이브러리가 설치되지 않았습니다.")
        print("설치 명령어: pip install ultralytics")
        return False

    # 모델 경로 설정
    model_path = "/Users/moss/PycharmProjects/PythonProject/training/traffic_sign_detection/yolov8_training8/weights/best.pt"

    # 모델 파일 존재 확인
    if not Path(model_path).exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("경로를 확인해주세요.")
        return False

    print(f"📁 모델 파일 경로: {model_path}")

    try:
        # YOLO 모델 로드
        print("🔄 YOLOv8 모델 로딩 중...")
        model = YOLO(model_path)
        print("✅ 모델 로드 완료")

        # ONNX로 변환 (Raspberry Pi에 최적화)
        print("🔄 ONNX 변환 중...")
        onnx_path = model.export(
            format='onnx',  # ONNX 형식
            imgsz=320,  # 입력 이미지 크기 (Raspberry Pi 최적화)
            optimize=True,  # 최적화 활성화
            int8=False,  # INT8 양자화 (선택사항)
            dynamic=False,  # 동적 배치 크기 비활성화
            simplify=True,  # 모델 단순화
            opset=11,  # ONNX opset 버전
        )

        print(f"✅ ONNX 변환 완료!")
        print(f"📁 변환된 파일: {onnx_path}")

        # 파일 크기 확인
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB

        print(f"📊 파일 크기 비교:")
        print(f"   원본 (.pt): {original_size:.2f} MB")
        print(f"   변환 (.onnx): {onnx_size:.2f} MB")

        return onnx_path

    except Exception as e:
        print(f"❌ 변환 중 오류 발생: {e}")
        return False


def convert_with_quantization():
    """INT8 양자화와 함께 변환 (더 작은 크기, 빠른 추론)"""

    try:
        from ultralytics import YOLO

        model_path = "/Users/moss/PycharmProjects/PythonProject/training/traffic_sign_detection/yolov8_training/weights/best.pt"

        if not Path(model_path).exists():
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            return False

        print("🔄 INT8 양자화와 함께 ONNX 변환 중...")
        model = YOLO(model_path)

        # INT8 양자화된 ONNX 변환
        onnx_path = model.export(
            format='onnx',
            imgsz=320,
            optimize=True,
            int8=True,  # INT8 양자화 활성화
            dynamic=False,
            simplify=True,
            opset=11,
        )

        print(f"✅ 양자화된 ONNX 변환 완료: {onnx_path}")
        return onnx_path

    except Exception as e:
        print(f"❌ 양자화 변환 중 오류: {e}")
        return False


def test_onnx_inference(onnx_path: str):
    """변환된 ONNX 모델 추론 테스트"""

    try:
        import onnxruntime as ort
        import numpy as np
        import cv2

        print("🧪 ONNX 모델 추론 테스트...")

        # ONNX Runtime 세션 생성
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        print(f"📋 모델 정보:")
        print(f"   입력 이름: {input_name}")
        print(f"   입력 크기: {input_shape}")

        # 테스트 이미지 생성
        test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

        # 전처리
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # 추론 시간 측정
        import time
        start_time = time.time()

        outputs = session.run(None, {input_name: input_tensor})

        inference_time = time.time() - start_time

        print(f"✅ 추론 테스트 성공!")
        print(f"⏱️ 추론 시간: {inference_time:.3f}초")
        print(f"📊 출력 형태: {outputs[0].shape}")

        # FPS 계산
        fps = 1.0 / inference_time
        print(f"🎯 예상 FPS: {fps:.2f}")

        return True

    except ImportError:
        print("❌ onnxruntime가 설치되지 않았습니다.")
        print("설치 명령어: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"❌ 추론 테스트 실패: {e}")
        return False


def convert_multiple_formats():
    """여러 형식으로 동시 변환"""

    try:
        from ultralytics import YOLO

        model_path = "/Users/moss/PycharmProjects/PythonProject/training/traffic_sign_detection/yolov8_training/weights/best.pt"

        if not Path(model_path).exists():
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            return False

        model = YOLO(model_path)

        formats_to_convert = [
            ('onnx', '일반 ONNX'),
            ('torchscript', 'TorchScript'),
            ('tflite', 'TensorFlow Lite')
        ]

        converted_files = {}

        for format_name, description in formats_to_convert:
            try:
                print(f"🔄 {description} 변환 중...")

                if format_name == 'onnx':
                    exported_path = model.export(
                        format=format_name,
                        imgsz=320,
                        optimize=True,
                        simplify=True
                    )
                else:
                    exported_path = model.export(
                        format=format_name,
                        imgsz=320
                    )

                converted_files[format_name] = exported_path

                # 파일 크기 확인
                file_size = os.path.getsize(exported_path) / (1024 * 1024)
                print(f"✅ {description} 변환 완료: {exported_path} ({file_size:.2f} MB)")

            except Exception as e:
                print(f"❌ {description} 변환 실패: {e}")

        return converted_files

    except Exception as e:
        print(f"❌ 다중 변환 중 오류: {e}")
        return {}


def create_conversion_script():
    """자동 변환 스크립트 생성"""

    script_content = '''#!/usr/bin/env python3
"""
YOLOv8 모델 자동 변환 스크립트
사용법: python convert_model.py --input path/to/best.pt
"""

import argparse
from ultralytics import YOLO
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 모델 ONNX 변환')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='입력 .pt 모델 파일 경로')
    parser.add_argument('--output', '-o', type=str, 
                       help='출력 .onnx 파일 경로 (선택사항)')
    parser.add_argument('--imgsz', type=int, default=320,
                       help='입력 이미지 크기 (기본값: 320)')
    parser.add_argument('--int8', action='store_true',
                       help='INT8 양자화 적용')

    args = parser.parse_args()

    # 입력 파일 확인
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        return

    # 모델 로드
    print(f"🔄 모델 로딩: {args.input}")
    model = YOLO(args.input)

    # 변환 설정
    export_args = {
        'format': 'onnx',
        'imgsz': args.imgsz,
        'optimize': True,
        'simplify': True,
        'int8': args.int8
    }

    # 변환 실행
    print(f"🔄 ONNX 변환 중... (이미지 크기: {args.imgsz})")
    onnx_path = model.export(**export_args)

    print(f"✅ 변환 완료: {onnx_path}")

    # 결과 파일을 지정된 경로로 이동 (선택사항)
    if args.output:
        import shutil
        shutil.move(onnx_path, args.output)
        print(f"📁 파일 이동 완료: {args.output}")

if __name__ == "__main__":
    main()
'''

    with open('convert_model.py', 'w', encoding='utf-8') as f:
        f.write(script_content)

    print("✅ 변환 스크립트 생성 완료: convert_model.py")
    print("📝 사용법:")
    print("   python convert_model.py --input best.pt")
    print("   python convert_model.py --input best.pt --int8")
    print("   python convert_model.py --input best.pt --output my_model.onnx")


def main():
    """메인 실행 함수"""

    print("🚀 YOLOv8 모델 변환 도구")
    print("=" * 50)

    while True:
        print("\n선택하세요:")
        print("1. 기본 ONNX 변환")
        print("2. INT8 양자화 ONNX 변환")
        print("3. 다중 형식 변환")
        print("4. ONNX 추론 테스트")
        print("5. 자동 변환 스크립트 생성")
        print("0. 종료")

        choice = input("\n선택 (0-5): ").strip()

        if choice == '1':
            result = convert_yolov8_to_onnx()
            if result:
                print(f"\n🎉 변환 성공! 파일: {result}")

        elif choice == '2':
            result = convert_with_quantization()
            if result:
                print(f"\n🎉 양자화 변환 성공! 파일: {result}")

        elif choice == '3':
            results = convert_multiple_formats()
            if results:
                print(f"\n🎉 다중 변환 완료! 총 {len(results)}개 파일")

        elif choice == '4':
            onnx_file = input("ONNX 파일 경로를 입력하세요: ").strip()
            if Path(onnx_file).exists():
                test_onnx_inference(onnx_file)
            else:
                print("❌ 파일을 찾을 수 없습니다.")

        elif choice == '5':
            create_conversion_script()

        elif choice == '0':
            print("👋 프로그램을 종료합니다.")
            break

        else:
            print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()