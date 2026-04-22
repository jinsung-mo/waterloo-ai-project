import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
# 학습된 best.pt 모델의 경로를 정확하게 지정합니다.
# 이 경로가 맞는지 확인해주세요.
model_path = '/Users/moss/PycharmProjects/PythonProject/training/traffic_sign_detection/yolov8_training13/weights/best.pt'
model = YOLO(model_path)

# 웹캠 설정
# 0번은 내장 웹캠을 의미합니다. 외장 웹캠은 1, 2 등으로 변경될 수 있습니다.
cap = cv2.VideoCapture(0)

# 웹캠이 제대로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

try:
    while True:
        # 웹캠에서 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # YOLOv8 모델로 객체 탐지
        # stream=True로 설정하여 더 효율적으로 처리합니다.
        results = model(frame, stream=True)

        # 결과 시각화
        # 결과를 프레임에 오버레이합니다.
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 종료 시 웹캠과 창 해제
    cap.release()
    cv2.destroyAllWindows()