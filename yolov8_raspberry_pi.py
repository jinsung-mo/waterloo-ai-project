import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import threading
import queue
from vilib import Vilib
import rospy
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json


class YOLOv8TrafficSignDetector:
    def __init__(self, model_path='best.pt', conf_threshold=0.25, device='cpu'):
        """
        YOLOv8 트래픽 사인 검출기 초기화

        Args:
            model_path: 학습된 YOLOv8 모델 경로
            conf_threshold: 신뢰도 임계값
            device: 추론 디바이스 ('cpu', 'cuda')
        """
        print("Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

        # 클래스 이름 매핑
        self.class_names = ['turn_left', 'turn_right', 'go_straight', 'stop']
        self.class_actions = {
            0: 'TURN_LEFT',
            1: 'TURN_RIGHT',
            2: 'GO_STRAIGHT',
            3: 'STOP'
        }

        # 예측 결과 필터링을 위한 버퍼
        self.prediction_buffer = deque(maxlen=10)  # 최근 10개 예측 저장
        self.confidence_weights = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])  # 최근 것일수록 높은 가중치

        # ROS 초기화
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=2)  # 최대 2개 이미지만 버퍼링

        # 성능 모니터링
        self.fps_counter = 0
        self.start_time = time.time()

    def filter_predictions(self, predictions):
        """
        예측 결과 필터링 (시간적 일관성 확보)

        Args:
            predictions: 현재 프레임의 예측 결과 리스트

        Returns:
            filtered_action: 필터링된 액션
            confidence: 신뢰도
        """
        if not predictions:
            self.prediction_buffer.append({'action': None, 'confidence': 0.0})
        else:
            # 가장 높은 신뢰도의 예측 선택
            best_pred = max(predictions, key=lambda x: x['confidence'])
            self.prediction_buffer.append(best_pred)

        # 최근 예측들의 가중 평균 계산
        class_votes = {i: 0.0 for i in range(4)}
        total_weight = 0.0

        buffer_size = len(self.prediction_buffer)
        weights = self.confidence_weights[-buffer_size:] if buffer_size <= 10 else self.confidence_weights

        for i, pred in enumerate(self.prediction_buffer):
            if pred['action'] is not None:
                weight = weights[i] * pred['confidence']
                class_votes[pred['action']] += weight
                total_weight += weight

        if total_weight == 0:
            return None, 0.0

        # 가장 높은 점수의 클래스 선택
        best_class = max(class_votes, key=class_votes.get)
        avg_confidence = class_votes[best_class] / total_weight if total_weight > 0 else 0.0

        # 임계값 이상일 때만 반환
        if avg_confidence > self.conf_threshold:
            return self.class_actions[best_class], avg_confidence
        else:
            return None, avg_confidence

    def detect_signs(self, image):
        """
        이미지에서 트래픽 사인 검출

        Args:
            image: OpenCV 이미지 (BGR)

        Returns:
            predictions: 검출 결과 리스트
            annotated_image: 어노테이션이 추가된 이미지
        """
        # YOLOv8 추론
        results = self.model(image, conf=self.conf_threshold, verbose=False)

        predictions = []
        annotated_image = image.copy()

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    # 박스 좌표 추출
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    predictions.append({
                        'action': class_id,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })

                    # 이미지에 박스와 라벨 그리기
                    label = f"{self.class_names[class_id]}: {confidence:.2f}"

                    # 박스 그리기
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # 라벨 배경 그리기
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_image, (int(x1), int(y1) - text_height - 5),
                                  (int(x1) + text_width, int(y1)), (0, 255, 0), -1)

                    # 라벨 텍스트 그리기
                    cv2.putText(annotated_image, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return predictions, annotated_image

    def process_frame(self, frame):
        """
        프레임 처리 및 결과 반환

        Args:
            frame: 입력 프레임

        Returns:
            result: 처리 결과 딕셔너리
        """
        # 프레임 크기 조정 (추론 속도 향상)
        height, width = frame.shape[:2]
        if width > 320:
            scale = 320 / width
            new_width = 320
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame

        # 트래픽 사인 검출
        predictions, annotated_frame = self.detect_signs(frame_resized)

        # 예측 결과 필터링
        filtered_action, avg_confidence = self.filter_predictions(predictions)

        # 결과 딕셔너리 생성
        result = {
            'action': filtered_action,
            'confidence': avg_confidence,
            'raw_predictions': predictions,
            'annotated_frame': annotated_frame,
            'timestamp': time.time()
        }

        # FPS 계산
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # 30프레임마다 FPS 출력
            elapsed = time.time() - self.start_time
            fps = 30 / elapsed
            print(f"FPS: {fps:.1f}")
            self.start_time = time.time()

        return result


class TrafficSignROSNode:
    def __init__(self):
        """ROS 노드 초기화"""
        rospy.init_node('traffic_sign_detector', anonymous=True)

        # YOLOv8 검출기 초기화
        model_path = rospy.get_param('~model_path', 'best.pt')
        conf_threshold = rospy.get_param('~conf_threshold', 0.25)

        self.detector = YOLOv8TrafficSignDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device='cpu'  # Raspberry Pi에서는 CPU 사용
        )

        # ROS Publishers
        self.action_pub = rospy.Publisher('/traffic_sign/action', String, queue_size=1)
        self.confidence_pub = rospy.Publisher('/traffic_sign/confidence', String, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/traffic_sign/debug_image', Image, queue_size=1)

        # ROS Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

        print("Traffic Sign ROS Node initialized")

    def image_callback(self, msg):
        """이미지 콜백 함수"""
        try:
            # ROS 이미지를 OpenCV로 변환
            cv_image = self.detector.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 프레임 처리
            result = self.detector.process_frame(cv_image)

            # 결과 퍼블리시
            if result['action']:
                action_msg = String()
                action_msg.data = result['action']
                self.action_pub.publish(action_msg)

                confidence_msg = String()
                confidence_msg.data = json.dumps({
                    'action': result['action'],
                    'confidence': result['confidence'],
                    'timestamp': result['timestamp']
                })
                self.confidence_pub.publish(confidence_msg)

                print(f"Detected: {result['action']} (confidence: {result['confidence']:.2f})")

            # 디버그 이미지 퍼블리시
            debug_msg = self.detector.bridge.cv2_to_imgmsg(result['annotated_frame'], "bgr8")
            self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")


class VilibTrafficSignDetector:
    def __init__(self, model_path='best.pt'):
        """Vilib을 사용한 트래픽 사인 검출기"""
        self.detector = YOLOv8TrafficSignDetector(
            model_path=model_path,
            conf_threshold=0.25,
            device='cpu'
        )

        print("Vilib Traffic Sign Detector initialized")

    def start_detection(self):
        """카메라 시작 및 검출 루프"""
        print("Starting camera...")
        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=False, web=True)

        try:
            while True:
                # 카메라에서 프레임 가져오기
                if Vilib.img_array is not None and len(Vilib.img_array) > 0:
                    frame = Vilib.img_array[0]

                    if frame is not None:
                        # 프레임 처리
                        result = self.detector.process_frame(frame)

                        # 결과 출력
                        if result['action']:
                            print(f"Action: {result['action']} (confidence: {result['confidence']:.2f})")

                        # 약간의 지연 (CPU 부하 줄이기)
                        time.sleep(0.1)

        except KeyboardInterrupt:
            print("Stopping detection...")
        finally:
            Vilib.camera_close()


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv8 Traffic Sign Detection')
    parser.add_argument('--mode', choices=['ros', 'vilib', 'test'], default='vilib',
                        help='실행 모드 선택')
    parser.add_argument('--model', default='best.pt',
                        help='YOLOv8 모델 경로')
    parser.add_argument('--test_image', default=None,
                        help='테스트 이미지 경로 (test 모드에서 사용)')

    args = parser.parse_args()

    if args.mode == 'ros':
        # ROS 모드
        try:
            node = TrafficSignROSNode()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass

    elif args.mode == 'vilib':
        # Vilib 모드
        detector = VilibTrafficSignDetector(args.model)
        detector.start_detection()

    elif args.mode == 'test':
        # 테스트 모드
        if args.test_image is None:
            print("테스트 모드에서는 --test_image 인자가 필요합니다.")
            return

        detector = YOLOv8TrafficSignDetector(args.model)

        # 테스트 이미지 로드
        image = cv2.imread(args.test_image)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {args.test_image}")
            return

        # 검출 수행
        result = detector.process_frame(image)

        # 결과 출력
        print(f"검출 결과: {result['action']} (신뢰도: {result['confidence']:.2f})")

        # 결과 이미지 표시
        cv2.imshow('Detection Result', result['annotated_frame'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()