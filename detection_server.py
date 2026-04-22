#!/usr/bin/env python3
"""
자율주행 로봇 - 방법 2: 원격 탐지 서버 (컴퓨터용)
GPU 가속 YOLO 추론 서버 - 완전한 버전
"""

import cv2
import numpy as np
import socket
import threading
import pickle
import struct
import time
import argparse
from ultralytics import YOLO


class DetectionServer:
    def __init__(self, host='0.0.0.0', port=8888, model_path='best.pt'):
        self.host = host
        self.port = port
        self.model_path = model_path

        # YOLO 모델 로드
        print("Loading YOLO model...")
        try:
            self.model = YOLO(model_path)
            # GPU 사용 가능하면 GPU로, 아니면 CPU로
            device = 'cuda' if self.model.device.type == 'cuda' else 'cpu'
            print(f"Model loaded successfully. Using device: {device}")

            # 모델 최적화
            if device == 'cuda':
                self.model.half()  # FP16 연산 (GPU에서만)

        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

        # 클래스 이름 매핑
        self.class_names = {0: 'turn_left', 1: 'turn_right', 2: 'go_straight', 3: 'stop'}

        # 서버 상태
        self.server_socket = None
        self.clients = []
        self.running = True
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'avg_inference_time': 0,
            'connected_clients': 0
        }

        print(f"Server initialized. Class mapping: {self.class_names}")

    def start_server(self):
        """서버 시작"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)

            print(f"🚀 Detection server started on {self.host}:{self.port}")
            print("Waiting for robot connections...")

            # 통계 출력 스레드
            stats_thread = threading.Thread(target=self.print_stats_periodically)
            stats_thread.daemon = True
            stats_thread.start()

            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"🤖 Robot connected from {address}")
                    self.stats['connected_clients'] += 1

                    # 클라이언트 처리 스레드 시작
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except Exception as e:
                    if self.running:
                        print(f"Accept error: {e}")

        except Exception as e:
            print(f"Server start error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
                print("Server socket closed")

    def handle_client(self, client_socket, address):
        """클라이언트 처리"""
        print(f"📡 Handling robot {address}")
        frame_count = 0

        try:
            while self.running:
                # 프레임 수신
                frame = self.receive_frame(client_socket)
                if frame is None:
                    print(f"❌ Failed to receive frame from {address}")
                    break

                frame_count += 1
                self.stats['total_frames'] += 1

                # YOLO 추론
                detection_result = self.detect_signs(frame, client_id=f"{address[0]}:{address[1]}")

                # 결과 전송
                if not self.send_detection(client_socket, detection_result):
                    print(f"❌ Failed to send result to {address}")
                    break

                # 주기적으로 클라이언트별 통계 출력
                if frame_count % 50 == 0:
                    print(f"📊 Robot {address}: {frame_count} frames processed")

        except Exception as e:
            print(f"❌ Client {address} error: {e}")
        finally:
            client_socket.close()
            self.stats['connected_clients'] -= 1
            print(f"🔌 Robot {address} disconnected (processed {frame_count} frames)")

    def receive_frame(self, client_socket):
        """프레임 수신"""
        try:
            # 데이터 크기 수신 (4-byte unsigned int)
            size_data = b""
            size_bytes_to_read = struct.calcsize("!I")  # Fix 1: Use fixed size
            while len(size_data) < size_bytes_to_read:
                chunk = client_socket.recv(size_bytes_to_read - len(size_data))
                if not chunk:
                    return None
                size_data += chunk

            size = struct.unpack("!I", size_data)[0]  # Fix 2: Unpack as 4-byte int

            # 크기 검증 (최대 10MB)
            if size > 10 * 1024 * 1024:
                print(f"⚠️  Frame size too large: {size} bytes")
                return None

            # 실제 데이터 수신
            data = b""
            while len(data) < size:
                remaining = size - len(data)
                chunk = client_socket.recv(min(remaining, 4096))
                if not chunk:
                    return None
                data += chunk

            # 역직렬화 및 디코딩
            buffer = pickle.loads(data)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

            if frame is None:
                print("⚠️  Failed to decode frame")
                return None

            return frame

        except Exception as e:
            print(f"Frame receive error: {e}")
            return None

    def send_detection(self, client_socket, detection_result):
        """탐지 결과 전송"""
        try:
            data = pickle.dumps(detection_result)
            size = struct.pack("!I", len(data))  # Fix 3: Pack as 4-byte int
            client_socket.sendall(size + data)
            return True
        except Exception as e:
            print(f"Detection send error: {e}")
            return False

    def detect_signs(self, frame, client_id="unknown"):
        """표지판 탐지"""
        try:
            start_time = time.time()

            # YOLO 추론
            results = self.model(frame, conf=0.6, iou=0.45, verbose=False)

            detected_signs = []
            detections_info = []

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])

                        if conf > 0.6:
                            sign_name = self.class_names.get(cls, 'unknown')
                            detected_signs.append(sign_name)

                            # 바운딩 박스 정보
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections_info.append({
                                'class': sign_name,
                                'class_id': cls,
                                'confidence': conf,
                                'bbox': [float(x1), float(y1), float(x2), float(y2)]
                            })

            inference_time = (time.time() - start_time) * 1000

            # 추론 시간 통계 업데이트
            if self.stats['avg_inference_time'] == 0:
                self.stats['avg_inference_time'] = inference_time
            else:
                self.stats['avg_inference_time'] = (self.stats['avg_inference_time'] * 0.9 + inference_time * 0.1)

            result = {
                'signs': detected_signs,
                'detections': detections_info,
                'inference_time': inference_time,
                'frame_shape': frame.shape,
                'timestamp': time.time()
            }

            if detected_signs:
                print(f"🎯 [{client_id}] Detected: {detected_signs} (inference: {inference_time:.1f}ms)")
                self.stats['total_detections'] += 1

            return result

        except Exception as e:
            print(f"❌ Detection error: {e}")
            return {
                'signs': [],
                'detections': [],
                'inference_time': 0,
                'frame_shape': None,
                'error': str(e)
            }

    def send_detection(self, client_socket, detection_result):
        """탐지 결과 전송"""
        try:
            data = pickle.dumps(detection_result)
            size = struct.pack("!I", len(data))
            client_socket.sendall(size + data)
            return True
        except Exception as e:
            print(f"Detection send error: {e}")
            return False

    def print_stats_periodically(self):
        """주기적으로 통계 출력"""
        while self.running:
            time.sleep(30)  # 30초마다
            if self.stats['total_frames'] > 0:
                print("\n" + "=" * 50)
                print(f"📈 Server Statistics:")
                print(f"   Connected Clients: {self.stats['connected_clients']}")
                print(f"   Total Frames: {self.stats['total_frames']}")
                print(f"   Total Detections: {self.stats['total_detections']}")
                print(f"   Avg Inference Time: {self.stats['avg_inference_time']:.1f}ms")
                print(f"   Detection Rate: {(self.stats['total_detections'] / self.stats['total_frames'] * 100):.1f}%")
                print("=" * 50 + "\n")

    def stop_server(self):
        """서버 중지"""
        print("\n🛑 Stopping detection server...")
        self.running = False
        if self.server_socket:
            self.server_socket.close()

        # 최종 통계
        print("\n📊 Final Statistics:")
        print(f"   Total Frames Processed: {self.stats['total_frames']}")
        print(f"   Total Signs Detected: {self.stats['total_detections']}")
        print(f"   Average Inference Time: {self.stats['avg_inference_time']:.1f}ms")
        print("Server stopped successfully.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='YOLO Detection Server for Autonomous Robot')
    parser.add_argument('--host', default='0.0.0.0', help='Server host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8888, help='Server port (default: 8888)')
    parser.add_argument('--model', default='best.pt', help='YOLO model path (default: best.pt)')

    args = parser.parse_args()

    print("🤖 Autonomous Robot Detection Server")
    print("=" * 40)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Model: {args.model}")
    print("=" * 40)

    # 모델 파일 존재 확인
    import os
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Please make sure the YOLO model file exists.")
        exit(1)

    # 서버 시작
    server = DetectionServer(host=args.host, port=args.port, model_path=args.model)

    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n⚠️  Received interrupt signal (Ctrl+C)")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        server.stop_server()


if __name__ == "__main__":
    main()