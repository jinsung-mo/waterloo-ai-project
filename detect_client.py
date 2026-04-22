#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자율주행 로봇 - 클라이언트 전용 (picarx 라이브러리 사용)
모델 감지 결과를 버퍼링하여 Stop 우선 → 최신 비-Stop 1개 실행
임계값 580 기준으로 라인 트레이싱 구현
좌회전/우회전 방향 수정
"""

import cv2
import time
import socket
import threading
import pickle
import struct
import queue
import argparse
from collections import deque
from picarx import Picarx

# 하드웨어 라이브러리 (클라이언트에서만 사용)
try:
    from picamera2 import Picamera2

    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("⚠️  Picamera2 library not available. Cannot run client mode.")


class RemoteRobotClient:
    def __init__(self, server_host='127.0.0.1', server_port=8888):
        if not HARDWARE_AVAILABLE:
            raise RuntimeError("Picamera2 library not available")

        # 라인 임계값 설정 (picarx용)
        self.LINE_THRESHOLD = 500  # picarx 센서 기준

        # picarx 하드웨어 초기화
        self.init_hardware()

        # 네트워크 설정
        self.server_host = server_host
        self.server_port = server_port
        self.client_socket = None
        self.connected = False

        # 상태 변수
        self.running = True
        self.intersection_detected = False

        # 표지판 버퍼(중복 없이 최대 2개), 타임아웃
        self.sign_buffer = deque(maxlen=2)
        self.last_sign_time = None
        self.SIGN_TIMEOUT = 12.0

        # 큐
        self.command_queue = queue.Queue(maxsize=5)

        # 통계
        self.stats = {'frames_sent': 0, 'detections_received': 0, 'network_errors': 0}

        print("🤖 Robot Client initialized (picarx)")
        print(f"Server: {server_host}:{server_port}")
        print(f"Line Detection Threshold: {self.LINE_THRESHOLD}")

    def init_hardware(self):
        """하드웨어 초기화 - picarx 사용"""
        print("🔧 Initializing picarx hardware...")
        try:
            # picarx 초기화
            self.px = Picarx()

            # 카메라 초기화
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(
                main={"format": 'RGB888', "size": (320, 240)}
            ))
            self.picam2.start()

            # 서보 초기화
            self.px.set_dir_servo_angle(0)
            time.sleep(1)
            print("✅ picarx hardware initialized successfully")
        except Exception as e:
            print(f"❌ Hardware initialization failed: {e}")
            raise

    def connect_to_server(self):
        """서버 연결"""
        try:
            if self.client_socket:
                self.client_socket.close()
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(5.0)
            self.client_socket.connect((self.server_host, self.server_port))
            self.connected = True
            print(f"✅ Connected to server {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            print(f"❌ Connection to server failed: {e}")
            self.connected = False
            return False

    def send_frame(self, frame):
        """프레임 전송"""
        try:
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 70]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            data = pickle.dumps(buffer)
            size = struct.pack("!I", len(data))
            self.client_socket.sendall(size + data)
            self.stats['frames_sent'] += 1
            return True
        except Exception as e:
            print(f"❌ Frame send error: {e}")
            self.connected = False
            self.stats['network_errors'] += 1
            return False

    def receive_detection(self):
        """탐지 결과 수신"""
        try:
            size_data = self.client_socket.recv(4)
            if not size_data:
                return None
            size = struct.unpack("!I", size_data)[0]
            data = b""
            while len(data) < size:
                chunk = self.client_socket.recv(min(size - len(data), 4096))
                if not chunk:
                    return None
                data += chunk
            detection_result = pickle.loads(data)
            self.stats['detections_received'] += 1
            return detection_result
        except Exception as e:
            print(f"❌ Detection receive error: {e}")
            self.connected = False
            self.stats['network_errors'] += 1
            return None

    def communication_thread(self):
        """통신 및 자동 재연결 스레드"""
        frame_count = 0
        while self.running:
            if not self.connected:
                print("🔄 Attempting to reconnect to the server...")
                self.connect_to_server()
                time.sleep(2)
                continue
            try:
                if frame_count % 3 == 0:
                    frame = self.picam2.capture_array()
                    if self.send_frame(frame):
                        result = self.receive_detection()
                        if result:
                            # 서버 클래스명: 'turn_left','turn_right','go_straight','stop'
                            signs = list(result.get('signs', []))
                            if signs:
                                self.push_signs(signs)
                frame_count += 1
                time.sleep(0.1)
            except Exception as e:
                print(f"❌ Communication error: {e}")
                self.connected = False
                if self.client_socket:
                    self.client_socket.close()
                time.sleep(1)

    def read_line_sensors(self):
        """라인 센서 읽기 - picarx 사용"""
        try:
            # picarx의 grayscale 센서 값 읽기
            sensor_values = self.px.get_grayscale_data()
            return sensor_values  # [left, center, right] 값 반환
        except Exception as e:
            print(f"센서 읽기 오류: {e}")
            return [1000, 1000, 1000]  # 기본값

    def get_line_status(self, sensor_values):
        """센서값이 임계값보다 크면 라인(True) - 흰색 라인 트레이싱용"""
        # 흰색 라인 트레이싱: 값이 클수록 흰색 라인(1), 작을수록 검은 바닥(0)
        # 임계값 580 기준: 580보다 크면 흰색 라인(1), 작으면 검은 바닥(0)
        return [
            sensor_values[0] > self.LINE_THRESHOLD,  # 왼쪽 센서
            sensor_values[1] > self.LINE_THRESHOLD,  # 중앙 센서
            sensor_values[2] > self.LINE_THRESHOLD  # 오른쪽 센서
        ]

    def detect_intersection(self, line_status):
        """교차로: L, C, R 모두 라인 위"""
        return all(line_status)

    def line_following(self, line_status):
        """기본 라인트레이싱 (교차로 아님) - 흰색 라인 기준"""
        left_on, center_on, right_on = line_status

        if not any(line_status):
            print("⚠️  Line lost!")
            self.px.stop()
            return False

        # 흰색 라인 트레이싱 로직 - 정지와 직진도 반대로 수정
        # [0,1,0] = [왼쪽 검은바닥, 중앙 흰선, 오른쪽 검은바닥] → 정지 (반대로 수정)
        if not left_on and center_on and not right_on:
            print("🛑 [0,1,0] → STOP")
            self.px.stop()

        # [1,0,1] = [왼쪽 흰선, 중앙 검은바닥, 오른쪽 흰선] → 직진 (반대로 수정)
        elif left_on and not center_on and right_on:
            print("➡️ [1,0,1] → FORWARD")
            self.px.set_dir_servo_angle(0)
            self.px.forward(15)

        # [1,1,1] = [왼쪽 흰선, 중앙 흰선, 오른쪽 흰선] → 직진 (반대로 수정)
        elif left_on and center_on and right_on:
            print("➡️ [1,1,1] → FORWARD")
            self.px.set_dir_servo_angle(0)
            self.px.forward(15)

        # [1,1,0] = [왼쪽 흰선, 중앙 흰선, 오른쪽 검은바닥] → 좌회전
        elif left_on and center_on and not right_on:
            print("🔄 [1,1,0] → LEFT TURN")
            self.px.set_dir_servo_angle(20)  # 양수로 변경 (좌회전)
            self.px.forward(12)

        # [0,1,1] = [왼쪽 검은바닥, 중앙 흰선, 오른쪽 흰선] → 우회전
        elif not left_on and center_on and right_on:
            print("🔄 [0,1,1] → RIGHT TURN")
            self.px.set_dir_servo_angle(-20)  # 음수로 변경 (우회전)
            self.px.forward(12)

        # [1,0,0] = [왼쪽 흰선, 중앙 검은바닥, 오른쪽 검은바닥] → 강한 좌회전
        elif left_on and not center_on and not right_on:
            print("🔄 [1,0,0] → STRONG LEFT TURN")
            self.px.set_dir_servo_angle(30)  # 양수로 변경 (강한 좌회전)
            self.px.forward(10)

        # [0,0,1] = [왼쪽 검은바닥, 중앙 검은바닥, 오른쪽 흰선] → 강한 우회전
        elif not left_on and not center_on and right_on:
            print("🔄 [0,0,1] → STRONG RIGHT TURN")
            self.px.set_dir_servo_angle(-30)  # 음수로 변경 (강한 우회전)
            self.px.forward(10)

        else:
            # 기타 상황 - 정지 (예: [0,0,0])
            print("⚠️ Unknown line status, stopping")
            self.px.stop()
            return False

        return True

    def push_sign_unique(self, tok: str):
        """중복 없이 버퍼에 추가 (최신 우선, 최대 2개 유지) - 로직 개선"""
        if tok in self.sign_buffer:
            # 이미 있으면 제거 (순서 보장)
            self.sign_buffer.remove(tok)
        # 새로 추가 (최신)
        self.sign_buffer.append(tok)

    def push_signs(self, signs):
        """서버 감지 결과를 버퍼에 반영"""
        valid = {'turn_left', 'turn_right', 'go_straight', 'stop'}
        for s in signs:
            if s in valid:
                self.push_sign_unique(s)
        if signs:
            self.last_sign_time = time.time()
        # 디버그
        if self.sign_buffer:
            print(f"🧭 Buffered signs (max2): {list(self.sign_buffer)}")

    def run_ordered_actions_once(self):
        """
        Stop 우선 → 가장 최신 비-Stop 1개 실행.
        실행 후 버퍼 초기화.
        """
        if not self.sign_buffer:
            return False

        did = False

        # 1) Stop 우선
        if 'stop' in self.sign_buffer:
            print("🛑 Executing STOP (3s)")
            self.px.stop()
            time.sleep(3.0)
            did = True

        # 2) 최신 비-Stop 1개
        non_stop = [a for a in list(self.sign_buffer)[::-1] if a != 'stop']
        if non_stop:
            action = non_stop[0]
            print(f"🚀 Executing action after STOP (or solo): {action}")
            self.execute_action(action)
            did = True

        # 3) 초기화
        self.sign_buffer.clear()
        self.last_sign_time = None
        return did

    def execute_action(self, action: str):
        """문자열 동작 실행 - picarx 함수 사용"""
        if action == 'turn_left':
            self.turn_left_intersection()
        elif action == 'turn_right':
            self.turn_right_intersection()
        elif action == 'go_straight':
            self.go_straight_intersection()
        else:
            # 안전 기본값
            self.go_straight_intersection()

    def turn_left_intersection(self):
        """좌회전 실행 - picarx 함수 사용 (반대로 수정)"""
        print("🔄 Executing LEFT TURN")
        self.px.set_dir_servo_angle(0)
        self.px.forward(15)
        time.sleep(0.5)
        self.px.set_dir_servo_angle(-45)  # 음수로 변경 (좌회전)
        self.px.forward(12)
        time.sleep(1.2)
        self.px.set_dir_servo_angle(0)

    def turn_right_intersection(self):
        """우회전 실행 - picarx 함수 사용 (반대로 수정)"""
        print("🔄 Executing RIGHT TURN")
        self.px.set_dir_servo_angle(0)
        self.px.forward(15)
        time.sleep(0.3)
        self.px.set_dir_servo_angle(45)  # 양수로 변경 (우회전)
        self.px.forward(12)
        time.sleep(1.2)
        self.px.set_dir_servo_angle(0)

    def go_straight_intersection(self):
        """직진 실행 - picarx 함수 사용"""
        print("➡️ Executing GO STRAIGHT")
        self.px.set_dir_servo_angle(0)
        self.px.forward(20)
        time.sleep(1.0)

    def run(self):
        print("🚀 Starting autonomous robot client (picarx)...")
        comm_thread = threading.Thread(target=self.communication_thread, daemon=True)
        comm_thread.start()
        print("✅ Robot main loop started! Press Ctrl+C to stop.")

        try:
            while self.running:
                sensor_values = self.read_line_sensors()
                line_status = self.get_line_status(sensor_values)

                # 디버그 출력 (주기적)
                if hasattr(self, '_debug_count'):
                    self._debug_count += 1
                else:
                    self._debug_count = 0

                if self._debug_count % 50 == 0:  # 50번마다 출력
                    print(f"[SENS] L:{sensor_values[0]} C:{sensor_values[1]} R:{sensor_values[2]} → Line:{line_status}")

                if self.detect_intersection(line_status):
                    if not self.intersection_detected:
                        print("🚦 INTERSECTION DETECTED!")
                        self.intersection_detected = True

                        # 최근 인식이 타임아웃 내에 있으면 우선순위 실행
                        if (self.last_sign_time is not None and
                                (time.time() - self.last_sign_time) <= self.SIGN_TIMEOUT and
                                self.sign_buffer):
                            executed = self.run_ordered_actions_once()
                            if not executed:
                                # 버퍼가 있었지만 실행 안 된 경우(이례적): 기본 직진
                                print("⚠️ No action executed, going straight by default")
                                self.go_straight_intersection()
                        else:
                            # 감지 없으면 기본 직진
                            print("⚠️ No signs detected, going straight by default")
                            self.go_straight_intersection()

                        time.sleep(0.5)
                else:
                    self.intersection_detected = False
                    self.line_following(line_status)

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n⚠️  Ctrl+C pressed.")
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n🧹 Cleaning up...")
        self.running = False
        try:
            self.px.stop()
            self.px.set_dir_servo_angle(0)
            self.picam2.stop()
            if self.client_socket:
                self.client_socket.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        print("✅ Cleanup completed.")


def main():
    """메인 함수 - 클라이언트 실행"""
    parser = argparse.ArgumentParser(description='Autonomous Robot Client (picarx)')
    parser.add_argument('--server', default='127.0.0.1',
                        help='Server IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8888,
                        help='Server port (default: 8888)')

    args = parser.parse_args()

    print("🤖 Autonomous Robot - Client Mode (picarx)")
    print("=" * 50)
    print(f"Server: {args.server}:{args.port}")
    print(f"Line Threshold: 580")
    print("=" * 50)

    if not HARDWARE_AVAILABLE:
        print("❌ Picamera2 library not available. Cannot run client mode.")
        print("Required: picamera2")
        return

    try:
        robot = RemoteRobotClient(server_host=args.server, server_port=args.port)
        robot.run()
    except Exception as e:
        print(f"❌ Client failed to start: {e}")


if __name__ == "__main__":
    main()