import torch
import cv2
import numpy as np
import time
import random
import json
import os
from djitellopy import Tello

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Rule-based Drone Agent with Bounding Box Obstacle Avoidance
class RuleBasedDroneAgent:
    def __init__(self):
        self.safety_threshold = 0.07      # 0.15 이하면 "위험한 거리" (기존 0.4 → 0.15)
        self.critical_threshold = 0.03    # 0.05 이하면 "매우 위험한 거리" (기존 0.2 → 0.05)  
        self.bbox_safety_threshold = 0.07 # 0.12 이하면 "바운딩박스 기준 위험" (기존 0.3 → 0.12)
        
        # 복귀 모드용 더 엄격한 임계값
        self.critical_threshold_return = 0.02  # 복귀 시 매우 위험한 거리
        
        # 연속 후진 방지를 위한 변수들
        self.last_action = None
        self.backward_count = 0
        self.max_consecutive_backwards = 2
        self.forward_preference = 0.0
        
        # 회전 동작 제거된 액션 리스트
        self.actions = [
            "Move Forward", "Move Backward", "Move Left", "Move Right",
            "Move Up", "Move Down", "Stay"
        ]
        print("Rule-based Agent initialized - Bounding Box Obstacle Avoidance!")
    
    def analyze_bounding_box_obstacles(self, depth_map, detections, frame_shape):
        """바운딩 박스 기반 장애물 분석"""
        obstacles = []
        H, W = depth_map.shape
        frame_h, frame_w = frame_shape[:2]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            conf = detection['confidence']
            class_id = detection['class']
            
            # 프레임 크기에 맞춰 바운딩 박스 좌표 스케일링
            x1 = int(x1 * W / frame_w)
            x2 = int(x2 * W / frame_w)
            y1 = int(y1 * H / frame_h)
            y2 = int(y2 * H / frame_h)
            
            # 바운딩 박스 영역의 깊이 정보 추출
            bbox_region = depth_map[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            
            if bbox_region.size > 0:
                bbox_depth_mean = np.mean(bbox_region)
                bbox_depth_min = np.min(bbox_region)
                bbox_depth_std = np.std(bbox_region)
                
                # 바운딩 박스 중심점과 크기 계산
                center_x = (x1 + x2) / 2 / W  # 정규화된 중심 x좌표 (0~1)
                center_y = (y1 + y2) / 2 / H  # 정규화된 중심 y좌표 (0~1)
                width = (x2 - x1) / W
                height = (y2 - y1) / H
                
                obstacle = {
                    'bbox': [x1, y1, x2, y2],
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'depth_mean': bbox_depth_mean,
                    'depth_min': bbox_depth_min,
                    'depth_std': bbox_depth_std,
                    'confidence': conf,
                    'class_id': class_id,
                    'is_dangerous': bbox_depth_mean < self.bbox_safety_threshold,
                    'is_critical': bbox_depth_min < self.critical_threshold
                }
                obstacles.append(obstacle)
                
        return obstacles
    
    def get_avoidance_direction(self, obstacles, frame_center_x=0.5):
        """장애물을 피하기 위한 방향 결정"""
        if not obstacles:
            return "forward"
        
        # 위험한 장애물들만 필터링
        dangerous_obstacles = [obs for obs in obstacles if obs['is_dangerous']]
        
        if len(dangerous_obstacles) <= 4:
            return "forward"
        
        # 전방 중앙 영역의 장애물 확인 (중앙 40% 영역)
        front_center_obstacles = [
            obs for obs in dangerous_obstacles 
            if 0.3 < obs['center_x'] < 0.7 and obs['center_y'] < 0.6
        ]
        
        if not front_center_obstacles:
            return "forward"
        
        # 좌우 공간 분석
        left_obstacles = [obs for obs in dangerous_obstacles if obs['center_x'] < 0.4]
        right_obstacles = [obs for obs in dangerous_obstacles if obs['center_x'] > 0.6]
        
        # 좌우 공간의 평균 거리 계산
        left_space = 1.0  # 기본값
        right_space = 1.0
        
        if left_obstacles:
            left_space = np.mean([obs['depth_mean'] for obs in left_obstacles])
        
        if right_obstacles:
            right_space = np.mean([obs['depth_mean'] for obs in right_obstacles])
        
        # 상하 공간 분석
        upper_obstacles = [obs for obs in dangerous_obstacles if obs['center_y'] < 0.4]
        lower_obstacles = [obs for obs in dangerous_obstacles if obs['center_y'] > 0.6]
        
        upper_space = 1.0
        lower_space = 1.0
        
        if upper_obstacles:
            upper_space = np.mean([obs['depth_mean'] for obs in upper_obstacles])
        
        if lower_obstacles:
            lower_space = np.mean([obs['depth_mean'] for obs in lower_obstacles])
        
        print(f"Space analysis - Left: {left_space:.3f}, Right: {right_space:.3f}, Up: {upper_space:.3f}, Down: {lower_space:.3f}")
        
        # 가장 넓은 공간으로 회피 결정
        spaces = {
            'left': left_space,
            'right': right_space,
            'up': upper_space,
            'down': lower_space
        }
        
        # 좌우 우선, 상하는 보조적으로 사용
        if left_space > right_space + 0.1:
            return "left"
        elif right_space > left_space + 0.1:
            return "right"
        elif upper_space > lower_space + 0.1:
            return "up"
        elif lower_space > upper_space + 0.1:
            return "down"
        else:
            # 비슷하면 교대로 움직임
            if not hasattr(self, 'alternate_direction'):
                self.alternate_direction = 'left'
            
            if self.alternate_direction == 'left':
                self.alternate_direction = 'right'
                return 'left'
            else:
                self.alternate_direction = 'left'
                return 'right'
    
    def act_return_mode(self, state, obstacles, intended_action):
        """복귀 모드 전용 액션 결정 - 장애물 회피 완전 비활성화"""
        print(f"Return mode: intended action = {self.get_action_name(intended_action)}")
        
        # 복귀 모드에서는 장애물 무시하고 의도된 액션 그대로 실행
        print(f"✓ RETURNING - Executing intended action without obstacle avoidance: {self.get_action_name(intended_action)}")
        return intended_action
    
    def act(self, state, obstacles, mission_state="EXPLORATION", intended_action=None):
        """바운딩 박스 기반 장애물 회피 의사결정"""
        
        # 복귀 모드일 때는 다른 로직 사용
        if mission_state == "RETURNING" and intended_action is not None:
            return self.act_return_mode(state, obstacles, intended_action)
        
        # 기존 탐색 모드 로직
        regions = state[:8]
        center_mean = state[8]
        center_min = state[9]
        left_mean = state[12]
        right_mean = state[13]
        
        # 전방 영역 분석
        front_regions = regions[:4]  
        front_min = np.min(front_regions)
        front_mean = np.mean(front_regions)
        
        # 연속 동작 체크
        if hasattr(self, 'last_action') and self.last_action is not None:
            if self.last_action == 1:  # 후진
                self.backward_count = getattr(self, 'backward_count', 0) + 1
            else:
                self.backward_count = 0
        else:
            self.backward_count = 0
            self.last_action = None
        
        print(f"General analysis - Front: min={front_min:.3f}, mean={front_mean:.3f} | Center: {center_mean:.3f}")
        print(f"Detected {len(obstacles)} objects, {len([obs for obs in obstacles if obs['is_dangerous']])} dangerous")
        
        # 바운딩 박스 기반 회피 방향 결정
        avoidance_direction = self.get_avoidance_direction(obstacles)
        
        # 매우 위험한 상황 체크 (critical threshold)
        critical_obstacles = [obs for obs in obstacles if obs['is_critical']]
        
        if critical_obstacles:
            print("⚠ CRITICAL - Very close obstacle detected!")
            # 즉시 후진
            if self.backward_count < self.max_consecutive_backwards:
                print("🚨 EMERGENCY BACKWARD")
                self.last_action = 1
                return 1
        
        # 일반적인 장애물 회피 로직
        if avoidance_direction == "forward":
            # 추가 안전 체크 - 일반 깊이 맵도 확인
            if front_mean > self.safety_threshold and center_mean > self.safety_threshold:
                print("✓ FORWARD - Path clear (bbox + depth check)")
                self.last_action = 0
                return 0
            elif center_mean > self.safety_threshold * 0.7:
                print("✓ FORWARD - Center acceptable")
                self.last_action = 0
                return 0
        
        # 회피 동작 실행
        if avoidance_direction == "left":
            print("← LEFT avoidance (bbox-based)")
            self.last_action = 2
            return 2
        elif avoidance_direction == "right":
            print("→ RIGHT avoidance (bbox-based)")
            self.last_action = 3
            return 3
        elif avoidance_direction == "up":
            print("↑ UP avoidance (bbox-based)")
            self.last_action = 4
            return 4
        elif avoidance_direction == "down":
            print("↓ DOWN avoidance (bbox-based)")
            self.last_action = 5
            return 5
        
        # 최후의 수단 - 후진
        if self.backward_count < self.max_consecutive_backwards:
            print("⏪ BACKWARD - Last resort")
            self.last_action = 1
            return 1
        
        # 강제 전진 (데드락 방지)
        print("🚀 FORCE FORWARD - Breaking deadlock")
        self.last_action = 0
        self.backward_count = 0
        return 0
    
    def get_action_name(self, action_idx):
        return self.actions[action_idx]
    
    def return_mode_init(self):
        """복귀 모드 초기화"""
        self.return_detour_count = 0

def extract_compact_state(depth_map, detections, drone_info):
    """Extract state information with corrected depth interpretation"""
    H, W = depth_map.shape
    regions = []
    
    # 깊이 맵 정규화 - 높은 값 = 멀리, 낮은 값 = 가까이로 일관성 유지
    depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-6)
    
    # MiDaS inverse depth 보정 - 실제 거리로 변환
    depth_corrected = 1.0 - depth_normalized
    
    # 8개 영역 분석 (상단 4개 = 전방, 하단 4개 = 측면/후방)
    for i in range(2):
        for j in range(4):
            h_start, h_end = i * H // 2, (i + 1) * H // 2
            w_start, w_end = j * W // 4, (j + 1) * W // 4
            region = depth_corrected[h_start:h_end, w_start:w_end]
            regions.append(np.mean(region))
    
    # 전방 중앙 영역 강조 분석
    center_h = slice(H//3, 2*H//3)
    center_w = slice(W//3, 2*W//3)
    center_region = depth_corrected[center_h, center_w]
    center_mean = np.mean(center_region)
    center_min = np.min(center_region)
    center_std = np.std(center_region)
    
    # 좌우 분석
    left_region = depth_corrected[:, :W//2]
    right_region = depth_corrected[:, W//2:]
    left_mean = np.mean(left_region)
    right_mean = np.mean(right_region)
    
    # 객체 감지 처리
    person_detected = 0
    person_distance = 15.0
    obstacle_count = 0
    
    for detection in detections:
        if detection['class'] == 0:  # person
            person_detected = 1
            person_distance = min(detection.get('distance', 15.0), 15.0)
        else:
            obstacle_count += 1
    
    # 배터리
    battery_level = drone_info.get('battery', 100) / 100.0
    
    # 상태 벡터
    state = np.array([
        *regions,  # 8개 영역 거리
        center_mean, center_min, center_std,  # 중앙 영역
        left_mean, right_mean,  # 좌우 영역
        np.mean(depth_corrected[:H//2, :]),  # 전방 전체 평균
        np.mean(depth_corrected[H//2:, :]),  # 후방 전체 평균
        np.min(depth_corrected[:, :W//2]),   # 좌측 최소
        np.min(depth_corrected[:, W//2:]),   # 우측 최소
        person_detected, person_distance / 15.0, min(obstacle_count, 10) / 10.0,
        battery_level, 
        np.mean(depth_corrected), np.min(depth_corrected), np.max(depth_corrected),
        np.std(depth_corrected),
        np.percentile(depth_corrected, 25),
        np.percentile(depth_corrected, 50),
        np.percentile(depth_corrected, 75),
        np.sum(depth_corrected < 0.3) / (H * W),  # 가까운 픽셀 비율
        np.sum(depth_corrected < 0.5) / (H * W),  # 중간 거리 픽셀 비율
        np.sum(depth_corrected > 0.7) / (H * W),  # 먼 픽셀 비율
        len(detections) / 10.0
    ], dtype=np.float32)
    
    return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)

# Main execution
print("Loading models...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).to(device).eval()
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

print("Initializing Rule-based Agent with Bounding Box Obstacle Avoidance...")
rule_agent = RuleBasedDroneAgent()

print("Connecting to Tello...")
tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"Battery: {battery}%")
if battery < 20:
    print("Battery too low! Please charge the drone.")
    exit()

print("Starting video stream...")
tello.streamon()
time.sleep(2)

print("Testing video stream - Press 't' to takeoff, 'q' to quit")
stream_test_active = True
while stream_test_active:
    frame = tello.get_frame_read().frame
    if frame is not None:
        display_frame = cv2.resize(frame, (640, 480))
        cv2.putText(display_frame, "BBox Obstacle Avoidance (No Return Avoidance) - Press 't' to takeoff", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Battery: {tello.get_battery()}% | WARNING: No obstacle avoidance during return!", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow('BBox Obstacle Avoidance System', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            stream_test_active = False
            print("Takeoff confirmed!")
        elif key == ord('q'):
            print("Operation cancelled.")
            tello.streamoff()
            cv2.destroyAllWindows()
            exit()
    else:
        print("Waiting for video stream...")
        time.sleep(0.1)

print("Taking off...")
tello.takeoff()
time.sleep(3)

# Mission variables
mission_state = "EXPLORATION"
path_history = []
person_detection_count = 0
person_detection_start_time = None
return_index = 0
start_time = time.time()

last_action_time = time.time()
action_interval = 0.8
frame_skip = 0
step_count = 0

print("Mission started: Bounding Box-based obstacle avoidance exploration")
print("⚠️ WARNING: Obstacle avoidance is DISABLED during return mode!")
print("Objective: Move forward, avoid obstacles using bbox analysis, find person for 3 seconds, then return")

def execute_drone_action(action_idx):
    """Execute drone action with safety checks"""
    global last_action_time, path_history
    current_time = time.time()
    if current_time - last_action_time < action_interval:
        return False
    
    action_name = rule_agent.get_action_name(action_idx)
    try:
        print(f"Executing action: {action_name}")
        # if action_name == "Move Forward":
        #     tello.move_forward(15)
        # elif action_name == "Move Backward":
        #     tello.move_back(30)
        # elif action_name == "Move Left":
        #     tello.move_left(25)
        # elif action_name == "Move Right":
        #     tello.move_right(25)
        # elif action_name == "Move Up":
        #     tello.move_up(20)
        # elif action_name == "Move Down":
        #     tello.move_down(20)
        # elif action_name == "Stay":
        #     pass

        time.sleep(2)
        
        last_action_time = current_time
        # EXPLORATION 모드에서만 경로 기록, Stay 제외
        if mission_state == "EXPLORATION" and action_name != "Stay":
            path_history.append(action_idx)
            print(f"Path recorded: {action_name} (Total: {len(path_history)})")
        return True
    except Exception as e:
        print(f"Action execution error: {e}")
        return False

# 복귀를 위한 반대 액션 매핑
def get_reverse_action(action_idx):
    """Get reverse action for returning"""
    reverse_map = {
        0: 1,  # Forward -> Backward
        1: 0,  # Backward -> Forward
        2: 3,  # Left -> Right
        3: 2,  # Right -> Left
        4: 5,  # Up -> Down
        5: 4,  # Down -> Up
        6: 6   # Stay -> Stay
    }
    return reverse_map.get(action_idx, 6)

try:
    print("Starting main control loop with Bounding Box Obstacle Avoidance...")
    while True:
        frame = tello.get_frame_read().frame
        if frame is None:
            continue
        
        frame_skip += 1
        if frame_skip % 2 != 0:
            continue
        
        # Battery check
        current_battery = tello.get_battery()
        if current_battery < 15:
            print("Low battery! Landing...")
            break
        
        # Object detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)
        detections = []
        person_detected_now = False
        
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf > 0.4:  # 신뢰도 임계값 약간 낮춤
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                detection = {
                    'box': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': class_id,
                    'distance': 5.0
                }
                detections.append(detection)
                if class_id == 0:  # person detected
                    person_detected_now = True
        
        # Depth estimation
        input_tensor = transform(rgb_frame).to(device)
        with torch.no_grad():
            depth = midas(input_tensor)
            depth_map = depth.squeeze().cpu().numpy()
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        
        # Extract state and analyze obstacles
        drone_info = {'battery': current_battery}
        current_state = extract_compact_state(depth_map, detections, drone_info)
        obstacles = rule_agent.analyze_bounding_box_obstacles(depth_map, detections, frame.shape)
        
        # Mission logic
        if mission_state == "EXPLORATION":
            # Person detection logic
            if person_detected_now:
                if person_detection_start_time is None:
                    person_detection_start_time = time.time()
                    person_detection_count = 1
                    print("🔍 Person detected! Starting 3-second countdown...")
                else:
                    person_detection_count += 1
                    elapsed = time.time() - person_detection_start_time
                    print(f"👤 Person tracking: {elapsed:.1f}/3.0 seconds")
                    if elapsed >= 3.0:
                        print("✅ Person detected for 3 seconds! Switching to return mode.")
                        mission_state = "RETURNING"
                        return_index = len(path_history) - 1
                        # 복귀 모드 초기화
                        rule_agent.return_mode_init()
                        print(f"🔙 Starting return journey from step {return_index}")
                        print("⚠️ WARNING: Obstacle avoidance is now DISABLED!")
            else:
                person_detection_start_time = None
                person_detection_count = 0
            
            # Bounding box-based action decision (탐색 모드)
            if person_detection_start_time is not None:
                # 사람을 발견했을 때는 정지하여 안정적으로 추적
                action = 6  # Stay
                print("👤 Staying to track person...")
            else:
                action = rule_agent.act(current_state, obstacles, mission_state)
            
            action_executed = execute_drone_action(action)
            
            if action_executed:
                action_name = rule_agent.get_action_name(action)
                dangerous_count = len([obs for obs in obstacles if obs['is_dangerous']])
                print(f"🚁 Exploration: {action_name} | Dangerous objects: {dangerous_count} | Step: {step_count}")
                step_count += 1
        
        elif mission_state == "RETURNING":
            if return_index >= 0:
                original_action = path_history[return_index]
                reverse_action = get_reverse_action(original_action)
                
                print(f"🔙 Return step {len(path_history) - return_index}/{len(path_history)}")
                print(f"Original: {rule_agent.get_action_name(original_action)} → Reverse: {rule_agent.get_action_name(reverse_action)}")
                
                # 복귀 모드에서는 장애물 회피 없이 그대로 실행
                final_action = rule_agent.act(current_state, obstacles, mission_state, reverse_action)
                action_executed = execute_drone_action(final_action)
                
                if action_executed:
                    action_name = rule_agent.get_action_name(final_action)
                    original_name = rule_agent.get_action_name(original_action)
                    reverse_name = rule_agent.get_action_name(reverse_action)
                    
                    # 복귀 진행 - 장애물 회피 없이 항상 계획대로 진행
                    print(f"✅ Return: {action_name} (NO OBSTACLE AVOIDANCE)")
                    return_index -= 1  # 정상적으로 한 스텝 되돌아감
                    
                    print(f"Return progress: {len(path_history) - return_index - 1}/{len(path_history)} steps completed")
            else:
                print("🏠 Return journey completed! Landing...")
                mission_state = "LANDING"
        
        elif mission_state == "LANDING":
            print("🛬 Mission completed - Landing...")
            break
        
        # Display frame with enhanced information
        display_frame = cv2.resize(frame, (960, 720))
        
        # Draw bounding boxes for detected objects
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            conf = detection['confidence']
            class_id = detection['class']
            
            # Scale coordinates to display frame
            scale_x = 960 / frame.shape[1]
            scale_y = 720 / frame.shape[0]
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            # Color coding: red for person, yellow for others
            color = (0, 0, 255) if class_id == 0 else (0, 255, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"Person {conf:.2f}" if class_id == 0 else f"Obj {conf:.2f}"
            cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw obstacle analysis overlay
        dangerous_obstacles = [obs for obs in obstacles if obs['is_dangerous']]
        critical_obstacles = [obs for obs in obstacles if obs['is_critical']]
        
        # Status information
        status_y = 30
        cv2.putText(display_frame, f"Mission: {mission_state}", (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        status_y += 30
        
        cv2.putText(display_frame, f"Battery: {current_battery}% | Step: {step_count}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 25
        
        # Add warning if in return mode
        if mission_state == "RETURNING":
            cv2.putText(display_frame, "⚠️ OBSTACLE AVOIDANCE DISABLED!", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            status_y += 25
        
        cv2.putText(display_frame, f"Objects: {len(detections)} | Dangerous: {len(dangerous_obstacles)} | Critical: {len(critical_obstacles)}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
        status_y += 25
        
        if mission_state == "EXPLORATION":
            if person_detected_now and person_detection_start_time:
                elapsed = time.time() - person_detection_start_time
                cv2.putText(display_frame, f"Person Tracking: {elapsed:.1f}/3.0s", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Exploring... Looking for person", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        elif mission_state == "RETURNING":
            remaining_steps = return_index + 1
            total_steps = len(path_history)
            progress = ((total_steps - remaining_steps) / total_steps) * 100 if total_steps > 0 else 100
            cv2.putText(display_frame, f"Returning: {progress:.1f}% ({total_steps - remaining_steps}/{total_steps})", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        
        # Draw depth visualization (small overlay)
        depth_vis = cv2.applyColorMap(((1.0 - depth_map) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth_vis_small = cv2.resize(depth_vis, (240, 180))
        display_frame[540:720, 720:960] = depth_vis_small
        cv2.putText(display_frame, "Depth (Red=Close)", (725, 535), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('BBox Obstacle Avoidance System', display_frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("🛑 Mission terminated by user")
            break
        elif key == ord('l'):  # Manual landing
            print("🛬 Manual landing requested")
            break
        elif key == ord('r') and mission_state == "EXPLORATION":  # Force return
            print("🔙 Forced return mode activated")
            mission_state = "RETURNING"
            return_index = len(path_history) - 1
            rule_agent.return_mode_init()
            print("⚠️ WARNING: Obstacle avoidance is now DISABLED!")
        
        # Safety timeout (30 minutes max)
        if time.time() - start_time > 1800:  # 30 minutes
            print("⏰ Mission timeout reached")
            break
        
        time.sleep(0.05)  # Small delay to prevent CPU overload

except KeyboardInterrupt:
    print("\n🛑 Mission interrupted by user")
except Exception as e:
    print(f"❌ Mission error: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("🛬 Landing drone...")
    try:
        tello.land()
        time.sleep(3)
        print("✅ Drone landed safely")
    except:
        print("⚠️ Landing command failed")
    
    print("📹 Stopping video stream...")
    try:
        tello.streamoff()
        cv2.destroyAllWindows()
        print("✅ Video stream stopped")
    except:
        print("⚠️ Stream stop failed")
    
    print("📊 Mission Summary:")
    print(f"   Total exploration steps: {len(path_history)}")
    print(f"   Mission state: {mission_state}")
    print(f"   Final battery: {tello.get_battery() if 'current_battery' in locals() else 'Unknown'}%")
    if mission_state == "RETURNING":
        completed_return_steps = len(path_history) - return_index - 1
        print(f"   Return progress: {completed_return_steps}/{len(path_history)} steps")
    print("⚠️ Note: Obstacle avoidance was DISABLED during return mode")
    print("🎯 Mission terminated successfully")