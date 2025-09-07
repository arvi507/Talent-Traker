import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import logging

class VideoAnalyzer:
    """Advanced video analysis using OpenCV and MediaPipe for sports assessment"""
    
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize face detection for verification
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
    def analyze_video(self, video_path: str) -> Dict:
        """
        Comprehensive video analysis including pose estimation, frame quality, and motion detection
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Unable to open video file")
        
        analysis_data = {
            'total_frames': 0,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'duration': 0,
            'resolution': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            'pose_detections': [],
            'frame_quality_scores': [],
            'motion_intensity': [],
            'face_detections': [],
            'key_movements': []
        }
        
        frame_count = 0
        pose_landmarks_history = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Pose detection
                pose_results = self.pose.process(rgb_frame)
                if pose_results.pose_landmarks:
                    landmarks = self._extract_pose_landmarks(pose_results.pose_landmarks)
                    analysis_data['pose_detections'].append(landmarks)
                    pose_landmarks_history.append(landmarks)
                
                # Face detection for verification
                face_results = self.face_detection.process(rgb_frame)
                if face_results.detections:
                    face_data = self._extract_face_data(face_results.detections)
                    analysis_data['face_detections'].append(face_data)
                
                # Frame quality assessment
                quality_score = self._assess_frame_quality(frame)
                analysis_data['frame_quality_scores'].append(quality_score)
                
                # Motion intensity calculation
                if frame_count > 1 and len(pose_landmarks_history) >= 2:
                    motion = self._calculate_motion_intensity(
                        pose_landmarks_history[-2], pose_landmarks_history[-1]
                    )
                    analysis_data['motion_intensity'].append(motion)
                
                # Process every 5th frame to reduce computation
                if frame_count % 5 != 0:
                    continue
                    
        except Exception as e:
            logging.error(f"Error processing frame {frame_count}: {str(e)}")
            
        finally:
            cap.release()
        
        # Calculate derived metrics
        analysis_data['total_frames'] = frame_count
        analysis_data['duration'] = frame_count / analysis_data['fps'] if analysis_data['fps'] > 0 else 0
        analysis_data['pose_detection_rate'] = len(analysis_data['pose_detections']) / frame_count if frame_count > 0 else 0
        analysis_data['average_frame_quality'] = np.mean(analysis_data['frame_quality_scores']) if analysis_data['frame_quality_scores'] else 0
        analysis_data['average_motion_intensity'] = np.mean(analysis_data['motion_intensity']) if analysis_data['motion_intensity'] else 0
        
        # Detect key movement phases
        analysis_data['key_movements'] = self._detect_movement_phases(pose_landmarks_history)
        
        return analysis_data
    
    def _extract_pose_landmarks(self, landmarks) -> Dict:
        """Extract key pose landmarks for analysis"""
        landmark_dict = {}
        
        # Key landmarks for fitness assessment
        key_landmarks = {
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE
        }
        
        for name, landmark_id in key_landmarks.items():
            landmark = landmarks.landmark[landmark_id]
            landmark_dict[name] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        return landmark_dict
    
    def _extract_face_data(self, detections) -> Dict:
        """Extract face detection data for verification"""
        face_data = []
        
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            face_data.append({
                'confidence': detection.score[0],
                'bbox': {
                    'x': bbox.xmin,
                    'y': bbox.ymin,
                    'width': bbox.width,
                    'height': bbox.height
                }
            })
        
        return face_data
    
    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess frame quality using multiple metrics"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # Brightness assessment
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0  # Optimal around 128
        
        # Contrast assessment
        contrast_score = np.std(gray) / 255.0
        
        # Combined quality score
        quality_score = (sharpness_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
        
        return min(quality_score, 1.0)
    
    def _calculate_motion_intensity(self, landmarks1: Dict, landmarks2: Dict) -> float:
        """Calculate motion intensity between two frames"""
        if not landmarks1 or not landmarks2:
            return 0.0
        
        total_movement = 0.0
        landmark_count = 0
        
        # Calculate movement for key body points
        key_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 
                     'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
        for point in key_points:
            if point in landmarks1 and point in landmarks2:
                # Calculate Euclidean distance
                dx = landmarks2[point]['x'] - landmarks1[point]['x']
                dy = landmarks2[point]['y'] - landmarks1[point]['y']
                dz = landmarks2[point]['z'] - landmarks1[point]['z']
                
                movement = np.sqrt(dx**2 + dy**2 + dz**2)
                total_movement += movement
                landmark_count += 1
        
        return total_movement / landmark_count if landmark_count > 0 else 0.0
    
    def _detect_movement_phases(self, pose_history: List[Dict]) -> List[Dict]:
        """Detect key movement phases in the exercise"""
        if len(pose_history) < 10:
            return []
        
        phases = []
        motion_sequence = []
        
        # Calculate motion sequence
        for i in range(1, len(pose_history)):
            motion = self._calculate_motion_intensity(pose_history[i-1], pose_history[i])
            motion_sequence.append(motion)
        
        if not motion_sequence:
            return phases
        
        # Find peaks and valleys in motion
        motion_array = np.array(motion_sequence)
        mean_motion = np.mean(motion_array)
        std_motion = np.std(motion_array)
        
        # Detect high activity phases
        threshold = mean_motion + 0.5 * std_motion
        
        in_high_activity = False
        phase_start = 0
        
        for i, motion in enumerate(motion_array):
            if motion > threshold and not in_high_activity:
                # Start of high activity phase
                in_high_activity = True
                phase_start = i
            elif motion <= threshold and in_high_activity:
                # End of high activity phase
                in_high_activity = False
                phases.append({
                    'type': 'high_activity',
                    'start_frame': phase_start,
                    'end_frame': i,
                    'duration': i - phase_start,
                    'intensity': np.mean(motion_array[phase_start:i])
                })
        
        return phases
    
    def extract_movement_metrics(self, video_path: str, test_type: str) -> Dict:
        """Extract test-specific movement metrics"""
        analysis = self.analyze_video(video_path)
        
        if test_type == "Vertical Jump":
            return self._analyze_vertical_jump(analysis)
        elif test_type == "Sit-ups (1 minute)":
            return self._analyze_situps(analysis)
        elif test_type == "50m Sprint":
            return self._analyze_sprint(analysis)
        elif test_type == "Push-ups":
            return self._analyze_pushups(analysis)
        elif test_type == "Flexibility Test":
            return self._analyze_flexibility(analysis)
        else:
            return {'error': f'Unknown test type: {test_type}'}
    
    def _analyze_vertical_jump(self, analysis: Dict) -> Dict:
        """Analyze vertical jump performance"""
        pose_detections = analysis.get('pose_detections', [])
        
        if not pose_detections:
            return {'error': 'No pose detections found'}
        
        # Track hip height over time
        hip_heights = []
        for detection in pose_detections:
            if 'left_hip' in detection and 'right_hip' in detection:
                # Average of left and right hip y-coordinates (lower y = higher position)
                avg_hip_y = (detection['left_hip']['y'] + detection['right_hip']['y']) / 2
                hip_heights.append(1.0 - avg_hip_y)  # Invert so higher values = higher jump
        
        if not hip_heights:
            return {'error': 'Could not track hip movement'}
        
        # Find the jump metrics
        baseline_height = np.mean(hip_heights[:10]) if len(hip_heights) >= 10 else hip_heights[0]
        max_height = max(hip_heights)
        jump_height = max_height - baseline_height
        
        # Detect takeoff and landing phases
        takeoff_frame = 0
        landing_frame = len(hip_heights) - 1
        
        for i, height in enumerate(hip_heights):
            if height > baseline_height + (jump_height * 0.1):  # 10% of jump height
                takeoff_frame = i
                break
        
        for i in range(len(hip_heights) - 1, -1, -1):
            if hip_heights[i] > baseline_height + (jump_height * 0.1):
                landing_frame = i
                break
        
        flight_time = (landing_frame - takeoff_frame) / analysis.get('fps', 30)
        
        return {
            'jump_height_normalized': jump_height,
            'flight_time_seconds': max(flight_time, 0),
            'takeoff_frame': takeoff_frame,
            'landing_frame': landing_frame,
            'baseline_height': baseline_height,
            'max_height': max_height,
            'total_frames': len(hip_heights)
        }
    
    def _analyze_situps(self, analysis: Dict) -> Dict:
        """Analyze sit-ups performance"""
        pose_detections = analysis.get('pose_detections', [])
        
        if not pose_detections:
            return {'error': 'No pose detections found'}
        
        # Track torso angle over time
        torso_angles = []
        rep_count = 0
        in_up_position = False
        
        for detection in pose_detections:
            if all(point in detection for point in ['left_shoulder', 'left_hip', 'left_knee']):
                # Calculate angle between shoulder-hip and hip-knee vectors
                shoulder = detection['left_shoulder']
                hip = detection['left_hip']
                knee = detection['left_knee']
                
                # Vector from hip to shoulder
                vec1 = np.array([shoulder['x'] - hip['x'], shoulder['y'] - hip['y']])
                # Vector from hip to knee  
                vec2 = np.array([knee['x'] - hip['x'], knee['y'] - hip['y']])
                
                # Calculate angle
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                torso_angles.append(angle)
                
                # Count reps based on angle thresholds
                if angle < 60 and not in_up_position:  # Sitting up
                    in_up_position = True
                elif angle > 90 and in_up_position:  # Lying down
                    in_up_position = False
                    rep_count += 1
        
        # Calculate average cadence
        duration = analysis.get('duration', 1)
        cadence = rep_count / duration * 60 if duration > 0 else 0  # reps per minute
        
        return {
            'rep_count': rep_count,
            'cadence_per_minute': cadence,
            'duration_seconds': duration,
            'average_angle': np.mean(torso_angles) if torso_angles else 0,
            'angle_range': max(torso_angles) - min(torso_angles) if torso_angles else 0
        }
    
    def _analyze_sprint(self, analysis: Dict) -> Dict:
        """Analyze sprint performance"""
        pose_detections = analysis.get('pose_detections', [])
        
        if not pose_detections:
            return {'error': 'No pose detections found'}
        
        # Track horizontal movement (assuming camera is stationary)
        positions = []
        for detection in pose_detections:
            if 'left_hip' in detection and 'right_hip' in detection:
                # Average hip position as center of mass proxy
                avg_x = (detection['left_hip']['x'] + detection['right_hip']['x']) / 2
                positions.append(avg_x)
        
        if len(positions) < 10:
            return {'error': 'Insufficient movement data'}
        
        # Calculate movement metrics
        total_displacement = abs(positions[-1] - positions[0])
        
        # Estimate speed (normalized units per second)
        duration = analysis.get('duration', 1)
        estimated_speed = total_displacement / duration if duration > 0 else 0
        
        # Calculate stride frequency
        motion_intensity = analysis.get('motion_intensity', [])
        if motion_intensity:
            avg_intensity = np.mean(motion_intensity)
            stride_frequency = avg_intensity * 30  # Approximate conversion
        else:
            stride_frequency = 0
        
        return {
            'estimated_speed': estimated_speed,
            'total_displacement': total_displacement,
            'duration_seconds': duration,
            'stride_frequency': stride_frequency,
            'movement_consistency': 1.0 - np.std(positions) if len(positions) > 1 else 1.0
        }
    
    def _analyze_pushups(self, analysis: Dict) -> Dict:
        """Analyze push-ups performance"""
        pose_detections = analysis.get('pose_detections', [])
        
        if not pose_detections:
            return {'error': 'No pose detections found'}
        
        # Track elbow angles and body alignment
        rep_count = 0
        in_down_position = False
        elbow_angles = []
        
        for detection in pose_detections:
            if all(point in detection for point in ['left_shoulder', 'left_elbow', 'left_wrist']):
                # Calculate elbow angle
                shoulder = detection['left_shoulder']
                elbow = detection['left_elbow']
                wrist = detection['left_wrist']
                
                # Vectors for angle calculation
                vec1 = np.array([shoulder['x'] - elbow['x'], shoulder['y'] - elbow['y']])
                vec2 = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])
                
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                elbow_angles.append(angle)
                
                # Count reps based on elbow angle
                if angle < 90 and not in_down_position:  # Arms bent (down position)
                    in_down_position = True
                elif angle > 150 and in_down_position:  # Arms extended (up position)
                    in_down_position = False
                    rep_count += 1
        
        # Calculate metrics
        duration = analysis.get('duration', 1)
        cadence = rep_count / duration * 60 if duration > 0 else 0
        
        return {
            'rep_count': rep_count,
            'cadence_per_minute': cadence,
            'duration_seconds': duration,
            'average_elbow_angle': np.mean(elbow_angles) if elbow_angles else 0,
            'angle_range': max(elbow_angles) - min(elbow_angles) if elbow_angles else 0
        }
    
    def _analyze_flexibility(self, analysis: Dict) -> Dict:
        """Analyze flexibility test performance"""
        pose_detections = analysis.get('pose_detections', [])
        
        if not pose_detections:
            return {'error': 'No pose detections found'}
        
        # Track forward reach distance
        reach_distances = []
        
        for detection in pose_detections:
            if all(point in detection for point in ['left_wrist', 'right_wrist', 'left_hip', 'right_hip']):
                # Calculate average wrist position
                avg_wrist_x = (detection['left_wrist']['x'] + detection['right_wrist']['x']) / 2
                avg_hip_x = (detection['left_hip']['x'] + detection['right_hip']['x']) / 2
                
                # Forward reach distance (relative to hip position)
                reach_distance = avg_wrist_x - avg_hip_x
                reach_distances.append(reach_distance)
        
        if not reach_distances:
            return {'error': 'Could not track reach movement'}
        
        # Find maximum reach
        max_reach = max(reach_distances)
        min_reach = min(reach_distances)
        flexibility_range = max_reach - min_reach
        
        return {
            'max_reach_distance': max_reach,
            'flexibility_range': flexibility_range,
            'reach_consistency': 1.0 - np.std(reach_distances) if len(reach_distances) > 1 else 1.0,
            'final_reach': reach_distances[-1] if reach_distances else 0
        }
