import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.video_analysis import VideoAnalyzer
import logging

class FitnessTestProcessor:
    """Process fitness test videos and extract performance metrics"""
    
    def __init__(self):
        self.video_analyzer = VideoAnalyzer()
        
        # Test-specific thresholds and parameters
        self.test_parameters = {
            'Vertical Jump': {
                'min_jump_height': 0.05,  # Minimum normalized jump height
                'max_flight_time': 2.0,   # Maximum reasonable flight time
                'baseline_frames': 10     # Frames to establish baseline
            },
            'Sit-ups (1 minute)': {
                'min_angle_down': 90,     # Minimum angle for down position
                'max_angle_up': 60,       # Maximum angle for up position
                'min_rep_duration': 0.5,  # Minimum time per rep
                'max_duration': 65        # Maximum test duration
            },
            '50m Sprint': {
                'min_displacement': 0.3,  # Minimum movement across frame
                'max_duration': 20,       # Maximum reasonable sprint time
                'min_speed': 0.01         # Minimum speed threshold
            },
            'Push-ups': {
                'min_angle_down': 90,     # Minimum elbow angle for down position
                'max_angle_up': 150,      # Maximum elbow angle for up position
                'min_rep_duration': 0.8   # Minimum time per rep
            },
            'Flexibility Test': {
                'min_reach_time': 2.0,    # Minimum time in reach position
                'stability_threshold': 0.05  # Maximum movement for stable position
            }
        }
    
    def process_test(self, video_path: str, test_type: str) -> Dict:
        """Main method to process fitness test video"""
        try:
            # Basic video analysis
            video_analysis = self.video_analyzer.analyze_video(video_path)
            
            # Extract test-specific metrics
            movement_metrics = self.video_analyzer.extract_movement_metrics(video_path, test_type)
            
            # Process based on test type
            if test_type == "Vertical Jump":
                results = self._process_vertical_jump(video_analysis, movement_metrics)
            elif test_type == "Sit-ups (1 minute)":
                results = self._process_situps(video_analysis, movement_metrics)
            elif test_type == "50m Sprint":
                results = self._process_sprint(video_analysis, movement_metrics)
            elif test_type == "Push-ups":
                results = self._process_pushups(video_analysis, movement_metrics)
            elif test_type == "Flexibility Test":
                results = self._process_flexibility(video_analysis, movement_metrics)
            else:
                results = {'error': f'Unknown test type: {test_type}'}
            
            # Add common video quality metrics
            results['video_quality'] = {
                'duration': video_analysis.get('duration', 0),
                'fps': video_analysis.get('fps', 0),
                'resolution': video_analysis.get('resolution', (0, 0)),
                'pose_detection_rate': video_analysis.get('pose_detection_rate', 0),
                'average_frame_quality': video_analysis.get('average_frame_quality', 0)
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing {test_type}: {str(e)}")
            return {'error': f'Processing failed: {str(e)}'}
    
    def _process_vertical_jump(self, video_analysis: Dict, movement_metrics: Dict) -> Dict:
        """Process vertical jump test"""
        if 'error' in movement_metrics:
            return movement_metrics
        
        params = self.test_parameters['Vertical Jump']
        
        # Extract metrics
        jump_height = movement_metrics.get('jump_height_normalized', 0)
        flight_time = movement_metrics.get('flight_time_seconds', 0)
        
        # Validate metrics
        validation_issues = []
        
        if jump_height < params['min_jump_height']:
            validation_issues.append('Jump height appears too low')
        
        if flight_time > params['max_flight_time']:
            validation_issues.append('Flight time appears unrealistic')
        
        if video_analysis.get('duration', 0) < 3:
            validation_issues.append('Video too short for proper analysis')
        
        # Calculate performance score (0-100)
        # Using relative jump height and flight time
        height_score = min(jump_height * 2000, 100)  # Normalize to 0-100 range
        time_score = min(flight_time * 200, 100)     # Normalize to 0-100 range
        performance_score = (height_score * 0.7 + time_score * 0.3)
        
        # Movement analysis
        movement_analysis = {
            'takeoff_quality': self._analyze_takeoff_form(movement_metrics),
            'landing_quality': self._analyze_landing_form(movement_metrics),
            'body_alignment': self._analyze_body_alignment(video_analysis),
            'preparation_phase': self._analyze_preparation_phase(movement_metrics)
        }
        
        return {
            'test_type': 'Vertical Jump',
            'metrics': {
                'jump_height_normalized': jump_height,
                'flight_time_seconds': flight_time,
                'performance_score': performance_score,
                'estimated_jump_height_cm': jump_height * 50,  # Rough conversion for display
            },
            'movement_analysis': movement_analysis,
            'validation_issues': validation_issues,
            'raw_data': movement_metrics
        }
    
    def _process_situps(self, video_analysis: Dict, movement_metrics: Dict) -> Dict:
        """Process sit-ups test"""
        if 'error' in movement_metrics:
            return movement_metrics
        
        params = self.test_parameters['Sit-ups (1 minute)']
        
        # Extract metrics
        rep_count = movement_metrics.get('rep_count', 0)
        cadence = movement_metrics.get('cadence_per_minute', 0)
        duration = movement_metrics.get('duration_seconds', 0)
        
        # Validate metrics
        validation_issues = []
        
        if duration > params['max_duration']:
            validation_issues.append('Test duration exceeds 1 minute limit')
        
        if duration < 30:
            validation_issues.append('Test duration appears too short')
        
        if cadence > 120:
            validation_issues.append('Cadence appears unrealistically high')
        
        # Calculate performance score
        # Standard sit-up scoring for general fitness
        performance_score = min(rep_count * 2, 100)  # 50 reps = 100 points
        
        # Movement analysis
        movement_analysis = {
            'form_consistency': self._analyze_situp_form(movement_metrics),
            'range_of_motion': self._analyze_range_of_motion(movement_metrics),
            'rhythm_consistency': self._analyze_rhythm_consistency(cadence, rep_count),
            'fatigue_analysis': self._analyze_fatigue_pattern(video_analysis)
        }
        
        return {
            'test_type': 'Sit-ups (1 minute)',
            'metrics': {
                'rep_count': rep_count,
                'cadence_per_minute': cadence,
                'duration_seconds': duration,
                'performance_score': performance_score,
                'reps_per_second': rep_count / duration if duration > 0 else 0
            },
            'movement_analysis': movement_analysis,
            'validation_issues': validation_issues,
            'raw_data': movement_metrics
        }
    
    def _process_sprint(self, video_analysis: Dict, movement_metrics: Dict) -> Dict:
        """Process sprint test"""
        if 'error' in movement_metrics:
            return movement_metrics
        
        params = self.test_parameters['50m Sprint']
        
        # Extract metrics
        speed = movement_metrics.get('estimated_speed', 0)
        duration = movement_metrics.get('duration_seconds', 0)
        displacement = movement_metrics.get('total_displacement', 0)
        
        # Validate metrics
        validation_issues = []
        
        if displacement < params['min_displacement']:
            validation_issues.append('Insufficient movement detected across frame')
        
        if duration > params['max_duration']:
            validation_issues.append('Sprint time appears unrealistic')
        
        if speed < params['min_speed']:
            validation_issues.append('Movement speed appears too low')
        
        # Calculate performance score
        # Higher speed = better score, typical range 0-15 units/second
        performance_score = min(speed * 400, 100)
        
        # Movement analysis
        movement_analysis = {
            'acceleration_phase': self._analyze_acceleration(video_analysis),
            'stride_consistency': self._analyze_stride_pattern(movement_metrics),
            'body_posture': self._analyze_running_posture(video_analysis),
            'finish_technique': self._analyze_finish_technique(movement_metrics)
        }
        
        return {
            'test_type': '50m Sprint',
            'metrics': {
                'estimated_speed': speed,
                'duration_seconds': duration,
                'total_displacement': displacement,
                'performance_score': performance_score,
                'estimated_time_50m': 50 / (speed * 100) if speed > 0 else 0  # Rough estimate
            },
            'movement_analysis': movement_analysis,
            'validation_issues': validation_issues,
            'raw_data': movement_metrics
        }
    
    def _process_pushups(self, video_analysis: Dict, movement_metrics: Dict) -> Dict:
        """Process push-ups test"""
        if 'error' in movement_metrics:
            return movement_metrics
        
        params = self.test_parameters['Push-ups']
        
        # Extract metrics
        rep_count = movement_metrics.get('rep_count', 0)
        cadence = movement_metrics.get('cadence_per_minute', 0)
        duration = movement_metrics.get('duration_seconds', 0)
        
        # Validate metrics
        validation_issues = []
        
        if cadence > 100:
            validation_issues.append('Push-up cadence appears unrealistic')
        
        if duration < 10:
            validation_issues.append('Test duration appears too short')
        
        # Calculate performance score
        performance_score = min(rep_count * 3, 100)  # ~33 reps = 100 points
        
        # Movement analysis
        movement_analysis = {
            'form_quality': self._analyze_pushup_form(movement_metrics),
            'depth_consistency': self._analyze_pushup_depth(movement_metrics),
            'body_alignment': self._analyze_plank_position(video_analysis),
            'endurance_pattern': self._analyze_endurance_pattern(video_analysis)
        }
        
        return {
            'test_type': 'Push-ups',
            'metrics': {
                'rep_count': rep_count,
                'cadence_per_minute': cadence,
                'duration_seconds': duration,
                'performance_score': performance_score,
                'average_rep_time': duration / rep_count if rep_count > 0 else 0
            },
            'movement_analysis': movement_analysis,
            'validation_issues': validation_issues,
            'raw_data': movement_metrics
        }
    
    def _process_flexibility(self, video_analysis: Dict, movement_metrics: Dict) -> Dict:
        """Process flexibility test"""
        if 'error' in movement_metrics:
            return movement_metrics
        
        params = self.test_parameters['Flexibility Test']
        
        # Extract metrics
        max_reach = movement_metrics.get('max_reach_distance', 0)
        flexibility_range = movement_metrics.get('flexibility_range', 0)
        consistency = movement_metrics.get('reach_consistency', 0)
        
        # Validate metrics
        validation_issues = []
        
        if video_analysis.get('duration', 0) < params['min_reach_time']:
            validation_issues.append('Test duration too short for proper measurement')
        
        if consistency < 0.8:
            validation_issues.append('Movement appears unstable during measurement')
        
        # Calculate performance score
        # Normalize reach distance to 0-100 scale
        reach_score = min(abs(max_reach) * 1000, 100)
        consistency_score = consistency * 100
        performance_score = (reach_score * 0.7 + consistency_score * 0.3)
        
        # Movement analysis
        movement_analysis = {
            'reach_stability': consistency,
            'movement_smoothness': self._analyze_movement_smoothness(video_analysis),
            'posture_maintenance': self._analyze_flexibility_posture(video_analysis),
            'progression_pattern': self._analyze_flexibility_progression(movement_metrics)
        }
        
        return {
            'test_type': 'Flexibility Test',
            'metrics': {
                'max_reach_distance': max_reach,
                'flexibility_range': flexibility_range,
                'reach_consistency': consistency,
                'performance_score': performance_score,
                'estimated_reach_cm': abs(max_reach) * 30  # Rough conversion
            },
            'movement_analysis': movement_analysis,
            'validation_issues': validation_issues,
            'raw_data': movement_metrics
        }
    
    # Helper methods for movement analysis
    
    def _analyze_takeoff_form(self, metrics: Dict) -> Dict:
        """Analyze takeoff form for vertical jump"""
        takeoff_frame = metrics.get('takeoff_frame', 0)
        baseline_height = metrics.get('baseline_height', 0)
        
        return {
            'explosive_power': min(metrics.get('jump_height_normalized', 0) * 1000, 100),
            'timing_consistency': 85,  # Placeholder - would need more detailed analysis
            'preparation_quality': 80   # Placeholder
        }
    
    def _analyze_landing_form(self, metrics: Dict) -> Dict:
        """Analyze landing form for vertical jump"""
        return {
            'stability': 85,           # Placeholder
            'control': 80,             # Placeholder
            'safety_score': 90         # Placeholder
        }
    
    def _analyze_body_alignment(self, analysis: Dict) -> Dict:
        """Analyze body alignment during movement"""
        pose_detections = analysis.get('pose_detections', [])
        
        if not pose_detections:
            return {'alignment_score': 50}
        
        # Simple alignment check based on shoulder-hip alignment
        alignment_scores = []
        
        for detection in pose_detections:
            if all(point in detection for point in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                # Check if shoulders and hips are level
                shoulder_level = abs(detection['left_shoulder']['y'] - detection['right_shoulder']['y'])
                hip_level = abs(detection['left_hip']['y'] - detection['right_hip']['y'])
                
                alignment_score = 100 - (shoulder_level + hip_level) * 500  # Normalize
                alignment_scores.append(max(alignment_score, 0))
        
        return {
            'alignment_score': np.mean(alignment_scores) if alignment_scores else 50
        }
    
    def _analyze_preparation_phase(self, metrics: Dict) -> Dict:
        """Analyze preparation phase of movement"""
        return {
            'preparation_time': metrics.get('takeoff_frame', 0) / 30,  # Convert to seconds
            'stability_score': 85,     # Placeholder
            'readiness_indicator': 90  # Placeholder
        }
    
    def _analyze_situp_form(self, metrics: Dict) -> Dict:
        """Analyze sit-up form quality"""
        avg_angle = metrics.get('average_angle', 0)
        angle_range = metrics.get('angle_range', 0)
        
        return {
            'range_of_motion_score': min(angle_range / 90 * 100, 100),
            'form_consistency': max(100 - abs(avg_angle - 75), 0),  # Optimal around 75 degrees
            'technique_score': 85  # Placeholder
        }
    
    def _analyze_range_of_motion(self, metrics: Dict) -> Dict:
        """Analyze range of motion in exercise"""
        angle_range = metrics.get('angle_range', 0)
        
        return {
            'full_rom_percentage': min(angle_range / 60 * 100, 100),  # 60 degrees as full ROM
            'consistency': 85,     # Placeholder
            'quality': 80          # Placeholder
        }
    
    def _analyze_rhythm_consistency(self, cadence: float, rep_count: int) -> Dict:
        """Analyze rhythm consistency"""
        if rep_count == 0:
            return {'rhythm_score': 0}
        
        expected_time_per_rep = 60 / cadence if cadence > 0 else 0
        
        return {
            'rhythm_score': min(85 + (cadence / 60 * 15), 100),  # Better rhythm with moderate cadence
            'tempo_consistency': 80,   # Placeholder
            'pacing_quality': 85       # Placeholder
        }
    
    def _analyze_fatigue_pattern(self, analysis: Dict) -> Dict:
        """Analyze fatigue pattern during exercise"""
        motion_intensity = analysis.get('motion_intensity', [])
        
        if not motion_intensity:
            return {'fatigue_resistance': 50}
        
        # Check if motion intensity decreases over time (indicating fatigue)
        if len(motion_intensity) > 10:
            early_intensity = np.mean(motion_intensity[:len(motion_intensity)//3])
            late_intensity = np.mean(motion_intensity[-len(motion_intensity)//3:])
            fatigue_ratio = late_intensity / early_intensity if early_intensity > 0 else 0
        else:
            fatigue_ratio = 1.0
        
        return {
            'fatigue_resistance': min(fatigue_ratio * 100, 100),
            'endurance_score': 80,     # Placeholder
            'consistency_rating': 85   # Placeholder
        }
    
    def _analyze_acceleration(self, analysis: Dict) -> Dict:
        """Analyze acceleration phase in sprint"""
        motion_intensity = analysis.get('motion_intensity', [])
        
        if len(motion_intensity) < 10:
            return {'acceleration_quality': 50}
        
        # Check if motion intensity increases in the beginning
        initial_phase = motion_intensity[:len(motion_intensity)//4]
        acceleration_trend = np.polyfit(range(len(initial_phase)), initial_phase, 1)[0] if len(initial_phase) > 1 else 0
        
        return {
            'acceleration_quality': min(acceleration_trend * 10000 + 50, 100),
            'explosive_start': 80,     # Placeholder
            'technique_score': 85      # Placeholder
        }
    
    def _analyze_stride_pattern(self, metrics: Dict) -> Dict:
        """Analyze stride pattern consistency"""
        stride_frequency = metrics.get('stride_frequency', 0)
        
        return {
            'stride_consistency': min(stride_frequency / 30 * 100, 100),
            'frequency_score': min(stride_frequency * 2, 100),
            'pattern_quality': 80      # Placeholder
        }
    
    def _analyze_running_posture(self, analysis: Dict) -> Dict:
        """Analyze running posture"""
        return self._analyze_body_alignment(analysis)
    
    def _analyze_finish_technique(self, metrics: Dict) -> Dict:
        """Analyze sprint finish technique"""
        return {
            'finish_quality': 85,      # Placeholder
            'deceleration_control': 80, # Placeholder
            'technique_maintenance': 90 # Placeholder
        }
    
    def _analyze_pushup_form(self, metrics: Dict) -> Dict:
        """Analyze push-up form quality"""
        avg_angle = metrics.get('average_elbow_angle', 0)
        angle_range = metrics.get('angle_range', 0)
        
        return {
            'form_score': max(100 - abs(avg_angle - 120), 0),  # Optimal around 120 degrees
            'depth_consistency': min(angle_range / 60 * 100, 100),
            'technique_quality': 85    # Placeholder
        }
    
    def _analyze_pushup_depth(self, metrics: Dict) -> Dict:
        """Analyze push-up depth consistency"""
        angle_range = metrics.get('angle_range', 0)
        
        return {
            'depth_score': min(angle_range / 80 * 100, 100),  # Full range expected
            'consistency': 85,         # Placeholder
            'quality': 80              # Placeholder
        }
    
    def _analyze_plank_position(self, analysis: Dict) -> Dict:
        """Analyze plank position maintenance"""
        return self._analyze_body_alignment(analysis)
    
    def _analyze_endurance_pattern(self, analysis: Dict) -> Dict:
        """Analyze endurance pattern"""
        return self._analyze_fatigue_pattern(analysis)
    
    def _analyze_movement_smoothness(self, analysis: Dict) -> Dict:
        """Analyze smoothness of movement"""
        motion_intensity = analysis.get('motion_intensity', [])
        
        if not motion_intensity:
            return {'smoothness_score': 50}
        
        # Calculate smoothness based on motion variance
        smoothness = 100 - min(np.std(motion_intensity) * 1000, 50) if len(motion_intensity) > 1 else 50
        
        return {
            'smoothness_score': smoothness,
            'fluidity': 80,            # Placeholder
            'control_quality': 85      # Placeholder
        }
    
    def _analyze_flexibility_posture(self, analysis: Dict) -> Dict:
        """Analyze posture during flexibility test"""
        return self._analyze_body_alignment(analysis)
    
    def _analyze_flexibility_progression(self, metrics: Dict) -> Dict:
        """Analyze flexibility progression pattern"""
        max_reach = metrics.get('max_reach_distance', 0)
        consistency = metrics.get('reach_consistency', 0)
        
        return {
            'progression_quality': min(abs(max_reach) * 500, 100),
            'improvement_pattern': 80,  # Placeholder
            'technique_score': consistency * 100
        }
