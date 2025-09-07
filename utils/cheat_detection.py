import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from utils.video_analysis import VideoAnalyzer

class CheatDetector:
    """Advanced cheat detection for fitness assessment videos"""
    
    def __init__(self):
        self.video_analyzer = VideoAnalyzer()
        
        # Detection thresholds and parameters
        self.detection_params = {
            'min_video_duration': 3.0,      # Minimum video duration in seconds
            'max_video_duration': 180.0,    # Maximum video duration in seconds
            'min_pose_detection_rate': 0.6, # Minimum pose detection rate
            'max_motion_variance': 2.0,     # Maximum motion variance threshold
            'min_frame_quality': 0.4,       # Minimum average frame quality
            'face_consistency_threshold': 0.7,  # Face consistency across frames
            'temporal_consistency_threshold': 0.8,  # Movement temporal consistency
        }
        
        # Anomaly detection weights
        self.anomaly_weights = {
            'video_manipulation': 0.25,
            'movement_anomalies': 0.25,
            'temporal_inconsistencies': 0.2,
            'pose_estimation_issues': 0.15,
            'environmental_factors': 0.15
        }
    
    def detect_anomalies(self, video_path: str, test_type: str) -> Dict:
        """
        Comprehensive anomaly detection for video authenticity
        
        Args:
            video_path: Path to the video file
            test_type: Type of fitness test being performed
            
        Returns:
            Dictionary containing anomaly analysis results
        """
        try:
            # Perform basic video analysis
            video_analysis = self.video_analyzer.analyze_video(video_path)
            
            # Initialize anomaly detection results
            anomaly_results = {
                'risk_score': 0.0,  # Overall risk score (0-1, higher = more suspicious)
                'confidence': 0.0,  # Confidence in detection (0-1)
                'checks': {},       # Individual check results
                'warnings': [],     # List of warnings
                'recommendations': []  # Recommendations for review
            }
            
            # Run individual detection methods
            video_checks = self._check_video_integrity(video_analysis)
            movement_checks = self._check_movement_patterns(video_analysis, test_type)
            temporal_checks = self._check_temporal_consistency(video_analysis)
            pose_checks = self._check_pose_estimation_quality(video_analysis)
            environmental_checks = self._check_environmental_factors(video_analysis)
            
            # Aggregate results
            anomaly_results['checks'].update({
                'video_integrity': video_checks,
                'movement_patterns': movement_checks,
                'temporal_consistency': temporal_checks,
                'pose_estimation': pose_checks,
                'environmental_factors': environmental_checks
            })
            
            # Calculate overall risk score
            risk_components = {
                'video_manipulation': video_checks.get('risk_score', 0),
                'movement_anomalies': movement_checks.get('risk_score', 0),
                'temporal_inconsistencies': temporal_checks.get('risk_score', 0),
                'pose_estimation_issues': pose_checks.get('risk_score', 0),
                'environmental_factors': environmental_checks.get('risk_score', 0)
            }
            
            overall_risk = sum(
                risk_components[component] * self.anomaly_weights[component]
                for component in risk_components
            )
            
            anomaly_results['risk_score'] = min(overall_risk, 1.0)
            anomaly_results['confidence'] = self._calculate_detection_confidence(anomaly_results)
            
            # Generate warnings and recommendations
            anomaly_results['warnings'] = self._generate_warnings(anomaly_results)
            anomaly_results['recommendations'] = self._generate_recommendations(anomaly_results)
            
            return anomaly_results
            
        except Exception as e:
            logging.error(f"Anomaly detection failed: {str(e)}")
            return {
                'risk_score': 0.5,  # Medium risk when detection fails
                'confidence': 0.0,
                'checks': {},
                'warnings': ['Anomaly detection system encountered an error'],
                'recommendations': ['Manual review recommended due to detection failure'],
                'error': str(e)
            }
    
    def _check_video_integrity(self, analysis: Dict) -> Dict:
        """Check for video manipulation and integrity issues"""
        checks = {
            'duration_check': {'passed': True, 'status': 'Normal'},
            'frame_consistency': {'passed': True, 'status': 'Normal'},
            'resolution_stability': {'passed': True, 'status': 'Normal'},
            'compression_artifacts': {'passed': True, 'status': 'Normal'},
            'risk_score': 0.0
        }
        
        risk_factors = []
        
        # Duration check
        duration = analysis.get('duration', 0)
        if duration < self.detection_params['min_video_duration']:
            checks['duration_check'] = {'passed': False, 'status': 'Too short'}
            risk_factors.append(0.3)
        elif duration > self.detection_params['max_video_duration']:
            checks['duration_check'] = {'passed': False, 'status': 'Unusually long'}
            risk_factors.append(0.1)
        
        # Frame quality consistency
        frame_qualities = analysis.get('frame_quality_scores', [])
        if frame_qualities:
            quality_variance = np.var(frame_qualities)
            if quality_variance > 0.1:  # High variance in quality
                checks['frame_consistency'] = {'passed': False, 'status': 'Inconsistent quality'}
                risk_factors.append(0.2)
        
        # Resolution check
        resolution = analysis.get('resolution', (0, 0))
        if resolution[0] < 480 or resolution[1] < 360:
            checks['resolution_stability'] = {'passed': False, 'status': 'Low resolution'}
            risk_factors.append(0.1)
        
        # Calculate risk score for video integrity
        checks['risk_score'] = min(sum(risk_factors), 1.0)
        
        return checks
    
    def _check_movement_patterns(self, analysis: Dict, test_type: str) -> Dict:
        """Check for unrealistic or impossible movement patterns"""
        checks = {
            'movement_realism': {'passed': True, 'status': 'Realistic'},
            'biomechanical_feasibility': {'passed': True, 'status': 'Feasible'},
            'performance_outliers': {'passed': True, 'status': 'Normal range'},
            'movement_smoothness': {'passed': True, 'status': 'Natural'},
            'risk_score': 0.0
        }
        
        risk_factors = []
        
        # Motion intensity analysis
        motion_intensity = analysis.get('motion_intensity', [])
        if motion_intensity:
            max_motion = max(motion_intensity)
            motion_variance = np.var(motion_intensity)
            
            # Check for unrealistic motion spikes
            if max_motion > 0.5:  # Threshold for unrealistic motion
                checks['movement_realism'] = {'passed': False, 'status': 'Unrealistic motion detected'}
                risk_factors.append(0.4)
            
            # Check for motion smoothness
            if motion_variance > self.detection_params['max_motion_variance']:
                checks['movement_smoothness'] = {'passed': False, 'status': 'Erratic movement pattern'}
                risk_factors.append(0.3)
        
        # Test-specific movement validation
        test_specific_risk = self._check_test_specific_movements(analysis, test_type)
        if test_specific_risk > 0:
            checks['biomechanical_feasibility'] = {'passed': False, 'status': 'Unusual for test type'}
            risk_factors.append(test_specific_risk)
        
        # Pose landmark consistency
        pose_detections = analysis.get('pose_detections', [])
        if pose_detections:
            consistency_score = self._calculate_pose_consistency(pose_detections)
            if consistency_score < 0.7:
                checks['performance_outliers'] = {'passed': False, 'status': 'Inconsistent pose data'}
                risk_factors.append(0.2)
        
        checks['risk_score'] = min(sum(risk_factors), 1.0)
        
        return checks
    
    def _check_temporal_consistency(self, analysis: Dict) -> Dict:
        """Check for temporal inconsistencies in the video"""
        checks = {
            'frame_rate_consistency': {'passed': True, 'status': 'Consistent'},
            'motion_continuity': {'passed': True, 'status': 'Continuous'},
            'timestamp_analysis': {'passed': True, 'status': 'Normal'},
            'speed_variations': {'passed': True, 'status': 'Natural'},
            'risk_score': 0.0
        }
        
        risk_factors = []
        
        # Frame rate analysis
        fps = analysis.get('fps', 0)
        if fps > 0:
            total_frames = analysis.get('total_frames', 0)
            expected_duration = total_frames / fps
            actual_duration = analysis.get('duration', 0)
            
            if abs(expected_duration - actual_duration) > 1.0:  # 1 second tolerance
                checks['frame_rate_consistency'] = {'passed': False, 'status': 'Frame timing issues'}
                risk_factors.append(0.3)
        
        # Motion continuity check
        motion_intensity = analysis.get('motion_intensity', [])
        if len(motion_intensity) > 10:
            # Check for abrupt changes that might indicate editing
            motion_diffs = np.diff(motion_intensity)
            large_jumps = np.sum(np.abs(motion_diffs) > 0.2)
            
            if large_jumps > len(motion_intensity) * 0.1:  # More than 10% large jumps
                checks['motion_continuity'] = {'passed': False, 'status': 'Abrupt motion changes'}
                risk_factors.append(0.25)
        
        # Speed variation analysis
        if motion_intensity:
            speed_variance = np.var(motion_intensity)
            if speed_variance > 0.5:  # High variance might indicate speed manipulation
                checks['speed_variations'] = {'passed': False, 'status': 'Unusual speed patterns'}
                risk_factors.append(0.2)
        
        checks['risk_score'] = min(sum(risk_factors), 1.0)
        
        return checks
    
    def _check_pose_estimation_quality(self, analysis: Dict) -> Dict:
        """Check quality and consistency of pose estimation"""
        checks = {
            'detection_rate': {'passed': True, 'status': 'Good detection'},
            'landmark_quality': {'passed': True, 'status': 'High quality'},
            'visibility_scores': {'passed': True, 'status': 'Good visibility'},
            'anatomical_consistency': {'passed': True, 'status': 'Consistent'},
            'risk_score': 0.0
        }
        
        risk_factors = []
        
        # Pose detection rate
        pose_detection_rate = analysis.get('pose_detection_rate', 0)
        if pose_detection_rate < self.detection_params['min_pose_detection_rate']:
            checks['detection_rate'] = {'passed': False, 'status': 'Low detection rate'}
            risk_factors.append(0.3)
        
        # Landmark quality analysis
        pose_detections = analysis.get('pose_detections', [])
        if pose_detections:
            # Analyze landmark visibility scores
            visibility_scores = []
            for detection in pose_detections:
                for landmark_name, landmark_data in detection.items():
                    if isinstance(landmark_data, dict) and 'visibility' in landmark_data:
                        visibility_scores.append(landmark_data['visibility'])
            
            if visibility_scores:
                avg_visibility = np.mean(visibility_scores)
                if avg_visibility < 0.6:
                    checks['visibility_scores'] = {'passed': False, 'status': 'Poor landmark visibility'}
                    risk_factors.append(0.2)
        
        # Anatomical consistency check
        anatomical_consistency = self._check_anatomical_consistency(pose_detections)
        if anatomical_consistency < 0.8:
            checks['anatomical_consistency'] = {'passed': False, 'status': 'Anatomically inconsistent'}
            risk_factors.append(0.25)
        
        checks['risk_score'] = min(sum(risk_factors), 1.0)
        
        return checks
    
    def _check_environmental_factors(self, analysis: Dict) -> Dict:
        """Check environmental factors that might indicate cheating"""
        checks = {
            'lighting_consistency': {'passed': True, 'status': 'Consistent'},
            'background_stability': {'passed': True, 'status': 'Stable'},
            'camera_movement': {'passed': True, 'status': 'Minimal'},
            'occlusion_analysis': {'passed': True, 'status': 'Clear view'},
            'risk_score': 0.0
        }
        
        risk_factors = []
        
        # Frame quality as proxy for lighting consistency
        frame_qualities = analysis.get('frame_quality_scores', [])
        if frame_qualities:
            quality_std = np.std(frame_qualities)
            if quality_std > 0.2:  # High variation in frame quality
                checks['lighting_consistency'] = {'passed': False, 'status': 'Variable lighting'}
                risk_factors.append(0.15)
        
        # Camera stability (using motion data as proxy)
        motion_intensity = analysis.get('motion_intensity', [])
        if motion_intensity:
            # Check for consistent high motion that might indicate camera shake
            high_motion_frames = sum(1 for m in motion_intensity if m > 0.3)
            if high_motion_frames > len(motion_intensity) * 0.8:
                checks['camera_movement'] = {'passed': False, 'status': 'Excessive camera movement'}
                risk_factors.append(0.1)
        
        # Pose detection consistency as proxy for occlusions
        pose_detection_rate = analysis.get('pose_detection_rate', 0)
        if pose_detection_rate < 0.8:
            checks['occlusion_analysis'] = {'passed': False, 'status': 'Frequent occlusions'}
            risk_factors.append(0.2)
        
        checks['risk_score'] = min(sum(risk_factors), 1.0)
        
        return checks
    
    def _check_test_specific_movements(self, analysis: Dict, test_type: str) -> float:
        """Check for test-specific movement anomalies"""
        risk_score = 0.0
        
        key_movements = analysis.get('key_movements', [])
        motion_intensity = analysis.get('motion_intensity', [])
        
        if test_type == "Vertical Jump":
            # Should have clear takeoff and landing phases
            if len(key_movements) < 1:
                risk_score += 0.3  # No clear jump phases detected
            
            # Check for reasonable motion pattern
            if motion_intensity:
                max_motion = max(motion_intensity)
                if max_motion < 0.1:  # Very low motion for a jump
                    risk_score += 0.2
        
        elif test_type == "Sit-ups (1 minute)":
            # Should have repetitive motion pattern
            if motion_intensity:
                # Look for periodicity in motion
                motion_fft = np.fft.fft(motion_intensity)
                if len(motion_fft) > 10:
                    dominant_freq = np.argmax(np.abs(motion_fft[1:10])) + 1
                    if dominant_freq < 1:  # No clear repetitive pattern
                        risk_score += 0.2
        
        elif test_type == "50m Sprint":
            # Should have sustained forward motion
            if motion_intensity:
                if np.mean(motion_intensity) < 0.05:  # Very low average motion
                    risk_score += 0.3
        
        elif test_type == "Push-ups":
            # Should have up-down repetitive motion
            if len(key_movements) < 2:  # Should have multiple movement phases
                risk_score += 0.2
        
        elif test_type == "Flexibility Test":
            # Should have gradual reaching motion
            if motion_intensity:
                motion_variance = np.var(motion_intensity)
                if motion_variance > 0.1:  # Too much variation for flexibility test
                    risk_score += 0.15
        
        return min(risk_score, 1.0)
    
    def _calculate_pose_consistency(self, pose_detections: List[Dict]) -> float:
        """Calculate consistency of pose landmarks across frames"""
        if len(pose_detections) < 10:
            return 0.5
        
        # Track key landmarks across frames
        landmark_trajectories = {}
        
        for detection in pose_detections:
            for landmark_name, landmark_data in detection.items():
                if isinstance(landmark_data, dict) and 'x' in landmark_data:
                    if landmark_name not in landmark_trajectories:
                        landmark_trajectories[landmark_name] = {'x': [], 'y': [], 'z': []}
                    
                    landmark_trajectories[landmark_name]['x'].append(landmark_data['x'])
                    landmark_trajectories[landmark_name]['y'].append(landmark_data['y'])
                    landmark_trajectories[landmark_name]['z'].append(landmark_data.get('z', 0))
        
        # Calculate smoothness of trajectories
        consistency_scores = []
        
        for landmark_name, trajectory in landmark_trajectories.items():
            if len(trajectory['x']) > 5:
                # Calculate smoothness using variance of derivatives
                x_smooth = 1.0 - min(np.var(np.diff(trajectory['x'])), 0.5) * 2
                y_smooth = 1.0 - min(np.var(np.diff(trajectory['y'])), 0.5) * 2
                
                landmark_consistency = (x_smooth + y_smooth) / 2
                consistency_scores.append(landmark_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _check_anatomical_consistency(self, pose_detections: List[Dict]) -> float:
        """Check for anatomically consistent pose relationships"""
        if not pose_detections:
            return 0.5
        
        consistency_scores = []
        
        for detection in pose_detections:
            frame_consistency = []
            
            # Check shoulder-hip alignment
            if all(point in detection for point in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                shoulder_width = abs(detection['left_shoulder']['x'] - detection['right_shoulder']['x'])
                hip_width = abs(detection['left_hip']['x'] - detection['right_hip']['x'])
                
                # Shoulders should generally be wider than hips
                if shoulder_width > hip_width * 0.8:
                    frame_consistency.append(1.0)
                else:
                    frame_consistency.append(0.5)
            
            # Check arm proportions
            if all(point in detection for point in ['left_shoulder', 'left_elbow', 'left_wrist']):
                upper_arm = np.sqrt(
                    (detection['left_shoulder']['x'] - detection['left_elbow']['x'])**2 +
                    (detection['left_shoulder']['y'] - detection['left_elbow']['y'])**2
                )
                forearm = np.sqrt(
                    (detection['left_elbow']['x'] - detection['left_wrist']['x'])**2 +
                    (detection['left_elbow']['y'] - detection['left_wrist']['y'])**2
                )
                
                # Upper arm and forearm should be similar lengths (within reason)
                if 0.5 < upper_arm / forearm < 2.0:
                    frame_consistency.append(1.0)
                else:
                    frame_consistency.append(0.3)
            
            if frame_consistency:
                consistency_scores.append(np.mean(frame_consistency))
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _calculate_detection_confidence(self, anomaly_results: Dict) -> float:
        """Calculate confidence in the anomaly detection"""
        checks = anomaly_results.get('checks', {})
        
        confidence_factors = []
        
        # Number of successful checks increases confidence
        total_checks = 0
        passed_checks = 0
        
        for check_category, check_results in checks.items():
            if isinstance(check_results, dict):
                for check_name, check_data in check_results.items():
                    if isinstance(check_data, dict) and 'passed' in check_data:
                        total_checks += 1
                        if check_data['passed']:
                            passed_checks += 1
        
        if total_checks > 0:
            check_success_rate = passed_checks / total_checks
            confidence_factors.append(check_success_rate)
        
        # Lower risk scores with consistent checks indicate higher confidence
        risk_score = anomaly_results.get('risk_score', 0.5)
        risk_confidence = 1.0 - abs(risk_score - 0.5) * 2  # Higher confidence at extremes
        confidence_factors.append(risk_confidence)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _generate_warnings(self, anomaly_results: Dict) -> List[str]:
        """Generate warnings based on anomaly detection results"""
        warnings = []
        
        risk_score = anomaly_results.get('risk_score', 0)
        checks = anomaly_results.get('checks', {})
        
        # Overall risk warnings
        if risk_score > 0.7:
            warnings.append("High risk of video manipulation or cheating detected")
        elif risk_score > 0.4:
            warnings.append("Moderate anomalies detected - manual review recommended")
        elif risk_score > 0.2:
            warnings.append("Minor anomalies detected - consider additional verification")
        
        # Specific check warnings
        for check_category, check_results in checks.items():
            if isinstance(check_results, dict):
                category_risk = check_results.get('risk_score', 0)
                if category_risk > 0.3:
                    category_name = check_category.replace('_', ' ').title()
                    warnings.append(f"{category_name} issues detected")
        
        return warnings
    
    def _generate_recommendations(self, anomaly_results: Dict) -> List[str]:
        """Generate recommendations based on anomaly detection results"""
        recommendations = []
        
        risk_score = anomaly_results.get('risk_score', 0)
        checks = anomaly_results.get('checks', {})
        
        if risk_score > 0.6:
            recommendations.extend([
                "Manual review by trained officials required",
                "Consider requesting video resubmission",
                "Verify athlete identity through additional means"
            ])
        elif risk_score > 0.3:
            recommendations.extend([
                "Additional automated verification recommended",
                "Consider video quality improvement suggestions"
            ])
        else:
            recommendations.append("Video appears authentic - standard processing can proceed")
        
        # Specific recommendations based on failed checks
        video_integrity = checks.get('video_integrity', {})
        if video_integrity.get('risk_score', 0) > 0.2:
            recommendations.append("Recommend higher quality video recording")
        
        movement_patterns = checks.get('movement_patterns', {})
        if movement_patterns.get('risk_score', 0) > 0.2:
            recommendations.append("Review movement execution technique")
        
        environmental_factors = checks.get('environmental_factors', {})
        if environmental_factors.get('risk_score', 0) > 0.2:
            recommendations.append("Improve recording environment and camera stability")
        
        return recommendations
    
    def generate_verification_report(self, anomaly_results: Dict, athlete_id: str, test_type: str) -> Dict:
        """Generate comprehensive verification report"""
        report = {
            'athlete_id': athlete_id,
            'test_type': test_type,
            'verification_timestamp': datetime.now().isoformat(),
            'overall_risk_assessment': self._get_risk_level(anomaly_results.get('risk_score', 0)),
            'confidence_level': anomaly_results.get('confidence', 0),
            'automated_decision': self._get_automated_decision(anomaly_results),
            'detailed_analysis': anomaly_results,
            'action_required': self._determine_required_action(anomaly_results),
            'verification_summary': self._create_verification_summary(anomaly_results)
        }
        
        return report
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to categorical assessment"""
        if risk_score > 0.7:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM" 
        elif risk_score > 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_automated_decision(self, anomaly_results: Dict) -> str:
        """Get automated verification decision"""
        risk_score = anomaly_results.get('risk_score', 0)
        confidence = anomaly_results.get('confidence', 0)
        
        if risk_score < 0.2 and confidence > 0.8:
            return "APPROVED"
        elif risk_score > 0.6 or confidence < 0.3:
            return "REJECTED"
        else:
            return "MANUAL_REVIEW_REQUIRED"
    
    def _determine_required_action(self, anomaly_results: Dict) -> str:
        """Determine what action is required based on results"""
        decision = self._get_automated_decision(anomaly_results)
        
        if decision == "APPROVED":
            return "No further action required"
        elif decision == "REJECTED":
            return "Video resubmission required"
        else:
            return "Manual review by qualified assessor required"
    
    def _create_verification_summary(self, anomaly_results: Dict) -> str:
        """Create human-readable verification summary"""
        risk_score = anomaly_results.get('risk_score', 0)
        warnings = anomaly_results.get('warnings', [])
        
        if risk_score < 0.2:
            return "Video analysis completed successfully. No significant anomalies detected. Assessment appears authentic and ready for scoring."
        elif risk_score < 0.4:
            return f"Video analysis completed with minor concerns. {len(warnings)} potential issues identified. Standard processing can proceed with notation."
        elif risk_score < 0.7:
            return f"Video analysis identified moderate anomalies. {len(warnings)} issues detected. Manual review recommended before final assessment."
        else:
            return f"Video analysis detected significant anomalies. {len(warnings)} serious issues identified. Video requires thorough manual review or resubmission."
