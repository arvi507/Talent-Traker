import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
from data.benchmarks import PERFORMANCE_BENCHMARKS, AGE_GROUPS, GENDER_FACTORS
from typing import List, Dict

class PerformanceScorer:
    """Calculate performance scores based on age, gender, and test-specific benchmarks"""
    
    def __init__(self):
        self.benchmarks = PERFORMANCE_BENCHMARKS
        self.age_groups = AGE_GROUPS
        self.gender_factors = GENDER_FACTORS
        
    def calculate_score(self, test_results: Dict, age: int, gender: str, test_type: str) -> Dict:
        """
        Calculate comprehensive performance score with benchmarking
        
        Args:
            test_results: Results from fitness test processing
            age: Athlete's age
            gender: Athlete's gender (Male/Female/Other)
            test_type: Type of fitness test
            
        Returns:
            Dictionary containing scores, percentiles, and comparisons
        """
        try:
            # Get age group
            age_group = self._get_age_group(age)
            
            # Get test-specific scoring method
            if test_type == "Vertical Jump":
                score_data = self._score_vertical_jump(test_results, age_group, gender)
            elif test_type == "Sit-ups (1 minute)":
                score_data = self._score_situps(test_results, age_group, gender)
            elif test_type == "50m Sprint":
                score_data = self._score_sprint(test_results, age_group, gender)
            elif test_type == "Push-ups":
                score_data = self._score_pushups(test_results, age_group, gender)
            elif test_type == "Flexibility Test":
                score_data = self._score_flexibility(test_results, age_group, gender)
            else:
                return {'error': f'Unknown test type: {test_type}'}
            
            # Calculate overall performance metrics
            overall_score = self._calculate_overall_score(score_data, test_results)
            percentile = self._calculate_percentile(overall_score, age_group, gender, test_type)
            category = self._get_performance_category(overall_score)
            
            # Generate benchmark comparison
            benchmark_comparison = self._generate_benchmark_comparison(
                score_data, age_group, gender, test_type
            )
            
            return {
                'overall_score': overall_score,
                'percentile': percentile,
                'category': category,
                'component_scores': score_data,
                'benchmark_comparison': benchmark_comparison,
                'age_group': age_group,
                'gender_adjusted': True,
                'scoring_factors': {
                    'age_factor': self._get_age_factor(age),
                    'gender_factor': self.gender_factors.get(gender, 1.0),
                    'test_difficulty': self._get_test_difficulty(test_type)
                }
            }
            
        except Exception as e:
            return {'error': f'Scoring calculation failed: {str(e)}'}
    
    def _get_age_group(self, age: int) -> str:
        """Determine age group category"""
        for age_range, group_name in self.age_groups.items():
            min_age, max_age = age_range
            if min_age <= age <= max_age:
                return group_name
        return "Adult"  # Default fallback
    
    def _score_vertical_jump(self, results: Dict, age_group: str, gender: str) -> Dict:
        """Score vertical jump performance"""
        metrics = results.get('metrics', {})
        movement_analysis = results.get('movement_analysis', {})
        
        # Primary metric: jump height
        jump_height = metrics.get('jump_height_normalized', 0)
        flight_time = metrics.get('flight_time_seconds', 0)
        
        # Get benchmarks for this age group and gender
        benchmarks = self.benchmarks['Vertical Jump'][age_group][gender]
        
        # Score components (0-100 each)
        height_score = self._score_against_benchmark(jump_height * 100, benchmarks['jump_height'])
        technique_score = self._score_technique_components(movement_analysis)
        consistency_score = self._score_consistency(results)
        
        return {
            'primary_metric_score': height_score,
            'technique_score': technique_score,
            'consistency_score': consistency_score,
            'flight_time_bonus': min(flight_time * 50, 10),  # Bonus points for flight time
            'components': {
                'explosive_power': height_score,
                'takeoff_technique': movement_analysis.get('takeoff_quality', {}).get('explosive_power', 70),
                'landing_control': movement_analysis.get('landing_quality', {}).get('stability', 70),
                'body_alignment': movement_analysis.get('body_alignment', {}).get('alignment_score', 70)
            }
        }
    
    def _score_situps(self, results: Dict, age_group: str, gender: str) -> Dict:
        """Score sit-ups performance"""
        metrics = results.get('metrics', {})
        movement_analysis = results.get('movement_analysis', {})
        
        # Primary metric: repetition count
        rep_count = metrics.get('rep_count', 0)
        cadence = metrics.get('cadence_per_minute', 0)
        
        # Get benchmarks
        benchmarks = self.benchmarks['Sit-ups (1 minute)'][age_group][gender]
        
        # Score components
        rep_score = self._score_against_benchmark(rep_count, benchmarks['rep_count'])
        form_score = self._score_technique_components(movement_analysis)
        endurance_score = self._score_endurance_pattern(movement_analysis, metrics.get('duration_seconds', 0))
        
        return {
            'primary_metric_score': rep_score,
            'technique_score': form_score,
            'endurance_score': endurance_score,
            'cadence_bonus': min(cadence / 60 * 10, 15),  # Optimal cadence bonus
            'components': {
                'core_strength': rep_score,
                'form_quality': movement_analysis.get('form_consistency', {}).get('form_consistency', 70),
                'range_of_motion': movement_analysis.get('range_of_motion', {}).get('full_rom_percentage', 70),
                'rhythm_consistency': movement_analysis.get('rhythm_consistency', {}).get('rhythm_score', 70)
            }
        }
    
    def _score_sprint(self, results: Dict, age_group: str, gender: str) -> Dict:
        """Score sprint performance"""
        metrics = results.get('metrics', {})
        movement_analysis = results.get('movement_analysis', {})
        
        # Primary metric: speed (higher is better)
        speed = metrics.get('estimated_speed', 0)
        duration = metrics.get('duration_seconds', 0)
        
        # Get benchmarks
        benchmarks = self.benchmarks['50m Sprint'][age_group][gender]
        
        # Score components
        speed_score = self._score_against_benchmark(speed * 100, benchmarks['speed'])
        technique_score = self._score_technique_components(movement_analysis)
        acceleration_score = movement_analysis.get('acceleration_phase', {}).get('acceleration_quality', 70)
        
        return {
            'primary_metric_score': speed_score,
            'technique_score': technique_score,
            'acceleration_score': acceleration_score,
            'components': {
                'maximum_speed': speed_score,
                'acceleration': acceleration_score,
                'running_form': movement_analysis.get('body_posture', {}).get('alignment_score', 70),
                'finish_technique': movement_analysis.get('finish_technique', {}).get('finish_quality', 70)
            }
        }
    
    def _score_pushups(self, results: Dict, age_group: str, gender: str) -> Dict:
        """Score push-ups performance"""
        metrics = results.get('metrics', {})
        movement_analysis = results.get('movement_analysis', {})
        
        # Primary metric: repetition count
        rep_count = metrics.get('rep_count', 0)
        
        # Get benchmarks
        benchmarks = self.benchmarks['Push-ups'][age_group][gender]
        
        # Score components
        rep_score = self._score_against_benchmark(rep_count, benchmarks['rep_count'])
        form_score = self._score_technique_components(movement_analysis)
        strength_endurance = self._score_endurance_pattern(movement_analysis, metrics.get('duration_seconds', 0))
        
        return {
            'primary_metric_score': rep_score,
            'technique_score': form_score,
            'strength_endurance_score': strength_endurance,
            'components': {
                'upper_body_strength': rep_score,
                'form_quality': movement_analysis.get('form_quality', {}).get('form_score', 70),
                'depth_consistency': movement_analysis.get('depth_consistency', {}).get('depth_score', 70),
                'body_alignment': movement_analysis.get('body_alignment', {}).get('alignment_score', 70)
            }
        }
    
    def _score_flexibility(self, results: Dict, age_group: str, gender: str) -> Dict:
        """Score flexibility test performance"""
        metrics = results.get('metrics', {})
        movement_analysis = results.get('movement_analysis', {})
        
        # Primary metric: reach distance
        reach_distance = abs(metrics.get('max_reach_distance', 0))
        consistency = metrics.get('reach_consistency', 0)
        
        # Get benchmarks
        benchmarks = self.benchmarks['Flexibility Test'][age_group][gender]
        
        # Score components
        flexibility_score = self._score_against_benchmark(reach_distance * 100, benchmarks['reach_distance'])
        stability_score = consistency * 100
        technique_score = self._score_technique_components(movement_analysis)
        
        return {
            'primary_metric_score': flexibility_score,
            'stability_score': stability_score,
            'technique_score': technique_score,
            'components': {
                'flexibility_range': flexibility_score,
                'movement_control': stability_score,
                'posture_quality': movement_analysis.get('posture_maintenance', {}).get('alignment_score', 70),
                'progression_technique': movement_analysis.get('progression_pattern', {}).get('technique_score', 70)
            }
        }
    
    def _score_against_benchmark(self, value: float, benchmark: Dict) -> float:
        """Score a value against benchmark ranges"""
        excellent = benchmark['excellent']
        good = benchmark['good']
        average = benchmark['average']
        poor = benchmark['poor']
        
        if value >= excellent:
            return 95 + min((value - excellent) / excellent * 5, 5)  # 95-100
        elif value >= good:
            return 80 + (value - good) / (excellent - good) * 15  # 80-95
        elif value >= average:
            return 60 + (value - average) / (good - average) * 20  # 60-80
        elif value >= poor:
            return 40 + (value - poor) / (average - poor) * 20  # 40-60
        else:
            return max(value / poor * 40, 0)  # 0-40
    
    def _score_technique_components(self, movement_analysis: Dict) -> float:
        """Score technique components from movement analysis"""
        if not movement_analysis:
            return 70.0  # Default score if no analysis available
        
        technique_scores = []
        
        # Extract technique-related scores
        for component, data in movement_analysis.items():
            if isinstance(data, dict):
                for metric, score in data.items():
                    if isinstance(score, (int, float)) and 0 <= score <= 100:
                        technique_scores.append(score)
        
        return np.mean(technique_scores) if technique_scores else 70.0
    
    def _score_consistency(self, results: Dict) -> float:
        """Score movement consistency"""
        validation_issues = results.get('validation_issues', [])
        
        # Base consistency score
        consistency_score = 90.0
        
        # Deduct points for validation issues
        consistency_score -= len(validation_issues) * 10
        
        # Check video quality metrics
        video_quality = results.get('video_quality', {})
        pose_detection_rate = video_quality.get('pose_detection_rate', 0)
        frame_quality = video_quality.get('average_frame_quality', 0)
        
        # Adjust based on detection quality
        consistency_score *= pose_detection_rate
        consistency_score *= frame_quality
        
        return max(consistency_score, 0)
    
    def _score_endurance_pattern(self, movement_analysis: Dict, duration: float) -> float:
        """Score endurance pattern for repetitive exercises"""
        if duration == 0:
            return 50.0
        
        # Base endurance score
        endurance_score = 75.0
        
        # Adjust based on fatigue analysis
        fatigue_analysis = movement_analysis.get('fatigue_analysis', {})
        if fatigue_analysis:
            fatigue_resistance = fatigue_analysis.get('fatigue_resistance', 75)
            endurance_score = (endurance_score + fatigue_resistance) / 2
        
        # Bonus for longer duration tests
        duration_bonus = min(duration / 60 * 10, 15)
        endurance_score += duration_bonus
        
        return min(endurance_score, 100)
    
    def _calculate_overall_score(self, score_data: Dict, test_results: Dict) -> float:
        """Calculate weighted overall score"""
        # Base weights for score components
        weights = {
            'primary_metric_score': 0.5,
            'technique_score': 0.3,
            'consistency_score': 0.1,
            'endurance_score': 0.1,
            'acceleration_score': 0.1,
            'strength_endurance_score': 0.1,
            'stability_score': 0.1
        }
        
        total_score = 0
        total_weight = 0
        
        for component, weight in weights.items():
            if component in score_data:
                total_score += score_data[component] * weight
                total_weight += weight
        
        # Add bonus components
        bonus_components = ['flight_time_bonus', 'cadence_bonus']
        for bonus in bonus_components:
            if bonus in score_data:
                total_score += score_data[bonus]
        
        # Normalize to 0-100 scale
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 50  # Default score if no components available
        
        # Apply validation penalties
        validation_issues = test_results.get('validation_issues', [])
        penalty = min(len(validation_issues) * 5, 25)  # Max 25 point penalty
        
        return max(min(final_score - penalty, 100), 0)
    
    def _calculate_percentile(self, score: float, age_group: str, gender: str, test_type: str) -> float:
        """Calculate percentile ranking based on score"""
        # Simplified percentile calculation
        # In a real system, this would be based on historical data
        
        if score >= 95:
            return 99
        elif score >= 90:
            return 95
        elif score >= 85:
            return 90
        elif score >= 80:
            return 85
        elif score >= 75:
            return 80
        elif score >= 70:
            return 70
        elif score >= 60:
            return 60
        elif score >= 50:
            return 50
        else:
            return max(score / 2, 5)
    
    def _get_performance_category(self, score: float) -> str:
        """Categorize performance level"""
        if score >= 90:
            return "Elite"
        elif score >= 80:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Average"
        elif score >= 50:
            return "Below Average"
        else:
            return "Needs Improvement"
    
    def _generate_benchmark_comparison(self, score_data: Dict, age_group: str, gender: str, test_type: str) -> Dict:
        """Generate detailed benchmark comparison"""
        benchmarks = self.benchmarks.get(test_type, {}).get(age_group, {}).get(gender, {})
        
        comparison = {}
        
        primary_score = score_data.get('primary_metric_score', 0)
        
        if primary_score >= 90:
            comparison['Performance Level'] = f"Exceeds excellent standards for {age_group} {gender.lower()} athletes"
        elif primary_score >= 80:
            comparison['Performance Level'] = f"Meets excellent standards for {age_group} {gender.lower()} athletes"
        elif primary_score >= 60:
            comparison['Performance Level'] = f"Meets good standards for {age_group} {gender.lower()} athletes"
        else:
            comparison['Performance Level'] = f"Below average for {age_group} {gender.lower()} athletes"
        
        # Add component-specific comparisons
        components = score_data.get('components', {})
        for component, score in components.items():
            if score >= 80:
                comparison[component] = "Above average"
            elif score >= 60:
                comparison[component] = "Average"
            else:
                comparison[component] = "Needs improvement"
        
        return comparison
    
    def _get_age_factor(self, age: int) -> float:
        """Get age-based performance factor"""
        # Peak performance typically between 20-25
        if 20 <= age <= 25:
            return 1.0
        elif 16 <= age < 20:
            return 0.95 + (age - 16) * 0.0125  # Gradual increase
        elif 25 < age <= 30:
            return 1.0 - (age - 25) * 0.01  # Gradual decrease
        elif 30 < age <= 35:
            return 0.95 - (age - 30) * 0.02  # Moderate decrease
        else:
            return 0.85  # Maintain minimum factor
    
    def _get_test_difficulty(self, test_type: str) -> float:
        """Get test difficulty factor"""
        difficulty_factors = {
            'Vertical Jump': 1.0,
            'Sit-ups (1 minute)': 0.9,
            '50m Sprint': 1.1,
            'Push-ups': 0.95,
            'Flexibility Test': 0.85
        }
        
        return difficulty_factors.get(test_type, 1.0)
    
    def compare_athletes(self, athlete1_scores: Dict, athlete2_scores: Dict) -> Dict:
        """Compare two athletes' performances"""
        comparison = {
            'athlete1_advantage': [],
            'athlete2_advantage': [],
            'similar_performance': []
        }
        
        score1 = athlete1_scores.get('overall_score', 0)
        score2 = athlete2_scores.get('overall_score', 0)
        
        # Overall comparison
        if abs(score1 - score2) < 5:
            comparison['overall'] = "Similar overall performance"
        elif score1 > score2:
            comparison['overall'] = f"Athlete 1 performs {score1 - score2:.1f} points better overall"
        else:
            comparison['overall'] = f"Athlete 2 performs {score2 - score1:.1f} points better overall"
        
        # Component comparisons
        components1 = athlete1_scores.get('component_scores', {}).get('components', {})
        components2 = athlete2_scores.get('component_scores', {}).get('components', {})
        
        for component in set(components1.keys()) | set(components2.keys()):
            score1_comp = components1.get(component, 0)
            score2_comp = components2.get(component, 0)
            
            if abs(score1_comp - score2_comp) < 5:
                comparison['similar_performance'].append(component)
            elif score1_comp > score2_comp:
                comparison['athlete1_advantage'].append(component)
            else:
                comparison['athlete2_advantage'].append(component)
        
        return comparison
    
    def get_improvement_recommendations(self, score_data: Dict, test_type: str) -> List[str]:
        """Generate improvement recommendations based on scores"""
        recommendations = []
        
        components = score_data.get('components', {})
        overall_score = score_data.get('overall_score', 0)
        
        # General recommendations based on overall score
        if overall_score < 60:
            recommendations.append("Focus on fundamental technique improvement")
            recommendations.append("Increase training frequency and intensity")
        
        # Component-specific recommendations
        for component, score in components.items():
            if score < 70:
                if 'strength' in component.lower():
                    recommendations.append(f"Improve {component} through targeted strength training")
                elif 'form' in component.lower() or 'technique' in component.lower():
                    recommendations.append(f"Work on {component} with proper coaching")
                elif 'endurance' in component.lower():
                    recommendations.append(f"Build {component} through progressive training")
                elif 'flexibility' in component.lower():
                    recommendations.append(f"Include regular stretching for {component}")
        
        # Test-specific recommendations
        test_recommendations = {
            'Vertical Jump': [
                "Include plyometric exercises in training",
                "Focus on explosive leg strength development",
                "Practice proper takeoff and landing technique"
            ],
            'Sit-ups (1 minute)': [
                "Strengthen core muscles with planks and variations",
                "Practice proper sit-up form to maximize efficiency",
                "Build muscular endurance through high-rep training"
            ],
            '50m Sprint': [
                "Work on acceleration and sprint technique",
                "Include speed development drills",
                "Focus on proper running mechanics"
            ],
            'Push-ups': [
                "Build upper body and core strength",
                "Practice proper push-up form and depth",
                "Gradually increase training volume"
            ],
            'Flexibility Test': [
                "Include daily stretching routine",
                "Focus on hamstring and back flexibility",
                "Practice proper sitting reach technique"
            ]
        }
        
        if test_type in test_recommendations:
            recommendations.extend(test_recommendations[test_type][:2])  # Add top 2 recommendations
        
        return list(set(recommendations))  # Remove duplicates
