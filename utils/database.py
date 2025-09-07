import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import threading

class SportsTalentDatabase:
    """Simple in-memory database with file persistence for sports talent assessment data"""
    
    def __init__(self, data_file: str = "data/database.json"):
        self.data_file = data_file
        self.lock = threading.Lock()
        
        # Initialize database structure
        self.db = {
            'athletes': [],
            'assessments': [],
            'officials': [],
            'benchmarks': [],
            'metadata': {
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
        }
        
        # Load existing data if available
        self._load_data()
    
    def _load_data(self):
        """Load data from file if it exists"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    loaded_data = json.load(f)
                    # Merge with default structure to handle schema updates
                    for key, value in loaded_data.items():
                        if key in self.db:
                            self.db[key] = value
        except Exception as e:
            print(f"Warning: Could not load database file: {e}")
    
    def _save_data(self):
        """Save data to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            
            # Update metadata
            self.db['metadata']['last_updated'] = datetime.now().isoformat()
            
            # Save to file
            with open(self.data_file, 'w') as f:
                json.dump(self.db, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save database file: {e}")
    
    def get_database(self) -> Dict:
        """Get a copy of the entire database"""
        with self.lock:
            return self.db.copy()
    
    def save_athlete(self, athlete_data: Dict) -> str:
        """
        Save athlete data and return athlete ID
        
        Args:
            athlete_data: Dictionary containing athlete information
            
        Returns:
            String athlete ID
        """
        with self.lock:
            # Check if athlete already exists (by name for simplicity)
            athlete_name = athlete_data.get('name', '').strip()
            
            for existing_athlete in self.db['athletes']:
                if (existing_athlete.get('name', '').strip().lower() == 
                    athlete_name.lower() and 
                    existing_athlete.get('age') == athlete_data.get('age')):
                    return existing_athlete['id']
            
            # Create new athlete record
            athlete_id = str(uuid.uuid4())
            athlete_record = {
                'id': athlete_id,
                'name': athlete_data.get('name', ''),
                'age': athlete_data.get('age', 0),
                'gender': athlete_data.get('gender', ''),
                'state': athlete_data.get('state', ''),
                'sport_interest': athlete_data.get('sport_interest', ''),
                'contact_info': athlete_data.get('contact_info', {}),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'status': 'active',
                'assessments_completed': 0,
                'best_scores': {},
                'notes': []
            }
            
            self.db['athletes'].append(athlete_record)
            self._save_data()
            
            return athlete_id
    
    def save_assessment(self, assessment_data: Dict) -> str:
        """
        Save assessment data and return assessment ID
        
        Args:
            assessment_data: Dictionary containing assessment information
            
        Returns:
            String assessment ID
        """
        with self.lock:
            assessment_id = assessment_data.get('id', str(uuid.uuid4()))
            
            assessment_record = {
                'id': assessment_id,
                'athlete_id': assessment_data.get('athlete_id', ''),
                'test_type': assessment_data.get('test_type', ''),
                'timestamp': assessment_data.get('timestamp', datetime.now().isoformat()),
                'overall_score': assessment_data.get('overall_score', 0),
                'component_scores': assessment_data.get('component_scores', {}),
                'metrics': assessment_data.get('metrics', {}),
                'movement_analysis': assessment_data.get('movement_analysis', {}),
                'video_analysis': assessment_data.get('video_analysis', {}),
                'verification_status': assessment_data.get('verification_status', 'pending'),
                'authenticity_score': assessment_data.get('authenticity_score', 0),
                'cheat_detection': assessment_data.get('cheat_detection', {}),
                'benchmarking': assessment_data.get('benchmarking', {}),
                'created_at': datetime.now().isoformat(),
                'processed_by': assessment_data.get('processed_by', 'system'),
                'review_notes': assessment_data.get('review_notes', []),
                'flags': assessment_data.get('flags', [])
            }
            
            self.db['assessments'].append(assessment_record)
            
            # Update athlete's assessment count and best scores
            self._update_athlete_stats(assessment_record)
            
            self._save_data()
            
            return assessment_id
    
    def get_athlete(self, athlete_id: str) -> Optional[Dict]:
        """Get athlete by ID"""
        with self.lock:
            for athlete in self.db['athletes']:
                if athlete['id'] == athlete_id:
                    return athlete.copy()
            return None
    
    def get_assessment(self, assessment_id: str) -> Optional[Dict]:
        """Get assessment by ID"""
        with self.lock:
            for assessment in self.db['assessments']:
                if assessment['id'] == assessment_id:
                    return assessment.copy()
            return None
    
    def get_athlete_assessments(self, athlete_id: str) -> List[Dict]:
        """Get all assessments for a specific athlete"""
        with self.lock:
            assessments = []
            for assessment in self.db['assessments']:
                if assessment['athlete_id'] == athlete_id:
                    assessments.append(assessment.copy())
            
            # Sort by timestamp, most recent first
            assessments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return assessments
    
    def update_assessment_status(self, assessment_id: str, status: str, 
                               notes: str = '', reviewer: str = 'system') -> bool:
        """Update assessment verification status"""
        with self.lock:
            for assessment in self.db['assessments']:
                if assessment['id'] == assessment_id:
                    assessment['verification_status'] = status
                    assessment['updated_at'] = datetime.now().isoformat()
                    
                    # Add review note
                    review_note = {
                        'timestamp': datetime.now().isoformat(),
                        'reviewer': reviewer,
                        'action': f'Status changed to {status}',
                        'notes': notes
                    }
                    assessment['review_notes'].append(review_note)
                    
                    self._save_data()
                    return True
            return False
    
    def search_athletes(self, criteria: Dict) -> List[Dict]:
        """
        Search athletes based on criteria
        
        Args:
            criteria: Dictionary with search parameters (name, state, sport_interest, etc.)
            
        Returns:
            List of matching athlete records
        """
        with self.lock:
            results = []
            
            for athlete in self.db['athletes']:
                match = True
                
                # Check each criteria
                for key, value in criteria.items():
                    if key in athlete:
                        athlete_value = str(athlete[key]).lower()
                        search_value = str(value).lower()
                        
                        if key == 'name':
                            # Partial match for name
                            if search_value not in athlete_value:
                                match = False
                                break
                        else:
                            # Exact match for other fields
                            if athlete_value != search_value:
                                match = False
                                break
                
                if match:
                    results.append(athlete.copy())
            
            return results
    
    def get_leaderboard(self, test_type: str = None, limit: int = 100) -> List[Dict]:
        """
        Get leaderboard data
        
        Args:
            test_type: Filter by specific test type (optional)
            limit: Maximum number of results
            
        Returns:
            List of top performers with athlete info
        """
        with self.lock:
            assessments = self.db['assessments'].copy()
            
            # Filter by test type if specified
            if test_type:
                assessments = [a for a in assessments if a.get('test_type') == test_type]
            
            # Filter verified assessments only
            assessments = [a for a in assessments if a.get('verification_status') == 'verified']
            
            # Sort by overall score, descending
            assessments.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
            
            # Limit results
            assessments = assessments[:limit]
            
            # Enrich with athlete data
            leaderboard = []
            for assessment in assessments:
                athlete = self.get_athlete(assessment['athlete_id'])
                if athlete:
                    entry = {
                        'assessment_id': assessment['id'],
                        'athlete_id': assessment['athlete_id'],
                        'athlete_name': athlete['name'],
                        'age': athlete['age'],
                        'gender': athlete['gender'],
                        'state': athlete['state'],
                        'sport_interest': athlete['sport_interest'],
                        'test_type': assessment['test_type'],
                        'overall_score': assessment['overall_score'],
                        'timestamp': assessment['timestamp'],
                        'rank': len(leaderboard) + 1
                    }
                    leaderboard.append(entry)
            
            return leaderboard
    
    def get_analytics_data(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        Get analytics data for dashboard
        
        Args:
            start_date: Filter start date (ISO format)
            end_date: Filter end date (ISO format)
            
        Returns:
            Dictionary containing analytics data
        """
        with self.lock:
            assessments = self.db['assessments'].copy()
            athletes = self.db['athletes'].copy()
            
            # Date filtering
            if start_date or end_date:
                filtered_assessments = []
                for assessment in assessments:
                    timestamp = assessment.get('timestamp', '')
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue
                    filtered_assessments.append(assessment)
                assessments = filtered_assessments
            
            # Calculate analytics
            analytics = {
                'total_athletes': len(athletes),
                'total_assessments': len(assessments),
                'verified_assessments': len([a for a in assessments if a.get('verification_status') == 'verified']),
                'pending_reviews': len([a for a in assessments if a.get('verification_status') == 'pending_review']),
                'average_score': 0,
                'score_distribution': {},
                'test_type_distribution': {},
                'gender_distribution': {},
                'state_distribution': {},
                'age_distribution': {},
                'performance_trends': {},
                'top_performers': []
            }
            
            if assessments:
                # Average score
                verified_assessments = [a for a in assessments if a.get('verification_status') == 'verified']
                if verified_assessments:
                    analytics['average_score'] = sum(a.get('overall_score', 0) for a in verified_assessments) / len(verified_assessments)
                
                # Score distribution
                score_ranges = {'0-20': 0, '21-40': 0, '41-60': 0, '61-80': 0, '81-100': 0}
                for assessment in verified_assessments:
                    score = assessment.get('overall_score', 0)
                    if score <= 20:
                        score_ranges['0-20'] += 1
                    elif score <= 40:
                        score_ranges['21-40'] += 1
                    elif score <= 60:
                        score_ranges['41-60'] += 1
                    elif score <= 80:
                        score_ranges['61-80'] += 1
                    else:
                        score_ranges['81-100'] += 1
                analytics['score_distribution'] = score_ranges
                
                # Test type distribution
                test_types = {}
                for assessment in assessments:
                    test_type = assessment.get('test_type', 'Unknown')
                    test_types[test_type] = test_types.get(test_type, 0) + 1
                analytics['test_type_distribution'] = test_types
            
            # Athlete demographics
            gender_dist = {}
            state_dist = {}
            age_dist = {'<16': 0, '16-20': 0, '21-25': 0, '26-30': 0, '30+': 0}
            
            for athlete in athletes:
                # Gender distribution
                gender = athlete.get('gender', 'Unknown')
                gender_dist[gender] = gender_dist.get(gender, 0) + 1
                
                # State distribution
                state = athlete.get('state', 'Unknown')
                state_dist[state] = state_dist.get(state, 0) + 1
                
                # Age distribution
                age = athlete.get('age', 0)
                if age < 16:
                    age_dist['<16'] += 1
                elif age <= 20:
                    age_dist['16-20'] += 1
                elif age <= 25:
                    age_dist['21-25'] += 1
                elif age <= 30:
                    age_dist['26-30'] += 1
                else:
                    age_dist['30+'] += 1
            
            analytics['gender_distribution'] = gender_dist
            analytics['state_distribution'] = state_dist
            analytics['age_distribution'] = age_dist
            
            # Top performers (top 10)
            leaderboard = self.get_leaderboard(limit=10)
            analytics['top_performers'] = leaderboard
            
            return analytics
    
    def _update_athlete_stats(self, assessment: Dict):
        """Update athlete statistics after new assessment"""
        athlete_id = assessment.get('athlete_id')
        if not athlete_id:
            return
        
        for athlete in self.db['athletes']:
            if athlete['id'] == athlete_id:
                # Increment assessment count
                athlete['assessments_completed'] = athlete.get('assessments_completed', 0) + 1
                
                # Update best scores
                test_type = assessment.get('test_type')
                overall_score = assessment.get('overall_score', 0)
                
                if test_type:
                    current_best = athlete.get('best_scores', {}).get(test_type, 0)
                    if overall_score > current_best:
                        if 'best_scores' not in athlete:
                            athlete['best_scores'] = {}
                        athlete['best_scores'][test_type] = overall_score
                
                athlete['updated_at'] = datetime.now().isoformat()
                break
    
    def add_athlete_note(self, athlete_id: str, note: str, author: str = 'system') -> bool:
        """Add a note to athlete record"""
        with self.lock:
            for athlete in self.db['athletes']:
                if athlete['id'] == athlete_id:
                    note_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'author': author,
                        'note': note
                    }
                    
                    if 'notes' not in athlete:
                        athlete['notes'] = []
                    
                    athlete['notes'].append(note_entry)
                    athlete['updated_at'] = datetime.now().isoformat()
                    
                    self._save_data()
                    return True
            return False
    
    def flag_assessment(self, assessment_id: str, flag_type: str, 
                       reason: str, flagged_by: str = 'system') -> bool:
        """Flag an assessment for review"""
        with self.lock:
            for assessment in self.db['assessments']:
                if assessment['id'] == assessment_id:
                    flag_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'type': flag_type,
                        'reason': reason,
                        'flagged_by': flagged_by,
                        'resolved': False
                    }
                    
                    if 'flags' not in assessment:
                        assessment['flags'] = []
                    
                    assessment['flags'].append(flag_entry)
                    
                    # Change status to require review
                    if assessment.get('verification_status') == 'verified':
                        assessment['verification_status'] = 'flagged'
                    
                    self._save_data()
                    return True
            return False
    
    def get_pending_reviews(self) -> List[Dict]:
        """Get assessments requiring manual review"""
        with self.lock:
            pending = []
            
            for assessment in self.db['assessments']:
                status = assessment.get('verification_status', '')
                if status in ['pending_review', 'flagged']:
                    # Enrich with athlete data
                    athlete = self.get_athlete(assessment['athlete_id'])
                    assessment_copy = assessment.copy()
                    if athlete:
                        assessment_copy['athlete_info'] = athlete
                    pending.append(assessment_copy)
            
            # Sort by timestamp, oldest first (FIFO)
            pending.sort(key=lambda x: x.get('timestamp', ''))
            return pending
    
    def export_data(self, export_type: str = 'json') -> Dict:
        """Export database for backup or analysis"""
        with self.lock:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'export_type': export_type,
                'data': self.db.copy()
            }
            return export_data
    
    def import_data(self, import_data: Dict, merge: bool = True) -> bool:
        """Import data from backup or external source"""
        try:
            with self.lock:
                if merge:
                    # Merge data, avoiding duplicates
                    imported_db = import_data.get('data', {})
                    
                    # Merge athletes
                    existing_athlete_ids = {a['id'] for a in self.db['athletes']}
                    for athlete in imported_db.get('athletes', []):
                        if athlete['id'] not in existing_athlete_ids:
                            self.db['athletes'].append(athlete)
                    
                    # Merge assessments
                    existing_assessment_ids = {a['id'] for a in self.db['assessments']}
                    for assessment in imported_db.get('assessments', []):
                        if assessment['id'] not in existing_assessment_ids:
                            self.db['assessments'].append(assessment)
                else:
                    # Replace entire database
                    self.db = import_data.get('data', self.db)
                
                self._save_data()
                return True
                
        except Exception as e:
            print(f"Import failed: {e}")
            return False

# Global database instance
_db_instance = None

def get_database() -> Dict:
    """Get the global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SportsTalentDatabase()
    return _db_instance.get_database()

def save_athlete(athlete_data: Dict) -> str:
    """Save athlete data using global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SportsTalentDatabase()
    return _db_instance.save_athlete(athlete_data)

def save_assessment(assessment_data: Dict) -> str:
    """Save assessment data using global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SportsTalentDatabase()
    return _db_instance.save_assessment(assessment_data)

def init_database():
    """Initialize the database with sample data if empty"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SportsTalentDatabase()
    
    # Check if database is empty and add sample data
    db = _db_instance.get_database()
    if len(db['athletes']) == 0:
        # Add sample athletes for demonstration
        sample_athletes = [
            {
                'name': 'Arjun Kumar',
                'age': 19,
                'gender': 'Male',
                'state': 'Maharashtra',
                'sport_interest': 'Athletics'
            },
            {
                'name': 'Priya Singh',
                'age': 17,
                'gender': 'Female',
                'state': 'Punjab',
                'sport_interest': 'Swimming'
            },
            {
                'name': 'Rajesh Patel',
                'age': 22,
                'gender': 'Male',
                'state': 'Gujarat',
                'sport_interest': 'Wrestling'
            }
        ]
        
        for athlete_data in sample_athletes:
            _db_instance.save_athlete(athlete_data)
