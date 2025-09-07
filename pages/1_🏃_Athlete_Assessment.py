import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
import uuid
from utils.video_analysis import VideoAnalyzer
from utils.fitness_tests import FitnessTestProcessor
from utils.scoring import PerformanceScorer
from utils.cheat_detection import CheatDetector
from utils.database import get_database, save_assessment, save_athlete

st.set_page_config(page_title="Athlete Assessment", page_icon="🏃", layout="wide")

def main():
    st.title("🏃‍♂️ Athletic Performance Assessment")
    st.markdown("Upload your fitness test video and get instant AI-powered analysis!")
    
    # Athlete registration
    st.sidebar.header("👤 Athlete Information")
    athlete_name = st.sidebar.text_input("Full Name", placeholder="Enter your name")
    age = st.sidebar.number_input("Age", min_value=12, max_value=40, value=18)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    state = st.sidebar.text_input("State", placeholder="e.g., Maharashtra")
    sport_interest = st.sidebar.selectbox("Sport of Interest", 
                                        ["Athletics", "Football", "Cricket", "Basketball", 
                                         "Swimming", "Wrestling", "Boxing", "Other"])
    
    # Test selection
    st.header("🎯 Select Fitness Test")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        test_type = st.selectbox("Choose Test Type", [
            "Vertical Jump",
            "Sit-ups (1 minute)",
            "50m Sprint",
            "Push-ups",
            "Flexibility Test"
        ])
    
    with col2:
        # Test instructions
        instructions = {
            "Vertical Jump": """
            📋 **Instructions:**
            1. Stand with feet shoulder-width apart
            2. Jump as high as possible
            3. Land safely on both feet
            4. Camera should capture full body profile
            """,
            "Sit-ups (1 minute)": """
            📋 **Instructions:**
            1. Lie on your back, knees bent at 90°
            2. Hands behind head or crossed on chest
            3. Perform maximum sit-ups in 60 seconds
            4. Camera should show side profile
            """,
            "50m Sprint": """
            📋 **Instructions:**
            1. Start in standing position
            2. Run 50 meters at maximum speed
            3. Camera should capture full run
            4. Ensure clear start and finish markers
            """,
            "Push-ups": """
            📋 **Instructions:**
            1. Start in plank position
            2. Lower body until chest nearly touches ground
            3. Push back up to starting position
            4. Camera should show side profile
            """,
            "Flexibility Test": """
            📋 **Instructions:**
            1. Sit with legs straight, feet against wall
            2. Reach forward as far as possible
            3. Hold position for 3 seconds
            4. Camera should show side profile
            """
        }
        
        st.info(instructions[test_type])
    
    # Video upload section
    st.header("📹 Upload Performance Video")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a clear video of your performance. Max file size: 100MB"
    )
    
    if uploaded_video and athlete_name:
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        # Display video
        st.video(uploaded_video)
        
        # Analysis button
        if st.button("🤖 Analyze Performance", type="primary"):
            with st.spinner("🔍 Analyzing your performance..."):
                try:
                    # Initialize processors
                    analyzer = VideoAnalyzer()
                    test_processor = FitnessTestProcessor()
                    scorer = PerformanceScorer()
                    cheat_detector = CheatDetector()
                    
                    # Step 1: Basic video analysis
                    st.write("🔄 Step 1: Processing video...")
                    video_data = analyzer.analyze_video(video_path)
                    
                    # Step 2: Test-specific analysis
                    st.write("🔄 Step 2: Analyzing movement patterns...")
                    test_results = test_processor.process_test(video_path, test_type)
                    
                    # Step 3: Cheat detection
                    st.write("🔄 Step 3: Verifying authenticity...")
                    cheat_analysis = cheat_detector.detect_anomalies(video_path, test_type)
                    
                    # Step 4: Scoring
                    st.write("🔄 Step 4: Calculating performance score...")
                    performance_score = scorer.calculate_score(test_results, age, gender, test_type)
                    
                    # Display results
                    display_results(test_results, performance_score, cheat_analysis, 
                                  athlete_name, age, gender, state, sport_interest, test_type)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("Please ensure your video is clear and follows the test instructions.")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(video_path):
                        os.unlink(video_path)
    
    elif uploaded_video and not athlete_name:
        st.warning("⚠️ Please enter your name in the sidebar to proceed with analysis.")
    
    # Sample videos section
    st.markdown("---")
    st.header("📺 Sample Videos")
    st.markdown("Watch these examples to understand proper test execution:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Vertical Jump Example**")
        st.info("🎥 Sample video would be displayed here")
    with col2:
        st.markdown("**Sit-ups Example**")
        st.info("🎥 Sample video would be displayed here")
    with col3:
        st.markdown("**Sprint Example**")
        st.info("🎥 Sample video would be displayed here")

def display_results(test_results, performance_score, cheat_analysis, 
                   athlete_name, age, gender, state, sport_interest, test_type):
    """Display comprehensive analysis results"""
    
    st.success("✅ Analysis Complete!")
    
    # Overall score card
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Overall Score", 
            f"{performance_score['overall_score']:.1f}/100",
            delta=f"{performance_score['percentile']:.0f}th percentile"
        )
    
    with col2:
        authenticity_score = 100 - (cheat_analysis.get('risk_score', 0) * 100)
        st.metric(
            "Authenticity Score",
            f"{authenticity_score:.1f}%",
            delta="Verified" if authenticity_score > 80 else "Review Needed"
        )
    
    with col3:
        grade = get_performance_grade(performance_score['overall_score'])
        st.metric("Performance Grade", grade, delta=performance_score['category'])
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Metrics", "🔍 Movement Analysis", 
                                      "🏆 Benchmarking", "🛡️ Verification"])
    
    with tab1:
        st.subheader("Performance Metrics")
        
        # Test-specific metrics
        metrics_data = test_results.get('metrics', {})
        
        for metric, value in metrics_data.items():
            if isinstance(value, (int, float)):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{metric.replace('_', ' ').title()}**")
                with col2:
                    st.write(f"{value:.2f}")
    
    with tab2:
        st.subheader("Movement Analysis")
        
        # Movement quality assessment
        movement_data = test_results.get('movement_analysis', {})
        
        if movement_data:
            for aspect, score in movement_data.items():
                progress = min(score / 100, 1.0) if isinstance(score, (int, float)) else 0.5
                st.write(f"**{aspect.replace('_', ' ').title()}**")
                st.progress(progress)
                st.write(f"Score: {score}/100" if isinstance(score, (int, float)) else f"Assessment: {score}")
        else:
            st.info("Movement analysis data will be displayed here")
    
    with tab3:
        st.subheader("Performance Benchmarking")
        
        # Comparison with benchmarks
        benchmark_data = performance_score.get('benchmark_comparison', {})
        
        if benchmark_data:
            st.write("**Comparison with Age/Gender Standards:**")
            for category, comparison in benchmark_data.items():
                st.write(f"- **{category}**: {comparison}")
        
        # Performance trends (if athlete has previous records)
        st.write("**Historical Performance:**")
        st.info("Track your progress over time with multiple assessments")
    
    with tab4:
        st.subheader("Video Verification")
        
        # Authenticity checks
        verification_results = cheat_analysis.get('checks', {})
        
        st.write("**Automated Verification Checks:**")
        for check, result in verification_results.items():
            icon = "✅" if result.get('passed', False) else "⚠️"
            st.write(f"{icon} **{check.replace('_', ' ').title()}**: {result.get('status', 'Unknown')}")
        
        if cheat_analysis.get('risk_score', 0) < 0.2:
            st.success("🛡️ Video appears authentic and follows test protocols")
        else:
            st.warning("⚠️ Some anomalies detected. Manual review may be required.")
    
    # Save assessment to database
    assessment_id = str(uuid.uuid4())
    athlete_id = save_athlete({
        'name': athlete_name,
        'age': age,
        'gender': gender,
        'state': state,
        'sport_interest': sport_interest
    })
    
    assessment_data = {
        'id': assessment_id,
        'athlete_id': athlete_id,
        'test_type': test_type,
        'timestamp': datetime.now().isoformat(),
        'overall_score': performance_score['overall_score'],
        'metrics': test_results.get('metrics', {}),
        'authenticity_score': authenticity_score,
        'verification_status': 'verified' if authenticity_score > 80 else 'pending_review'
    }
    
    save_assessment(assessment_data)
    
    # Gamification elements
    st.markdown("---")
    st.subheader("🏆 Achievements")
    
    # Award badges based on performance
    badges = award_badges(performance_score['overall_score'], authenticity_score)
    
    if badges:
        cols = st.columns(len(badges))
        for i, badge in enumerate(badges):
            with cols[i]:
                st.success(f"🏅 {badge}")
    else:
        st.info("Keep training to unlock achievement badges!")
    
    # Next steps
    st.markdown("---")
    st.subheader("🎯 Next Steps")
    
    if performance_score['overall_score'] >= 80:
        st.success("🌟 Excellent performance! You may be eligible for advanced training programs.")
    elif performance_score['overall_score'] >= 60:
        st.info("👍 Good performance! Focus on specific areas for improvement.")
    else:
        st.warning("💪 Keep training! Regular practice will help improve your scores.")
    
    st.write("**Recommendations:**")
    recommendations = generate_recommendations(performance_score, test_type)
    for rec in recommendations:
        st.write(f"- {rec}")

def get_performance_grade(score):
    """Convert numeric score to letter grade"""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B+"
    elif score >= 60:
        return "B"
    elif score >= 50:
        return "C"
    else:
        return "D"

def award_badges(performance_score, authenticity_score):
    """Award badges based on performance"""
    badges = []
    
    if performance_score >= 90:
        badges.append("Elite Performer")
    elif performance_score >= 80:
        badges.append("High Achiever")
    elif performance_score >= 70:
        badges.append("Strong Performer")
    
    if authenticity_score >= 95:
        badges.append("Verified Athlete")
    
    return badges

def generate_recommendations(performance_score, test_type):
    """Generate training recommendations"""
    recommendations = []
    
    score = performance_score['overall_score']
    
    if test_type == "Vertical Jump":
        if score < 70:
            recommendations.extend([
                "Focus on plyometric exercises like box jumps",
                "Strengthen your leg muscles with squats and lunges",
                "Work on explosive power training"
            ])
        else:
            recommendations.append("Maintain current training regimen")
    
    elif test_type == "Sit-ups (1 minute)":
        if score < 70:
            recommendations.extend([
                "Build core strength with planks and crunches",
                "Practice proper sit-up form",
                "Increase endurance with interval training"
            ])
    
    elif test_type == "50m Sprint":
        if score < 70:
            recommendations.extend([
                "Work on acceleration drills",
                "Improve running technique",
                "Build leg strength and power"
            ])
    
    # General recommendations
    recommendations.extend([
        "Take another assessment in 4-6 weeks",
        "Consider consulting with a sports trainer",
        "Maintain regular fitness routine"
    ])
    
    return recommendations

if __name__ == "__main__":
    main()
