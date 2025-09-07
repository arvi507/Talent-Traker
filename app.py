import streamlit as st
import pandas as pd
import plotly.express as px
from utils.database import get_database, init_database
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="SAI Talent Assessment Platform",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database safely
try:
    init_database()
    db = get_database()
except Exception as e:
    st.error(f"Database init failed: {e}")
    db = {"athletes": [], "assessments": []}  # fallback empty data

def main():
    st.title("🏅 Sports Authority of India")
    st.subheader("AI-Powered Athletic Talent Assessment Platform")
    
    # Hero section with stats
    col1, col2, col3, col4 = st.columns(4)
    
    
    total_athletes = len(db['athletes'])
    total_assessments = len(db['assessments'])
    avg_score = sum([a.get('overall_score', 0) for a in db['assessments']]) / max(len(db['assessments']), 1)
    
    with col1:
        st.metric("Total Athletes", total_athletes, delta=5)
    with col2:
        st.metric("Assessments Completed", total_assessments, delta=12)
    with col3:
        st.metric("Average Performance Score", f"{avg_score:.1f}", delta=2.3)
    with col4:
        st.metric("Active Tests", "5", delta=0)
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Platform Features
        
        **For Athletes:**
        - 📱 Mobile-first video recording interface
        - 🤖 AI-powered performance analysis
        - 📈 Real-time feedback and scoring
        - 🏆 Gamified progress tracking
        - 📊 Performance benchmarking
        
        **For Officials:**
        - 🔍 Advanced assessment review
        - 📋 Athlete profiling and comparison
        - 🛡️ Automated cheat detection
        - 📈 Performance analytics dashboard
        - 📑 Detailed reporting system
        """)
        
        st.markdown("### 🏃‍♂️ Available Fitness Tests")
        
        tests_df = pd.DataFrame({
            'Test Name': ['Vertical Jump', 'Sit-ups (1 min)', '50m Sprint', 'Push-ups', 'Flexibility Test'],
            'Category': ['Power', 'Core Strength', 'Speed', 'Upper Body', 'Mobility'],
            'Duration': ['5 seconds', '60 seconds', '10 seconds', '60 seconds', '30 seconds'],
            'Equipment': ['Smartphone', 'Smartphone + Mat', 'Smartphone + Track', 'Smartphone + Mat', 'Smartphone']
        })
        
        st.dataframe(tests_df, use_container_width=True)
    
    with col2:
        st.markdown("### 📱 How It Works")
        
        st.markdown("""
        1. **📹 Record Performance**
           Upload video of your fitness test
           
        2. **🤖 AI Analysis**
           Our algorithms analyze your movement
           
        3. **📊 Get Scored**
           Receive instant performance metrics
           
        4. **🏆 Track Progress**
           Monitor improvement over time
           
        5. **🎯 Get Selected**
           Top performers get SAI attention
        """)
        
        st.info("💡 **Tip:** Ensure good lighting and clear camera angle for best results!")
    
    # Recent activity
    st.markdown("---")
    st.subheader("📊 Platform Analytics")
    
    # Sample analytics data
    if db['assessments']:
        # Performance distribution chart
        scores = [a.get('overall_score', 0) for a in db['assessments']]
        fig = px.histogram(x=scores, nbins=10, title="Performance Score Distribution")
        fig.update_xaxes(title="Score")
        fig.update_yaxes(title="Number of Athletes")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📈 Analytics will appear here once assessments are completed")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🇮🇳 <strong>Sports Authority of India</strong> | Democratizing Athletic Talent Discovery</p>
        <p><em>Powered by AI • Built for India • Designed for Excellence</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
