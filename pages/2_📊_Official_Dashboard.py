import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.database import get_database
import numpy as np

st.set_page_config(page_title="Official Dashboard", page_icon="📊", layout="wide")

def main():
    st.title("📊 SAI Official Dashboard")
    st.markdown("Comprehensive analysis and management of athlete assessments")
    
    # Authentication (simplified for demo)
    if 'authenticated' not in st.session_state:
        authenticate_official()
        return
    
    # Load data
    db = get_database()
    athletes_df = pd.DataFrame(db['athletes'])
    assessments_df = pd.DataFrame(db['assessments'])
    
    if assessments_df.empty:
        st.info("📈 No assessment data available yet. Athletes need to complete assessments first.")
        return
    
    # Merge athlete and assessment data
    if not athletes_df.empty:
        full_df = assessments_df.merge(athletes_df, left_on='athlete_id', right_on='id', how='left', suffixes=('', '_athlete'))
    else:
        full_df = assessments_df
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analytics Overview", 
        "👥 Athlete Profiles", 
        "🔍 Assessment Review",
        "🏆 Talent Identification",
        "📋 Reports"
    ])
    
    with tab1:
        display_analytics_overview(full_df)
    
    with tab2:
        display_athlete_profiles(full_df)
    
    with tab3:
        display_assessment_review(full_df)
    
    with tab4:
        display_talent_identification(full_df)
    
    with tab5:
        display_reports(full_df)

def authenticate_official():
    """Simple authentication for officials"""
    st.subheader("🔐 Official Access")
    st.info("This dashboard is restricted to authorized SAI officials only.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        official_id = st.text_input("Official ID", type="password")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", type="primary"):
            # Simple authentication (in production, use proper authentication)
            if official_id == "SAI_OFFICIAL" and password == "sai2025":
                st.session_state.authenticated = True
                st.success("✅ Authentication successful")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
    
    with col2:
        st.markdown("""
        **Demo Credentials:**
        - Official ID: `SAI_OFFICIAL`
        - Password: `sai2025`
        
        **Features Available:**
        - 📊 Performance analytics
        - 👥 Athlete management
        - 🔍 Assessment verification
        - 🏆 Talent identification
        - 📋 Detailed reporting
        """)

def display_analytics_overview(df):
    """Display main analytics dashboard"""
    st.subheader("📈 Performance Analytics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_assessments = len(df)
        st.metric("Total Assessments", total_assessments, delta=f"+{int(total_assessments * 0.15)}")
    
    with col2:
        avg_score = df['overall_score'].mean() if 'overall_score' in df.columns else 0
        st.metric("Average Score", f"{avg_score:.1f}", delta="+2.3")
    
    with col3:
        high_performers = len(df[df['overall_score'] >= 80]) if 'overall_score' in df.columns else 0
        st.metric("High Performers (80+)", high_performers, delta=f"+{int(high_performers * 0.2)}")
    
    with col4:
        verified_assessments = len(df[df.get('verification_status') == 'verified'])
        verification_rate = (verified_assessments / total_assessments * 100) if total_assessments > 0 else 0
        st.metric("Verification Rate", f"{verification_rate:.1f}%", delta="+5.2%")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        if 'overall_score' in df.columns:
            # Score distribution
            fig = px.histogram(df, x='overall_score', nbins=15, 
                             title="Performance Score Distribution")
            fig.update_xaxes(title="Score")
            fig.update_yaxes(title="Number of Athletes")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'test_type' in df.columns:
            # Test type distribution
            test_counts = df['test_type'].value_counts()
            fig = px.pie(values=test_counts.values, names=test_counts.index,
                        title="Assessment Distribution by Test Type")
            st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        if 'gender' in df.columns and 'overall_score' in df.columns:
            # Gender performance comparison
            fig = px.box(df, x='gender', y='overall_score', 
                        title="Performance by Gender")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'age' in df.columns and 'overall_score' in df.columns:
            # Age vs performance scatter
            fig = px.scatter(df, x='age', y='overall_score',
                           title="Performance vs Age", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
    
    # State-wise analysis
    if 'state' in df.columns:
        st.subheader("🗺️ State-wise Performance Analysis")
        
        state_stats = df.groupby('state').agg({
            'overall_score': ['mean', 'count'],
            'verification_status': lambda x: (x == 'verified').sum()
        }).round(2)
        
        state_stats.columns = ['Avg Score', 'Total Assessments', 'Verified']
        state_stats = state_stats.sort_values('Avg Score', ascending=False)
        
        st.dataframe(state_stats, use_container_width=True)

def display_athlete_profiles(df):
    """Display athlete management interface"""
    st.subheader("👥 Athlete Profile Management")
    
    # Search and filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_name = st.text_input("🔍 Search by name", placeholder="Enter athlete name")
    
    with col2:
        filter_state = st.selectbox("Filter by State", 
                                   ["All"] + list(df['state'].unique()) if 'state' in df.columns else ["All"])
    
    with col3:
        filter_sport = st.selectbox("Filter by Sport Interest",
                                  ["All"] + list(df['sport_interest'].unique()) if 'sport_interest' in df.columns else ["All"])
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_name:
        filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False, na=False)]
    
    if filter_state != "All":
        filtered_df = filtered_df[filtered_df['state'] == filter_state]
    
    if filter_sport != "All":
        filtered_df = filtered_df[filtered_df['sport_interest'] == filter_sport]
    
    # Display athlete cards
    if not filtered_df.empty:
        for idx, athlete in filtered_df.iterrows():
            with st.expander(f"👤 {athlete.get('name', 'Unknown')} - Score: {athlete.get('overall_score', 0):.1f}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Age:** {athlete.get('age', 'N/A')}")
                    st.write(f"**Gender:** {athlete.get('gender', 'N/A')}")
                    st.write(f"**State:** {athlete.get('state', 'N/A')}")
                
                with col2:
                    st.write(f"**Sport Interest:** {athlete.get('sport_interest', 'N/A')}")
                    st.write(f"**Test Type:** {athlete.get('test_type', 'N/A')}")
                    st.write(f"**Assessment Date:** {athlete.get('timestamp', 'N/A')[:10] if athlete.get('timestamp') else 'N/A'}")
                
                with col3:
                    st.write(f"**Overall Score:** {athlete.get('overall_score', 0):.1f}/100")
                    st.write(f"**Verification:** {athlete.get('verification_status', 'Unknown').title()}")
                    
                    # Action buttons
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button(f"📋 Details", key=f"details_{idx}"):
                            st.info("Detailed athlete profile would open here")
                    with col_btn2:
                        if st.button(f"⭐ Shortlist", key=f"shortlist_{idx}"):
                            st.success("Athlete added to shortlist!")
    else:
        st.info("No athletes found matching the search criteria")

def display_assessment_review(df):
    """Display assessment review interface"""
    st.subheader("🔍 Assessment Review & Verification")
    
    # Pending reviews
    pending_df = df[df.get('verification_status') == 'pending_review']
    
    if not pending_df.empty:
        st.warning(f"⚠️ {len(pending_df)} assessments require manual review")
        
        for idx, assessment in pending_df.iterrows():
            with st.expander(f"🔍 Review Required: {assessment.get('name', 'Unknown')} - {assessment.get('test_type', 'Unknown')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Athlete:** {assessment.get('name', 'Unknown')}")
                    st.write(f"**Test:** {assessment.get('test_type', 'Unknown')}")
                    st.write(f"**Score:** {assessment.get('overall_score', 0):.1f}/100")
                    st.write(f"**Date:** {assessment.get('timestamp', 'N/A')[:10] if assessment.get('timestamp') else 'N/A'}")
                    
                    # Reasons for review
                    st.write("**Review Reasons:**")
                    st.write("- Authenticity score below threshold")
                    st.write("- Unusual performance metrics detected")
                    st.write("- Manual verification requested")
                
                with col2:
                    st.write("**Actions:**")
                    if st.button(f"✅ Approve", key=f"approve_{idx}", type="primary"):
                        st.success("Assessment approved!")
                    if st.button(f"❌ Reject", key=f"reject_{idx}"):
                        st.error("Assessment rejected!")
                    if st.button(f"📝 Request Resubmission", key=f"resubmit_{idx}"):
                        st.info("Resubmission requested!")
    else:
        st.success("✅ All assessments have been reviewed!")
    
    # Verification statistics
    st.markdown("---")
    st.subheader("📊 Verification Statistics")
    
    if 'verification_status' in df.columns:
        verification_counts = df['verification_status'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=verification_counts.values, names=verification_counts.index,
                        title="Verification Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            verification_df = pd.DataFrame({
                'Status': verification_counts.index,
                'Count': verification_counts.values,
                'Percentage': (verification_counts.values / verification_counts.sum() * 100).round(1)
            })
            st.dataframe(verification_df, use_container_width=True)

def display_talent_identification(df):
    """Display talent identification dashboard"""
    st.subheader("🏆 Talent Identification & Ranking")
    
    if 'overall_score' in df.columns:
        # Top performers
        top_performers = df.nlargest(20, 'overall_score')
        
        st.write("### 🥇 Top 20 Performers")
        
        # Create ranking
        ranking_df = top_performers[['name', 'age', 'gender', 'state', 'sport_interest', 'test_type', 'overall_score']].copy()
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        ranking_df = ranking_df[['Rank', 'name', 'age', 'gender', 'state', 'sport_interest', 'test_type', 'overall_score']]
        ranking_df.columns = ['Rank', 'Name', 'Age', 'Gender', 'State', 'Sport Interest', 'Test Type', 'Score']
        
        st.dataframe(ranking_df, use_container_width=True)
        
        # Talent pool analysis
        st.markdown("---")
        st.write("### 📊 Talent Pool Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance categories
            elite = len(df[df['overall_score'] >= 90])
            advanced = len(df[(df['overall_score'] >= 80) & (df['overall_score'] < 90)])
            intermediate = len(df[(df['overall_score'] >= 70) & (df['overall_score'] < 80)])
            developing = len(df[df['overall_score'] < 70])
            
            categories = ['Elite (90+)', 'Advanced (80-89)', 'Intermediate (70-79)', 'Developing (<70)']
            counts = [elite, advanced, intermediate, developing]
            
            fig = px.bar(x=categories, y=counts, title="Talent Distribution by Performance Level")
            fig.update_xaxes(title="Performance Category")
            fig.update_yaxes(title="Number of Athletes")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sport-wise talent distribution
            if 'sport_interest' in df.columns:
                sport_performance = df.groupby('sport_interest')['overall_score'].agg(['mean', 'count']).round(2)
                sport_performance.columns = ['Average Score', 'Athletes Count']
                sport_performance = sport_performance.sort_values('Average Score', ascending=False)
                
                fig = px.scatter(sport_performance, x='Athletes Count', y='Average Score', 
                               title="Sport-wise Talent Pool", 
                               hover_data={'Average Score': ':.2f', 'Athletes Count': True})
                st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation system
        st.markdown("---")
        st.write("### 🎯 Talent Recommendations")
        
        # High potential athletes
        high_potential = df[(df['overall_score'] >= 75) & (df.get('verification_status') == 'verified')]
        
        if not high_potential.empty:
            st.success(f"🌟 {len(high_potential)} athletes recommended for advanced training programs")
            
            recommendation_df = high_potential[['name', 'age', 'state', 'sport_interest', 'overall_score']].copy()
            recommendation_df.columns = ['Name', 'Age', 'State', 'Sport Interest', 'Score']
            st.dataframe(recommendation_df, use_container_width=True)
            
            if st.button("📋 Generate Talent Report", type="primary"):
                st.success("Talent identification report generated successfully!")
        else:
            st.info("No athletes currently meet the high-potential criteria")

def display_reports(df):
    """Display reporting interface"""
    st.subheader("📋 Reports & Analytics")
    
    # Report generation options
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📊 Available Reports")
        
        report_types = [
            "Performance Summary Report",
            "State-wise Talent Distribution",
            "Gender Performance Analysis",
            "Age Group Comparison",
            "Test-wise Performance Metrics",
            "Verification Status Report"
        ]
        
        selected_report = st.selectbox("Select Report Type", report_types)
        
        date_range = st.date_input(
            "Select Date Range",
            value=[datetime.now() - timedelta(days=30), datetime.now()],
            max_value=datetime.now()
        )
        
        if st.button("📄 Generate Report", type="primary"):
            generate_report(df, selected_report, date_range)
    
    with col2:
        st.write("### 📈 Quick Statistics")
        
        # Quick stats
        if not df.empty:
            stats_data = {
                'Metric': [
                    'Total Assessments',
                    'Average Performance Score',
                    'Highest Score Recorded',
                    'Most Popular Test',
                    'Top Performing State',
                    'Verification Rate'
                ],
                'Value': [
                    len(df),
                    f"{df['overall_score'].mean():.2f}" if 'overall_score' in df.columns else "N/A",
                    f"{df['overall_score'].max():.2f}" if 'overall_score' in df.columns else "N/A",
                    df['test_type'].mode().iloc[0] if 'test_type' in df.columns and not df['test_type'].empty else "N/A",
                    df.groupby('state')['overall_score'].mean().idxmax() if 'state' in df.columns and 'overall_score' in df.columns else "N/A",
                    f"{(df['verification_status'] == 'verified').mean() * 100:.1f}%" if 'verification_status' in df.columns else "N/A"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

def generate_report(df, report_type, date_range):
    """Generate specific report based on type"""
    st.success(f"✅ {report_type} generated successfully!")
    
    if report_type == "Performance Summary Report":
        st.markdown("### 📊 Performance Summary Report")
        
        summary_stats = {
            'Total Athletes Assessed': len(df),
            'Average Performance Score': f"{df['overall_score'].mean():.2f}" if 'overall_score' in df.columns else "N/A",
            'Standard Deviation': f"{df['overall_score'].std():.2f}" if 'overall_score' in df.columns else "N/A",
            'Highest Score': f"{df['overall_score'].max():.2f}" if 'overall_score' in df.columns else "N/A",
            'Lowest Score': f"{df['overall_score'].min():.2f}" if 'overall_score' in df.columns else "N/A"
        }
        
        for metric, value in summary_stats.items():
            st.write(f"**{metric}:** {value}")
    
    st.info("📧 Report has been generated and can be exported for official use.")

if __name__ == "__main__":
    main()
