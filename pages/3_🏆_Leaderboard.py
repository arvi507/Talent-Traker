import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.database import get_database
from datetime import datetime, timedelta

st.set_page_config(page_title="Leaderboard", page_icon="🏆", layout="wide")

def main():
    st.title("🏆 SAI Talent Leaderboard")
    st.markdown("Discover India's rising athletic talent across various fitness assessments")
    
    # Load data
    db = get_database()
    athletes_df = pd.DataFrame(db['athletes'])
    assessments_df = pd.DataFrame(db['assessments'])
    
    if assessments_df.empty:
        st.info("🏃‍♂️ Leaderboard will appear here once athletes complete their assessments")
        display_placeholder_leaderboard()
        return
    
    # Merge data
    if not athletes_df.empty:
        full_df = assessments_df.merge(athletes_df, left_on='athlete_id', right_on='id', how='left', suffixes=('', '_athlete'))
    else:
        full_df = assessments_df
    
    # Leaderboard filters
    st.sidebar.header("🎯 Leaderboard Filters")
    
    # Test type filter
    test_types = ["All Tests"] + list(full_df['test_type'].unique()) if 'test_type' in full_df.columns else ["All Tests"]
    selected_test = st.sidebar.selectbox("Filter by Test Type", test_types)
    
    # Gender filter
    genders = ["All Genders"] + list(full_df['gender'].unique()) if 'gender' in full_df.columns else ["All Genders"]
    selected_gender = st.sidebar.selectbox("Filter by Gender", genders)
    
    # Age group filter
    if 'age' in full_df.columns:
        age_groups = {
            "All Ages": (0, 100),
            "Under 16": (0, 15),
            "16-20": (16, 20),
            "21-25": (21, 25),
            "26-30": (26, 30),
            "Above 30": (31, 100)
        }
        selected_age_group = st.sidebar.selectbox("Filter by Age Group", list(age_groups.keys()))
        age_range = age_groups[selected_age_group]
    else:
        selected_age_group = "All Ages"
        age_range = (0, 100)
    
    # State filter
    states = ["All States"] + list(full_df['state'].unique()) if 'state' in full_df.columns else ["All States"]
    selected_state = st.sidebar.selectbox("Filter by State", states)
    
    # Apply filters
    filtered_df = full_df.copy()
    
    if selected_test != "All Tests":
        filtered_df = filtered_df[filtered_df['test_type'] == selected_test]
    
    if selected_gender != "All Genders":
        filtered_df = filtered_df[filtered_df['gender'] == selected_gender]
    
    if selected_age_group != "All Ages" and 'age' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
    
    if selected_state != "All States":
        filtered_df = filtered_df[filtered_df['state'] == selected_state]
    
    # Display leaderboards
    tab1, tab2, tab3, tab4 = st.tabs(["🥇 Overall Rankings", "🏃‍♂️ Test-wise Rankings", "📊 Statistics", "🌟 Achievements"])
    
    with tab1:
        display_overall_rankings(filtered_df)
    
    with tab2:
        display_test_wise_rankings(filtered_df)
    
    with tab3:
        display_statistics(filtered_df)
    
    with tab4:
        display_achievements(filtered_df)

def display_overall_rankings(df):
    """Display overall performance rankings"""
    st.subheader("🥇 Top Performers - Overall Rankings")
    
    if df.empty or 'overall_score' not in df.columns:
        st.info("No performance data available for the selected filters")
        return
    
    # Top 50 performers
    top_performers = df.nlargest(50, 'overall_score').reset_index(drop=True)
    
    # Create podium for top 3
    if len(top_performers) >= 3:
        st.markdown("### 🏆 Champions Podium")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:  # Gold - 1st place (center)
            gold_athlete = top_performers.iloc[0]
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #FFD700, #FFA500); border-radius: 10px; margin: 10px;'>
                <h2>🥇 1st Place</h2>
                <h3>{gold_athlete.get('name', 'Unknown')}</h3>
                <h4>Score: {gold_athlete.get('overall_score', 0):.1f}/100</h4>
                <p>{gold_athlete.get('state', 'Unknown')} | {gold_athlete.get('test_type', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col1:  # Silver - 2nd place (left)
            silver_athlete = top_performers.iloc[1]
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #C0C0C0, #A9A9A9); border-radius: 10px; margin: 10px 10px 30px 10px;'>
                <h3>🥈 2nd Place</h3>
                <h4>{silver_athlete.get('name', 'Unknown')}</h4>
                <p>Score: {silver_athlete.get('overall_score', 0):.1f}/100</p>
                <small>{silver_athlete.get('state', 'Unknown')}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:  # Bronze - 3rd place (right)
            bronze_athlete = top_performers.iloc[2]
            st.markdown(f"""
            <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #CD7F32, #A0522D); border-radius: 10px; margin: 10px 10px 30px 10px;'>
                <h3>🥉 3rd Place</h3>
                <h4>{bronze_athlete.get('name', 'Unknown')}</h4>
                <p>Score: {bronze_athlete.get('overall_score', 0):.1f}/100</p>
                <small>{bronze_athlete.get('state', 'Unknown')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Full rankings table
    st.markdown("### 📋 Complete Rankings")
    
    # Prepare rankings data
    rankings_df = top_performers.copy()
    rankings_df['Rank'] = range(1, len(rankings_df) + 1)
    
    # Select and rename columns for display
    display_columns = ['Rank', 'name', 'age', 'gender', 'state', 'test_type', 'overall_score']
    available_columns = [col for col in display_columns if col in rankings_df.columns]
    rankings_display = rankings_df[available_columns].copy()
    
    # Rename columns for better display
    column_names = {
        'name': 'Name',
        'age': 'Age', 
        'gender': 'Gender',
        'state': 'State',
        'test_type': 'Test Type',
        'overall_score': 'Score'
    }
    rankings_display = rankings_display.rename(columns=column_names)
    
    # Style the dataframe
    def highlight_top_performers(row):
        if row.name == 0:  # 1st place
            return ['background-color: #FFD700; font-weight: bold'] * len(row)
        elif row.name == 1:  # 2nd place
            return ['background-color: #C0C0C0; font-weight: bold'] * len(row)
        elif row.name == 2:  # 3rd place
            return ['background-color: #CD7F32; font-weight: bold'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = rankings_display.style.apply(highlight_top_performers, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Performance distribution chart
    st.markdown("### 📊 Score Distribution")
    fig = px.histogram(top_performers, x='overall_score', nbins=15, 
                      title=f"Performance Score Distribution (Top {len(top_performers)} Athletes)")
    fig.update_xaxes(title="Performance Score")
    fig.update_yaxes(title="Number of Athletes")
    st.plotly_chart(fig, use_container_width=True)

def display_test_wise_rankings(df):
    """Display rankings for each test type"""
    st.subheader("🏃‍♂️ Test-wise Performance Rankings")
    
    if df.empty or 'test_type' not in df.columns:
        st.info("No test-specific data available")
        return
    
    test_types = df['test_type'].unique()
    
    for test_type in test_types:
        with st.expander(f"🏆 {test_type} Rankings"):
            test_df = df[df['test_type'] == test_type].nlargest(20, 'overall_score')
            
            if not test_df.empty:
                # Test-specific podium
                if len(test_df) >= 3:
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (col, medal, place) in enumerate([(col2, "🥇", "1st"), (col1, "🥈", "2nd"), (col3, "🥉", "3rd")]):
                        if i < len(test_df):
                            athlete = test_df.iloc[i]
                            with col:
                                st.markdown(f"""
                                **{medal} {place} Place**  
                                **{athlete.get('name', 'Unknown')}**  
                                Score: {athlete.get('overall_score', 0):.1f}/100  
                                {athlete.get('state', 'Unknown')} | Age {athlete.get('age', 'Unknown')}
                                """)
                
                # Full test rankings
                test_rankings = test_df.copy()
                test_rankings['Rank'] = range(1, len(test_rankings) + 1)
                
                display_cols = ['Rank', 'name', 'age', 'gender', 'state', 'overall_score']
                available_cols = [col for col in display_cols if col in test_rankings.columns]
                test_display = test_rankings[available_cols]
                
                col_names = {'name': 'Name', 'age': 'Age', 'gender': 'Gender', 'state': 'State', 'overall_score': 'Score'}
                test_display = test_display.rename(columns=col_names)
                
                st.dataframe(test_display, use_container_width=True, hide_index=True)

def display_statistics(df):
    """Display comprehensive statistics"""
    st.subheader("📊 Performance Statistics & Analytics")
    
    if df.empty:
        st.info("No statistical data available")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_athletes = len(df['name'].unique()) if 'name' in df.columns else len(df)
        st.metric("Total Athletes", total_athletes)
    
    with col2:
        avg_score = df['overall_score'].mean() if 'overall_score' in df.columns else 0
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with col3:
        highest_score = df['overall_score'].max() if 'overall_score' in df.columns else 0
        st.metric("Highest Score", f"{highest_score:.1f}")
    
    with col4:
        score_std = df['overall_score'].std() if 'overall_score' in df.columns else 0
        st.metric("Score Std Dev", f"{score_std:.1f}")
    
    # Detailed analytics
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance by gender
        if 'gender' in df.columns and 'overall_score' in df.columns:
            gender_stats = df.groupby('gender')['overall_score'].agg(['mean', 'count', 'std']).round(2)
            gender_stats.columns = ['Average Score', 'Count', 'Std Dev']
            
            st.markdown("### 👥 Performance by Gender")
            st.dataframe(gender_stats)
            
            fig = px.box(df, x='gender', y='overall_score', title="Score Distribution by Gender")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance by age group
        if 'age' in df.columns and 'overall_score' in df.columns:
            df_copy = df.copy()
            df_copy['age_group'] = pd.cut(df_copy['age'], bins=[0, 16, 20, 25, 30, 100], 
                                        labels=['<16', '16-20', '21-25', '26-30', '30+'])
            
            age_stats = df_copy.groupby('age_group')['overall_score'].agg(['mean', 'count']).round(2)
            age_stats.columns = ['Average Score', 'Count']
            
            st.markdown("### 📅 Performance by Age Group")
            st.dataframe(age_stats)
            
            fig = px.bar(age_stats.reset_index(), x='age_group', y='Average Score', 
                        title="Average Score by Age Group")
            st.plotly_chart(fig, use_container_width=True)
    
    # State-wise performance
    if 'state' in df.columns and 'overall_score' in df.columns:
        st.markdown("### 🗺️ State-wise Performance Analysis")
        
        state_stats = df.groupby('state').agg({
            'overall_score': ['mean', 'count', 'max'],
            'name': 'nunique'
        }).round(2)
        
        state_stats.columns = ['Avg Score', 'Total Tests', 'Highest Score', 'Unique Athletes']
        state_stats = state_stats.sort_values('Avg Score', ascending=False)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(state_stats)
        
        with col2:
            fig = px.scatter(state_stats.reset_index(), 
                           x='Unique Athletes', y='Avg Score', 
                           size='Total Tests',
                           hover_name='state',
                           title="State Performance Overview")
            st.plotly_chart(fig, use_container_width=True)

def display_achievements(df):
    """Display achievements and badges"""
    st.subheader("🌟 Achievements & Recognition")
    
    if df.empty or 'overall_score' not in df.columns:
        st.info("Achievements will be displayed once performance data is available")
        return
    
    # Achievement categories
    achievements = {
        "🏆 Elite Performers (90+ Score)": df[df['overall_score'] >= 90],
        "⭐ High Achievers (80-89 Score)": df[(df['overall_score'] >= 80) & (df['overall_score'] < 90)],
        "💪 Strong Performers (70-79 Score)": df[(df['overall_score'] >= 70) & (df['overall_score'] < 80)],
        "🌟 Perfect Score Achievers": df[df['overall_score'] >= 99],
        "🔥 State Champions": df.groupby('state')['overall_score'].idxmax() if 'state' in df.columns else pd.Series(dtype=int)
    }
    
    # Display achievement counts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        elite_count = len(achievements["🏆 Elite Performers (90+ Score)"])
        st.metric("Elite Performers", elite_count, delta=f"{elite_count/len(df)*100:.1f}% of total" if len(df) > 0 else "0%")
    
    with col2:
        high_count = len(achievements["⭐ High Achievers (80-89 Score)"])
        st.metric("High Achievers", high_count, delta=f"{high_count/len(df)*100:.1f}% of total" if len(df) > 0 else "0%")
    
    with col3:
        perfect_count = len(achievements["🌟 Perfect Score Achievers"])
        st.metric("Perfect Scores", perfect_count, delta="Exceptional!" if perfect_count > 0 else "None yet")
    
    # Achievement details
    for achievement_name, achievement_df in achievements.items():
        if len(achievement_df) > 0:
            with st.expander(f"{achievement_name} ({len(achievement_df)} athletes)"):
                if achievement_name == "🔥 State Champions":
                    # Special handling for state champions
                    if isinstance(achievement_df, pd.Series) and not achievement_df.empty:
                        champions_df = df.loc[achievement_df]
                        display_cols = ['name', 'state', 'test_type', 'overall_score']
                        available_cols = [col for col in display_cols if col in champions_df.columns]
                        if available_cols:
                            champions_display = champions_df[available_cols]
                            champions_display.columns = ['Name', 'State', 'Test Type', 'Score']
                            st.dataframe(champions_display, use_container_width=True, hide_index=True)
                else:
                    # Regular achievement display
                    display_cols = ['name', 'age', 'gender', 'state', 'test_type', 'overall_score']
                    available_cols = [col for col in display_cols if col in achievement_df.columns]
                    if available_cols:
                        achievement_display = achievement_df[available_cols].copy()
                        col_names = {
                            'name': 'Name', 'age': 'Age', 'gender': 'Gender',
                            'state': 'State', 'test_type': 'Test Type', 'overall_score': 'Score'
                        }
                        achievement_display = achievement_display.rename(columns=col_names)
                        st.dataframe(achievement_display, use_container_width=True, hide_index=True)
    
    # Special recognitions
    st.markdown("### 🎖️ Special Recognitions")
    
    if 'state' in df.columns:
        # Most improved state (placeholder logic)
        top_state = df.groupby('state')['overall_score'].mean().idxmax()
        st.success(f"🏅 **Top Performing State:** {top_state}")
    
    if 'test_type' in df.columns:
        # Most popular test
        popular_test = df['test_type'].mode().iloc[0]
        test_count = df['test_type'].value_counts().iloc[0]
        st.info(f"🎯 **Most Popular Test:** {popular_test} ({test_count} assessments)")

def display_placeholder_leaderboard():
    """Display placeholder leaderboard when no data is available"""
    st.markdown("### 🏃‍♂️ Sample Leaderboard Preview")
    
    st.info("This is how the leaderboard will look once athletes start completing assessments!")
    
    # Sample data for demonstration
    sample_data = {
        'Rank': [1, 2, 3, 4, 5],
        'Name': ['Athlete 1', 'Athlete 2', 'Athlete 3', 'Athlete 4', 'Athlete 5'],
        'State': ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Punjab', 'Kerala'],
        'Test Type': ['Vertical Jump', 'Sprint', 'Sit-ups', 'Push-ups', 'Flexibility'],
        'Score': [95.2, 92.8, 91.5, 89.3, 87.6]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Coming Soon Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Real-time Rankings:**
        - Live updates as assessments are completed
        - Filter by test type, gender, age group
        - State-wise and national rankings
        """)
    
    with col2:
        st.markdown("""
        **Achievement System:**
        - Performance badges and certificates
        - Recognition for top performers
        - Progress tracking and milestones
        """)

if __name__ == "__main__":
    main()
