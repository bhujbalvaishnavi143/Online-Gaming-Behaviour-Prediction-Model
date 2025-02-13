import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model (Replace 'model.pkl' with your actual model file)

model = joblib.load("random_forest_model.pkl")
scaler=joblib.load('scaler.pkl')

# Title
st.title("Gaming Behavior Engagement Level Prediction")

# User Inputs
st.sidebar.header("Enter Player Details")

def user_input():
    age = st.sidebar.slider("Age", 10, 60, 25)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    location = st.sidebar.selectbox("Location", ['Other', 'USA', 'Europe' ,'Asia'])
    game_genre = st.sidebar.selectbox("Game Genre", ["Action", "RPG", "Sports", "Strategy", "Simulation"])
    playtime_hours = st.sidebar.slider("Play Time (Hours)", 0, 100, 10)
    in_game_purchases = st.sidebar.selectbox("In-Game Purchases", [0,1])
    game_difficulty = st.sidebar.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])
    sessions_per_week = st.sidebar.slider("Sessions Per Week", 1, 50, 5)
    avg_session_duration = st.sidebar.slider("Avg Session Duration (Minutes)", 5, 300, 60)
    player_level = st.sidebar.slider("Player Level", 1, 100, 10)
    achievements_unlocked = st.sidebar.slider("Achievements Unlocked", 0, 50, 5)
    #engagement_level = st.sidebar.selectbox("Engagement Level", ["Low", "Medium", "High"])
    
    # Encoding categorical variables
    gender_map = {"Male": 0, "Female": 1}
    game_difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}
    location_map={'Other': 0, 'USA': 1, 'Europe': 2, 'Asia': 3}
    GameGenre_map= {'Strategy': 0, 'Sports': 1, 'Action': 2, 'RPG': 3, 'Simulation': 4}
    
    data = {
        "Age": age,  # Normalized
        "Gender": gender_map[gender],
        "Location": location_map[location],  # Placeholder, needs encoding
        "GameGenre": GameGenre_map[game_genre],  # Placeholder, needs encoding
        "PlayTimeHours": playtime_hours,  # Normalized
        "InGamePurchases": in_game_purchases,
        "GameDifficulty": game_difficulty_map[game_difficulty],
        "SessionsPerWeek": sessions_per_week,  # Normalized
        "AvgSessionDurationMinutes": avg_session_duration,  # Normalized
        "PlayerLevel": player_level ,  # Normalized
        "AchievementsUnlocked": achievements_unlocked,  # Normalized
        #"EngagementLevel": engagement_level_map[engagement_level]
    }
    return pd.DataFrame([data])

# Get user input
input_data = user_input()
numerical_columns = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])


# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0]==0:
        st.write("## Predicted Output:","Low")
    elif prediction[0]==1:
         st.write("## Predicted Output:","Medium")
    elif prediction[0]==2:
        st.write("## Predicted Output:","High")

      
         

st.write("### Sample Input Data:")
st.write(input_data)
