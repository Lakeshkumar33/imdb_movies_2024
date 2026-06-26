import pandas as pd
import numpy as np
import streamlit as st
import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("Movie Plot Recommendation")
st.write("Similar storyline")

try:
    tfidf = joblib.load(os.path.join(BASE_DIR,"tfidf_vectorizer.joblib"))
    X = joblib.load(os.path.join(BASE_DIR,"tfidf_matrix.joblib"))
    df = joblib.load(os.path.join(BASE_DIR,"movies_dataframe.joblib"))
        
except FileNotFoundError:
    st.error("Model matrices or Dataframe joblib files not found in your project")
    st.stop()

st.write("Enter the keywords or storyline concept to find the top 10 movies")

movie_1 = movie_2 = movie_3 = movie_4 = movie_5 = ""
movie_6 = movie_7 = movie_8 = movie_9 = movie_10 = ""

with st.form("movie_similarity_form"):

    user_input = st.text_input("Enter keywords or Movie Concept:", value = "fade celebrity take black market drug cell rep")

    if st.form_submit_button("Predict Similar Storylines"):

        user_vector = tfidf.transform([user_input]). toarray()

        scores = cosine_similarity(user_vector, X).flatten()

        top_10_indices = scores.argsort()[::-1][:10]

        movie_names_list = df.iloc[top_10_indices]["Movie Name"].tolist()
        storylines_list = df.iloc[top_10_indices]["Storyline"].tolist()
        scores_list = scores[top_10_indices].tolist()

       
        movie_1 = f"**{movie_names_list[0]}**\n Plot: *{storylines_list[0]}*"
        movie_2 = f"**{movie_names_list[1]}**\n Plot: *{storylines_list[1]}*"
        movie_3 = f"**{movie_names_list[2]}**\n Plot: *{storylines_list[2]}*"
        movie_4 = f"**{movie_names_list[3]}**\n Plot: *{storylines_list[3]}*"
        movie_5 = f"**{movie_names_list[4]}**\n Plot: *{storylines_list[4]}*"
        movie_6 = f"**{movie_names_list[5]}**\n Plot: *{storylines_list[5]}*"
        movie_7 = f"**{movie_names_list[6]}**\n Plot: *{storylines_list[6]}*"
        movie_8 = f"**{movie_names_list[7]}**\n Plot: *{storylines_list[7]}*"
        movie_9 = f"**{movie_names_list[8]}**\n Plot: *{storylines_list[8]}*"
        movie_10 = f"**{movie_names_list[9]}**\n Plot: *{storylines_list[9]}*"


    if movie_1 !="":
        st.success(f"""**Top 10 Similar Movies Found:**
        1. {movie_1}
        2. {movie_2}
        3. {movie_3}
        4. {movie_4}
        5. {movie_5}
        6. {movie_6}
        7. {movie_7}
        8. {movie_8}
        9. {movie_9}
        10. {movie_10}""")
    
st.divider()

st.write("Eucliden Distances Score")

with st.form("Eucliden distance_form"):

    user_input = st.text_input("Enter keywords or Movie Concept:", value = "fade celebrity take black market drug cell rep")

    if st.form_submit_button("Predict Eucliden Distances Score"):

        user_vector = tfidf.transform([user_input]). toarray()

        distances = euclidean_distances(user_vector, X).flatten()

        top_10_indices = distances.argsort()[:10]

        movie_names_list = df.iloc[top_10_indices]["Movie Name"].tolist()
        storylines_list = df.iloc[top_10_indices]["Storyline"].tolist()
        scores_list = distances[top_10_indices].tolist()

       
        movie_1 = f"{movie_names_list[0]} (Distance: {scores_list[0]:.4f})"
        movie_2 = f"{movie_names_list[1]} (Distance: {scores_list[1]:.4f})"
        movie_3 = f"{movie_names_list[2]} (Distance: {scores_list[2]:.4f})"
        movie_4 = f"{movie_names_list[3]} (Distance: {scores_list[3]:.4f})"
        movie_5 = f"{movie_names_list[4]} (Distance: {scores_list[4]:.4f})"
        movie_6 = f"{movie_names_list[5]} (Distance: {scores_list[5]:.4f})"
        movie_7 = f"{movie_names_list[6]} (Distance: {scores_list[6]:.4f})"
        movie_8 = f"{movie_names_list[7]} (Distance: {scores_list[7]:.4f})"
        movie_9 = f"{movie_names_list[8]} (Distance: {scores_list[8]:.4f})"
        movie_10 = f"{movie_names_list[9]} (Distance: {scores_list[9]:.4f})"

    if movie_1 !="":
        st.success(f"""**Top 10 Similar Movies Found:**
        1. {movie_1}
        2. {movie_2}
        3. {movie_3}
        4. {movie_4}
        5. {movie_5}
        6. {movie_6}
        7. {movie_7}
        8. {movie_8}
        9. {movie_9}
        10. {movie_10}""")


