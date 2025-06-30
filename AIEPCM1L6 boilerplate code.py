import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init, Fore
import time
import sys


# Initialize colorama
init(autoreset=True)

# Load and preprocess the dataset
def load_data(file_path='imdb_top_1000.csv'):
    try:
        df = pd.read_csv(file_path)
        df['combined_features'] = df['Genre'].fillna('') + ' ' + df['Overview'].fillna('')
        return df
    except FileNotFoundError:
        print(Fore.RED + f"Error: The file '{file_path}' was not found.")
        exit()

movies_df = load_data()

# Vectorize the combined features and compute cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movies(genre=None, mood=None, rating=None, top_n=5 ):
    filtered_df=movies_df
    if genre:
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(genre, case=False, na=False)]
    if rating:
        filtered_df = filtered_df[filtered_df['IMDB_Rating'] >= rating]
    recommendations=[]
    for idx, row in filtered_df.iterrows():
        overview = row['Overview']
        if pd.isna(overview):
            continue
        polarity = TextBlob(overview).sentiment.polarity
        if (mood and ((TextBlob(mood).sentiment.polarity < 0 and polarity > 0) or polarity >= 0)) or not mood:
            recommendations.append((row['Series_Title'], polarity))
        if len(recommendations) == top_n:
            break

    return recommendations if recommendations else "No suitable movie recommendations found."
        
        
def handle_ai(name):
    print(Fore.BLUE + "\n🔍 Let's find the perfect movie for you!\n")
    # Show genres in a single line
    print(Fore.GREEN + "Available Genres: ", end="")
    print("The genres avalible are: 1.History")
    print(" 2.Action")
    print(" 3.Drama")
    print(" 4.Romance")
    print(" 5.Crime")
    print(" 6.Documentary")
    print(" 7.Fantasy")
    print(" 8.Sci-Fi")
    print(" 9.Family")
    print(" 10.Mystery")
    print(" 11.Adventure")
    
    genre= input("Enter your choice").strip()
    mood = input(Fore.YELLOW + "How do you feel today? (Describe your mood): ").strip()
    print(Fore.BLUE + "\nAnalyzing mood", end="", flush=True)
    polarity = TextBlob(mood).sentiment.polarity
    
    mood_desc = "positive 😊" if polarity > 0 else "negative 😞" if polarity < 0 else "neutral 😐"
    print(f"\n{Fore.GREEN}Your mood is {mood_desc} (Polarity: {polarity:.2f}).\n")
    rating_input = float(input(Fore.YELLOW + "Enter minimum IMDB rating (7.6-9.3) or '0'if you don't want to enter a rating: "))
    print(f"{Fore.BLUE}\nFinding movies for {name}", end=" ")
    recs = recommend_movies(genre=genre, mood=mood, rating=rating_input, top_n=5)
    print(recs)
   

   
# Main program 
if __name__=="__main__":
    print(Fore.BLUE + "🎥 Welcome to your Personal Movie Recommendation Assistant! 🎥\n")
    name = input(Fore.YELLOW + "What's your name? ").strip()

    print(f"\n{Fore.GREEN}Great to meet you, {name}!\n")
    handle_ai(name)
    
    
another=input(f"{Fore.MAGENTA} would you like another recommendation based on your preferences? (yes/no): ").strip().lower()  
if another == "yes":
    # Ask for preferences again or reuse previous ones as needed
    genre = input("Enter your preferred genre: ").strip()
    mood = input("How do you feel today? (Describe your mood): ").strip()
    rating_input = float(input("Enter minimum IMDB rating (7.6-9.3) or '0' if you don't want to enter a rating: "))
    rec = recommend_movies(genre=genre, mood=mood, rating=rating_input, top_n=1)
    print(rec)  
    
else:
    print(f"{Fore.GREEN}Thank you for using the Movie Reccommendation AI assistant {name}!")    