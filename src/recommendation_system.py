import pandas as pd
import numpy as np
import plotly.offline as plt
import plotly.graph_objs as go
plt.init_notebook_mode()
import os
import random
import math
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import theano
theano.config.compute_test_value = 'raise'
from multiprocessing import Pool, cpu_count
import uuid


DATA_DIR = "../data/"
SONGS_FILE = "songs.csv"
NFEATURE = 21  # Number of Features
S = 50  # Hyper Parameter
totReco = 0  # Number of total recommendation till now
startConstant = 5  # for low penalty in starting phase

# Read data
Songs = pd.read_csv(DATA_DIR + SONGS_FILE, index_col=0)

ratedSongs = set()

# User data structure
users = {}

def register_user():
    name = input("Enter your name: ")
    if name in users:
        print("User already exists. Please login.")
        return None
    user_id = str(uuid.uuid4())
    users[user_id] = {"name": name, "features": np.zeros(NFEATURE), "rated_songs": set()}
    print(f"User {name} registered successfully. Your user ID is {user_id}")
    return user_id

def login_user():
    user_id = input("Enter your user ID: ")
    if user_id not in users:
        print("Invalid user ID. Please register first.")
        return None
    print(f"Welcome back, {users[user_id]['name']}!")
    return user_id

def get_user_data(user_id):
    if user_id not in users:
        return None
    return users[user_id]

def compute_utility(user_features, song_features, epoch, s=S):
    """ Compute utility U based on user preferences and song preferences """
    user_features = user_features.copy()
    song_features = song_features.copy()
    dot = user_features.dot(song_features)
    ee = (1.0 - 1.0 * math.exp(-1.0 * epoch / s))
    res = dot * ee
    return res

def get_song_features(song):
    """ Feature of particular song """
    if isinstance(song, pd.Series):
        return song[-NFEATURE:]
    elif isinstance(song, pd.DataFrame):
        return get_song_features(pd.Series(song.loc[song.index[0]]))
    else:
        raise TypeError("{} should be a Series or DataFrame".format(song))

def get_song_genre(song):
    genres = []
    for genre in ["Pop", "Rock", "Country", "Folk", "Dance", "Grunge", "Love", "Metal", "Classic", "Funk", "Electric", "Acoustic", "Indie", "Jazz", "SoundTrack", "Rap"]:
        if song[genre] == 1:
            genres.append(genre)
    return genres

def best_recommendation(user_features, epoch, s):
    global Songs
    Songs = Songs.copy()
    """ Song with highest utility """
    utilities = np.zeros(Songs.shape[0])

    for i, (Title, song) in enumerate(Songs.iterrows()):
        song_features = get_song_features(song)
        utilities[i] = compute_utility(user_features, song_features, epoch - song.last_t, s)
    return Songs[Songs.index == Songs.index[utilities.argmax()]]

def all_recommendation(user_features):
    """ Top 10 songs with using exploration and exploitation """
    global Songs
    Songs = Songs.copy()
    i = 0
    recoSongs = []
    while i < 10:
        song = greedy_choice_no_t(user_features, totReco, S)
        recoSongs.append(song)
        Songs.loc[Songs.index.isin(song.index), 'last_t'] = totReco
        i += 1
    return recoSongs

def random_choice():
    """ Random songs which aren't been rated yet """
    global Songs
    Songs = Songs.copy()
    song = Songs.sample()
    while (song.index[0] in ratedSongs):
        song = Songs.sample()
    return song

def greedy_choice(user_features, epoch, s):
    """ greedy approach to the problem """
    global totReco
    epsilon = 1 / math.sqrt(epoch + 1)
    totReco = totReco + 1
    if random.random() > epsilon:  # choose the best
        return best_recommendation(user_features, epoch, s)
    else:
        return random_choice()

def greedy_choice_no_t(user_features, epoch, s, epsilon=0.3):
    """ greedy approach to the problem. After some iteration value of epsilon will be constant """
    global totReco
    totReco = totReco + 1
    if random.random() > epsilon:  # choose the best
        return best_recommendation(user_features, epoch, s)
    else:
        return random_choice()

def iterative_mean(old, new, t):
    """ Compute the new mean, Added startConstant for low penalty in starting phase """
    t += startConstant
    return ((t - 1) / t) * old + (1 / t) * new

def update_features(user_features, song_features, rating, t):
    return iterative_mean(user_features, song_features * rating, 1.0 * float(t) + 1.0)

def reinforcement_learning(user_id, s=200, N=5):
    global Songs
    Songs = Songs.copy()

    # Use user's features and rated songs
    user_data = get_user_data(user_id)
    user_features = user_data["features"]
    ratedSongs = user_data["rated_songs"]

    print("Select song features that you like")
    Features = ["1980s", "1990s", "2000s", "2010s", "2020s", "Pop", "Rock", "Country", "Folk", "Dance", "Grunge",
                "Love", "Metal", "Classic", "Funk", "Electric", "Acoustic", "Indie", "Jazz", "SoundTrack", "Rap"]
    for i in range(0, len(Features)):
        print(str(i + 1) + ". " + Features[i])
    choice = "y"
    likedFeat = set()
    while (choice.lower().strip() == "y"):
        num = input("Enter number associated with feature: ")  # Changed raw_input to input
        likedFeat.add(Features[int(num) - 1])
        choice = input("Do you want to add another feature? (y/n) ")  # Changed raw_input to input
    for i in range(0, len(Features)):
        if (Features[i] in likedFeat):
            user_features[i] = 1.0 / len(likedFeat)

    print("\n\nRate following " + str(N) + " songs. So that we can know your taste.\n")  # Changed print statement
    for t in range(N):
        if (t >= 10):
            recommendation = greedy_choice_no_t(user_features, t + 1, s, 0.3)
        else:
            recommendation = greedy_choice(user_features, t + 1, s)
        recommendation_features = get_song_features(recommendation)
        user_rating = input('How much do you like "' + recommendation.index[0] + '" (1-10): ')
        user_rating = int(user_rating)
        user_rating = 1.0 * user_rating / 10.0
        user_features = update_features(user_features, recommendation_features, user_rating, t)
        utility = compute_utility(user_features, recommendation_features, t, s)
        Songs.loc[Songs.index.isin(recommendation.index), 'last_t'] = t + 1
        ratedSongs.add(recommendation.index[0])
        user_data["features"] = user_features
        user_data["rated_songs"] = ratedSongs

    print("\n\nBased on your preferences, here are some recommendations for you:\n")  # Changed print statement
    for i, song in enumerate(all_recommendation(user_features)):
        print(f"{i+1}. {song.index[0]}")


def main():
    user_id = register_user()  # You can also use login_user() if the user already exists
    if user_id:
        reinforcement_learning(user_id)

if __name__ == "__main__":
    main()
