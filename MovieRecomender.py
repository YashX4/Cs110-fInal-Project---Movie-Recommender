import csv
import os
import math

#folder adress and file adress 
DataDirectory = os.path.join(os.path.dirname(__file__), "ml-32m")
MoviesFile= DataDirectory + "/movies.csv"
RatingFile= DataDirectory + "/ratings.csv"

#Numbers of ratings laoded,
SAMPLE_N = 2000000 # set None to load full 32M, I dont want to use to many resources unnessiarliy, realisicly a sample size of 2M or so should be accurate for most users

#Loads movies from movie file into a dictionary that has id and title.
def load_movies(path):
    movies = {}
    #opens file with name path(movies file), ensures it can read non expected charecters, removes blank lines with newline = " ", and also uses utf-8 to read window paths. 
    with open(path, newline='', encoding='utf-8') as f:
        #Reads the CSV file row by row, giving each row as a dictionary where the keys are the column names
        reader = csv.DictReader(f)
        for row in reader:
            movies[row['movieId']] = row['title']
    return movies

# Loads the N number of smapled ratings
def load_ratings(path, sample_n=None):
    ratings = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if sample_n is not None and i >= sample_n:
                break
            try:
                ratings.append((row['userId'], row['movieId'], float(row['rating'])))
            except ValueError:
                continue
    return ratings #returns list of tuples with this structure (userId, movieId, rating)

# loads your file into a structure to be comparable to other users in the dataabase
def InputFile(adress, movies_map=None):
    InputedRatings = {}

    # rows we can't or couldn't read due to misinput or malformed data
    seen = 0
    skipped = 0

    # candidate column names for movie id and rating
    movie_id_candidates = {'movie_id', 'movieid'}
    rating_candidates = {'rating'}

    with open(adress, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # find actual header names present in this file for movie and rating
        header_fields = reader.fieldnames or []
        lowered = {h.lower(): h for h in header_fields}

        # find the movie column
        movie_col = None
        for cand in movie_id_candidates:
            if cand in lowered:
                movie_col = lowered[cand]
                break

        # find the rating column
        rating_col = None
        for cand in rating_candidates:
            if cand in lowered:
                rating_col = lowered[cand]
                break

        if movie_col is None or rating_col is None:
            raise ValueError("Could not find movie id or rating column. Found headers: {"+header_fields+"}")

        # read rows
        for row in reader:
            seen += 1
            try:
                raw_mid = row[movie_col].strip()
                # normalize movie id to string (MovieLens movie ids are integers but your user_ratings keys are strings)
                mID = str(int(raw_mid))
                r = float(row[rating_col])
            except Exception:
                skipped += 1
                continue

            InputedRatings[mID] = r

    #Warns about movies not present in MovieLens
    if movies_map is not None:
        missing = [m for m in InputedRatings.keys() if m not in movies_map]
        if missing:
            print(f"Warning: {len(missing)} of your {len(InputedRatings)} rated movie IDs were not found in the MovieLens movies map. "
                  "They will be ignored in recommendations if not present in the dataset.")
            for m in missing:
                InputedRatings.pop(m, None)

    print(f"Loaded personal ratings: {len(InputedRatings)} items (skipped {skipped} malformed rows out of {seen}).")
    return InputedRatings


#This is the formula that returns the simlarity between two users ratings. Returns a number between 0 and 1 indicating similarity. 
# I am treating each rating as a vector using this function and performing a cosine simiarity formula to derive a normalized value for how simlar any two users are 
def cosine_similarity(ratings1, ratings2):
    # ratings1 and ratings2 are dictionaries {movieId: rating}
    #Gets the movies rated by both users
    common_movies = set(ratings1.keys()) & set(ratings2.keys())
    if not common_movies:
        return 0  # no common films between users 
    
    dot_product = 0    # sum of (r1 * r2)
    sum1 = 0           # sum of squares of ratings1
    sum2 = 0           # sum of squares of ratings2
    
    for movie in common_movies:
        r1 = ratings1[movie]
        r2 = ratings2[movie]
        
        dot_product += r1 * r2
        sum1 += r1 ** 2
        sum2 += r2 ** 2
    
    # Calcs vector leaghts
    norm1 = math.sqrt(sum1)
    norm2 = math.sqrt(sum2)
    
    # Check for zero length vectors to avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0
    
    #Compute cosine similarity
    similarity = dot_product / (norm1 * norm2)
    
    return similarity

#Returns the top N users most similar to personal_ratings.
def top_similar_users(personal_ratings, user_ratings, topN=5):
    similarities = []
    for user, ratings in user_ratings.items():
        sim = cosine_similarity(personal_ratings, ratings)
        similarities.append((user, sim))

    def get_similarity(item):
        return item[1] 
    
    # sort by similarity descending
    similaritiesSorted = sorted(similarities, key=get_similarity, reverse=True)
    return similaritiesSorted[:topN]

def recommend_movies(personal_ratings, user_ratings, movies, top_n_users=5, top_n_movies=10):
    similar_users = top_similar_users(personal_ratings, user_ratings, top_n_users)
    
    # weighted scores for each movie
    scores = {}
    for user, sim in similar_users:
        for movie, rating in user_ratings[user].items():
            if movie not in personal_ratings:  # only consider movies you haven't rated
                if movie not in scores:
                    scores[movie] = 0
                scores[movie] += sim * rating  # weight by similarity
    
    def get_score(item):
        return item[1] 
    # sort by score
    recommended = sorted(scores.items(), key=get_score, reverse=True)
    
    # return top movie titles
    return [(movies[movie_id], score) for movie_id, score in recommended[:top_n_movies]]

def save_recommendations(recommendations, outputFile="recommendations.csv"):
    with open(outputFile, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Movie Title", "Score"])
        for i, (title, score) in enumerate(recommendations, start=1):
            writer.writerow([i, title, round(score, 2)])
    print("\nRecommendations saved to {"+outputFile+"}")


if __name__ == "__main__":
    # existence checks
    if not os.path.exists(MoviesFile):
        raise SystemExit(f"Missing {MoviesFile} — edit DataDirectory if necessary.")
    if not os.path.exists(RatingFile):
        raise SystemExit(f"Missing {RatingFile} — edit DataDirectory if necessary.")

    print("Loading movies...")
    movies = load_movies(MoviesFile)
    print("Loaded {"+str(len(movies))+"} movies. Example (first 5):")
    for i, (mID, title) in enumerate(movies.items()):
        print("{"+mID+"} -> {"+title+"}")
        if i == 5:
            break

    print("\nLoading ratings...")
    ratings = load_ratings(RatingFile, sample_n=SAMPLE_N)
    print("Loaded " + str(len(ratings)) + " ratings (sample_n=" + str(SAMPLE_N) + "). Example rows (first 10):")
    for r in ratings[:10]:
        print("  ", r)

    #combining the ratings and movies into user rating dictiorary that makes it easy to perform compute similarity
    user_ratings = {}
    #Loop through all rating tuples
    for user_id, movie_id, rating in ratings:
        if user_id not in user_ratings:
            user_ratings[user_id] = {}  # first time we see this user, create empty dict for this user
        user_ratings[user_id][movie_id] = rating #formats into 'user_id': {'Movie_id': rating,...},

    
    # quick stats
    user_set = set()
    movie_set = set()
    for u, m, _ in ratings:
        user_set.add(u)
        movie_set.add(m)
    print(f"\nUnique users in loaded sample: {len(user_set)}")
    print(f"Unique movies in loaded sample: {len(movie_set)}")

    personal_file = input("Enter the path to your personal ratings CSV: ").strip().strip('"')
    
    if not os.path.exists(personal_file):
        raise SystemExit("Personal ratings file not found. Exiting.")
    
    personal_ratings = InputFile(personal_file)
    print(f"Loaded your ratings for {len(personal_ratings)} movies.")
    
    recommendations = recommend_movies(personal_ratings, user_ratings, movies,top_n_users=5, top_n_movies=10)
    
    # Print recommendations
    print("\nTop movie recommendations for you:")
    for i, (title, score) in enumerate(recommendations, start=1):
         print(str(i) + ". " + title + " (score: " + str(round(score, 2)) + ")")
    
    # Save recommendations to CSV
    save_recommendations(recommendations)