import pandas as pd
from surprise import Dataset, Reader, SVD, dump

rating = pd.read_csv('rating.csv')


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)

trainset = data.build_full_trainset()

model = SVD()
model.fit(trainset)


dump.dump('recommendation_model', algo=model)


user_id = 1  # Replace with the user ID you want to recommend movies for

movie_ids = rating['movieId'].unique()

user_predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids]

user_predictions.sort(key=lambda x: x.est, reverse=True)
top_n = user_predictions[:10]

movie_titles = [rating[rating['movieId'] == prediction.iid]['title'].values[0] for prediction in top_n]


for i, movie_title in enumerate(movie_titles):
    print(f"Recommendation {i + 1}: {movie_title}")
