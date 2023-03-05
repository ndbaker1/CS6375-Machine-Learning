'''Collaborative Filtering Algorithm implementation and test suite'''

import math
from collections import defaultdict
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error


class CollaborativeFiltering:

    def __init__(self, train_x, train_y) -> None:

        self.movie_ids = set(movie_id for movie_id, _ in train_x)

        customer_votes = defaultdict(list)
        for i, (_, customer_id) in enumerate(train_x):
            customer_votes[customer_id].append(train_y[i])

        self.vote_database = {
            pair: rating
            for pair, rating in zip(train_x, train_y)
        }

        self.vote_means = {
            k: sum(v) / len(v)
            for k, v in customer_votes.items()
        }

        indices = ["movie_id", "customer_id"]

        self.customer_df = pd.DataFrame(
            train_x,
            columns=indices,
        ).groupby(indices).size().unstack(fill_value=0)

        # array of vectors for each customer representing their movie voting
        # needs to be transposed because customer_id is how we want to index the data in the sparse matrix
        self.customer_matrix = sparse.csr_matrix(
            self.customer_df.transpose().values)

        # train kNN model
        self.database_knn = NearestNeighbors(metric="cosine",
                                             algorithm="brute")
        self.database_knn.fit(self.customer_matrix)

    def predict(self, test_x, k=1) -> list:

        predictions = list()

        for movie_id, customer_id in test_x:
            customer_index = self.customer_df.columns.get_loc(customer_id)
            neighbor_sets = self.database_knn.kneighbors(
                self.customer_matrix[customer_index],
                return_distance=False,
            )

            neighbors = [self.customer_df.columns[i] for i in neighbor_sets[0]]

            prediction = self.vote_means[customer_id] + k * sum(
                self.correlation(customer_id, neighbor) * (self.vote_database[
                    (movie_id, neighbor)] - self.vote_means[neighbor])
                for neighbor in neighbors if
                (movie_id, neighbor) in self.vote_database)

            predictions.append(prediction)

        return predictions

    def correlation(self, c1, c2):

        def both_voted(movie_id):
            c1_entry, c2_entry = (movie_id, c1), (movie_id, c2)
            return c1_entry in self.vote_database and c2_entry in self.vote_database

        distances = [(
            self.vote_database[(movie_id, c1)] - self.vote_means[c1],
            self.vote_database[(movie_id, c2)] - self.vote_means[c2],
        ) for movie_id in self.movie_ids if both_voted(movie_id)]

        if len(distances) == 0:
            return 0

        c1_distances, c2_distances = zip(*distances)

        numer = sum(
            c1_distance * c2_distance
            for c1_distance, c2_distance in zip(c1_distances, c2_distances))

        denom = math.sqrt(
            sum(math.pow(dist, 2) for dist in c1_distances) *
            sum(math.pow(dist, 2) for dist in c2_distances))

        if denom <= 0:
            return 0

        return numer / denom


if __name__ == "__main__":

    train_x, train_y = [], []
    test_x, test_y = [], []

    with open("./netflix/TrainingRatings.txt") as training_ratings_file:
        for movie_id, customer_id, rating in (
                l.split(",")
                for l in training_ratings_file.read().splitlines()
                if len(l) > 0):

            train_x.append((movie_id, customer_id))
            train_y.append(float(rating))

    with open("./netflix/TestingRatings.txt") as testing_ratings_file:
        for movie_id, customer_id, rating in (
                l.split(",") for l in testing_ratings_file.read().splitlines()
                if len(l) > 0):

            test_x.append((movie_id, customer_id))
            test_y.append(float(rating))

    print("training...")
    collaborative_filter = CollaborativeFiltering(train_x, train_y)

    # compute and print evaluation metrics
    print("testing...")
    predictions = collaborative_filter.predict(test_x)

    mean_abs_error, root_mean_sqr_error = (
        mean_absolute_error(test_y, predictions),
        mean_squared_error(test_y, predictions, squared=False),
    )

    print(f"mean absolute error: {mean_abs_error}")
    print(f"root mean squared error: {root_mean_sqr_error}")
