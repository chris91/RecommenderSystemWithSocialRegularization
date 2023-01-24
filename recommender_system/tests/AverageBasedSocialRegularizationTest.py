import unittest
from collections import defaultdict

import pandas as pd
from surprise import Reader, Dataset, KNNBasic, accuracy, SVD
from surprise.model_selection import train_test_split, cross_validate

from recommender_system.hybrid_with_social_regularization.AverageBasedSocialRegularization import \
    AverageBasedSocialRegularization

THE_MATRIX = "the_matrix"
PULP_FICTION = "pulp_fiction"
INTERSTELLAR = "interstellar"
THE_DEPARTED = "the_departed"
THE_USUAL_SUSPECTS = "the_usual_suspects"
THE_GODFATHER = "the_godfather"
PARASITE = "parasite"
JOKER = "joker"
ALIEN = "alien"
THE_BIG_LEBOWSKI = "the_big_lebowski"
SILICON_VALLEY = "silicon_valley"


class TestAverageBasedRecommender(unittest.TestCase):
    # Format of User-Item Ratings matrix: userId, movieName, rating (1-5)

    ratings = pd.DataFrame(
        [[1, THE_MATRIX, 4],
         [1, PULP_FICTION, 5],
         [1, THE_BIG_LEBOWSKI, 5],
         [1, THE_DEPARTED, 3],
         [1, PARASITE, 5],
         [1, JOKER, 5],
         [1, THE_USUAL_SUSPECTS, 5],
         [1, ALIEN, 2],
         [1, THE_GODFATHER, 3],
         [1, SILICON_VALLEY, 4],

         [2, THE_MATRIX, 5],
         [2, PULP_FICTION, 5],
         [2, THE_BIG_LEBOWSKI, 4],
         [2, PARASITE, 3],
         [2, THE_USUAL_SUSPECTS, 5],
         [2, INTERSTELLAR, 5],

         [3, THE_MATRIX, 4],
         [3, THE_BIG_LEBOWSKI, 5],
         [3, PARASITE, 4],
         [3, JOKER, 3],
         [3, INTERSTELLAR, 5],
         [3, SILICON_VALLEY, 3],

         [4, THE_MATRIX, 3],
         [4, PULP_FICTION, 3],
         [4, THE_BIG_LEBOWSKI, 4],
         [4, PARASITE, 2],
         [4, JOKER, 3],
         [4, ALIEN, 3],
         [4, INTERSTELLAR, 4],
         [4, SILICON_VALLEY, 3],

         [5, THE_MATRIX, 2],
         [5, THE_BIG_LEBOWSKI, 3],
         [5, THE_DEPARTED, 4],
         [5, PARASITE, 1],
         [5, JOKER, 5],
         [5, THE_USUAL_SUSPECTS, 3],
         [5, ALIEN, 1],
         [5, INTERSTELLAR, 3],
         [5, THE_GODFATHER, 5],

         [6, THE_MATRIX, 4],
         [6, PULP_FICTION, 3],
         [6, THE_BIG_LEBOWSKI, 5],
         [6, THE_DEPARTED, 4],
         [6, PARASITE, 2],
         [6, JOKER, 4],
         [6, THE_GODFATHER, 4],

         [7, THE_BIG_LEBOWSKI, 5],
         [7, PARASITE, 2],
         [7, THE_GODFATHER, 1],
         [7, ALIEN, 3],
         [7, INTERSTELLAR, 5],
         [7, SILICON_VALLEY, 2],

         [8, THE_MATRIX, 4],
         [8, PULP_FICTION, 3],
         [8, THE_BIG_LEBOWSKI, 3],
         [8, THE_DEPARTED, 2],
         [8, PARASITE, 4],
         [8, JOKER, 4],
         [8, ALIEN, 4],
         [8, INTERSTELLAR, 3],
         [8, THE_GODFATHER, 5],

         [9, PULP_FICTION, 3],
         [9, THE_BIG_LEBOWSKI, 5],
         [9, THE_DEPARTED, 4],
         [9, PARASITE, 2],
         [9, JOKER, 2],
         [9, THE_USUAL_SUSPECTS, 5],
         [9, INTERSTELLAR, 1],
         [9, THE_GODFATHER, 3],
         [9, SILICON_VALLEY, 5],

         [10, THE_BIG_LEBOWSKI, 3],
         [10, JOKER, 5],
         [10, THE_USUAL_SUSPECTS, 3],
         [10, INTERSTELLAR, 4],

         ], columns=['userID', 'itemID', 'rating'])

    edges = pd.DataFrame(
        [
            [1, 2],
            [1, 4],
            [1, 6],
            [1, 7],
            [1, 8],

            [2, 3],
            [2, 4],
            [2, 6],
            [2, 9],
            [2, 10],

            [3, 5],
            [3, 4],
            [3, 1],
            [3, 8],
            [3, 10],

            [4, 1],
            [4, 3],
            [4, 2],
            [4, 9],
            [4, 5],
        ], columns=['from_user_id', 'to_user_id'])

    def test_average_based_social_regularization(self):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings, reader)

        trainset, testset = train_test_split(data, test_size=.25)

        algoABSR = AverageBasedSocialRegularization(user_max=10)
        algoABSR.fit(trainset)

        predictions = algoABSR.test(testset)

        accuracy.rmse(predictions)
        accuracy.mae(predictions)

    def test_with_cross_validation(self):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings, reader)

        algo = AverageBasedSocialRegularization(user_max=10)

        cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
