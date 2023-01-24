from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate, KFold

from recommender_system.main_builder import read_ratings_from_csv, read_ratings_from_db


def recommender(option=1):
    reader = Reader(rating_scale=(1, 5))
    # ratings = read_ratings_from_csv()
    ratings = read_ratings_from_db()
    data = Dataset.load_from_df(ratings, reader)

    if option == 1:
        trainset, testset = train_test_split(data, test_size=.25)

        algo = SVD(verbose=True)
        algo.fit(trainset)

        predictions = algo.test(testset)
        accuracy.rmse(predictions)
        accuracy.mae(predictions)

    else:
        # define a cross-validation iterator
        kf = KFold(n_splits=5)
        # kf = LeaveOneOut(n_splits=5)
        # kf = RepeatedKFold(n_splits=5, n_repeats=10)

        algo = SVD(verbose=True)

        for trainset, testset in kf.split(data):
            # train and test algorithm.
            algo.fit(trainset)
            predictions = algo.test(testset)

            # Compute and print Root Mean Squared Error
            accuracy.rmse(predictions, verbose=True)
            accuracy.mae(predictions, verbose=True)

if __name__ == '__main__':
    print("Singular Value Decomposition Recommender System starting...")
    recommender(2)
