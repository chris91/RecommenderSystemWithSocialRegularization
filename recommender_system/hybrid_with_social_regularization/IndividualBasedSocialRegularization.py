from time import sleep

import numpy as np
from surprise import AlgoBase, KNNBasic
from surprise.utils import get_rng

from recommender_system.main_builder import get_friends_for_user


class IndividualBasedSocialRegularization(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, init_mean=0,
                 init_std_dev=.1, learning_rate=.005, regularization=.02,
                 beta_reg=.001, user_max=757,
                 random_state=None, verbose=False):

        self.friends = {}
        self.social_difference = None
        self.social_regularization = None
        self.beta_reg = beta_reg
        self.user_max = user_max
        self.user_similarities = None
        self.user_regularization = None
        self.global_mean = None

        self.item_factors = None
        self.user_factors = None
        self.item_bias = None
        self.user_bias = None

        self.n_factors = n_factors
        self.n_epochs = n_epochs

        self.init_mean = init_mean
        self.init_std_dev = init_std_dev

        self.learning_rate = learning_rate
        self.regularization = regularization

        self.random_state = random_state
        self.verbose = verbose

        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        # call KNNBasic for User to create user_similarities
        sim_options = {'name': 'cosine',
                       'user_based': True
                       }
        algoKNN = KNNBasic(sim_options=sim_options, verbose=True)
        algoKNN.fit(trainset)
        simsMatrix = algoKNN.compute_similarities()
        self.user_similarities = simsMatrix

        rng = get_rng(self.random_state)

        self.user_bias = np.zeros(trainset.n_users, np.double)
        self.item_bias = np.zeros(trainset.n_items, np.double)
        self.user_factors = rng.normal(self.init_mean, self.init_std_dev,
                                       (trainset.n_users, self.n_factors))
        self.item_factors = rng.normal(self.init_mean, self.init_std_dev,
                                       (trainset.n_items, self.n_factors))
        self.user_regularization = np.zeros((trainset.n_users, self.n_factors), np.double)
        self.global_mean = self.trainset.global_mean

        self.social_difference = np.zeros((trainset.n_users, self.n_factors), np.double)
        self.social_regularization = np.zeros(trainset.n_users, np.double)

        for user in trainset.all_users():
            user_raw = trainset.to_raw_uid(user)

            # reading friends of user from db
            friends = get_friends_for_user(user_raw, self.user_max)

            friends_inner = []
            for friend in friends:
                friend_inner = trainset.to_inner_uid(friend[0])
                friends_inner.append(friend_inner)

            self.friends[user] = friends_inner

        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            for user in trainset.all_users():
                sum_of_friends = 0
                temp_matrix = np.zeros((trainset.n_users, self.n_factors), np.double)
                friends_similarities = np.zeros((trainset.n_users, self.n_factors), np.double)
                for friend in self.friends[user]:
                    if friend != user:
                        for factor in range(self.n_factors):
                            temp_matrix[friend][factor] = self.user_factors[user][factor] - self.user_factors[friend][
                                factor]

                        friends_similarities[friend] = self.user_similarities[user][friend] * temp_matrix[friend]

                        sum_of_squares = 0
                        for factor in range(self.n_factors):
                            sum_of_squares += pow(temp_matrix[friend][factor], 2)
                        sum_of_friends += self.user_similarities[user][friend] * sum_of_squares

                self.social_difference[user] = np.sum(friends_similarities, axis=0)
                self.social_regularization[user] = self.beta_reg * sum_of_friends

            for user, item, rating in trainset.all_ratings():

                # compute current error
                dot = 0
                for factor in range(self.n_factors):
                    dot += self.item_factors[item, factor] * self.user_factors[user, factor]

                err = rating - (self.global_mean + self.user_bias[user] + self.item_bias[item] + dot
                                + self.social_regularization[user])

                # update biases
                self.user_bias[user] += self.learning_rate * (err - self.regularization * self.user_bias[user])
                self.item_bias[item] += self.learning_rate * (err - self.regularization * self.item_bias[item])

                # update social regularization
                self.social_regularization[user] = self.learning_rate * (
                        err - self.beta_reg * self.social_regularization[user])

                # update factors
                for factor in range(self.n_factors):
                    user_factor = self.user_factors[user, factor]
                    item_factor = self.item_factors[item, factor]
                    self.user_factors[user, factor] += self.learning_rate * (
                            err * item_factor - self.regularization * user_factor
                            - self.beta_reg * self.social_difference[user][factor])
                    self.item_factors[item, factor] += self.learning_rate * (
                            err * user_factor - self.regularization * item_factor)

    def estimate(self, user, item):

        known_user = self.trainset.knows_user(user)
        known_item = self.trainset.knows_item(item)

        est = self.trainset.global_mean

        if known_user:
            est += self.user_bias[user]

        if known_item:
            est += self.item_bias[item]

        if known_user and known_item:
            est += np.dot(self.item_factors[item], self.user_factors[user])
            est += self.social_regularization[user]

        return est
