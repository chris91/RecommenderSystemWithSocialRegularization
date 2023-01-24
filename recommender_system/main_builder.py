import pandas as pd

from recommender_system import db_controller


def read_ratings_from_csv():
    df = pd.read_csv('../datasets/movielens_ratings.csv', usecols=["userId", "movieId", "rating"])

    return df


def read_ratings_from_db():
    conn = db_controller.create_connection("../my.db")

    ratings = []
    cur = conn.cursor()
    cur.execute("SELECT user_id, movie_id, rating FROM ratings WHERE id <= 100000")

    rows = cur.fetchall()
    for row in rows:
        ratings.append(row)

    ratings_df = pd.DataFrame.from_records(ratings, columns=['user_id', 'movie_id', 'rating'])

    return ratings_df


def read_ratings_for_current_user_from_db(user_id):
    conn = db_controller.create_connection("../my.db")

    ratings = []
    cur = conn.cursor()
    cur.execute("SELECT user_id, movie_id, rating FROM ratings WHERE user_id = ?", (int(user_id),))

    rows = cur.fetchall()
    for row in rows:
        ratings.append(row)

    ratings_df = pd.DataFrame.from_records(ratings, columns=['user_id', 'movie_id', 'rating'])

    return ratings_df

def read_similarities_from_db():
    conn = db_controller.create_connection("../my.db")

    similarities = []
    cur = conn.cursor()
    cur.execute("SELECT from_movie_id, to_movie_id, similarity FROM movie_similarities")

    rows = cur.fetchall()
    for row in rows:
        similarities.append(row)

    similarities_df = pd.DataFrame.from_records(similarities, columns=['from_movie_id', 'to_movie_id', 'similarity'])

    return similarities_df


def read_similarities_from_db_where(user_rated_movies):
    conn = db_controller.create_connection("../my.db")

    movie_ids = user_rated_movies

    similarities = []
    cur = conn.cursor()

    for movie_id in movie_ids['movie_id']:
        cur.execute(
            "SELECT from_movie_id, to_movie_id, similarity FROM movie_similarities WHERE from_movie_id = ? AND to_movie_id != ?",
            (int(movie_id), int(movie_id),))

        rows = cur.fetchall()
        for row in rows:
            similarities.append(row)

    similarities_df = pd.DataFrame.from_records(similarities, columns=['from_movie_id', 'to_movie_id', 'similarity'])

    return similarities_df


def write_similarities_to_db(df):
    conn = db_controller.create_connection("../my.db")
    cur = conn.cursor()

    query_delete_similarities = "DELETE FROM movie_similarities"
    cur.execute(query_delete_similarities)

    query_insert_similarities = """ INSERT INTO movie_similarities(from_movie_id, to_movie_id, similarity) VALUES(?, ?, ?) """

    for index, row in df.iterrows():
        cur.execute(query_insert_similarities,
                    (int(row['from_movie_id']), int(row['to_movie_id']), float(row['similarity']),))

    conn.commit()


def get_friends_for_user(user_id, max_user):
    conn = db_controller.create_connection("../my.db")

    friends = []
    query_search_from_user = "SELECT to_user_id FROM user_edges WHERE from_user_id = ? AND to_user_id <= ?"

    query_search_to_user = "SELECT from_user_id FROM user_edges WHERE to_user_id = ? AND from_user_id <= ?"

    cur = conn.cursor()
    cur.execute(query_search_from_user, (int(user_id), (int(max_user)),))

    rows = cur.fetchall()
    for row in rows:
        friends.append(row)

    cur.execute(query_search_to_user, (int(user_id), (int(max_user)),))
    rows = cur.fetchall()
    for row in rows:
        friends.append(row)

    return friends


def write_user_similarities_to_db(df):
    conn = db_controller.create_connection("../my.db")

    query_insert_user_similarities = """ INSERT INTO user_similarities(from_user_id, to_user_id, similarity) VALUES(?, ?, ?) """

    cur = conn.cursor()
    for index, row in df.iterrows():
        cur.execute(query_insert_user_similarities,
                    (int(row['from_user_id']), int(row['to_user_id']), float(row['similarity']),))

    conn.commit()

