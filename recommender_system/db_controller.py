import sqlite3
from sqlite3 import Error

import numpy as np
import pandas as pd

sql_create_users_table = """ CREATE TABLE IF NOT EXISTS users (
                                        user_id integer PRIMARY KEY
                                ); """

sql_create_movies_table = """ CREATE TABLE IF NOT EXISTS movies (
                                        movie_id integer PRIMARY KEY
                                ); """

sql_create_ratings_table = """ CREATE TABLE IF NOT EXISTS ratings (
                                        id integer PRIMARY KEY AUTOINCREMENT,
                                        user_id integer NOT NULL,
                                        movie_id integer NOT NULL,
                                        rating decimal,
                                        FOREIGN KEY (user_id) REFERENCES users (user_id),
                                        FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
                                ); """

sql_create_user_edges_table = """ CREATE TABLE IF NOT EXISTS user_edges (
                                        id integer PRIMARY KEY AUTOINCREMENT,
                                        from_user_id integer NOT NULL,
                                        to_user_id integer NOT NULL,
                                        FOREIGN KEY (from_user_id) REFERENCES users (user_id),
                                        FOREIGN KEY (to_user_id) REFERENCES users (user_id)
                                ); """

sql_create_movie_similarities_table = """ CREATE TABLE IF NOT EXISTS movie_similarities (
                                        id integer PRIMARY KEY AUTOINCREMENT,
                                        from_movie_id integer NOT NULL,
                                        to_movie_id integer NOT NULL,
                                        similarity decimal,
                                        FOREIGN KEY (from_movie_id) REFERENCES movies (movie_id),
                                        FOREIGN KEY (to_movie_id) REFERENCES movies (movie_id)
                                ); """

sql_create_user_similarities_table = """ CREATE TABLE IF NOT EXISTS user_similarities (
                                        id integer PRIMARY KEY AUTOINCREMENT,
                                        from_user_id integer NOT NULL,
                                        to_user_id integer NOT NULL,
                                        similarity decimal,
                                        FOREIGN KEY (from_user_id) REFERENCES users (user_id),
                                        FOREIGN KEY (to_user_id) REFERENCES users (user_id)
                                ); """


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        # print("Connected to:\t", db_file)
        return conn
    except Error as e:
        print(e)
    return conn


def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def read_from_csv_to_dataframe(filename):
    df = pd.read_csv(filename)
    print(df)

    return df


def initialize_db(conn):
    create_table(conn, sql_create_users_table)
    create_table(conn, sql_create_movies_table)
    create_table(conn, sql_create_ratings_table)
    create_table(conn, sql_create_movie_similarities_table)
    create_table(conn, sql_create_user_similarities_table)

    cur = conn.cursor()

    query_insert_movies = """ INSERT INTO movies(movie_id) VALUES(?) """

    movies_df = pd.read_csv("./datasets/movielens_movies.csv", usecols=["movieId"])

    for index, row in movies_df.iterrows():
        cur.execute(query_insert_movies, (int(row['movieId']),))

    query_insert_users = """ INSERT INTO users(user_id) VALUES(?) """
    query_insert_ratings = """ INSERT INTO ratings(user_id, movie_id, rating) VALUES(?, ?, ?) """

    user_ids = []
    ratings_df = pd.read_csv("./datasets/movielens_ratings.csv", usecols=["userId", "movieId", "rating"])
    for index, row in ratings_df.iterrows():
        if row['userId'] not in user_ids:
            user_ids.append(row['userId'])
            cur.execute(query_insert_users, (int(row['userId']),))

        cur.execute(query_insert_ratings, (int(row['userId']), int(row['movieId']), float(row['rating']),))

    conn.commit()


def insert_user_edges_to_db(conn):
    create_table(conn, sql_create_user_edges_table)

    query_insert_users = """ INSERT INTO user_edges(from_user_id, to_user_id) VALUES(?, ?) """

    user_edges_df = pd.read_csv("./datasets/youtube_ungraph.csv", sep="\t", usecols=["FromNodeId", "ToNodeId"])

    cur = conn.cursor()
    for index, row in user_edges_df.iterrows():
        cur.execute(query_insert_users, (int(row['FromNodeId']), int(row['ToNodeId']),))

    conn.commit()


if __name__ == '__main__':

    conn = create_connection("my.db")

    if conn is not None:
        initialize_db(conn)
        insert_user_edges_to_db(conn)
        pass

    else:
        print("Error, cannot establish connection to db")
