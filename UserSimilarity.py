import collections
import math
from collections import defaultdict

def calculate_user_similarity(trainset, use_iif_similarity=False):
    print('building movie-users inverse table...')
    movie2users = collections.defaultdict(set)
    movie_popular = defaultdict(int)

    for user, movies in trainset.items():
        for movie in movies:
            movie2users[movie].add(user)
            movie_popular[movie] += 1
    print('building movie-users inverse table success.')
    movie_count = len(movie2users)
    print('total movie number = %d' % movie_count)
    print('generate user co-rated movies similarity matrix...')
    usersim_mat = {}
    for movie, users in movie2users.items():
        for user1 in users:
            usersim_mat.setdefault(user1, defaultdict(int))
            for user2 in users:
                if user1 == user2:
                    continue
                if use_iif_similarity:
                    usersim_mat[user1][user2] += 1 / math.log(1 + len(users))
                else:
                    usersim_mat[user1][user2] += 1

    print('calculate user-user similarity matrix...')
    for user1, related_users in usersim_mat.items():
        len_user1 = len(trainset[user1])
        for user2, count in related_users.items():
            len_user2 = len(trainset[user2])
            usersim_mat[user1][user2] = count / math.sqrt(len_user1 * len_user2)

    print('calculate user-user similarity matrix success.')
    return usersim_mat, movie_popular, movie_count