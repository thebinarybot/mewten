import collections
import math
from collections import defaultdict

def claculateMoviePopular(trainset):
    moviePopular = defaultdict(int)
    print("Calculating number of movies and Popularity...")

    for user, movies in trainset.items():
        for movie in movies:
            moviePopular[movie] += 1

    movieCount = len(moviePopular)
    print("Total Movie Count = %d" % movieCount)
    return moviePopular, movieCount

def calculateItemSimilarity(trainset, iufSimilarity = False):
    moviePopular, movieCount = claculateMoviePopular(trainset)
    print("Generating Item Similarity Matrix...")

    simMat = {}
    for user, movies in trainset.items():
        for movie1 in movies:
            simMat.setdefault(movie1, defaultdict(int))
            for movie2 in movies:
                if movie1 == movie2:
                    continue
                if iufSimilarity:
                    simMat[movie1][movie2] += 1 / math.log(1 + len(movies))
                else:
                    simMat[movie1][movie2] += 1

    for movie1, relatedItems in simMat.items():
        len_movie1 = moviePopular[movie1]
        for movie2, count in relatedItems.items():
            lenUser2 = moviePopular[movie2]
            simMat[movie1][movie2] = count / math.sqrt(len_movie1 * lenUser2)

    return simMat, moviePopular, movieCount