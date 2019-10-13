import collections
import math
from operator import itemgetter
from collections import defaultdict
import ItemSimilarity
import utils


class ItemBasedCF:

    def __init__(self, kSimilarMovies = 20, nRecommendMovies = 10, iufSimilarity=False, saveModel = True):
        print("ItemBasedCF start...\n")
        self.kSimilarMovies = kSimilarMovies
        self.nRecommendMovies = nRecommendMovies
        self.trainset = None
        self.saveModel = saveModel
        self.iufSimilarity = iufSimilarity

    def fit(self, trainset):
        modelManager = utils.ModelManager()
        try:
            self.simMat = modelManager.loadModel('simMat-iif' if self.iufSimilarity else 'simMat')
            self.moviePopular = modelManager.loadModel('moviePopular')
            self.movieCount = modelManager.loadModel('movieCount')
            self.trainset = modelManager.loadModel('trainset')

        except OSError:
            print('No model saved before.\nTrain a new model...')
            self.simMat, self.moviePopular, self.movieCount = \
                ItemSimilarity.calculateItemSimilarity(trainset=trainset, iufSimilarity=self.iufSimilarity)
            self.trainset = trainset
            if self.saveModel:
                modelManager.saveModel(self.simMat, 'simMat-iif' if self.iufSimilarity else 'simMat')
                modelManager.saveModel(self.moviePopular, 'moviePopular')
                modelManager.saveModel(self.movieCount, 'movieCount')
                modelManager.saveModel(self.trainset, 'trainset')

    def recommend(self, user):
        if not self.simMat or not self.nRecommendMovies or \
                not self.trainset or not self.moviePopular or not self.movieCount:
            raise NotImplementedError('ItemCF has not init or fit method has not called yet.')
        K = self.kSimilarMovies
        N = self.nRecommendMovies
        predictScore = collections.defaultdict(int)
        if user not in self.trainset:
            print('The user (%s) not in trainset.' % user)
            return
        watchedMovies = self.trainset[user]
        for movie, rating in watchedMovies.items():
            for related_movie, similarity_factor in sorted(self.simMat[movie].items(), key=itemgetter(1), reverse=True)[0:K]:
                if related_movie in watchedMovies:
                    continue
                predictScore[related_movie] += similarity_factor * rating
        return [movie for movie, _ in sorted(predictScore.items(), key=itemgetter(1), reverse=True)[0:N]]

    def test(self, testset):
        if not self.nRecommendMovies or not self.trainset or not self.moviePopular or not self.movieCount:
            raise ValueError('ItemCF has not init or fit method has not called yet.')
        self.testset = testset
        print('Test recommendation system start...')
        N = self.nRecommendMovies
        hit = 0
        recommendCount = 0
        testCount = 0
        allRecommendedMovies = set()
        popularSum = 0

        for i, user in enumerate(self.trainset):
            testMovies = self.testset.get(user, {})
            recMovies = self.recommend(user)  
            for movie in recMovies:
                if movie in testMovies:
                    hit += 1
                allRecommendedMovies.add(movie)
                popularSum += math.log(1 + self.moviePopular[movie])
            recommendCount += N
            testCount += len(testMovies)
        precision = hit / (1.0 * recommendCount)
        recall = hit / (1.0 * testCount)
        coverage = len(allRecommendedMovies) / (1.0 * self.movieCount)
        popularity = popularSum / (1.0 * recommendCount)

        print('Test recommendation system success.')

        print('Precision=%.4f\tRecall=%.4f\tCoverage=%.4f\tPopularity=%.4f\n' % (precision, recall, coverage, popularity))

    def predict(self, testset):
        movies_recommend = defaultdict(list)
        print('Predict scores start...')
        predict_time = LogTime(print_step=500)
        for i, user in enumerate(testset):
            recMovies = self.recommend(user)  
            movies_recommend[user].append(recMovies)
            predict_time.count_time()
        print('Predict scores success.')
        predict_time.finish()
        return movies_recommend
