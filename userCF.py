import collections
import math
import UserSimilarity
import utils
from operator import itemgetter
from collections import defaultdict

class UserBasedCF:   
    def __init__(self, k_sim_user=20, n_rec_movie=10, use_iif_similarity=False, save_model=True):       
        print("UserBasedCF start...\n")
        self.k_sim_user = k_sim_user
        self.n_rec_movie = n_rec_movie
        self.trainset = None
        self.save_model = save_model
        self.use_iif_similarity = use_iif_similarity

    def fit(self, trainset):
        model_manager = utils.ModelManager()
        try:
            self.user_sim_mat = model_manager.loadModel(
                'user_sim_mat-iif' if self.use_iif_similarity else 'user_sim_mat')
            self.movie_popular = model_manager.loadModel('movie_popular')
            self.movie_count = model_manager.loadModel('movie_count')
            self.trainset = model_manager.loadModel('trainset')
            print('User origin similarity model has saved before.\nLoad model success...\n')
        except OSError:
            print('No model saved before.\nTrain a new model...')
            self.user_sim_mat, self.movie_popular, self.movie_count = \
                UserSimilarity.calculate_user_similarity(trainset=trainset,
                                                     use_iif_similarity=self.use_iif_similarity)
            self.trainset = trainset
            print('Train a new model success.')
            if self.save_model:
                model_manager.saveModel(self.user_sim_mat,
                                         'user_sim_mat-iif' if self.use_iif_similarity else 'user_sim_mat')
                model_manager.saveModel(self.movie_popular, 'movie_popular')
                model_manager.saveModel(self.movie_count, 'movie_count')
            print('The new model has saved success.\n')

    def recommend(self, user):
        if not self.user_sim_mat or not self.n_rec_movie or \
                not self.trainset or not self.movie_popular or not self.movie_count:
            raise NotImplementedError('UserCF has not init or fit method has not called yet.')
        K = self.k_sim_user
        N = self.n_rec_movie
        predict_score = collections.defaultdict(int)
        if user not in self.trainset:
            print('The user (%s) not in trainset.' % user)
            return
        watched_movies = self.trainset[user]
        for similar_user, similarity_factor in sorted(self.user_sim_mat[user].items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for movie, rating in self.trainset[similar_user].items():
                if movie in watched_movies:
                    continue            
                predict_score[movie] += similarity_factor * rating    
        return [movie for movie, _ in sorted(predict_score.items(), key=itemgetter(1), reverse=True)[0:N]]

    def test(self, testset):
        if not self.n_rec_movie or not self.trainset or not self.movie_popular or not self.movie_count:
            raise ValueError('UserCF has not init or fit method has not called yet.')
        self.testset = testset
        print('Test recommendation system start...')
        N = self.n_rec_movie
        hit = 0
        rec_count = 0
        test_count = 0
        all_rec_movies = set()
        popular_sum = 0


        for i, user in enumerate(self.trainset):
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user) 
            for movie in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
            rec_count += N
            test_count += len(test_movies)
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)

        print('Test recommendation system success.')

        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f\n' %
              (precision, recall, coverage, popularity))
        
    def predict(self, testset):
        movies_recommend = defaultdict(list)
        print('Predict scores start...')
        for i, user in enumerate(testset):
            rec_movies = self.recommend(user) 
            movies_recommend[user].append(rec_movies)
        print('Predict scores success.')
        return movies_recommend
