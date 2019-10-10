import collections
import itertools
import random
import os
from collections import namedtuple

datasetTuple = namedtuple('datasetTuple', ['url', 'path', 'sep', 'reader_params'])

availableDatasets = {
    'ml-100k':
        datasetTuple(
            url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            path='data/ml-100k/u.data',
            sep='\t',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='\t')
        ),
    'ml-1m'  :
        datasetTuple(
            url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            path='data/ml-1m/ratings.dat',
            sep='::',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='::')
        )
}

random.seed(0)

class DataSet:

    def __init__(self):
        pass

    @classmethod
    def parseLine(obj, line:str, sep:str):
        user, movie, rate = line.strip('\r\n').split(sep)[:3]
        return user, movie, rate

    @classmethod
    def loadDataset(obj, name = 'ml-100k'):
        try:
            dataset = availableDatasets[name]
        except KeyError:
            raise ValueError('unknown dataset ' + name + '. Accepted values are ' + ', '.join(BUILTIN_DATASETS.keys()) + '.')

        if not os.path.isfile(dataset.path):
            raise OSError("Dataset data/" + name + " could not be found in this project.\nPlease download it from " + dataset.url + " manually and unzip it to data/ directory.")
        
        with open(dataset.path) as f:
            ratings = [obj.parseLine(line, dataset.sep) for line in itertools.islice(f, 0, None)]

        print("Loading " + name + " Dataset success.")
        return ratings
    
    @classmethod
    def splitData(obj, ratings, testSize = 0.2):
        train, test = collections.defaultdict(dict), collections.defaultdict(dict)
        trainset_len = 0
        testset_len = 0
        for user, movie, rate in ratings:
            if random.random() <= testSize:
                test[user][movie] = int(rate)
                testset_len += 1
            else:
                train[user][movie] = int(rate)
                trainset_len += 1
        print("Spliting data to training set and test set success.")
        print("Train set size = %s" % trainset_len)
        print("Test set size = %s\n" % testset_len)
        return train, test