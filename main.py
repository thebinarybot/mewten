import utils
from itemCF import ItemBasedCF
from dataset import DataSet


def runModel(model_name, dataset_name, testSize=0.3, clean=False):
    print('*' * 70)
    print('\tThis is %s model trained on %s with testSize = %.2f' % (model_name, dataset_name, testSize))
    print('*' * 70 + '\n')
    modelManager = utils.ModelManager(dataset_name, testSize)
    try:
        trainset = modelManager.loadModel('trainset')
        testset = modelManager.loadModel('testset')
    except OSError:
        ratings = DataSet.loadDataset(name=dataset_name)
        trainset, testset = DataSet.splitData(ratings, testSize=testSize)
        modelManager.saveModel(trainset, 'trainset')
        modelManager.saveModel(testset, 'testset')

    modelManager.clearWorkspace(clean)
    
    model = ItemBasedCF(iufSimilarity = True)

    model.fit(trainset)
    recommendTest(model, [1, 100, 233, 666, 888])
    model.test(trainset)

def recommendTest(model, user_list):
    for user in user_list:
        recommend = model.recommend(str(user))
        print("recommend for userid = %s:" % user)
        print(recommend)
        print()


if __name__ == '__main__':
    dataset_name = 'ml-100k'
    modelType = 'ItemCF'
    testSize = 0.1
    runModel(modelType, dataset_name, testSize, False)