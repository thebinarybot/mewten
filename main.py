import utils
from itemCF import ItemBasedCF
from userCF import UserBasedCF
from dataset import DataSet


def runModel(model_name, dataset_name, testSize=0.3, clean=False):
    print('*' * 60)
    print('This is %s model trained on %s with testSize = %.2f' % (model_name, dataset_name, testSize))
    print('*' * 60 + '\n')
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
    
    if modelType == 'ItemCF':
        model = ItemBasedCF(iufSimilarity = True)
    elif modelType == 'UserCF':
        model = UserBasedCF(use_iif_similarity = True)

    model.fit(trainset)
    recommendTest(model, [1, 100, 233, 666])
    model.test(trainset)

def recommendTest(model, userList):
    for user in userList:
        recommend = model.recommend(str(user))
        print("recommend for userid = %s:" % user)
        print(recommend)
        print()

if __name__ == '__main__':
    dataset_name = 'ml-100k'
    modelType = 'UserCF'
    testSize = 0.2
    runModel(modelType, dataset_name, testSize, False)
