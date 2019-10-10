import pickle
import os
import shutil

class ModelManager:
    
    pathName = ''
    @classmethod
    def __init__(obj, dataSet = None, testSize = 0.3):
        if not obj.pathName:
            obj.pathName = "Models/" + dataSet + "withTestSize" + str(testSize)
    
    def saveModel(self, model, name:str):
        if not os.path.exists('Models'):
            os.mkdir('Models')
        pickle.dump(model, open(self.pathName + "-%s" % name, "wb"))
    
    def loadModel(self, name:str):
        if not os.path.exists(self.pathName + "-%s" % name):
            raise OSError('There is no model named %s in model/ dir' % name)
        return pickle.load(open(self.path_name + "-%s" % name, "rb"))

    @staticmethod
    def clearWorkspace(clean = False):
        if clean and os.path.exists('Models'):
            shutil.rmtree('Models')