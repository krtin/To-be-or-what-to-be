import pandas as pd
import numpy as np
import pickle
import os

class simplemodel(object):
    """docstring for simplemodel."""
    def __init__(self):
        super (simplemodel, self).__init__()
        if(os.path.exists("dataset_window_train.pkl")):
            with open("dataset_window_train.pkl", 'rb') as modelfile:
                [traindata, vocab] = pickle.load(modelfile)
        else:
            raise Exception("Training Dataset needs to be created first")

        self.vocab = vocab
        self.traindata = traindata

    def preparedata(self):
        traindata = pd.DataFrame(self.traindata)
        vocab = self.vocab
        vocab.remove('None')
        vocablen = len(vocab)
        

    def train(self):
        [X, Y] = self.preparedata()
