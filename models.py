import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class simplemodel(object):
    """docstring for simplemodel."""
    def __init__(self, algo="window"):
        super (simplemodel, self).__init__()
        if(algo=="window"):
            if(os.path.exists("dataset_window_train.pkl")):
                with open("dataset_window_train.pkl", 'rb') as modelfile:
                    [traindata, vocab] = pickle.load(modelfile)
            else:
                raise Exception("Training Dataset needs to be created first")
        else:
            if(os.path.exists("dataset_dtree_train.pkl")):
                with open("dataset_dtree_train.pkl", 'rb') as modelfile:
                    [traindata, vocab] = pickle.load(modelfile)
            else:
                raise Exception("Dtree Training Dataset needs to be created first")

        self.algo = algo
        self.vocab = vocab
        self.traindata = traindata

    def preparedata(self):
        traindata = pd.DataFrame(self.traindata)
        vocab = self.vocab
        if(self.algo=="window"):
            vocab.remove('None')
        else:
            vocab.remove('NaN')
        vocablen = len(vocab)


        if(self.algo=="window"):
            #separate out Y
            Y = traindata[0]
            del traindata[0]
        else:
            #separate out Y
            Y = traindata["y"]
            del traindata["y"]
            traindata = traindata.fillna("None")

        #list of possible Y
        yvocab = list(Y.unique())

        #create vocab mapping using vocab
        vocabmap = {}
        for word, num in zip(sorted(vocab), range(1, vocablen+1)):
            vocabmap[word] = num

        vocabmap["None"] = 0


        self.vocabmap = vocabmap
        self.inv_vocabmap = {v: k for k, v in vocabmap.items()}
        #map y to numbers
        ymap = {}
        for word, num in zip(sorted(yvocab), range(1, len(yvocab)+1)):
            ymap[word] = num
        self.ymap = ymap
        self.inv_ymap = {v: k for k, v in ymap.items()}


        traindata = traindata.applymap(lambda x: vocabmap[x])
        Y = Y.map(ymap)
        self.columns = traindata.columns
        if(self.algo!="window"):
            self.deletecolumns = ["root", "p_root"]
            traindata.drop(self.deletecolumns, axis=1, inplace=True)
            self.newcolumns = traindata.columns


        traindata = np.array(traindata).astype(float)
        Y = np.array(Y)

        #print(vocabmap)
        #print(Y)
        return [traindata, Y]


    def train(self):
        [X, y] = self.preparedata()
        #X = StandardScaler().fit_transform(X)
        #feature_selection = SelectKBest(chi2, k=16)
        #X = feature_selection.fit_transform(X, y)
        if(self.algo=="window"):
            manualsel = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
            X = np.transpose(np.transpose(X)[[manualsel]])
            self.selected_feats = manualsel
        #print(X.shape)
        #self.selected_feats = np.array(feature_selection.get_support(indices=True))


        #print(X.shape)
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=.2, random_state=42)



        clf = DecisionTreeClassifier(max_depth=6)
        clf.fit(X_train, y_train)
        self.learnedmodel = clf
        #score = clf.score(X_dev, y_dev)
        #score2 = clf.score(X_train, y_train)

        #print(score, score2)

        '''
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            score = clf.score(X_dev, y_dev)
            score2 = clf.score(X_train, y_train)
            print(name, score, score2)
        '''

    def predict(self, data):
        if(self.algo=="window"):
            y = data[0]
            X = np.array(data[1:])
        else:
            y = data["y"]
            del data["y"]
            columns = self.newcolumns
            X = []
            #print(columns)
            for col in columns:
                if(col in data.keys()):
                    X.append(data[col])
                else:
                    X.append("None")

            X = np.array(X)

        #y = self.ymap[y]


        #print(X)
        mapper = np.vectorize(lambda k: self.vocabmap[k])
        X = mapper(X)
        if(self.algo=="window"):
            X = X[self.selected_feats]
        #print(self.vocabmap)
        predict = self.learnedmodel.predict([X])
        probs = self.learnedmodel.predict_proba([X])
        max_prob = max(probs[0])
        predict = self.inv_ymap[predict[0]]
        #print(predict , y)

        return [predict, max_prob]

#sm = simplemodel("dep")
#sm.train()
