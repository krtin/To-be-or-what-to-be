
import os
from collections import Counter
import pickle
#for probability model with Dependency
import train_deptree as traindtrees
#for probability model with windowing
import train_window as trainwindow
#for decision tree with windowing
import windowfeatureextraction as winfeatext
#for decision tree with Dependency
import depfeatureextraction as dtreefeatext
import numpy as np
#decision tree 
import models

possible_words = ['am','are','were','was','is','been','being','be']

inputdir = 'input'
outputdir = 'output'
inputfiles = [f for f in os.listdir(inputdir) if os.path.isfile(os.path.join(inputdir, f))]
outputfiles = [f for f in os.listdir(outputdir) if os.path.isfile(os.path.join(outputdir, f))]

#store count for Dependency tree for Probability model
if(os.path.exists("deptree_level1_counts.pkl") is False):
    print("Training using deptree on corpus")
    with open('corpus.txt', 'r') as f:
        corpus = f.read()
    traindtrees.getDepFeaturesTrain(corpus, possible_words)

#store counts for window of size 2 for Probability model
if(os.path.exists("window_size2_counts.pkl") is False):
    print("Training using windowing on corpus")
    with open('corpus.txt', 'r') as f:
        corpus = f.read()
    trainwindow.getWindowFeaturesTrain(corpus, possible_words)

#store dataset for training window decision tree with window size of 8
if(os.path.exists("dataset_window_train.pkl") is False):
    print("Creating Dataset using windowing on corpus")
    with open('corpus.txt', 'r') as f:
        corpus = f.read()
    winfeatext.saveWindowFeaturesTrain(corpus, possible_words, win_size=8)

#store dataset for training Dependency based decision tree with depth of 2
if(os.path.exists("dataset_dtree_train.pkl") is False):
    print("Creating Dataset using dtree on corpus")
    with open('corpus.txt', 'r') as f:
        corpus = f.read()
    dtreefeatext.saveTreeFeaturesTrain(corpus, possible_words, depth=2)

#load counts for dep tree
with open("deptree_level1_counts.pkl", 'rb') as modelfile:
    deptree_counts = pickle.load(modelfile)

#load counts for windowing
with open("window_size2_counts.pkl", 'rb') as modelfile:
    window_counts = pickle.load(modelfile)


#generate total counts of output vocab
word_counts = Counter()
for counter_key in deptree_counts.keys():
    #print(counter_key, deptree_counts[counter_key])
    word_counts[counter_key[2]] += deptree_counts[counter_key]

#generate total counts of output vocab
window_word_counts = Counter()
for counter_key in window_counts.keys():
    #print(counter_key, deptree_counts[counter_key])
    window_word_counts[counter_key[2]] += window_counts[counter_key]

#train window based decision tree
sm = models.simplemodel()
sm.train()
#train Dependency based decision tree
dtreemodel = models.simplemodel("dtree")
dtreemodel.train()

dep_total_correct = 0
win_total_correct = 0
mlmodel_total_correct = 0
dtreemodel_total_correct = 0
dep_hrank = 0
win_hrank = 0
mlmodel_hrank = 0
dtreemodel_hrank = 0
testlen = 0

#loop through test data and perform prediction for each model
for inputfile, outputfile in zip(sorted(inputfiles), sorted(outputfiles)):
    #input data
    with open(os.path.join(inputdir, inputfile), 'r') as f:
        noof_blanks = int(f.readline().strip('\n'))
        para = f.read().strip('\n')
    #output data
    with open(os.path.join(outputdir, outputfile), 'r') as f:
        labels = f.read().strip('\n').split('\n')

    #get features corresponding to each type of model
    wind_ext_feats = winfeatext.getWindowFeaturesTest(para, '----', noof_blanks, possible_words, win_size=8)
    dtree_ext_feats = dtreefeatext.getDepFeaturesTest(para, '----', noof_blanks, possible_words, depth=2)
    window_features = trainwindow.getWindowFeaturesTest(para, '----', noof_blanks, possible_words)
    features = traindtrees.getDepFeaturesTest(para, '----', noof_blanks, possible_words)

    correct = 0
    win_correct = 0
    mlmodel_correct = 0
    dtreemodel_correct = 0

    #loop through features, one set of feature for each blank
    for feature, label, win_feature, wind_ext_feat, dtree_ext_feat in zip(features, labels, window_features, wind_ext_feats, dtree_ext_feats):
        max_prob = 0
        prediction = ""
        max_feat = ""
        win_max_prob = 0
        win_pred = ""
        sum_count_win = 0
        sum_count_dep = 0
        #find the max probability for probability based models
        for word in possible_words:
            current_count = deptree_counts[feature['dep'], feature['gov_pos'], word]
            current_prob = float(current_count)
            #/float(np.sum(list(word_counts.values())))
            #/float(word_counts[word])
            win_feature[2] = word
            win_curr_count = window_counts[tuple(win_feature)]
            win_curr_prob = float(win_curr_count)
            sum_count_dep += current_count
            sum_count_win += win_curr_count
            #/float(np.sum(list(window_word_counts.values())))
            #/float(window_word_counts[word])
            #print(feature['dep'], feature['gov_pos'], word, current_prob)

            #prediction using deptree
            if(current_prob>=max_prob):
                max_prob = current_prob
                prediction = word
                max_feat = [feature['dep'], feature['gov_pos'], word]

            #prediction using window
            if(win_curr_prob>=win_max_prob):
                win_max_prob = win_curr_prob
                win_pred = word


        wind_ext_feat[0] = label
        #prediction through decision tree model using windowing
        [mlmodel_pred, mlmodel_prob]= sm.predict(wind_ext_feat)
        #prediction through decision tree model using Dependency
        [dtreemodel_predict, dtree_prob] = dtreemodel.predict(dtree_ext_feat)


        if(sum_count_win!=0):
            win_max_prob = float(win_max_prob)/float(sum_count_win)
        if(sum_count_dep!=0):
            max_prob = float(max_prob)/float(sum_count_dep)

        #pick the better probability based model
        if(sum_count_win>1 and (win_max_prob>max_prob)):
            prediction = win_pred

        if(win_pred==label):
            win_correct += 1

        if(prediction==label):
            correct += 1

        if(mlmodel_pred==label):
            mlmodel_correct += 1

        if(dtreemodel_predict==label):
            dtreemodel_correct += 1



        #print(prediction, win_pred, label, max_feat, max_prob, win_max_prob, sum_count_win)

    #print("Accuracy for Dep and window: %d out of %d (%.2f)"%(correct, noof_blanks, float(correct)/float(noof_blanks)))
    #print("Accuracy for Window: %d out of %d (%.2f)"%(win_correct, noof_blanks, float(win_correct)/float(noof_blanks)))

    #just for output score
    dep_hrank += float(correct)/float(noof_blanks)*10.
    win_hrank += float(win_correct)/float(noof_blanks)*10.
    mlmodel_hrank += float(mlmodel_correct)/float(noof_blanks)*10.
    dtreemodel_hrank += float(dtreemodel_correct)/float(noof_blanks)*10.
    dep_total_correct += correct
    win_total_correct += win_correct
    mlmodel_total_correct += mlmodel_correct
    dtreemodel_total_correct += dtreemodel_correct
    testlen += noof_blanks

#final result
print("Accuracy for Dep and window: %d out of %d (%.2f) Hackerrank score: %d"%(dep_total_correct, testlen, float(dep_total_correct)/float(testlen)*100., dep_hrank))
print("Accuracy for Window: %d out of %d (%.2f) Hackerrank score: %d"%(win_total_correct, testlen, float(win_total_correct)/float(testlen)*100., win_hrank))
print("Accuracy for Window ML Model: %d out of %d (%.2f) Hackerrank score: %d"%(mlmodel_total_correct, testlen, float(mlmodel_total_correct)/float(testlen)*100., mlmodel_hrank))
print("Accuracy for Dep ML Model: %d out of %d (%.2f) Hackerrank score: %d"%(dtreemodel_total_correct, testlen, float(dtreemodel_total_correct)/float(testlen)*100., dtreemodel_hrank))
