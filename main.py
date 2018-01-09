
import os
from collections import Counter
import pickle
import train_deptree as traindtrees
import train_window as trainwindow
import windowfeatureextraction as winfeatext
import numpy as np
possible_words = ['am','are','were','was','is','been','being','be']

#corpus = "When the modern Olympics began in 1896, the initiators and organizers were looking for a great popularizing event, recalling the ancient glory of Greece."


inputdir = 'input'
outputdir = 'output'
inputfiles = [f for f in os.listdir(inputdir) if os.path.isfile(os.path.join(inputdir, f))]
outputfiles = [f for f in os.listdir(outputdir) if os.path.isfile(os.path.join(outputdir, f))]

if(os.path.exists("deptree_level1_counts.pkl") is False):
    print("Training using deptree on corpus")
    with open('corpus.txt', 'r') as f:
        corpus = f.read()
    traindtrees.getDepFeaturesTrain(corpus, possible_words)

if(os.path.exists("window_size2_counts.pkl") is False):
    print("Training using windowing on corpus")
    with open('corpus.txt', 'r') as f:
        corpus = f.read()
    trainwindow.getWindowFeaturesTrain(corpus, possible_words)

if(os.path.exists("dataset_window_train.pkl") is False):
    print("Creating Dataset using windowing on corpus")
    with open('corpus.txt', 'r') as f:
        corpus = f.read()
    winfeatext.saveWindowFeaturesTrain(corpus, possible_words, win_size=6)


with open("deptree_level1_counts.pkl", 'rb') as modelfile:
    deptree_counts = pickle.load(modelfile)

with open("window_size2_counts.pkl", 'rb') as modelfile:
    window_counts = pickle.load(modelfile)



word_counts = Counter()
for counter_key in deptree_counts.keys():
    #print(counter_key, deptree_counts[counter_key])
    word_counts[counter_key[2]] += deptree_counts[counter_key]

window_word_counts = Counter()
for counter_key in window_counts.keys():
    #print(counter_key, deptree_counts[counter_key])
    window_word_counts[counter_key[2]] += window_counts[counter_key]



dep_total_correct = 0
win_total_correct = 0
testlen = 0

for inputfile, outputfile in zip(sorted(inputfiles), sorted(outputfiles)):
    with open(os.path.join(inputdir, inputfile), 'r') as f:
        noof_blanks = int(f.readline().strip('\n'))
        para = f.read().strip('\n')

    with open(os.path.join(outputdir, outputfile), 'r') as f:
        labels = f.read().strip('\n').split('\n')


'''
    window_features = trainwindow.getWindowFeaturesTest(para, '----', noof_blanks, possible_words)


    features = traindtrees.getDepFeaturesTest(para, '----', noof_blanks, possible_words)

    correct = 0
    win_correct = 0

    for feature, label, win_feature in zip(features, labels, window_features):
        max_prob = 0
        prediction = ""
        max_feat = ""
        win_max_prob = 0
        win_pred = ""
        sum_count_win = 0
        sum_count_dep = 0
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

        if(sum_count_win!=0):
            win_max_prob = float(win_max_prob)/float(sum_count_win)
        if(sum_count_dep!=0):
            max_prob = float(max_prob)/float(sum_count_dep)

        if(sum_count_win>1 and (win_max_prob>max_prob)):
            prediction = win_pred

        if(win_pred==label):
            win_correct += 1

        if(prediction==label):
            correct += 1



        #print(prediction, win_pred, label, max_feat, max_prob, win_max_prob, sum_count_win)

    #print("Accuracy for Dep and window: %d out of %d (%.2f)"%(correct, noof_blanks, float(correct)/float(noof_blanks)))
    #print("Accuracy for Window: %d out of %d (%.2f)"%(win_correct, noof_blanks, float(win_correct)/float(noof_blanks)))
    dep_total_correct += correct
    win_total_correct += win_correct
    testlen += noof_blanks

print("Accuracy for Dep and window: %d out of %d (%.2f)"%(dep_total_correct, testlen, float(dep_total_correct)/float(testlen)*100.))
print("Accuracy for Window: %d out of %d (%.2f)"%(win_total_correct, testlen, float(win_total_correct)/float(testlen)*100.))
'''
