from pycorenlp import StanfordCoreNLP
from collections import defaultdict, Counter
import pickle
from nltk import sent_tokenize
from progress import printProgressBar
import pandas as pd
import config

nlp = StanfordCoreNLP(config.corenlppath)

#extract pos tags as feature for given window size, window size is for only one side
def saveWindowFeaturesTrain(text, word_list, win_size):

    sents = sent_tokenize(text)
    flag = 0
    traindata = []
    progress = 0
    totalen = len(sents)
    featvocab = set({"None"})
    for sent in sents:

        parsed_sents = nlp.annotate(sent, properties={
                  'annotators': 'tokenize,ssplit,pos',
                  'outputFormat': 'json'
                })

        for parsed_sent in parsed_sents['sentences']:
            totalwords = len(parsed_sent['tokens'])
            for item in parsed_sent['tokens']:
                if(item["word"] in word_list):
                    #print(item)
                    word_index = item["index"]
                    feature = ["None" for x in range(win_size*2+1)]
                    #first will be the word itself
                    feature[0] = item["word"]

                    for i in range(1, win_size+1):

                        if(word_index-i>0):
                            feature[i] = parsed_sent['tokens'][word_index-i-1]["pos"]
                            featvocab.add(feature[i])

                        if(word_index+i<=totalwords):
                            feature[win_size + i] = parsed_sent['tokens'][word_index + i -1]["pos"]
                            featvocab.add(feature[win_size + i])

                    traindata.append(feature)

                    #flag += 1
                    #break
        #if(flag==2):
        #    break
        progress += 1
        printProgressBar(progress, totalen)


    #print(traindata)
    #print(featvocab)
    with open("dataset_window_train.pkl", 'wb') as modelfile:
        pickle.dump([traindata, featvocab], modelfile)

#extract feature from single instance of test input can contain multiple blanks
def getWindowFeaturesTest(para, blank, noof_blanks, word_list, win_size):

    tagged_para = nlp.annotate(para, properties={
              'annotators': 'tokenize,ssplit,pos',
              'outputFormat': 'json'
            })


    features = []

    for sent in tagged_para['sentences']:
        totalwords = len(sent['tokens'])
        for item in sent['tokens']:
            if(item['originalText']==blank):
                #print(item)
                word_index = item["index"]
                feature = ["None" for x in range(win_size*2+1)]
                #first will be the word itself
                feature[0] = item["word"]

                for i in range(1, win_size+1):
                    if(word_index-i>0):
                        feature[i] = sent['tokens'][word_index-i-1]["pos"]

                    if(word_index+i<=totalwords):
                        feature[win_size + i] = sent['tokens'][word_index+i-1]["pos"]
                features.append(feature)



    if(len(features)!=noof_blanks):
        raise Exception("Number of blanks should be equal to number of items in features")

    return features
