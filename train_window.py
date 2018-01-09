from pycorenlp import StanfordCoreNLP
from collections import defaultdict, Counter
import pickle
from nltk import sent_tokenize
from progress import printProgressBar

nlp = StanfordCoreNLP('http://localhost:9000')

def getWindowFeaturesTrain(text, word_list):

    sents = sent_tokenize(text)
    flag = 0
    counts = Counter()
    progress = 0
    totalen = len(sents)

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
                    features = ["None" for x in range(5)]
                    features[2] = item["word"]
                    if(word_index-2>0):
                        #two words exists before
                        features[0] = parsed_sent['tokens'][word_index-3]["pos"]
                        features[1] = parsed_sent['tokens'][word_index-2]["pos"]
                    elif(word_index-1>0):
                        #one word exists before
                        features[1] = parsed_sent['tokens'][word_index-2]["pos"]

                    if(word_index+2<=totalwords):
                        #two words exists ahead
                        features[4] = parsed_sent['tokens'][word_index+1]["pos"]
                        features[3] = parsed_sent['tokens'][word_index]["pos"]
                    elif(word_index+1<=totalwords):
                        #one word exists ahead
                        features[3] = parsed_sent['tokens'][word_index]["pos"]

                    counts[tuple(features)] += 1
                    #print(counts)
                    flag = 1
                    #break
        #if(flag):
        #    break
        progress += 1
        printProgressBar(progress, totalen)

    with open("window_size2_counts.pkl", 'wb') as modelfile:
        pickle.dump(counts, modelfile)


def getWindowFeaturesTest(para, blank, noof_blanks, word_list):

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
                feature = ["None" for x in range(5)]
                feature[2] = item["word"]
                if(word_index-2>0):
                    #two words exists before
                    feature[0] = sent['tokens'][word_index-3]["pos"]
                    feature[1] = sent['tokens'][word_index-2]["pos"]
                elif(word_index-1>0):
                    #one word exists before
                    feature[1] = sent['tokens'][word_index-2]["pos"]

                if(word_index+2<=totalwords):
                    #two words exists ahead
                    feature[4] = sent['tokens'][word_index+1]["pos"]
                    feature[3] = sent['tokens'][word_index]["pos"]
                elif(word_index+1<=totalwords):
                    #one word exists ahead
                    feature[3] = sent['tokens'][word_index]["pos"]
                features.append(feature)

    if(len(features)!=noof_blanks):
        raise Exception("Number of blanks should be equal to number of items in features")

    return features
