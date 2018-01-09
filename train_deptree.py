from pycorenlp import StanfordCoreNLP
from collections import defaultdict, Counter
import pickle
from nltk import sent_tokenize
from progress import printProgressBar

nlp = StanfordCoreNLP('http://localhost:9000')


def getDepFeaturesTrain(text, word_list):

    sents = sent_tokenize(text)
    flag = 0
    counts = Counter()
    progress = 0
    totalen = len(sents)

    for sent in sents:
        parsed_sents = nlp.annotate(sent, properties={
                  'annotators': 'tokenize,ssplit,pos,depparse',
                  'outputFormat': 'json'
                })

        for parsed_sent in parsed_sents['sentences']:
            for item in parsed_sent['basicDependencies']:
                if(item["dependentGloss"] in word_list):
                    gov_index = item["governor"]
                    if(gov_index==0):
                        continue
                    dep = item["dep"]
                    gov = parsed_sent['tokens'][gov_index-1]
                    if(gov_index!=gov['index']):
                        raise Exception('Governor Index dont match')
                    gov_pos = gov['pos']
                    word = item["dependentGloss"]
                    #print(sent)
                    #print(word, dep, gov_pos)
                    counts[dep, gov_pos, word] += 1
                    flag=1
                    #break
                    #params[word["originalText"]][]

        #if(flag):
        #    break
        progress += 1
        printProgressBar(progress, totalen)

    with open("deptree_level1_counts.pkl", 'wb') as modelfile:
        pickle.dump(counts, modelfile)




def getDepFeaturesTest(para, blank, noof_blanks, word_list):


    tokenized_para = nlp.annotate(para, properties={
              'annotators': 'tokenize,ssplit',
              'outputFormat': 'json'
            })
    searched_blanks = defaultdict(list)
    noof_found = 0

    #first just search the position of the blanks as per stanford tokenization
    for sent in tokenized_para['sentences']:
        sentence_index = sent['index']

        for word in sent['tokens']:
            if(word["originalText"]==blank):
                blank_position = word['index']
                searched_blanks[sentence_index].append(blank_position)
                noof_found += 1

        #if all blanks have been found quit
        if(noof_found==noof_blanks):
            break

    if(noof_found!=noof_blanks):
        raise Exception('Could not find the correct number of blanks')

    #replace all the blanks in the paragraph with any possible word
    para = para.replace(blank, word_list[-1])
    parsed_para = nlp.annotate(para, properties={
              'annotators': 'tokenize,ssplit,pos,depparse',
              'outputFormat': 'json'
            })


    features = []

    for sent in parsed_para['sentences']:

        if(len(searched_blanks[sent['index']])>0):
            for item in sent['basicDependencies']:
                for searched_pos in searched_blanks[sent['index']]:
                    if(item['dependent']==searched_pos):
                        dep = item['dep']
                        gov_index = item['governor']
                        gov = sent['tokens'][gov_index-1]
                        if(gov_index!=gov['index']):
                            raise Exception('Governer Index dont match')
                        gov_pos = gov['pos']
                        features.append({"dep":dep, "gov_pos":gov_pos})
                        searched_blanks[sent['index']].remove(searched_pos)

                if(len(searched_blanks[sent['index']])==0):
                    break


    if(len(features)!=noof_blanks):
        raise Exception("Number of blanks should be equal to number of items in features")

    return features
