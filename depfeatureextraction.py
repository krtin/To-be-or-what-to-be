from pycorenlp import StanfordCoreNLP
from collections import defaultdict, Counter
import pickle
from nltk import sent_tokenize
from progress import printProgressBar
import config

nlp = StanfordCoreNLP(config.corenlppath)

#this is not being used currently
def getFeatures(currentdepth, maxdepth, features, node, parsed_sent):

    #set the current gov index
    gov_index = node["governor"]
    #if we reach root break
    if(gov_index==0):
        return features
    gov = parsed_sent['tokens'][gov_index-1]
    if(gov_index!=gov['index']):
        raise Exception('Governor Index dont match')
    gov_pos = gov['pos']
    dep = node["dep"]
    features[dep] = gov_pos
    #print(currentdepth, features)


    if(currentdepth<maxdepth):
        for node_item in parsed_sent['basicDependencies']:
            if(node_item["dependent"]==gov_index):

                ret_features = getFeatures(currentdepth+1, maxdepth, {}, node_item, parsed_sent)
                #print(ret_features)
                features.update(ret_features)

    return features

#extracts the Dependency of the parent word and all other children of the parent, the Dependency is used as a feature type and pos tag is used as feature value
def saveTreeFeaturesTrain(text, word_list, depth):

    sents = sent_tokenize(text)
    flag = 0
    counts = Counter()
    progress = 0
    totalen = len(sents)
    features = []
    ignore_deps = ["punct", "cc"]
    featvocab = set({"NaN"})
    for sent in sents:

        parsed_sents = nlp.annotate(sent, properties={
                  'annotators': 'tokenize,ssplit,pos,depparse',
                  'outputFormat': 'json'
                })

        for parsed_sent in parsed_sents['sentences']:
            for item in parsed_sent['basicDependencies']:
                if(item["dependentGloss"] in word_list):
                    feature = {}
                    #print(sent)
                    #print(parsed_sent['basicDependencies'])
                    #the first feature is always the word itself
                    feature["y"] = item["dependentGloss"]

                    gov_index = item["governor"]
                    #if we reach root break
                    if(gov_index==0):
                        continue
                    gov = parsed_sent['tokens'][gov_index-1]
                    if(gov_index!=gov['index']):
                        raise Exception('Governor Index dont match')
                    gov_pos = gov['pos']
                    dep = "p_"+item["dep"]
                    feature[dep] = gov_pos
                    featvocab.add(gov_pos)

                    for node_item in parsed_sent['basicDependencies']:
                        if(node_item["governor"]==gov_index and node_item["dependent"]!=item['dependent']):
                            #print(node_item)
                            dep = node_item["dep"]
                            if(dep in ignore_deps):
                                continue
                            dep_index = node_item["dependent"]
                            dep_item = parsed_sent['tokens'][dep_index-1]
                            if(dep_index!=dep_item['index']):
                                raise Exception('Dependency Index dont match')
                            pos = dep_item["pos"]
                            feature[dep] = pos
                            featvocab.add(pos)

                    #features = getFeatures(1, depth, feature, item, parsed_sent)
                    features.append(feature)
                    #flag+=1
                    #break
                    #params[word["originalText"]][]

        #if(flag==6):
        #    break
        progress += 1
        printProgressBar(progress, totalen)

    #print(featvocab)
    with open("dataset_dtree_train.pkl", 'wb') as modelfile:
        pickle.dump([features, featvocab], modelfile)



#does the same feature extraction but for incoming single instance of testset
#Dependency of target blank is found by replacing one of the possible words assuming it will have minimum effect on Dependency
def getDepFeaturesTest(para, blank, noof_blanks, word_list, depth=2):


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
    ignore_deps = ["punct", "cc"]
    for sent in parsed_para['sentences']:

        if(len(searched_blanks[sent['index']])>0):
            for item in sent['basicDependencies']:
                for searched_pos in searched_blanks[sent['index']]:
                    if(item['dependent']==searched_pos):

                        feature = {}
                        #print(sent)
                        #print(parsed_sent['basicDependencies'])
                        #the first feature is always the word itself
                        feature["y"] = item["dependentGloss"]

                        gov_index = item["governor"]
                        #if we reach root break
                        if(gov_index==0):
                            continue
                        gov = sent['tokens'][gov_index-1]
                        if(gov_index!=gov['index']):
                            raise Exception('Governor Index dont match')
                        gov_pos = gov['pos']
                        dep = "p_"+item["dep"]
                        feature[dep] = gov_pos

                        for node_item in sent['basicDependencies']:
                            if(node_item["governor"]==gov_index and node_item["dependent"]!=item['dependent']):
                                #print(node_item)
                                dep = node_item["dep"]
                                if(dep in ignore_deps):
                                    continue
                                dep_index = node_item["dependent"]
                                dep_item = sent['tokens'][dep_index-1]
                                if(dep_index!=dep_item['index']):
                                    raise Exception('Dependency Index dont match')
                                pos = dep_item["pos"]
                                feature[dep] = pos

                        #features = getFeatures(1, depth, feature, item, parsed_sent)
                        features.append(feature)
                        searched_blanks[sent['index']].remove(searched_pos)

                if(len(searched_blanks[sent['index']])==0):
                    break


    if(len(features)!=noof_blanks):
        raise Exception("Number of blanks should be equal to number of items in features")

    return features
