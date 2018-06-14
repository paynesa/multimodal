"""
A playground for building models
"""
from load_data import parse_args

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model

import numpy as np
import pandas as pd
from scipy import stats
import os
from pymagnitude import *
import pathlib
import pickle 

class MultimodalEmbedding:
    """
    This class builds a linear model and a neural net that learns the mapping from word embeddings to image embeddings
    """
    def __init__(self, x_train, y_train, args):
        self.x_train = x_train 
        self.y_train = y_train
        self.args = args
        self.model = None

    def build_linear_model(self):
        self.model = Sequential()
        self.model.add(Dense(4096, input_shape=(300,)))
        self.model.add(Dropout(0.1)) 
        self.model.summary()
        sgd = SGD(lr=self.args.lr)
        self.model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["accuracy"])

    def build_neural_net(self):
        self.model = Sequential()
        self.model.add(Dense(self.args.u, activation="tanh", input_shape=(300,)))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(4096))
        self.model.summary()

        sgd = SGD(lr=self.args.lr)
        self.model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["accuracy"])
    
    def start_training(self, model):
        if model == "linear":
            self.build_linear_model()
        elif model == "neural":
            self.build_neural_net()
        else:
            # TODO: edit error message
            printf("model: linear/neural")
        
        print("Training initialized...")
        history = self.model.fit(self.x_train, self.y_train, epochs=self.args.e, verbose=1)
        print("Training complete")

        try:
            self.model.save(self.args.s+'.h5')
            print("Model saved")
        except:
            raise Exception("Error saving model")
    
    def load_model(self):
        self.model = load_model(self.args.l+'.h5')
 
    def predict(self, x):
        """
        @param x: a set of word embeddings
        """
        try:
            learned_embedding = self.model.predict(x)
            return learned_embedding
        except:
            raise Exception("Error loading model")

def create_fused_embedding(words, word_dict):
    """
    concatenate word embeddings with learned embeddings
    @param words: the dataset of words in the training dataset
    @param word_dict: dictionary: keys: words, values: learned embeddings
    @return the concatenated embeddings
    """
    fused_word_dict = {}
    for i in range(words.shape[0]): 
        if os.path.exists(words[i][0]):
            with open(words[i][0] + "/word.p", 'rb') as fp:
                word_embedding = pickle.load(fp)
            learned_embedding = word_dict[words[i][0]]
            # concatenate word embedding with L2-normalized learned embedding 
            norm_learned = np.linalg.norm(learned_embedding, axis=1)
            # TODO: norm_learned == 0??
            norm_inv = 1 / norm_learned
            learned_embedding = np.multiply(learned_embedding, norm_inv[np.newaxis,:])
            fused_embedding = np.zeros((428,))
            fused_embedding[:300] = word_embedding
            fused_embedding[300:] = learned_embedding

            fused_word_dict[words[i][0]] = fused_embedding[i]
    
    return fused_word_dict

def get_simlex():
    """
    SimLex-999 
    """
    simlex = pd.read_csv('/data1/minh/evaluation/SimLex-999/SimLex-999.txt', sep="\t", header=0)
    return simlex.as_matrix()

def get_wordsim_all():
    """
    WordSim-353
    """
    wordsim = pd.read_csv('/data1/minh/evaluation/WordSim/combined.csv')
    return wordsim.as_matrix()

def get_wordsim_sim():
    """
    Wordim353-sim 
    """
    wordsim = pd.read_csv('/data1/minh/evaluation/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt', sep='\t', header=None)
    print(wordsim.head())
    return wordsim.as_matrix()

def get_wordsim_rel():
    """
    Wordim353-rel
    """
    wordsim = pd.read_csv('/data1/minh/evaluation/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt', sep='\t', header=None)
    return wordsim.as_matrix()

def get_semvis():
    """
    SemSim / VisSim
    """
    semsim = pd.read_csv('/data1/minh/evaluation/SemSim/SemSim.txt', sep='\t', header=0)
    semsim['WORD1'], semsim['WORD2'] = semsim['WORDPAIR'].str.split('#', 1).str
    sem = semsim[['WORD1', 'WORD2', 'SEMANTIC']]
    sim = semsim[['WORD1', 'WORD2', 'VISUAL']] 
    return sem.as_matrix(), sim.as_matrix()

def get_men():
    """
    MEN natural form (no POS)
    """
    men = pd.read_csv('/data1/minh/evaluation/MEN/MEN_dataset_natural_form_full', sep= " ", header=None)
    return men.as_matrix()

def split_eval(eval_set_list):
    """
    Split each eval set into a VIS set and a ZS set
    """
    path = '/data1/minh/evaluation/'
    vis_words = pd.read_csv('/data1/minh/multimodal/words_processed.txt', header=None).as_matrix()
    counter = 0 # to mark eval set 

    for eval_set in eval_set_list:
        for i in range(eval_set.shape[0]):
            # check if both words in the word pair have image embeddings 
            if eval_set[i][0] in vis_words and eval_set[i][1] in vis_words:
                with open(path+str(counter)+'_vis.txt', 'a') as f:
                    np.savetxt(f, eval_set[i].reshape(1, eval_set[i].shape[0]), fmt='%s')
            else:
                with open(path+str(counter)+'_zs.txt', 'a') as f:
                    np.savetxt(f, eval_set[i].reshape(1, eval_set[i].shape[0]), fmt='%s')
        counter += 1

def aggregate_set(eval_set_type):
    """
    aggregate all zs words and their word embeddings into a file, for prediction 
    similiar for vis words 
    @returns vis_set: a numpy array of zs/vis words, associated with word embeddings
    """
    path = '/data1/minh/evaluation/'
    word_dict = Magnitude('/data1/embeddings/pymagnitude/word.magnitude')
    check_duplicates_dict = {}
    # open all _zs and _vis.txt files
    for i in range(5):
        if eval_set_type == 'vis':
            eval_set = pd.read_csv(path+str(i)+'_vis.txt', sep=' ', header=None).as_matrix()
        elif eval_set_type == 'zs':
            eval_set = pd.read_csv(path+str(i) + '_zs.txt', sep= ' ', header=None).as_matrix()
        
        for i in range(eval_set.shape[0]):
            # if this word has never been added to the prediction set
            if check_duplicates_dict.get(eval_set[i][0]) is None:
                word1 = word_dict.query(eval_set[i][0]).astype('<U100')
                word1 = np.insert(word1, 0, eval_set[i][0])
                with open('/data1/minh/multimodal/pred_set_'+eval_set_type+'.txt', 'a') as f:
                    np.savetxt(f, word1.reshape(1, word1.shape[0]), fmt='%s')
                check_duplicates_dict[eval_set[i][0]] = 1

            if check_duplicates_dict.get(eval_set[i][1]) is None:
                word2 = word_dict.query(eval_set[i][1]).astype('<U100')
                word2 = np.insert(word2, 0, eval_set[i][1])
                with open('/data1/minh/multimodal/pred_set_'+eval_set_type+'.txt', 'a') as f:
                    np.savetxt(f, word2.reshape(1, word2.shape[0]), fmt='%s')
                check_duplicates_dict[eval_set[i][1]] = 1
    
def save_prediction(word_list, list_type, pred_embedding):
    # save dictionary of predicted embeddings
    word_dict = dict(zip(word_list, pred_embedding))
    if self.args.s:
        path = self.args.s 
    else:
        path = self.args.l

    with open(path+"_"+list_type+".p", 'wb') as fp:
        pickle.dump(word_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

def compute_pair_sim(word1, word2): 
    """
    compute cosine similarity between two words
    """
    dot_product = np.dot(word1, word2)
    length_word1 = np.linalg.norm(word1)
    length_word2 = np.linalg.norm(word2)
    return dot_product/(length_word1 * length_word2)

def compute_sim(eval_set, word_dict):
    """
    compute similarity for all words in the evaluation set
    @param word_dict: dictionary: keys: words, values: learned embeddings
    @param eval_set_type: type of evaluation set (zs/vis) 
    @return a numpy array of word similarity
    """ 
    word_sim = []
    for i in range(eval_set.shape[0]):
        embedding1 = word_dict[eval_set[i][0]]
        embedding2 = word_dict[eval_set[i][1]]
        pair_sim = compute_pair_sim(embedding1, embedding2)
        word_sim.append(pair_sim)
    word_sim = np.asarray(word_sim)
    
    return word_sim

def evaluate(eval_set_type, word_dict):
    """
    Print out evaluation results (correlation, P-value) for all sets, either of type ZS or VIS 
    @param eval_set_type: Type of eval set (ZS/VIS)
    @param word_dict: corresponding dictionary: keys: zs/vis words, values: predicted embeddings 
    """ 
    path = '/data1/minh/evaluation/'
    for i in range(6):
        eval_set = pd.read_csv(path+str(i)+'_'+eval_set_type+'.txt', sep=' ', header=None).as_matrix()
        model_sim = compute_sim(eval_set, word_dict)
        cor, pval = stats.spearmanr(model_sim, eval_set[:,2])
        print("Correlation for {} ({}): {}, P-value: {}".format(str(i), eval_set_type, cor, pval))

def main():
    args = parse_args()
    if args.s == None and args.l == None:
        raise Exception("Either save or load a model")
    x_train = pd.read_csv("/data1/minh/multimodal/x_train.txt", sep=" ", header=None)
    y_train = pd.read_csv("/data1/minh/multimodal/y_train.txt", sep=" ", header=None)
    print("Done loading x_train and y_train")
    model = MultimodalEmbedding(x_train, y_train, args)
    
    # load evaluation sets
    wordsim_sim = get_wordsim_sim()
    wordsim_rel = get_wordsim_rel()
    simlex = get_simlex()
    sem_sim, vis_sim = get_semvis()
    men = get_men()
    eval_set_list = [wordsim_sim, wordsim_rel, simlex, men, sem_sim, vis_sim]
    # uncomment if haven't splitted the eval set
    # split_eval(eval_set_list)
    # print("Done splitting eval sets into zs and vis.")
    # print("Each eval set should have a separate zs and vis in evaluation folder.")
    
    # uncomment if haven't created prediction set 
    # aggregate_set('vis')
    # aggregate_set('zs')
    # print("Done aggregate zs and vis sets")
    # print("All ZS/VIS words are collected in pred_set in multimodal folder.")

    # if args.s: train, save and load model in one go
    # if args.l: load an old model for prediction
    if args.s:
        model_path = args.s
        model.start_training(args.model)
    if args.l:
        model_path = args.l
        model.load_model()
    
    # uncomment for prediction
    path = '/data1/minh/multimodal/'
    # vis_pred_set = pd.read_csv(path+'pred_set_vis.txt', sep=' ', header=None).as_matrix() 
    # zs_pred_set = pd.read_csv(path+'pred_set_zs.txt', sep=' ', header=None).as_matrix()
    # vis_embedding = model.predict(vis_pred_set[:, 1:])
    # zs_embedding = model.predict(zs_pred_set[:, 1:])

    # uncomment if haven't accumulated all words into a word_dict dictionary
    # save_prediction(zs_pred_set[:, 0], "zs", zs_embedding)
    # save_prediction(vis_pred_set[:, 0], "vis", vis_embedding)
    # print("All predicted embeddings are saved in word_dict in multimodal folder.")    
    
    with open(model_path+"_vis.p", 'rb') as fp:
        word_dict_vis = pickle.load(fp)
    with open(model_path+"_zs.p", 'rb') as fp:
        word_dict_zs = pickle.load(fp)
    
    evaluate('vis', word_dict_vis)
    evaluate('zs', word_dict_zs)

if __name__ == '__main__':
    main()

