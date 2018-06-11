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
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

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

        sgd = optimizers.SGD(lr=self.args.lr)
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
        print(self.args.e)
        history = self.model.fit(self.x_train, self.y_train, epochs=self.args.e, verbose=1)
        print("Training complete")

        try:
            self.model.save(self.args.s)
            print("Model saved")
        except:
            raise Exception("Error saving model")

    def predict(self, x):
        """
        @param x: a set of word embeddings
        """
        try:
            # self.args.s: train, save, then load the model in one go 
            # self.args.l: load an old model for prediction
            if self.args.s:
                self.model = load_model(self.args.s)
            elif self.args.l:
                self.model = load_model(self.args.l)
            
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
    semsim = semsim[['WORD1', 'WORD2', 'SEMANTIC', 'VISUAL', 'WORDPAIR']]
    
    return semsim.as_matrix()

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
    vis_words = pd.read_csv('words_processed.txt', header=None).as_matrix()
    counter = 0 # to mark eval set 

    for eval_set in eval_set_list:
        for i in range(eval_set.shape[0]):
            # check if both words in the word pair have image embeddings 
            if eval_set[i][0] in vis_words and eval_set[i][1] in vis_words:
                with open(str(counter)+'_vis.txt', 'a') as f:
                    np.savetxt(f, eval_set[i].reshape(1, eval_set[i].shape[0]), fmt='%s')
            else:
                with open(str(counter)+'_zs.txt', 'a') as f:
                    np.savetxt(f, eval_set[i].reshape(1, eval_set[i].shape[0]), fmt='%s')
        counter += 1

def create_zs_set_individual(eval_set):
    """
    Zero-shot: dataset that contains word embeddings of words with no visual info
    VIS: dataset that contains word embeddings of words with visual info
    """
    img_dict = KeyedVectors.load_word2vec_format("image.txt", binary=False)
    word_dict = Magnitude("/data1/minh/word.magnitude")
    # list of all words in the eval_set 
    zs_words = []
    vis_words = []

    for i in range(eval_set.shape[0]):
        # check if image vectors exist for both words
        word1 = word_dict[eval_set[i][0]]
        word2 = word_dict.query(eval_set[i][1])
        if img_dict.query(eval_set[i][0]) == False or img_dict.query(eval_set[i][1]) == False: 
            # add word to zs_words 
            zs_words.append(eval_set[i][0])
            try: 
                zs_set = np.vstack([zs_set, word1])
                zs_set = np.vstack([zs_set, word2])
            except:
                zs_set = word1
                zs_set = np.vstack([zs_set, word2])
        else:  
            # add word to vis_words 
            vis_words.append(eval_set[i][0])
            try: 
                vis_set = np.vstack([vis_set, word1])
                vis_set = np.vstack([vis_set, word2])
            except:
                vis_set = word1
                vis_set = np.vstack([vis_set, word2])
    return zs_set, vis_set, zs_words, vis_words

def create_zs_set_all(eval_set_list):
    """
    Aggregate VIS and ZS words of all test sets into 2 big sets
    """
    zs_words_all = []
    vis_words_all = []

    for s in eval_set_list:
        zs, vis, zs_words, vis_words = create_zs_set_individual(s)
        # aggregate zs, vis words of all eval sets
        zs_words_all += zs_words
        vis_words_all += vis_words
        try:
            zs_set = np.vstack([zs_set, zs])
            vis_set = np.vstack([vis_set, vis])
        except:
            zs_set = zs 
            vis_set = vis
    return zs_set, vis_set, zs_words_all, vis_words_all

def save_prediction(word_list, list_type, pred_embedding, word_dict):
    # create a directory for each word, save predicted embeddings
    for i in range(len(word_list)):
        folder_path = word_list[i] + " (" + list_type + ")" 
        pathlib.Path(folder_name).mkdir()
        with open(folder_path + "/ex1.p", 'wb') as fp:
          pickle.dump(pred_embedding[i], fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        word_dict[word_list[i]] = pred_embedding[i]
    return word_dict 

def compute_pair_sim(word1, word2): 
    """
    compute cosine similarity between two words
    """
    dot_product = np.dot(word1, word2)
    length_word1 = np.linalg.norm(word1)
    length_word2 = np.linalg.norm(word2)
    return dot_product/(length_word1 * length_word2)

def compute_sim(word_dict, eval_set):
    """
    compute similarity for all words in the evaluation set
    @param word_dict: dictionary: keys: words, values: learned embeddings
    @param eval_set: the evaluation set 
    @return a numpy array of word similarity
    """
    zs_sim = []
    vis_sim = []
    img_dict = Magnitude("image.magnitude")
    for i in range(eval_set.shape[0]):
        embedding1 = word_dict[eval_set[i][0]]
        embedding2 = word_dict[eval_set[i][1]]
        pair_sim = compute_pair_sim(embedding1, embedding2)
        # if this word is ZS 
        if img_dict.query(eval_set[i][0]) == False or img_dict.query(eval_set[i][1]) == False:
            zs_sim.append(pair_sim)
        else:
            # TODO: split human ratings for zs and vis, then save them 
            vis_sim.append(pair_sim)
    zs_sim = np.asarray(zs_sim) 
    vis_sim = np.asarray(vis_sim)
    return zs_sim, vis_sim

def evaluate_cor(model_sim, human_sim):
    """
    Compute Spearman rank correlation between two sets of similarity ratings
    @param model_sim: similarity ratings computed by model
    @param human_sim: similarity ratings rated by humans
    @return rank correlation and its P-value
    """
    cor, pval = stats.spearmanr(model_sim, human_sim)
    return cor, pval

def main():
    args = parse_args()
    if args.s == None and args.l == None:
        raise Exception("Either save or load a model")
    x_train = pd.read_csv("x_train.txt", sep=" ", header=None)
    y_train = pd.read_csv("y_train.txt", sep=" ", header=None)
    model = MultimodalEmbedding(x_train, y_train, args)
    
    # load evaluation sets
    wordsim_sim = get_wordsim_sim()
    wordsim_rel = get_wordsim_rel()
    simlex = get_simlex()
    semvis = get_semvis()
    men = get_men()
    eval_set_list = [wordsim_sim, wordsim_rel, simlex, semvis, men]
    # uncomment if haven't splitted the eval set
    # split_eval(eval_set_list)
    # zs_set, vis_set, zs_words, vis_words = create_zs_set_all(eval_set_list)

    # if args.s: train, save and load model in one go
    # if args.l: load an old model for prediction
    if args.s:
        model.start_training(args.model)
    vis_embedding = model.predict(vis_set)
    zs_embedding = model.predict(zs_set)

    # save embeddings to word's directory and accumulate all words into a word_dict dictionary
    word_dict = {}
    word_dict = save_prediction(zs_words, "zs", zs_embedding, word_dict)
    word_dict = save_prediction(vis_words, "vis", vis_embedding, word_dict)
    
    # evaluate against simlex
    simlex = get_simlex()
    zs_sim, vis_sim = compute_sim(word_dict, simlex)
    cor, pval = evaluate_cor(model_sim, simlex[:,3])
    print("Correlation for SimLex (VIS): {}, P-value: {}".format(cor, pval))

    #evaluate against wordsim
    wordsim = get_wordsim_all()
    model_sim = compute_sim(word_dict, wordsim)
    cor, pval = evaluate_cor(model_sim, wordsim[:,2])
    print("Correlation for WordSim (VIS): {}, P-value: {}".format(cor, pval))

if __name__ == '__main__':
    main()
