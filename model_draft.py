"""
A playground for building models
"""
from load_data import parse_args
from load_data import create_word_embedding

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model

import numpy as np
import pandas as pd
from scipy import stats
import os
# from pymagnitude import *

class MultimodalEmbedding:
    """
    This class builds a linear model and a neural net that learns the mapping from word embeddings to image embeddings
    """
    def _init__(self, x_train, y_train, args):
        self.x_train = x_train 
        self.y_train = y_train
        self.args = args
        self.model = None

    def build_linear_model(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(300,)))
        self.model.add(Dropout(0.1)) 
        self.model.summary()

        sgd = optimizers.SGD(lr=self.args.lr)
        self.model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["accuracy"])

    def build_neural_net(self):
        self.model = Sequential()
        self.model.add(Dense(self.args.u, activation="tanh", input_shape=(300,)))
        self.model.add(Dropout(0.25)) #is this where dropout layer should be??
        self.model.add(Dense(128))
        self.model.summary()

        sgd = optimizers.SGD(lr=self.args.lr)
        self.model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["accuracy"])
    
    def start_training(self, model):
        if model == "linear":
            build_linear_model()
        elif model == "neural":
            build_neural_net()
        else:
            # TODO: edit error message
            printf("model: linear/neural")
        
        print("Training initialized...")
        history = self.model.fit(self.x_train, self.y_train, self.args.e, verbose=1)
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
    simlex = pd.read_csv('/scratch/mnguyen7/re_experiments/evaluation/SimLex-999/SimLex-999.txt', sep="\t", header=0)
    return simlex.as_matrix()

def get_wordsim_all():
    """
    WordSim-353
    """
    wordsim = pd.read_csv('/scratch/mnguyen7/re_experiments/evaluation/WordSim/combined.csv')
    return wordsim.as_matrix()

def get_semvis():
    """
    SemSim / VisSim
    """
    semsim = pd.read_csv('/scratch/mnguyen7/re_experiments/evaluation/SemSim/SemSim.txt', sep='\t', header=0)
    semsim['WORD1'], semsim['WORD2'] = semsim['WORDPAIR'].str.split('#', 1).str
    semsim = semsim[['WORD1', 'WORD2', 'SEMANTIC', 'VISUAL', 'WORDPAIR']]
    
    return semsim.as_matrix()

def create_zs_set_individual(eval_set):
    """
    Zero-shot: dataset that contains words with no visual info
    VIS: dataset that contains words with visual info
    """
    img_dict = Magnitude("path/to/image.magnitude")
    word_dict = Magnitude("path/to/word.magnitude")
    # list of all words in the eval_set 
    zs_words = []
    vis_words = []

    for i in range(eval_set.shape[0]):
        # check if image vectors exist for both words
        word1 = word_dict.query(eval_set[i][0]) 
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
    compute similarity for all words in the evaluation set (maybe?)
    @param word_dict: dictionary: keys: words, values: learned embeddings
    @param eval_set: the evaluation set 
    @return a numpy array of word similarity
    """
    # TODO: figure out how to handle words in the eval set that are not in the training set  
    # TODO: might need to separate eval set into vis and zs
    sim = []
    for i in range(eval_set.shape[0]):
        if os.path.exists(eval_set[i][0]) and os.path.exists(eval_set[i][1]):
            embedding1 = word_dict[eval_set[i][0]]
            embedding2 = word_dict[eval_set[i][1]]
            pair_sim = compute_pair_sim(embedding1, embedding2)
            sim.append(pair_sim)
    sim = np.asarray(sim) 
    
    return sim 

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
    model = MultimodalEmbedding(x_train, y_train, args)
    
    # load evaluation sets
    wordsim = get_wordsim_all()
    simlex = get_simlex()
    semvis = get_semvis()
    semsim = semvis[:,2]
    vissim = semvis[:,3]
    eval_set_list = [wordsim, simlex, semsim, vissim]
    zs_set, vis_set, zs_words, vis_words = create_zs_set_all(eval_set_list)

    # train, save and load model in one go
    if args.s:
        model.start_training(args.model)
        vis_embedding = model.predict(vis_set)
        zs_embedding = model.predict(zs_set)
    # load an old model for prediction
    elif args.l:
        vis_embedding = model.predict(vis_set)
        zs_embedding = model.predict(zs_set)

    # save embeddings to word's directory and accumulate all words into a word_dict dictionary
    words = pd.read_csv('/nlp/data/bcal/features/word_absolute_paths.tsv', sep='\t')
    word_dict = {}
    word_dict = save_prediction(zs_list_all, "zs", zs_embedding, word_dict)
    word_dict = save_prediction(vis_list_all, "vis", vis_embedding, word_dict)
    
    # evaluate against simlex
    simlex = get_simlex()
    model_sim = compute_sim(word_dict, simlex)
    cor, pval = evaluate_cor(model_sim, simlex[:,3])
    print("Correlation for SimLex (VIS): {}, P-value: {}".format(cor, pval))

    #evaluate against wordsim
    wordsim = get_wordsim_all()
    model_sim = compute_sim(word_dict, wordsim)
    cor, pval = evaluate_cor(model_sim, wordsim[:,2])
    print("Correlation for WordSim (VIS): {}, P-value: {}".format(cor, pval))

if __name__ == '__main__':
    main()

