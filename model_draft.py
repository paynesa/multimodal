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

class MultimodalEmbedding:
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

    def predict(self):
        try:
            # self.args.s: train, save, then load the model in one go 
            # self.args.l: load an old model for prediction
            if self.args.s:
                self.model = load_model(self.args.s)
            elif self.args.l:
                self.model = load_model(self.args.l)
            
            learned_embedding = self.model.predict(self.x_train)
            return learned_embedding
        except:
            raise Exception("Error loading model")

def get_simlex():
    simlex = pd.read_csv('/scratch/mnguyen7/re_experiments/evaluation/SimLex-999/SimLex-999.txt', sep="\t", header=0)
    return simlex.as_matrix()

def word_sim():
    wordsim = pd.read_csv('/scratch/mnguyen7/re_experiments/evaluation/WordSim/combined.csv')
    return wordsim.as_matrix()

def compute_pair_sim(word1, word2): 
    dot_product = np.dot(word1, word2)
    length_word1 = np.linalg.norm(word1)
    length_word2 = np.linalg.norm(word2)
    return dot_product/(length_word1 * length_word2)

def compute_sim(word_dict, eval_set):
    # TODO: figure out how to handle words in the eval set that are not in the training set  
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
    cor, pval = stats.spearmanr(model_sim, human_sim)
    return cor, pval

def main():
    args = parse_args()
    model = MultimodalEmbedding(x_train, y_train, args)
    # train, save and load model in one go
    if args.s:
        model.start_training(args.model)
        embedding = model.predict()
    # load an old model for prediction
    elif args.l:
        embedding = model.predict()

    # save the learned embedding to word's directory 
    words = pd.read_csv('/nlp/data/bcal/features/word_absolute_paths.tsv', sep='\t')
    word_dict = {}
    for i in range(words.shape[0]):
        if os.path.exists(words[i][0]):
            with open(words[i][0] + "learned" + "ex1.p", 'wb') as fp:
                pickle.dump(embedding[i], fp, protocol=pickle.HIGHEST_PROTOCOL)
            word_dict[words[i][0]] = embedding[i]
    
    # evaluate against simlex
    simlex = get_simlex()
    model_sim = compute_sim(word_dict, simlex)
    cor, pval = evaluate_cor(model_sim, simlex[:,3])
    print("Correlation for SimLex: {}, P-value: {}".format(cor, pval))

    #evaluate against wordsim
    wordsim = get_wordsim()
    model_sim = compute_sim(word_dict, wordsim)
    cor, pval = evaluate_cor(model_sim, wordsim[:,2])
    print("Correlation for WordSim: {}, P-value: {}".format(cor, pval))
    
if __name__ == '__main__':
    main()

