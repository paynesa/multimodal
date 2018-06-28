"""
Purpose:
- Classifier that predicts word similarity scores 
"""

from evaluation import compute_sim_magnitude
from process_eval_set import get_eval_set_list, process_word, get_eval_set_missing

import numpy as np
import pandas as pd
from scipy import stats
from pymagnitude import *
from argparse import ArgumentParser
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model 
from sklearn.linear_model import LinearRegression

class SimClassifier:
    """
    This class combines concreteness ratings and various word similarity scores
    """
    def __init__(self, args):
        self.args = args
        self.model = None 

    def build_classifier(self):
        self.model = Sequential()
        self.model.add(Dense(self.args.u, activation='relu', input_shape=(6,)))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1, activation='relu'))
        
        self.model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])

    def start_training(self, x_train, y_train, x_test, y_test, path):
        self.build_classifier()

        print("Training initialized...")
        history = self.model.fit(x_train, y_train, epochs=self.args.e, verbose=1)
        print('Training complete')
        self.evaluate(x_test, y_test)

        try:
            self.model.save(path+'.h5')
            print('Model saved')
        except:
            raise Exception('Error saving model')

#def get_similarity()
# glove, img, pred, concat, conc1, conc2

def build_input_scores(num):
    vis_input = pd.read_csv('/data1/minh/multimodal/sim_classifier/'+str(num)+'_vis_concat.txt', sep=' ', header=None).values
    zs_input = pd.read_csv('/data1/minh/multimodal/sim_classifier/'+str(num)+'_zs_concat.txt', sep=' ', header=None).values
    vis_set = pd.read_csv('/data1/minh/evaluation/concrete/'+str(num)+'_vis.txt', sep=' ', header=None).values
    zs_set = pd.read_csv('/data1/minh/evaluation/concrete/'+str(num)+'_zs.txt', sep=' ', header=None).values
    
    # vis/zs ratio is the same for train set and test set 
    vis_idx = int(0.5*len(vis_input))
    zs_idx = int(0.5*len(zs_input))
    x_train = np.vstack((vis_input[:vis_idx][:,:4], zs_input[:zs_idx][:,:4]))
    x_test = np.vstack((vis_input[vis_idx:][:,:4], zs_input[zs_idx:][:,:4]))
    y_train = np.hstack((vis_set[:,2][:vis_idx], zs_set[:,2][:zs_idx]))
    y_test = np.hstack((vis_set[:,2][vis_idx:], zs_set[:,2][zs_idx:]))

    return x_train, y_train, x_test, y_test 

def main():
    eval_set_list = get_eval_set_list()
    eval_set_list = get_eval_set_missing(eval_set_list)
    
    eval_set_list = [eval_set_list[0]]
    lr = LinearRegression()
    counter = 0

    for eval_set in eval_set_list:
        x_train, y_train, x_test, y_test = build_input_scores(counter)
        print("Shape of x_train: {}, shape of x_test: {}".format(x_train.shape, x_test.shape))
        lr.fit(x_train, y_train)
        print("Score: {}".format(lr.score(x_test, y_test)))
        pred_scores = lr.predict(x_test)
        cor, pval = stats.spearmanr(pred_scores, y_test)
        print("Correlation: {}, P-value: {}".format(cor, pval))
        print()
        counter += 1

if __name__=='__main__':
    main()
