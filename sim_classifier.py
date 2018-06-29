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
import pickle
from sklearn.linear_model import LinearRegression
from argparse import ArgumentParser

#def get_similarity()
# glove, img, pred, concat, conc1, conc2

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-r', default=0.5, type=float, help='vis/zs ratio')
    parser.add_argument('-n', type=int, help='eval set')
    args = parser.parse_args()
    return args 

def build_input_scores(args):
    vis_input = pd.read_csv('/data1/minh/multimodal/sim_classifier/'+str(args.n)+'_vis_concat.txt', sep=' ', header=None).values
    zs_input = pd.read_csv('/data1/minh/multimodal/sim_classifier/'+str(args.n)+'_zs_concat.txt', sep=' ', header=None).values
    vis_set = pd.read_csv('/data1/minh/evaluation/concrete/'+str(args.n)+'_vis.txt', sep=' ', header=None).values
    zs_set = pd.read_csv('/data1/minh/evaluation/concrete/'+str(args.n)+'_zs.txt', sep=' ', header=None).values
    
    # vis/zs ratio is the same for train set and test set 
    vis_idx = int(args.r*len(vis_input))
    zs_idx = int(args.r*len(zs_input))
    x_train = np.vstack((vis_input[:vis_idx], zs_input[:zs_idx]))
    x_test = np.vstack((vis_input[vis_idx:], zs_input[zs_idx:]))
    y_train = np.hstack((vis_set[:,2][:vis_idx], zs_set[:,2][:zs_idx]))
    y_test = np.hstack((vis_set[:,2][vis_idx:], zs_set[:,2][zs_idx:]))

    return x_train, y_train, x_test, y_test 

def main():
    args = parse_args()
    eval_set_list = get_eval_set_list()
    eval_set_list = get_eval_set_missing(eval_set_list)[args.n]
    lr = LinearRegression()

    x_train, y_train, x_test, y_test = build_input_scores(args)
    print("Shape of x_train: {}, shape of x_test: {}".format(x_train.shape, x_test.shape))
    lr.fit(x_train, y_train)
    print("Score: {}".format(lr.score(x_test, y_test)))
    pred_scores = lr.predict(x_test)
    cor, pval = stats.spearmanr(pred_scores, y_test)
    print("Correlation for test set: {}, P-value: {}".format(cor, pval))

    pred_scores = lr.predict(np.vstack((x_train, x_test)))
    output_scores = np.hstack((y_train, y_test))
    cor, pval = stats.spearmanr(pred_scores, output_scores)
    print("Correlation for full set: {}, P-value: {}".format(cor, pval))
    print()

if __name__=='__main__':
    main()
