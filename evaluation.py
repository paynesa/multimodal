import numpy as np
import pandas as pd
from scipy import stats

def get_simlex():
    simlex = pd.read_csv('/scratch/mnguyen7/re_experiments/evaluation/SimLex-999/SimLex-999.txt', sep="\t", header=0)
    simlex = simlex.as_matrix()
    return simlex

def get_wordsim():
    wordsim = pd.read_csv('/scratch/mnguyen7/re_experiments/evaluation/WordSim/combined.csv')
    wordsim = wordsim.as_matrix()
    return wordsim

def rank_cor(): 
    simlex = get_simlex()
    print(np.append(simlex[:,0], simlex[:,1]))
    #print(get_wordsim()[:,2])
    # compute spearman correlation 
    cor, pval = stats.spearmanr(simlex[:,3], simlex[:,4])
    #print(cor)
    #print(pval)

rank_cor()
