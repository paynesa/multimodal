"""
Purpose:
- Accumulate 6 eval sets 
- Split each eval set into VIS and ZS subsets 
- Aggregate all VIS/ZS words across sets to avoid prediction for duplicated words
"""

import numpy as np
import pandas as pd 

def get_eval_set_list():
    """
    Wordsim-sim, wordsim-rel, simlex, MEN, SemSim, VisSim
    """
    simlex = pd.read_csv('/data1/minh/evaluation/SimLex-999/SimLex-999.txt', sep="\t", header=0)
    simlex = simlex[['word1', 'word2', 'SimLex999']].as_matrix()

    wordsim_sim = pd.read_csv('/data1/minh/evaluation/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt', sep='\t', header=None).as_matrix()
    
    wordsim_rel = pd.read_csv('/data1/minh/evaluation/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt', sep='\t', header=None).as_matrix()
    
    semsim = pd.read_csv('/data1/minh/evaluation/SemSim/SemSim.txt', sep='\t', header=0)
    semsim['WORD1'], semsim['WORD2'] = semsim['WORDPAIR'].str.split('#', 1).str
    sem = semsim[['WORD1', 'WORD2', 'SEMANTIC']].as_matrix()
    sim = semsim[['WORD1', 'WORD2', 'VISUAL']].as_matrix()
    
    men = pd.read_csv('/data1/minh/evaluation/MEN/MEN_dataset_natural_form_full', sep= " ", header=None).as_matrix()
    
    eval_set_list = [wordsim_sim, wordsim_rel, simlex, men, sem, sim]
    return eval_set_list

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

def main():
    # load evaluation sets
    eval_set_list = get_eval_set_list()
    split_eval(eval_set_list)
    print("Done splitting eval sets into zs and vis.")
    print("Each eval set should have a separate zs and vis in evaluation folder.")
    
    aggregate_set('vis')
    aggregate_set('zs')
    print("Done aggregate zs and vis sets")
    print("All ZS/VIS words are collected in pred_set in multimodal folder.")

if __name__ == '__main__':
    main()

