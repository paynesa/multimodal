"""
Purpose:
- Accumulate 6 eval sets 
- Split each eval set into VIS (subset of words with image vecs) and ZS (subset of zero-shot words)
- Aggregate all VIS/ZS words across sets to avoid prediction for duplicated words
"""
#TODO deal with hard-coding
import numpy as np
import pandas as pd
from pymagnitude import *
import pickle

#wordsim-sim, wordsim-rel, simlex, men, semsim, vissim
def get_eval_set_list():
	simlex = pd.read_csv('/data1/dkeren/evaluation_metrics/SimLex-999/SimLex-999.txt', sep="\t", header=0)
	simlex = simlex[['word1', 'word2', 'SimLex999']].values
	wordsim_sim = pd.read_csv("/data1/dkeren/evaluation_metrics/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt", sep='\t', header=None, engine = 'python').values
	wordsim_rel = pd.read_csv('/data1/dkeren/evaluation_metrics/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt', sep='\t', header=None).values
	semsim = pd.read_csv('/data1/dkeren/evaluation_metrics/SemSim/SemSim.txt', sep='\t', header=0)
	semsim['WORD1'], semsim['WORD2'] = semsim['WORDPAIR'].str.split('#', 1).str
	sem = semsim[['WORD1', 'WORD2', 'SEMANTIC']].values
	sim = semsim[['WORD1', 'WORD2', 'VISUAL']].values
	men = pd.read_csv('/data1/dkeren/evaluation_metrics/MEN/MEN_dataset_natural_form_full', sep= " ", header=None).values
	eval_set_list = [wordsim_sim, wordsim_rel, simlex, men, sem, sim]
	return eval_set_list

#split each eval set into a VIS set and a ZS set
def split_eval(eval_set_list):
	path = '/data1/minh/evaluation/'
	vis_words = pd.read_csv('/data1/minh/multimodal/words_processed.txt', header=None).values
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

#aggregate all zs words and their embeddings into a file for prediction, for prediction similar for vis words
def aggregate_set(eval_set_type):
	path = '/data1/minh/evaluation/'
	word_dict = Magnitude('/data1/embeddings/pymagnitude/word.magnitude')
	check_duplicates_dict = {}
	# open all _zs and _vis.txt files
	for i in range(5):
		if eval_set_type == 'vis':
			eval_set = pd.read_csv(path+str(i)+'_vis.txt', sep=' ', header=None).values
		elif eval_set_type == 'zs':
			eval_set = pd.read_csv(path+str(i) + '_zs.txt', sep= ' ', header=None).values
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

########################################################################################################################################################################
####################All functions below process data for word similarity classifier experiment, which doesn't improve similarity scores#################################
########################################################################################################################################################################

#turn plurals into singulars and remove anxiliary info
def process_word(word):
	if '_' in word:
		word = word.split('_')[0]
	if word[len(word)-1] == 's':
		word = word[:len(word)-1]
	return word

def remove_missing_concrete(eval_set_list, ratings_dict):
	counter = 0 # to mark eval set
	for eval_set in eval_set_list:
		for i in range(eval_set.shape[0]):
			word1 = process_word(eval_set[i][0])
			word2 = process_word(eval_set[i][1])
			if word1 in ratings_dict and word2 in ratings_dict:
				with open('/data1/minh/evaluation/concrete/'+str(counter)+'_missing_concrete.txt', 'a') as f:
					np.savetxt(f, eval_set[i].reshape(1, eval_set[i].shape[0]), fmt='%s')
	counter += 1

#Get eval set that removes all words missing concreteness ratings
def get_eval_set_missing(eval_set_list):
	with open('/data1/minh/multimodal/ratings_dict.p', 'rb') as fp:
		ratings_dict = pickle.load(fp)
	#uncomment if haven't built eval_set 
	#remove_missing_concrete(eval_set_list, ratings_dict)

	final_list = []
	for i in range(6):
		processed_set = pd.read_csv('/data1/minh/evaluation/concrete/'+str(i)+'_missing_concrete.txt', sep=' ', header=None).values
		final_list.append(processed_set)
	return final_list

def main():
	#load evaluation sets
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

