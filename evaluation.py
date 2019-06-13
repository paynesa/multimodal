import os, pickle, subprocess
from process_eval_set import get_eval_set_list
from argparse import ArgumentParser
from scipy import stats
from pymagnitude import *
import numpy as np
import pandas as pd

l = ["Wordsim_sim", "Wordsim_rel", "Simlex", "MEN", "SemSim", "VisSim"]

#convert the dictionary that contains all predicted embeddings into magnitude format
def convert_dict_to_magnitude(path):
	with open(path+'.p', 'rb') as fp:
		word_dict = pickle.load(fp)
	for k, v in word_dict.items():
		v = v.astype('<U100')
		v = np.insert(v, 0, k)
		with open(path+'temp.txt', 'a') as f:
			np.savetxt(f, v.reshape(1, v.shape[0]), fmt='%s')
	i = path+'temp.txt'
	o = path+'.magnitude'
	cmd = ("python3.6 -m pymagnitude.converter -i "+i+" -o "+o).split()
	try:
		subprocess.check_output(cmd)
		print("Magnitude conversion successful")
	except:
		raise Exception("Problem converting to magnitude, please convert from the command line")


#compute the cosine similarity between two words
def compute_pair_sim(word1, word2):
	dot_product = np.dot(word1, word2)
	mag_word1 = np.linalg.norm(word1)
	mag_word2 = np.linalg.norm(word2)
	return (dot_product/(mag_word1 * mag_word2))


#compute similarity for all words in the evaluation set
def compute_sim(eval_set, word_dict):
	word_sim = []
	for i in range(eval_set.shape[0]):
		embedding1 = word_dict[eval_set[i][0]]
		embedding2 = word_dict[eval_set[i][1]]
		pair_sim = compute_pair_sim(embedding1, embedding2)
		word_sim.append(pair_sim)
	word_sim = np.asarray(word_sim)
	return word_sim


#compute similarity for all words in the evaluation set when in magnitude
def compute_sim_magnitude(eval_set, word_dict):
	word_sim = []
	for i in range(eval_set.shape[0]):
		embedding1 = word_dict.query(eval_set[i][0])
		embedding2 = word_dict.query(eval_set[i][1])
		pair_sim = compute_pair_sim(embedding1, embedding2)
		word_sim.append(pair_sim)
	word_sim = np.asarray(word_sim)
	return word_sim


#Print out evaluation results (correlation, P-value) for all sets, either of type ZS or VIS 
def evaluate(eval_set_type, word_dict, dict_format):
	path = '/data1/minh/evaluation/'
	for i in range(6):
		eval_set = pd.read_csv(path+str(i)+'_'+eval_set_type+'.txt', sep=' ', header=None).as_matrix()
		if (dict_format == "dict"):
			model_sim = compute_sim(eval_set, word_dict)
		else:
			model_sim = compute_sim_magnitude(eval_set, word_dict)
		cor, pval = stats.spearmanr(model_sim, eval_set[:,2])
		print("Correlation for {} ({}): {:.3f}, P-value: {:.3f}".format(str(l[i]), eval_set_type, cor, pval))
	print()


#Print out evaluation results (correlation, P-value) for all sets (full set)
def evaluate_all(eval_set_list, word_dict, dict_format):
	for i in range(len(eval_set_list)):
		if dict_format == 'dict':
			model_sim = compute_sim(eval_set_list[i], word_dict)
		else:
			model_sim = compute_sim_magnitude(eval_set_list[i], word_dict)
		cor, pval = stats.spearmanr(model_sim, eval_set_list[i][:,2])
		print("Correlation for {} (all): {:.3f}, P-value: {:.3f}".format(str(l[i]), cor, pval))
	print()

def parse_args():
	parser = ArgumentParser()
	parser.add_argument('model', default='normal', type=str, help='[normal, c_linear, c_neural]')
	parser.add_argument('path', type=str, help='path to dictionary that contains predicted embeddings')
	parser.add_argument('--w', type=str, help='path to word vectors if evaluating a concatenated model')
	args = parser.parse_args()
	return args

def main():
	#parse the arguments and make sure that they are valid
	args = parse_args()
	if (args.path == None):
		raise Exception("You must input a valid path to your predicted embeddings")
	if (args.model != "c_linear") and (args.model != "c_neural") and (args.model != "normal"):
		raise Exception("Invalid evaluation type")

	#load the evaluation sets
	print("Loading evaluation sets...")
	eval_set_list = get_eval_set_list()
	print("Done loading evaluation sets")
	
	#evaluate a normal model (not concatenated)
	if (args.model == "normal"):
		print("Loading predictions...")
		try:
			with open(args.path+"_vis.p", 'rb') as fp:
				word_dict_vis = pickle.load(fp)
			with open(args.path+"_zs.p", 'rb') as fp:
				word_dict_zs = pickle.load(fp)
			with open(args.path+"_all.p", 'rb') as fp:
				word_dict_all = pickle.load(fp)
			print("Predictions loaded")
		except:
			raise Exception("There was a problem loading your predictions. Please check the filepath")
		evaluate('vis', word_dict_vis, 'dict')
		evaluate('zs', word_dict_zs, 'dict')
		evaluate_all(eval_set_list, word_dict_all, 'dict')

	#evaluate concatenated model
	else:
		print("Converting to magnitude")
		convert_dict_to_magnitude(args.path+'_all')
		try:
			print("Concatenating vectors...")
			pred_dict = Magnitude(args.path+'_all.magnitude')
			word_dict = Magnitude(args.w)
			fused_dict = Magnitude(word_dict, pred_dict)
		except:
			raise Exception("There was a problem loading your word embeddings. Please check the filepath")
		print("Dimension of concatenated vectors: {}".format(fused_dict.dim))
		evaluate('vis', fused_dict, 'magnitude')
		evaluate('zs', fused_dict, 'magnitude')
		evaluate_all(eval_set_list, fused_dict, 'magnitude')

if __name__ == '__main__':
	main()
