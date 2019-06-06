#TODO add try except for issues with files, raise exceptions
import sys, pickle, os, subprocess
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pymagnitude import *

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--i", default=None, type=str, help = "the directory containing the unprocessed image embeddings")
	parser.add_argument("--o", default=None, type=str, help = "the directory where you would like to save your training sets")
	parser.add_argument("avg_iter", default=None, type=str, help = "how you would like to have your repeated embeddings handled (avg to average, iter to consider 		each one sepearately)")
	parser.add_argument("--w", default=None, type=str, help = "the location of your word magnitude embeddings, if different from where you want to save your 		training sets")
	args = parser.parse_args()
	return args

def avg_embeddings(source, datapath):
	#iterate through the files
	folders = os.listdir(source)
	for f in folders:
		print("Folder name: {}".format(f))
		words = pd.read_csv(source+'/'+f, sep=' ', header=None).values
		print("Done loading ", f, " from pandas")

		start = 0
		for i in range(words.shape[0]-1):
			if words[i][0] != words[1+i][0]:
				end = i+1

				#average pooling
				img_embedding = words[start:end, 1:]
				average_embedding = img_embedding.sum(axis=0) /img_embedding.shape[0]

				#check for NaNs and save embeddings to a txt file:
				if False in pd.isnull(average_embedding):
					average_embedding = np.insert(average_embedding, 0, words[i][0])
					try:
						with open(datapath+'/imageembeddings.txt', 'a') as f:
							np.savetxt(f, average_embedding.reshape(1, average_embedding.shape[0]), fmt="%s")
						with open(datapath+'words.txt', 'a') as f:
							f.write("{}\n".format(words[i][0]))
					except:
						raise Exception("Error saving embeddings. Check file path")
				start = end

			#stop once non-desired word appears
			if 'column-' in words[i+1][0]:
				print("Done loading desired (English) words from", f)
				break 


def iter_embeddings(source, datapath):
	#iterate through the files
	folders = os.listdir(source)
	for f in folders:
		print("Folder name: {}".format(f))
		words = pd.read_csv(source+'/'+f, sep=' ', header=None).values
		print("Done loading ", f, " from pandas")

		start = 0
		for i in range(words.shape[0]-1):
			if words[i][0] != words[1+i][0]:
				end = i+1

				#iterate through the words to count their occurrences and label them accordingly
				b = str(words[i][0])
				for j in range(end-start):

					#check for NaNs and write to txt file
					if False in pd.isnull(words[j, 1:]):
						new = str(b) + "-" + str(j)
						words[start+j][0] = new
						z = words[start+j, :]
						try:
							with open(datapath+'/imageembeddings.txt', 'a') as f:
								np.savetxt(f, z.reshape(1, z.shape[0]), fmt = "%s")
							with open(datapath+'/words.txt', 'a') as f:
								f.write("{}\n".format(new))
						except:
							raise Exception("Error saving embeddings. Check file path")		
	
				start = end

			#stop as soon as a non-English word is reached
			if 'column-' in words[i+1][0]:
				print("Done loading desired (English) words from", f)
				break

def create_training_set(datapath, w = None):
	oov_counter = 0
	img_dict = Magnitude(datapath+'/image.magnitude')
	if (w == None):
		word_dict = Magnitude(datapath+'/word.magnitude')
	else:
		word_dict = Magnitude(w)

	#process the words
	for line in open(datapath+'/words.txt', 'r'):
		word = line.strip()
		if "row" in word:
			phrase = word.split('-')[1]
		else:
			phrase = word.split('-')[0]
		if "_" in phrase:
			word_list = phrase.split('_')
			phrase = ""
			for i in range(len(word_list)):
				phrase += word_list[i]
				if i < len(word_list)-1:
					phrase += " "
		try:
			with open(datapath+'/words_processed.txt', 'a') as f:
				f.write("{}\n".format(phrase))
		except:
			raise Exception("Error saving processed words")

		#Controlling for OOV words and writing the embeddings to the corresponding training files
		if (phrase in word_dict) and (word in img_dict):
			word_embedding = word_dict.query(phrase)
			img_embedding = img_dict.query(word)
			if (False in pd.isnull(np.asarray(word_embedding))) and (False in pd.isnull(np.asarray(img_embedding))):
				try:
					with open(datapath+'/x_train.txt', 'a') as f:
						np.savetxt(f, word_embedding.reshape(1, word_embedding.shape[0]))
					with open(datapath+'/y_train.txt', 'a') as f:
						np.savetxt(f, img_embedding.reshape(1, img_embedding.shape[0]))
				except:
					raise Exception("error saving training sets")
		else:
			oov_counter += 1
	print(oov_counter, "out-of-vocabulary words found")

def main():
	#define vars
	args = parse_args()
	src = args.i
	data = args.o
	avg = args.avg_iter
	wordmag = args.w

	#check arguments for null arguments and invalid averaging arg
	if (src == None) or (data == None) or ((avg!= 'avg') and (avg!='iter')):
		raise Exception("ERROR: You must pass in 3-4 command-line arguments:\n --i The location of the files\n --o The path to the directory where you would like the 			embeddings and dictionaries to be written\n 3. The averaging of the vectors:\n\t\'avg\' for averaging\n\t\'iter\' for non-averaging and labelling with 			iteration\n 4. --w (optional): location of the word embeddings, which should be labelled 'word.magnitude', if different than the directory given in 2")
	else:
		print("Loading your embeddings...")

	########################### BEGIN COMMENTING HERE IF MAGNITUDE CONVERSION FAILS #######################################################
	#load and process the image vectors

	if avg == 'avg':
		avg_embeddings(src, data)
		print("Done average pooling. Your averaged embeddings have been saved to ", data)
	if avg == 'iter': 
		iter_embeddings(src, data)
		print("Done iterating. Your iterated embeddings have been saved to ", data)

	#convert the resulting file to magnitude
	print("Converting to magnitude format...")
	i = data+"/imageembeddings.txt"
	o = data+"/image.magnitude"
	cmd = ("python3.6 -m pymagnitude.converter -i "+i+" -o "+o).split()
	try:
		subprocess.check_output(cmd)
	except:
		raise Exception("There was an error converting your file ", data+'/imageembeddings.txt', " to the magnitude format. Please try to do so manually")
	print("Magnitude conversion successful!")

	########################### END COMMENTING HERE IF MAGNITUDE CONVERSION FAILS #######################################################		
	#create the training set
	print("Creating training sets... (this could take a while)")
	create_training_set(data, wordmag)
	print("Training set creation complete!")

if __name__ == '__main__':
	main()
