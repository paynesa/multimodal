'''
Arg vecs: 
avg or not (default = don't average?) 
source =  /data1/minh/data
datapath /data1/minh/multimodal/img_embedding.txt
'''
import sys
import numpy as np
import pickle
import pandas as pd 
import os
import subprocess

if (len(sys.argv) !=4) and (len(sys.argv) != 5):
	print("ERROR: You must pass in 3-4 command-line arguments:\n 1. The location of the files\n 2. The path to the directory where you would like the embeddings and dictionaries to be written\n 3. The averaging of the vectors:\n\t\'avg\' for averaging\n\t\'iter\' for non-averaging and labelling with iteration\n 4. Optional: location of the word embeddings, which should be labelled 'word.magnitude', if different than the directory given in 2")
	quit()

source = sys.argv[1]
datapath = sys.argv[2]
avg = sys.argv[3]

def avg_embeddings():
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
				if False in np.isnan(average_embedding):
					average_embedding = np.insert(average_embedding, 0, words[i][0])
					with open(datapath+'/imageembeddings.txt', 'a') as f:
						np.savetxt(f, average_embedding.reshape(1, average.shape[0]), fmt="%s")
					with open(datapath+'words.txt', 'a') as f: 
						f.write("{}\n".format(words[i][0]))
				start = end
			
			#stop once non-English words appear; minh had print("Number of English words: {}".format(i/10)) -- but why are we dividing by 10
			if 'column-' in words[i+1][0]:
				print("Done loading English words")
				break

		#process remaining word if there is one
		if start != words.shape[0]: 
			img_embedding = words[start:, 1:]
			average_embedding = img_embedding.sum(axis=0) /img_embedding.shape[0]
			#check for NaNs and save embeddings to a txt file:
			if False in np.isnan(average_embedding):
				average_embedding = np.insert(average_embedding, 0, words[i][0])
				with open(datapath+'/imageembeddings.txt', 'a') as f:
					np.savetxt(f, average_embedding.reshape(1, average.shape[0]), fmt="%s")
				with open(datapath+'words.txt', 'a') as f: 
					f.write("{}\n".format(words[i][0]))
	print("Done average pooling")




def non_avg_embeddings():
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
					if False in np.isnan(words[j, 1:]):
						new = str(b) + "-" + str(j)
						words[start+j][0] = new
						#TODO check this -- weird behaviour
						with open(datapath+'/imageembeddings.txt', 'a') as f:
							np.savetxt(f, words[start+j, :], fmt="%s")
						with open(datapath+'words.txt', 'a') as f: 
							f.write("{}\n".format(words[i][0]))
				start = end

			#stop as soon as a non-English word is reached
			if 'column-' in words[i+1][0]:
				print("Done loading English words")
				break
		
		#process remaining word if there is one
		if start != words.shape[0]: 
			b = words[start][0]
			for i in range(words.shape[0] - start):
				#check for NaNs and write to txt file
				if False in np.isnan(words[j, 1:]):
					new = str(b) + "-" + str(j)
					words[start+j][0] = new
					with open(datapath+'/imageembeddings.txt', 'a') as f:
						np.savetxt(f, words[start+j, :], fmt="%s")
					with open(datapath+'words.txt', 'a') as f: 
						f.write("{}\n".format(words[i][0]))


if avg == 'avg':
	avg_embeddings()
	print("Your averaged embeddings have been saved to ", datapath)
elif avg == 'iter': 
	non_avg_embeddings()
	print("Your iterated embeddings have been saved to ", datapath)
else:
	print("ERROR: You must pass in a valid argument for iteration:\n\t\'avg\' for averaging\n\t\'iter\' for non-averaging and labelling with iteration")
	quit()

#try to convert the file to magnitude, quit if fails
try:
	subprocess.check_output(['python3.6', '- m', 'pymagnitude.converter', '-i', datapath+'/imageembeddings.txt', '-o', datapath+'/image.magnitude'])
except:
	print("There was an error converting your file ",  datapath+'/imageembeddings.txt', " to the magnitude format. Please try to do so manually")
	quit()


#if the conversion was successful, load the magnitude files
img_dict = Magnitude(datapath+'/image.magnitude')
if len(sys.argv) == 4:
	word_dict = Magnitude(datapath+'/word.magnitude')
else:
	word_dict = Magnitude(sys.argv[4])


#process the words
for line in open(datapath+'/words.txt', 'r'):
	word = str(words[i][0])
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
	with open(datapath+'/words_processed.txt', 'a') as f:
       		f.write("{}\n".format(phrase))
	
	#Controlling for OOV words and writing the embeddings to the corresponding training files
	#TODO: compare results when using OOV functionality to not
	if (phrase in word_dict) and (word in img_embedding):
		word_embedding = word_dict.query(phrase)
		img_embedding = img_dict.query(word)
		if (False in np.asarray(word_embedding)) and (False in np.asarray(img_embedding)):
			with open(datapath+'/x_train.txt', 'a') as f:
           				np.savetxt(f, word_embedding.reshape(1, word_embedding.shape[0]))
			with open(datapath+'/y_train.txt', 'a') as f:
           				np.savetxt(f, img_embedding.reshape(1, img_embedding.shape[0]))



