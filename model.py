"""
Purpose:
- Train models 
- Predict embeddings
- Save embeddings to dictionaries 
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model

from argparse import ArgumentParser
import pandas as pd
import pickle


#This class builds a linear model and a neural net that learns the mappting from word embeddigns to image embeddings
class MultimodalEmbedding:

	def __init__(self, x_train, y_train, args):
		self.x_train = x_train
		self.y_train = y_train
		self.args = args
		self.model = None

	def build_linear_model(self):
		self.model = Sequential()
		if (self.args.u == None):
			self.model.add(Dense(4096, input_shape=(300,)))
		else:
			self.model.add(Dense(4096, input_shape=(self.args.u,)))
		self.model.add(Dropout(0.1))
		self.model.summary()
		sgd = SGD(lr=self.args.lr)
		self.model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["accuracy"])

	def build_neural_net(self):
		self.model = Sequential()
		self.model.add(Dense(self.args.u, activation="tanh", input_shape=(300,)))
		self.model.add(Dropout(0.25))
		self.model.add(Dense(4096))
		self.model.summary()
		sgd = SGD(lr=self.args.lr)
		self.model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["accuracy"])

	def start_training(self):
		if (self.args.model == "linear"):
			self.build_linear_model()
		else:
			self.build_neural_net()
		print("Training initialized...")
		if (self.args.e != None):
			history = self.model.fit(self.x_train, self.y_train, epochs=self.args.e, verbose=1)
		elif (self.args.model == linear):
			history = self.model.fit(self.x_train, self.y_train, epochs=175, verbose=1)
		else:
			history = self.model.fit(self.x_train, self.y_train, epochs=25, verbose=1)
		print("Training complete. Saving model..")
		try:
			self.model.save(self.args.s+'.h5')
			print("Model saved!")
		except:
			raise Exception("Error saving model")

	def load_model(self):
		try:
			self.model = load_model(self.args.l+'.h5')
			print("Model loaded")
		except:
			raise Exception("Error loading model. Check file path.")

	def predict(self, word_embeddings):
		learned_embedding = self.model.predict(word_embeddings)
		return learned_embedding

# save dictionary of predicted embeddings
def save_prediction(word_list, list_type, pred_embedding, path):
	word_dict = dict(zip(word_list, pred_embedding))
	with open(path+"_"+list_type+".p", 'wb') as fp:
		try:
			pickle.dump(word_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
			print("Predictions saved")
		except:
			raise Exception("There was a problem saving your predictions")
	return word_dict

#merge two dictionaries into a pickle file
def merge_dict(dict1, dict2, path):
	dict_all = {**dict1, **dict2}
	with open(path+'_all.p', 'wb') as fp:
		try:
			pickle.dump(dict_all, fp, protocol=pickle.HIGHEST_PROTOCOL)
			print("Merged predictions saved")
		except:
			raise Exception("There was a problem saving your merged predictions")


#parse arguments set by user
def parse_args():
	parser = ArgumentParser()
	parser.add_argument("model", default=None, type=str, help="[linear, neural]")
	parser.add_argument("--lr", default=0.1, type=int, help="learning rate, default=0.1 for both models")
	parser.add_argument("--u", default=300, type=int, help="num of hidden units for neural net")
	parser.add_argument("--e", default=25, type=int, help="num of epochs for training, default=25 for neural net, 175 for linear")
	parser.add_argument("--s", type=str, help="path for saving model")
	parser.add_argument("--l", type=str, help="path for loading model")
	parser.add_argument("--i", type=str, help="path to the directory containing x_train and y_train")
	parser.add_argument("--p", type=str, help="path to the directory containing prediction sets if different than the path given in --i")
	args = parser.parse_args()
	return args

def main():
	#raise an exception if the user does not have a valid model type
	args = parse_args()	
	if (args.model != "linear") and (args.model != "neural"):
		raise Exception("You must input a valid model type (linear or neural)")
	if (args.p == None) and (args.i == None):
		raise Exception("You must give a location of your prediction files")

	#if args.s: train, save and load model in one go. if args.l: load an old model for prediction
	if (args.s != None):
		model_path = args.s
		#load the training sets
		print("Loading x_train and y_train...")
		datapath = args.i
		x_train = pd.read_csv(datapath+"/x_train.txt", sep = " ", header=None)
		y_train = pd.read_csv(datapath+"/y_train.txt", sep = " ", header=None)
		print("Done loading x_train and y_train")
		#initialize and train the model
		model = MultimodalEmbedding(x_train, y_train, args)
		model.start_training()
	elif (args.l != None):
		model_path = args.l
		model = MultimodalEmbedding(None, None, args)
		model.load_model()
	else:
		raise Exception("You must either load (--l) or save (--s) a model")
		

	#predict the embeddings
	print("Predicting...")
	if (args.p == None):
		path = args.i
	else:
		path = args.p
	vis_pred_set = pd.read_csv(path+'pred_set_vis.txt', sep=' ', header=None).values
	zs_pred_set = pd.read_csv(path+'pred_set_zs.txt', sep=' ', header=None).values
	vis_embedding = model.predict(vis_pred_set[:, 1:])
	zs_embedding = model.predict(zs_pred_set[:, 1:])

	#accumulate all words into word_dict dictionary
	word_dict_zs = save_prediction(zs_pred_set[:, 0], "zs", zs_embedding, model_path)
	word_dict_vis = save_prediction(vis_pred_set[:, 0], "vis", vis_embedding, model_path)

	# merge word_dict_zs and word_dict_vis 
	merge_dict(word_dict_zs, word_dict_vis, model_path)

if __name__=='__main__':
	main()
