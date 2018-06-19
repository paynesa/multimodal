"""
Purpose:
- Train models 
- Predict embeddings
- Save embeddings to dictionaries 
"""
from load_data import parse_args

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model

import pandas as pd
import pickle

class MultimodalEmbedding:
    """
    This class builds a linear model and a neural net that learns the mapping from word embeddings to image embeddings
    """
    def __init__(self, x_train, y_train, args):
        self.x_train = x_train 
        self.y_train = y_train
        self.args = args
        self.model = None

    def build_linear_model(self):
        self.model = Sequential()
        self.model.add(Dense(4096, input_shape=(300,)))
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
    
    def start_training(self, model):
        if model == "linear":
            self.build_linear_model()
        elif model == "neural":
            self.build_neural_net()
        else:
            # TODO: edit error message
            printf("model: linear/neural")
        
        print("Training initialized...")
        history = self.model.fit(self.x_train, self.y_train, epochs=self.args.e, verbose=1)
        print("Training complete")

        try:
            self.model.save(self.args.s+'.h5')
            print("Model saved")
        except:
            raise Exception("Error saving model")
    
    def load_model(self):
        try:
            self.model = load_model(self.args.l+'.h5')
        except:
            raise Exception("Error loading model")
 
    def predict(self, x):
        """
        @param x: a set of word embeddings
        """
        learned_embedding = self.model.predict(x)
        return learned_embedding

def save_prediction(word_list, list_type, pred_embedding, args):
    # save dictionary of predicted embeddings
    word_dict = dict(zip(word_list, pred_embedding))
    if args.s:
        path = args.s
    else:
        path = args.l                            
    
    with open(path+"_"+list_type+".p", 'wb') as fp:
        pickle.dump(word_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    return word_dict

def merge_dict(dict1, dict2, path):
    """merge 2 dictionaries and save to a pickle file"""
    dict_all = {**dict1, **dict2}
    with open(path+'_all.p', 'wb') as fp:
        pickle.dump(dict_all, fp, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    args = parse_args()
    if args.s == None and args.l == None:
        raise Exception("Either save or load a model")
    
    x_train = pd.read_csv("/data1/minh/multimodal/x_train.txt", sep=" ", header=None)
    y_train = pd.read_csv("/data1/minh/multimodal/y_train.txt", sep=" ", header=None)
    print("Done loading x_train and y_train")
    model = MultimodalEmbedding(x_train, y_train, args)
    
    # if args.s: train, save and load model in one go
    # if args.l: load an old model for prediction
    if args.s:
        model_path = args.s
        model.start_training(args.model)
    if args.l:
        model_path = args.l
        model.load_model() 
    
    path = '/data1/minh/multimodal/' 
    vis_pred_set = pd.read_csv(path+'pred_set_vis.txt', sep=' ', header=None).as_matrix()
    zs_pred_set = pd.read_csv(path+'pred_set_zs.txt', sep=' ', header=None).as_matrix()
    vis_embedding = model.predict(vis_pred_set[:, 1:])
    zs_embedding = model.predict(zs_pred_set[:, 1:])
     
    # uncomment if haven't accumulated all words into a word_dict dictionary
    word_dict_zs = save_prediction(zs_pred_set[:, 0], "zs", zs_embedding, args)
    word_dict_vis = save_prediction(vis_pred_set[:, 0], "vis", vis_embedding, args)        
    print("All predicted embeddings are saved in word_dict in multimodal folder.")

    # merge word_dict_zs and word_dict_vis 
    merge_dict(word_dict_zs, word_dict_vis, model_path)

if __name__=='__main__':
    main()
