import numpy as np
import pickle
from argparse import ArgumentParser

def build_embedding():
    """
    Return a dictionary, keys: words, values: word vectors 
    """
    # load Glove word vectors 
    GLOVE_300D = "/scratch/mnguyen7/re_experiments/glove.840B.300d.txt"
    word_vectors_gl = {}

    with open(GLOVE_300D, encoding='utf-8') as f:
        for line in f:
            word = line.rstrip().rsplit(' ')
            word_vectors_gl[word[0]] = np.asarray(word[1:], dtype='float32')
    
    # save the dictionary to a file 
    #with open('/scratch/mnguyen7/re_experiments/word_vectors_gl', 'wb') as fp:
        #pickle.dump(word_vectors_gl, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # open the file 
    with open('/scratch/mnguyen7/re_experiments/word_vectors_gl', 'rb') as fp:
        word_vectors_gl = pickle.load(fp)
    print("Shape of each word vector: {}".format(word_vectors_gl["something"].shape))
    
    return word_vectors_gl

# TODO: create a training set. x: word vector, y: image vector
# def create_train_set:
    # embedding = build_embedding()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("model", default="neural", type=str, help="[linear, neural]")
    parser.add_argument("-lr", default=0.1, type=int, help="learning rate, default=0.1 for both models")
    parser.add_argument("-u", default=300, type=int, help="num of hidden units for neural net")
    parser.add_argument("-e", default=25, type=int, help="num of epochs for training, default=25 for neural net, 175 for linear")
    parser.add_argument("-s", type=str, help="path for saving model")
    parser.add_argument("-l", type=str, help="path for loading model")
    args = parser.parse_args()
    # train_set = create_train_set()
    return args
    # return train_set, args
    
    return train_set, args

parse_args()
