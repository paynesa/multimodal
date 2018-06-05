import numpy as np
import pickle
import pandas as pd 
from argparse import ArgumentParser
import os
import pathlib
from pymagnitude import *

"""
Purpose:
  - Create the training set (x_train, y_train)
  - Create a directory for each word, saving word embedding and image embedding 
"""
# don't even need this anymore hahaha
def create_word_embedding():
    """
    Return a dictionary from Glove word vectors, keys: words, values: word vectors 
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

# TODO: test the training set. x: word vector, y: image vector  
def create_image_embedding():
    """
    create one image embedding for each word by average pooling all image feature vectors
    @return img_list: a numpy array of image embeddings 
    """
    # read the file that contains words and paths to image directories 
    words = pd.read_csv('/nlp/data/bcal/features/word_absolute_paths.tsv', sep='\t')
    for i in range(words.shape[0]):
        directory = words[i][1]
        for f in os.listdir(directory):
            # open the pickle file that contains image embeddings
            with open(f, 'rb') as fp:
                img = pickle.load(fb)
                try:
                    img_embedding = np.stack((img_embedding, img), axis=-1)
                except:
                    img_embedding = img

            # average pooling to create one single image embedding
            average_embedding = img_embedding.sum(axis=1) / img_embedding.shape[1]
            average_embedding.astype("<U100")
            average_embedding = np.insert(average_embedding, 0, words[i][1])
            if i == 0:
                img_list = average_embedding
            else:
                img_list = np.vstack((img_list, average_embedding))

            # save all embeddings to txt, convert txt to magnitude in cmd line 
            np.savetxt("img_embedding.txt", img_list)

def create_train_set():
    """
    for each word, if its image vector does not consists only of NaN values, 
    the word and image vectors are saved to the word's directory

    create the train set (x_train, y_train)
    @return x_train, y_train
    """
    words = pd.read_csv('/nlp/data/bcal/features/word_absolute_paths.tsv', sep='\t')
    word_dict = Magnitude('/path/to/glove.magnitude')
    create_image_embedding()
    img_dict = Magnitude('/path/to/image.magnitude')

    for i in range(words.shape[0]):
        # create a directory for each word
        # pathlib.Path(words[i][0]).mkdir()
        # handle OOV words
        word_embedding = word_dict.query(words[i][0])
        img_embedding = img_dict.query(words[i][0])
        
        # check if a word has valid image vectors 
        # valid: image vectors doesn't contain all NaNs
        check_nan = np.isnan(img_embedding)
        all_nan = check_nan[check_nan==True].shape[0]
        if all_nan == img_embedding.shape[0]:
            # with Magnitude, saving img and word vectors is not necessary
            # save word and image vectors to corresponding words' directories
            # with open(words[i][0] + '/word.p', 'wb') as fp:
                # pickle.dump(word_embedding, fb, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(words[i][0] + '/image.p', 'wb') as fp:
                # pickle.dump(img_embedding, fb, protocol=pickle.HIGHEST_PROTOCOL)
            
            # add to x_train and y_train
            try:
                x_train = np.vstack([x_train, word_embedding])
                y_train = np.vstack([y_train, img_embedding])
            except:
                x_train = word_embedding
                y_train = img_embedding 
    
    return x_train, y_train

def parse_args():
    """
    parse parameters set by user
    @return x_train, y_train, args (the parameters)
    """
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

parse_args()
