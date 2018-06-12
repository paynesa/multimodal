import numpy as np
import pickle
import pandas as pd 
from argparse import ArgumentParser
import os
import pathlib
from pymagnitude import *
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

"""
Purpose:
  - Create the training set (x_train, y_train)
"""
def create_image_embedding():
    """
    create one image embedding for each word by average pooling all image feature vectors
    @save img_embedding: a numpy array of image embeddings 
    """
    # read the files that contain words and their image embeddings
    # TODO: handle duplicates or not (maybe Magnitude will eventually handle this? 
    folders = os.listdir('/data1/minh/data')
    for f in folders:
        word_dir = pd.read_csv('/data1/minh/data/'+f, sep=' ', header=None).as_matrix()
        try:
            words = np.vstack((words, word_dir))
        except:
            words = word_dir
        print("done with {}".format(f)) 
    # img embeddings before average pooling
    np.savetxt('/data1/minh/data/unprocessed_img_embedding.txt', words, fmt='%s')
    print("Done opening all files")
    
    for i in range(0, words.shape[0], 10):
        img_embedding = words[i:i+10,1:]
        # average pooling to create one single image embedding
        average_embedding = img_embedding.sum(axis=0) / img_embedding.shape[0]
        average_embedding = average_embedding.astype("<U100")
        average_embedding = np.insert(average_embedding, 0, words[i][0])
        if i == 0:
            img_list = average_embedding
        else:
            img_list = np.vstack((img_list, average_embedding))

    # save all embeddings to txt, convert txt to magnitude in cmd line 
    np.savetxt("/data1/minh/data/img_embedding.txt", img_list, fmt="%s")
    print("Done average pooling")

def create_train_set():
    """
    for each word, if its image vector does not consists only of NaN values, 
    the word and image vectors are saved to the word's directory

    create the train set (x_train, y_train)
    @return x_train, y_train
    """
    words = pd.read_csv('/data1/minh/data/unprocessed_img_embedding.txt', sep=' ', header=None).as_matrix()
    # save all words in a txt file 
    np.savetxt('words.txt', words[:,0], fmt="%s")
    word_dict = Magnitude('/data1/embeddings/pymagnitude/word.magnitude')
    img_dict = Magnitude('image.magnitude')
    
    # create a file of processed words (no annotations of translation)
    for i in range(0, words.shape[0], 10):
        if "row" in words[i][0] or "column" in words[i][0]:
            word = words[i][0].split('-')[1]
        if "_" in words[i][0]:
            word_list = word.split('_')
            word = ""
            for i in range(len(word_list)):
                word += word_list[i]
                if i < len(word_list)-1:
                    word += " "
        # uncomment if haven't created words_processed 
        # with open('words_processed.txt', 'a') as f:
            # f.write("{}\n".format(word))
    # TODO: skip over words with all NaNs    

    # uncomment if haven't saved image embeddings to txt 
    # create_image_embedding()
    for i in range(0, words.shape[0], 10):
        # handle OOV words
        # convert word, e.g row-writings to writings 
        if "row" in words[i][0] or "column" in words[i][0]:
            phrase = words[i][0].split('-')[1]
        if "_" in words[i][0]:
            word_list = phrase.split('_')
            word = ""
            for i in range(len(word_list)):
                word += word_list[i]
                if i < len(word_list)-1:
                    word += " "
            phrase = word 
        word_embedding = word_dict.query(phrase)
        img_embedding = img_dict.query(words[i][0])

        # check if a word has valid image vectors 
        # valid: image vectors doesn't contain all NaNs
        # check_nan = np.isnan(img_embedding)
        # all_nan = check_nan[check_nan==True].shape[0]
        # if all_nan == img_embedding.shape[0]:
            
        # add to x_train and y_train
        try:
            x_train = np.vstack([x_train, word_embedding])
            y_train = np.vstack([y_train, img_embedding])
        except:
            x_train = word_embedding
            y_train = img_embedding 
    # uncomment if haven't created x_train, y_train
    # np.savetxt("x_train.txt", x_train)
    # np.savetxt("y_train.txt", y_train)

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
    return args

create_image_embedding()
