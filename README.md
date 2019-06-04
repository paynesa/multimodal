# Multimodal Embeddings
This code offers a collection of models which can be trained to create multimodal embeddings for a variety of applications. We also offer options to 

You should place all of your unprocessed image embeddings in .txt files in single directory with nothing else in it. These text files must be readable by [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html). 

Your word embeddings should be in the [magnitude](https://www.cis.upenn.edu/~ccb/publications/magnitude-fast-efficient-vector-embeddings-in-python.pdf) format, which can be achieved by converting them from the command line following the directions in the link above (you should name your output file word.magnitude). We recommend that you place word embeddings in a directory with nothing else in it (seperate from the location of your unprocessed image embeddings). It will then be possible to save your processed image embeddings to this directory, as well as to eventually place your training sets here.

## load_data.py
This file loads and cleans your data before creating the training sets for the model creation. 

load_data.py takes in 3 mandatory command-line arguments, and one optional argument. They are as follows: 
1. The path to the directory in which you placed your unprocessed image embeddings
2. The path to the directory where you would like the processed magnitude files and training sets to be saved
3. How you would like the image embeddings to be processed:
* 'avg' if you would like to average all vectors corresponding to the same word
* 'iter' if you would like vectors corresponding to the same words to appear separately in the training set
4. Optional: the location of the word embeddings, if it is different from the location given in #2

#### Example 1:
```
python3 load_data.py /home/data /home/results 'avg'
```
Will load the image embeddings located in the 'data' folder and save the processed embeddings and training sets to 'results.' Repeated words' embeddings will be averaged. No 4th argument was given, so the word embeddings are located in 'results.'

#### Example 2: 

```
python3 load_data.py /home/data /home/results 'iter' /home/embeddings
```
Will load the image embeddings located in the 'data' folder and save the processed embeddings and training sets to 'results.' Repeated words will not be averaged. A 4th argument was given, so the word embeddings are located in 'embeddings.'

##### Output
You will be notified as each file containing image embeddings is processed. Once all of the files have been processed, they will be converted to the magnitude format, and you will be notified of this as well. Finally, upon successful conversion, the training sets X_TRAIN and Y_TRAIN will be created, and you will be ready to work with the model of your choice. 

## Authors
This code was developed by Minh Nguyen (Swarthmore) and Sarah Payne (University of Pennsylvania).
