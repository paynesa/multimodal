"""
A playground for building models
"""
from load_data import parse_args

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np

class MultimodalEmbedding:
    def _init__(self, x_train, y_train, args):
        self.x_train = x_train 
        self.y_train = y_train
        self.args = args
        self.model = None

    def build_linear_model(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(300,)))
        self.model.add(Dropout(0.1)) 
        self.model.summary()

        sgd = optimizers.SGD(lr=self.args.lr)
        self.model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["accuracy"])

    def build_neural_net(self):
        self.model = Sequential()
        self.model.add(Dense(self.args.u, activation="tanh", input_shape=(300,)))
        self.model.add(Dropout(0.25)) #is this where dropout layer should be??
        self.model.add(Dense(128))
        self.model.summary()

        sgd = optimizers.SGD(lr=0.1)
        self.model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["accuracy"])
    
    def start_training(self, model):
        if model == "linear":
            build_linear_model()
        elif model == "neural":
            build_neural_net()
        else:
            # TODO: edit error message
            printf("model: linear/neural")
        
        print("Training initialized...")
        history = self.model.fit(self.x_train, self.y_train, self.args.e, verbose=1)
        print("Training complete")

        try:
            self.model.save(self.args.s)
            print("Model saved")
        except:
            raise Exception("Error saving model")

    def predict(self):
        try:
            # self.args.s: train, save, then load the model in one go 
            # self.args.l: load an old model for prediction
            if self.args.s:
                self.model = load_model(self.args.s)
            elif self.args.l:
                self.model = load_model(self.args.l)
            
            learned_embedding = self.model.predict(self.x_train)
            # TODO: figure out how to incorporate learned image embeddings and fused 
            # embeddings into the dataset
        except:
            raise Exception("Error loading model")

    def cosine_similarity(word1, word2):
        dot_product = np.dot(word1, word2)
        length_word1 = np.linalg.norm(word1)
        length_word2 = np.linalg.norm(word2)
        return dot_product/(length_word1 * length_word2)

def main():
    args = parse_args()
    print(args)
    model = MultimodalEmbedding(x_train, y_train, args)
    # train, save and load model in one go
    if args.s:
        model.start_training(args.model)
        model.predict()
    # load an old model for prediction
    elif args.l:
        model.predict()

if __name__ == '__main__':
    main()

