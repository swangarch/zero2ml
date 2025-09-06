from logregClass import logreg, save_weights
from train_util import count_correct
import numpy as np
from numpy import ndarray as array
import pandas as pd
from load_csv import load

class logregall:
    """logistic regression for training all features"""

    def __init__(self, df, feature_names, classname, goals):
        """Init logistic regression for all features"""

        self.df = df
        self.feature_names = feature_names
        self.lgs = []
        self.goals = goals
        self.classname = classname
        for i, goal in enumerate(self.goals):
            self.lgs.append(logreg(self.df, self.feature_names, self.classname, goal))
        
        self.weights = []

    def train_all(self, leanrning_rate=0.0005, max_iter=100000, Debug=False):
        """Train models for all class, save weights for classifcation."""

        for i, goal in enumerate(self.goals):
            # self.lgs.append(logreg(self.df, self.feature_names, self.classname, goal))
            weights = self.lgs[i].train(leanrning_rate, max_iter, Debug)
            self.weights.append(weights)

        print(self.weights)
        save_weights(np.array(self.weights), "weights.csv")
        
    def predict_test(self, df_test):
        """Predict for test dataset."""

        if len(self.weights) == 0:
            raise("No model weights yet.")
        predictions = []
        for i in range(len(self.lgs)):	
            predictions.append(self.lgs[i].predict(df_test))
        final_prediction = np.stack(predictions).transpose()

        final_index = []
        for i, arr in enumerate(final_prediction):
            max = 0
            max_j = 0
            for j, num in enumerate(arr):
                if num > max:
                    max_j = j
                    max = num
            final_index.append(max_j)

        print("[FINAL PREDICTION SAVED]")
        final = [self.goals[i] for i in final_index]
        truth = list(df_test[self.classname])

        print(count_correct(final, truth))

    def predict_new(self, df_new):
        """Predict for a new dataset."""

        if len(self.weights) == 0:
            raise("No model weights yet.")
        
        predictions = []
        for i in range(len(self.lgs)):	
            predictions.append(self.lgs[i].predict_new(df_new))
        final_prediction = np.stack(predictions).transpose()

        final_index = []
        for i, arr in enumerate(final_prediction):
            max = 0
            max_j = 0
            for j, num in enumerate(arr):
                if num > max:
                    max_j = j
                    max = num
            final_index.append(max_j)

        print("[FINAL PREDICTION SAVED]")

        final = []
        for count, index in enumerate(final_index):
            final.append([count, self.goals[index]]) 
        # print(final)
        df = pd.DataFrame(final, columns=["Index", "Hogwarts House"])
        df.to_csv("house.csv", index=False)

    def load_weights(self, path):
        """Load weights from file."""

        # df_weights = load(path)
        self.weights = np.loadtxt("weights.csv", delimiter=",", dtype=float).tolist()
        print(self.weights)
