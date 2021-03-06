### NN imports ###
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Activation, Dropout, Flatten, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

import tensorflow.keras.metrics as tkm
import tensorflow as tf
import random

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import pickle
import os
import glob
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

random.seed(2)        # Python
np.random.seed(2)     # numpy
#tf.set_random_seed(2) # tensorflow
tf.random.set_seed(2)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report

import time
from datetime import date, datetime

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# kernel_initializer = 'uniform'
# https://stackoverflow.com/questions/43235531/convolutional-neural-network-conv1d-input-shape
class NN():

    def __init__(self, n_features, n_classes):
        print("##################### Init NN #####################")
        self.N_FEATURES = n_features
        self.N_CLASSES  = n_classes
        self.METRICS    =  ['accuracy',
                              tkm.TruePositives(),
                              tkm.FalsePositives(name='fp'),
                              tkm.TrueNegatives(name='tn'),
                              tkm.FalseNegatives(name='fn'),
                              #tkm.BinaryAccuracy(name='accuracy'),
                              tkm.Precision(name='precision'),
                              tkm.Recall(name='recall'),
                              tkm.AUC(name='auc')]


        self.DATE = datetime.now().strftime("%d-%m_%H%M%S")
        create_dir(self.DATE)

    def prepare_data(self, X, y):
        print("##################### Prepare Data #####################")
        # np.bincount(np.hstack(y.values))
        tmp_X = np.vstack([v for v in X.accelerations])

        print("    *Oversampling*")
        smote = SMOTE('all', k_neighbors=3,random_state=2)
        tmp_X, tmp_y = smote.fit_sample(tmp_X, y.values)
        tmp_y = to_categorical(tmp_y, num_classes=self.N_CLASSES)

        print("    *Normalisation*")
        scaler = StandardScaler()
        tmp_X = scaler.fit_transform(tmp_X)


        self.seq_len = tmp_X.shape[1]
        self.X, self.X_val, self.y, self.y_val = train_test_split(tmp_X, tmp_y,
                                                                 test_size=0.15,
                                                                 random_state=2,
                                                                 shuffle=True)
        self.true_y_val = np.asarray(list(map(np.argmax, self.y_val)))
        self.true_y     = np.asarray(list(map(np.argmax, self.y)))
        self.CLASS_WEIGHTS = compute_class_weight('balanced',
                                                 np.unique(self.true_y),
                                                 self.true_y)

        self.X     = np.expand_dims(self.X, axis=2)
        self.X_val = np.expand_dims(self.X_val, axis=2)

    def make_model1(self):
        print("##################### Make Model #####################")
        model = Sequential()
        model.add(Conv1D(filters=20,
                         kernel_size=10,
                         input_shape=(self.seq_len, 1),
                         activation= 'relu',
                         )) #kernel_regularizer=regularizers.l2(0.002)
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(rate=0.3,seed=1))
        model.add(Dense(self.N_CLASSES, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.00001),
                      metrics=self.METRICS)
        self.model = model
        self.MODEL_NAME = "Model1"

    #https://github.com/ni79ls/har-keras-cnn/blob/master/20180903_Keras_HAR_WISDM_CNN_v1.0_for_medium.py

    # https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
    def make_model2(self):
        print("##################### Make Model #####################")
        model = Sequential()
        model.add(Conv1D(30, 10, activation='relu', input_shape=(self.seq_len, 1)))
        model.add(Conv1D(30, 10, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(60, 10, activation='relu'))
        model.add(Conv1D(60, 10, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(rate=0.5, seed=2))
        model.add(Dense(self.N_CLASSES, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=self.METRICS)
        self.model = model
        self.MODEL_NAME = "Model2"

    def make_model3(self):
        print("##################### Make Model #####################")
        model = Sequential()
        model.add(Conv1D(30, 10, activation='relu', input_shape=(self.seq_len, 1)))
        model.add(Conv1D(60, 10, activation='relu'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(rate=0.4, seed=2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(rate=0.4, seed=2))
        model.add(Dense(self.N_CLASSES, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=self.METRICS)

        self.model = model
        self.MODEL_NAME = "Model3"

    def fit(self, epochs, batch_size):
        print("##################### Fit Model #####################")
        #####
        #### use {} format
        ###
        checkpoint     = ModelCheckpoint(self.DATE + "/w_ckp_" + self.MODEL_NAME +  ".hdf5",
                                         monitor='val_loss',
                                         verbose=2,
                                         save_best_only=True,
                                         mode='min')

        early_stopping = EarlyStopping(monitor='val_loss',
                                        verbose=1,
                                        patience=6,
                                        mode='min',
                                        restore_best_weights=True)

        reduce_lr      = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.8,
                                           patience=4,
                                           #min_delta = 1e-04,
                                           verbose=2,
                                           mode='min')
        callbacks      = [early_stopping, checkpoint]

        start = time.time()
        self.history = self.model.fit(
                            self.X, self.y,
                            validation_data=(self.X_val, self.y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            #class_weight=self.CLASS_WEIGHTS,
                            verbose=2,
                            shuffle=False,
                            callbacks=callbacks)
        end = time.time()
        print("Running time: \n", (end - start)/60)


    def predict(self):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, D]
        depending on the activation function
        """
        print("##################### Make Prediction #####################")

        tmp_prediction = self.model.predict(self.X_val)

        ### softmax and categorical
        self.prediction = np.asarray(list(map(np.argmax, tmp_prediction)))

    def measure_performance(self, performance_function):
        print("##################### Measure Performance #####################")
        self.SCORE = performance_function(self.true_y_val, self.prediction)
        print("Validation score: \n", self.SCORE)

    def load_weights(self, weights_file):
        print("##################### Load Model #####################")
        self.model.load_weights(weights_file)

    def load_model_(self, model_file):
        return load_model(model_file)

    def save_model(self):
        print("##################### Save Model #####################")
        score = str(self.SCORE)[:4]
        self.model.save(self.DATE + "/SAVED_" + self.MODEL_NAME + "_" + score + ".hdf5")

    def save_history(self):
        print("##################### Save History #####################")
        pickle.dump(self.history.history,
                    open(self.DATE + "/history" + ".pickle", "wb" ))

    def save_prediction_to_csv(self, name):
        print("##################### Save Prediction #####################")
        df_predict = pd.DataFrame(columns=['id', 'y'])

        df_predict['id'] = np.arange(len(self.prediction))
        df_predict['y']  = self.prediction

        name0 = "CNN"
        name1 = name
        name2 = ""
        filename = "" + name0 + name1 + name2 + ".csv"
        df_predict.to_csv(filename)

    def plot_metrics(self):
        print("##################### Plot Metrics #####################")
        history = self.history
        for n, metric in enumerate(self.model.metrics_names):
            name = metric.replace("_"," ").capitalize()
            plt.subplot(5,2,n+1)

            plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_'+metric],
                     color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, 3])
            elif ("true_positives" in metric) or ("fn" in metric) or ("fp" in metric):
                plt.ylim([0,10000])
            elif ("tn" in metric):
                plt.ylim([0,20000])
            elif metric == 'auc':
                plt.ylim([0.8,1])
            else:
                plt.ylim([0,1])

            plt.legend()

    def show_confusion_matrix(self):
        print("##################### Confusiion Matrix #####################")
        matrix = metrics.confusion_matrix(self.true_y_val,
                                          self.prediction,
                                          labels=np.arange(self.N_CLASSES))
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot()

        heatmap = sns.heatmap(matrix,
                    cmap="coolwarm",
                    linecolor='white',
                    linewidths=1,
                    xticklabels=np.arange(self.N_CLASSES),
                    yticklabels=np.arange(self.N_CLASSES),
                    annot=True,
                    fmt="d", ax=ax)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()


    def test(self):
        print("##################### Test #####################")
        print(self.model.layers[0].get_config())
        print(self.model.layers[0].output)
        print(self.model.layers[0].get_weights())
        print(self.model.layers[0].weights)
