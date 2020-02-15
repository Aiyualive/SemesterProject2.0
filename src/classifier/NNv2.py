### NN imports ###
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Activation, Dropout, Flatten, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import tensorflow.keras.metrics as tkm
import tensorflow as tf

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import time
from datetime import date, datetime
import random
import pickle
import os
import glob
import re
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

random.seed(2)        # Python
np.random.seed(2)     # numpy
tf.set_random_seed(2) # tensorflow
#tf.random.set_seed(2)

def create_dir(folder):
  if not os.path.exists(folder):
    os.makedirs(folder)

#############
### ModelMaker class ###
#############
class ModelMaker():
    # seq len, n classes, metrics
    def __init__(self, seq_len, n_classes, metrics):
        self.seq_len   = seq_len
        self.n_classes = n_classes
        self.metrics   = metrics

    def model_chooser(self, model_name, verbose = 2):
        """
        Returns the chosen model
        params:
            model_name: name of model
        """
        # Could instead use a dictionary containing lambda funcs?
        print("########## Make " + model_name + " ##########")
        if model_name == "model1":
            self._makeModel1()
        elif model_name == "model2":
            self._makeModel2()
        elif model_name == "model3":
            self._makeModel3()
        elif model_name == "model4":
            self._makeModel4()
        else:
            raise Warning("Model name does not exist")

        if verbose:
            self.model.summary()

        return self.model

    def _makeModel1(self):
        model = Sequential()
        model.add(Conv1D(filters=20,
                         kernel_size=10,
                         input_shape=(self.seq_len, 1),
                         activation= 'relu')) #kernel_regularizer=regularizers.l2(0.002)
        model.add(MaxPooling1D(pool_size=3))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(rate=0.3,seed=1))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.0001),
                      metrics=self.metrics)
        self.model = model

    def _makeModel2(self):
        """
        https://github.com/ni79ls/har-keras-cnn/blob/master/20180903_Keras_HAR_WISDM_CNN_v1.0_for_medium.py
        https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
        """
        model = Sequential()
        model.add(Conv1D(30, 10, activation='relu', input_shape=(self.seq_len, 1)))
        model.add(Conv1D(30, 10, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(60, 10, activation='relu'))
        model.add(Conv1D(60, 10, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(rate=0.5, seed=2))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=self.metrics)
        self.model = model

    def _makeModel3(self):
        model = Sequential()
        model.add(Conv1D(30, 10, activation='relu', input_shape=(self.seq_len, 1)))
        model.add(Conv1D(60, 10, activation='relu'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(rate=0.4, seed=2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(rate=0.4, seed=2))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=self.metrics)
        self.model = model

    def _makeModel4(self):
        model = Sequential()
        model.add(Conv1D(filters=20,
                         kernel_size=10,
                         input_shape=(self.seq_len, 1),
                         activation= 'relu'))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Conv1D(filters=20,
                         kernel_size=10,
                         activation= 'relu'))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(rate=0.3,seed=1))
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.01),
                      metrics=self.metrics)
        self.model = model

################
### NN class ###
################
class NN():
    def __init__(self, epochs, batch_size, create_dir_bool = True, verbose = 2):
        print("########## Init NN ##########")
        self.metrics = [tkm.CategoricalAccuracy(name='ACC'),
                        #'accuracy',
                        tkm.TruePositives(name='TP'),
                        tkm.FalsePositives(name='FP'),
                        tkm.TrueNegatives(name='TN'),
                        tkm.FalseNegatives(name='FN'),
                        tkm.Precision(name='Precision'),
                        tkm.Recall(name='Recall'),
                        tkm.AUC(name='AUC')]

        self.exp_desc = ""
        self.verbose  = verbose
        self.epochs   = epochs
        self.batch_size = batch_size

        self.date = datetime.now().strftime("%d-%m_%H%M%S")
        if (create_dir_bool):
            create_dir(self.date)

    def prepare_data(self, X, y, oversampling = False):
        """
        Prepare data; normalisation, validation split
        """
        print("########## Prepare Data ##########")
        self.n_features = len(X.columns)
        self.n_classes  = len(np.unique(y))

        tmp_X = np.vstack([v for v in X.accelerations])
        # Maybe dont need flatten, minus 2 since labels start from 2
        tmp_y = to_categorical(y.values.flatten(), num_classes=self.n_classes)

        if oversampling:
            print("    *Oversampling*")
            smote = SMOTE('all', k_neighbors=3,random_state=2)
            tmp_X, tmp_y = smote.fit_sample(tmp_X, y.values)

        print("    *Normalisation*")
        scaler = StandardScaler()
        tmp_X = scaler.fit_transform(tmp_X)

        self.seq_len = tmp_X.shape[1]
        self.X, self.X_val, self.y, self.y_val = train_test_split(tmp_X, tmp_y,
                                                                 test_size=0.15,
                                                                 random_state=2,
                                                                 shuffle=True)

        self.true_y_val    = self._convertFromCategorical(self.y_val)
        self.true_y        = self._convertFromCategorical(self.y)
        self._printClassDist() # Could be done depending on verbose

        self.class_weights = compute_class_weight('balanced',
                                                 np.unique(self.true_y),
                                                 self.true_y)

        # Need to expand dim in order for conv1D to work
        self.X     = np.expand_dims(self.X, axis=2)
        self.X_val = np.expand_dims(self.X_val, axis=2)

    def make_model(self, model_name):
        """
        Makes a specified model from, using ModelMaker class
        params:
            model: the model to be made
        """
        model_maker = ModelMaker(self.seq_len, self.n_classes, self.metrics)
        self.model = model_maker.model_chooser(model_name, self.verbose)
        self.model_name = model_name
        self._set_callbacks() #self.verbose not set in here

    def fit(self):
        """
        Fit the model; define callbacks
        """
        print("########## Fit Model ##########")
        start = time.time()
        self.history = self.model.fit(
                            self.X, self.y,
                            validation_data=(self.X_val, self.y_val), # categorical
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            class_weight=self.class_weights,
                            verbose=self.verbose,
                            shuffle=False,
                            callbacks=self.callbacks)
        end = time.time()
        print("Running time: ", (end - start)/60)

    def classify(self):
        """
        Computes predictions for the validation set
        """
        print("########## Make Prediction ##########")
        tmp_prediction = self.model.predict(self.X_val)
        self.prediction = self._convertFromCategorical(tmp_prediction)

    def measure_performance(self):
        """
        Evaluate model on validation set.
        """
        print("########## Measure Performance ##########")
        self.results = self.model.evaluate(self.X_val, self.y_val,
                                           batch_size=self.batch_size,
                                           verbose=self.verbose)

    def load_weights(self, weights_file):
        """
        Loads weights into the same model that the weights was created from.
        So the prevous steps of creating the correct model has to be performed
        """
        print("########## Load Model ##########")
        self.model.load_weights(weights_file)

    def load_model_(self, model_file):
        """
        Model_file has to be generated with save_model; have not tested.
        """
        print("########## Load Model ##########")
        self.model = load_model(model_file)

    def save_model(self):
        print("########## Save Model ##########")
        self.model.save(self.date + "/SAVE_MODEL_%s_%s_%3.2f.hdf5"%(self.model_name, self.exp_desc, self.results[1]))

    def save_history(self):
        """
        Saves the training history into pickle file
        """
        print("########## Save History ##########")
        pickle.dump(self.history.history,
                    open(self.date + "/SAVE_HISTORY_%s_%s.pickle"%(self.model_name, self.exp_desc), "wb" ))

    def save_classification_to_csv(self, name):
        print("########## Save Prediction ##########")
        df_predict = pd.DataFrame(columns=['id', 'y'])

        df_predict['id'] = np.arange(len(self.prediction))
        df_predict['y']  = self.prediction

        name0 = "CNN"
        name1 = name
        name2 = ""
        filename = "" + name0 + name1 + name2 + ".csv"
        df_predict.to_csv(filename)

    def plot_metrics(self):
        print("########## Plot Metrics ##########")
        history = self.history
        for n, metric in enumerate(self.model.metrics_names):
            fig = plt.figure(figsize=(20,25))
            #name = re.sub(r"\d+", "", metric.replace("_", " ").capitalize())
            name = re.sub(r"\d+", "", metric.replace("true_positives_", "Tp").upper())

            train_history = history.history[metric]
            val_history   = history.history['val_' + metric]
            ylim0 = min(train_history + val_history)
            ylim1 = max(train_history + val_history)

            #print("lims")
            #print(ylim0, ylim1)

            #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places

            plt.subplot(5, 2, n + 1)
            plt.plot(history.epoch, train_history,
                     color=colors[0], label='Train')
            plt.plot(history.epoch, val_history,
                     color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.ylim([ylim0, ylim1])
            plt.legend()
            fig.set_size_inches(20, 25, forward=True)
            #fig.savefig(self.date + "/METRIC_%s_%s"%(metric, self.exp_desc),  bbox_inches='tight')

    def plot_confusion_matrix(self,):
        print("########## Confusiion Matrix ##########")
        matrix = confusion_matrix(self.true_y_val,
                                  self.prediction,
                                  labels=np.arange(self.n_classes))
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot()

        heatmap = sns.heatmap(matrix,
                    cmap="coolwarm",
                    linecolor='white',
                    linewidths=1,
                    xticklabels=np.arange(self.n_classes),
                    yticklabels=np.arange(self.n_classes),
                    annot=True,
                    fmt="d", ax=ax)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()
        fig.savefig(self.date + "/CONFMAT_%s"%(self.exp_desc),  bbox_inches='tight')

    def test(self):
      print("########## Test ##########")
      print(self.model.layers[0].get_config())
      print(self.model.layers[0].output)
      print(self.model.layers[0].get_weights())
      print(self.model.layers[0].weights)

    """Inspired by https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/"""
    def run_experiment(self, repetitions=10):
        scores = []
        experiment_history = []
        self.verbose = 0
        for i in range(repetitions):
            print("******************* EXP " + str(i) + " *******************")
            self.exp_desc = "EXP" + str(i)
            self.fit()
            self.classify()
            self.save_history()
            self.measure_performance()
            self.save_model()
            #self.plot_metrics()
            #self.plot_confusion_matrix()
            for n, metric in enumerate(self.model.metrics_names):
                print(">%d %s: %3.2f"%(i, metric, self.results[n]))
            scores.append(self.results)
            experiment_history.append(self.history)
            print("*********************************************\n")
        self._summarize_results(scores)
        self._summarize_plots(experiment_history)

    def _set_callbacks(self):
        checkpoint     = ModelCheckpoint(self.date + "/w_ckp_%s.hdf5"%(self.model_name),
                                        monitor='val_loss',
                                        verbose=0,
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

        self.callbacks      = [early_stopping, checkpoint]

    def _convertFromCategorical(self, arr):
        """
        Convert array from categorical to ordinary class integers
        """
        return np.asarray(list(map(np.argmax, arr)))

    def _printClassDist(self):
        true_cnt = np.bincount(self.true_y)
        true_pct = true_cnt/sum(true_cnt)
        true_cl  = np.nonzero(true_pct)[0]

        val_cnt = np.bincount(self.true_y_val)
        val_pct = val_cnt/sum(val_cnt)
        val_cl  = np.nonzero(val_pct)[0]

        dist_file = self.date + "/class_dist.txt"
        with open(dist_file, "w") as f:
            print("Traning set class distribution")
            f.write("Traning set class distribution")
            for i in true_cl:
                f.write("    %d: %d, %3.2f%%"%(i, true_cnt[i], true_pct[i]*100))
                print("    %d: %d, %3.2f%%"%(i, true_cnt[i], true_pct[i]*100))
            print()

            print("Validation set class distribution")
            f.write("Validation set class distribution")
            for i in val_cl:
                f.write("    %d: %d, %3.2f%%"%(i, val_cnt[i], val_pct[i]*100))
                print("    %d: %d, %3.2f%%"%(i, val_cnt[i], val_pct[i]*100))
            print()

    def _summarize_results(self, scores):
        print(">>> Summarize results <<<")
        m, s = np.mean(scores, axis=0), np.std(scores, axis=0)
        print(scores)
        for i, metric in enumerate(self.model.metrics_names):
            print('Overall %s: %.3f%% (+/-%.3f)' % (metric, m[i], s[i]))
        print()


    def _summarize_plots(self, history):
        print(">>> Summarized training plots <<<")
        # exp: metric for each experiment
        train_metrics = { i: [] for i in self.model.metrics_names} # for getting the ylimits
        val_metrics   = { 'val_' + i: [] for i in self.model.metrics_names}

        # Convert into metric: experiments[epochs[]]
        repetitions = len(history)
        for n, metric in enumerate(self.model.metrics_names):
            val_metric = 'val_' + metric
            for i in range(repetitions):
                train_vals = history[i].history[metric]
                val_vals   = history[i].history[val_metric]
                train_metrics[metric].append(train_vals)
                val_metrics[val_metric].append(val_vals)

        # metric: std/average for each epochs
        #print(train_metrics)
        #list(map(lambda x: (float(5)/9)*(x-32), fahrenheit.values()))
        train_avg = { key: np.mean(list(train_metrics[key]), axis=0) for key in train_metrics}

        train_std = { key: np.std(list(val), axis=0) for key, val in train_metrics.items()}

        val_avg = { key: np.mean(list(val), axis=0) for key, val in val_metrics.items()}
        val_std = { key: np.std(list(val), axis=0) for key, val in val_metrics.items()}


        # Average metric plot
        print(">>> Average metric plot")
        for n, metric in enumerate(self.model.metrics_names):
            val_metric = 'val_' + metric
            fig = plt.figure(figsize=(10,6))
            plt.errorbar(np.arange(self.epochs), train_avg[metric], train_std[metric],
                         marker = '.', linestyle='none', capsize=3, label = 'Train')
            plt.errorbar(np.arange(self.epochs)+0.01, val_avg[val_metric], val_std[val_metric],
                         marker = '.', linestyle='none', capsize=3, label = 'Val')
            plt.title('Average ' + metric)
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend(loc = 1)
            plt.show()

        # Metric plot for each experiment
        print(">>> Metric plot for each experiment")
        flatten = lambda l: [item for sublist in l for item in sublist]
        for i in range(repetitions):
            print(">>> Exp " + str(i))
            for n, metric in enumerate(self.model.metrics_names):
                val_metric = 'val_' + metric
                t = train_metrics[metric]
                v = val_metrics[val_metric]
                flat = flatten(t) + flatten(v)
                ybot, ytop = (min(flat), max(flat))

                fig = plt.figure(figsize=(10,6))
                plt.plot(np.arange(self.epochs), t[i], label='Train')
                plt.plot(np.arange(self.epochs), v[i], linestyle="--", label='Val')
                plt.title("Exp" + str(i) + ": " + metric)
                plt.xlabel('Epoch')
                plt.ylabel(metric)
                plt.ylim([ybot, ytop])
                plt.legend(loc = 1)
                plt.show()