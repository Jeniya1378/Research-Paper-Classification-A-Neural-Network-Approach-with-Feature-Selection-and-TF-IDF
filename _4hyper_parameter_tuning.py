"""
//********************************************************************
// CSC790: Information Retrieval and Web Search
// Project Title: Research Paper Classification
// by Team 1 (Sujung Choi, Jeniya Sultana)
// May 2, 2024
//
// Description: This file is used to conduct hyperparameter tuning for the neural network model.
// Note: Please install the package 'keras-tuner' before running this code.
// type 'pip install keras-tuner' in the terminal to install the package.
//********************************************************************
"""
from kerastuner.tuners import RandomSearch
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.feature_selection import SelectKBest, chi2
import random
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras import regularizers

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

def f1_score_metric(y_true, y_pred):
    """
    # f1_score_metric function calculates the F1 score metric for the model.
    """
    y_pred = tf.round(y_pred)
    true_positives = tf.keras.backend.sum(tf.round(y_true) * tf.round(y_pred))
    predicted_positives = tf.keras.backend.sum(tf.round(y_pred))
    possible_positives = tf.keras.backend.sum(tf.round(y_true))
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
     
    return f1


def create_model(input_size, num_classes, learning_rate=0.001, hidden_units1=400, hidden_units2=200, hidden_units3=100):
    """
    # create_model function creates a neural network model with the specified input size, number of classes, learning rate, and hidden units.
    """
    inputs = Input(shape=(input_size,))  
      
    L1 = Dense(hidden_units1, activation='relu')(inputs)
    L2 = Dense(hidden_units2, activation='relu')(L1)
    L3 = Dense(hidden_units3, activation='relu')(L2)

    L4 = Dense(num_classes, activation='softmax')(L3)
        
    nn_model = Model(inputs=inputs, outputs=L4)
    
    # compile the model with the specified loss function, optimizer, and metrics
    nn_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    nn_model.summary() # print the model summary
        
    return nn_model

def build_model(hp, input_size, num_classes):
    """
    # build_model function uses different hyperparameters to find the best model.
    # hp is the hyperparameters object in the keras-tuner library.
    # it sets the number of units for each layer and the learning rate as hyperparameters.
    """
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', min_value=200, max_value=600, step=200), activation='relu', input_shape=(input_size,)))
    model.add(Dense(units=hp.Int('units2', min_value=100, max_value=300, step=100), activation='relu'))
    model.add(Dense(units=hp.Int('units3', min_value=50, max_value=150, step=50), activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def save_models(trained_model, file_name):
    """
    # save_models function saves the trained model to a file.
    """
    trained_model.save(file_name)

def main():

    # set the number of classes
    num_classes = 4

    # load the data from the numpy files
    X = np.load('X_4car.npy')
    y = np.load('y_4car.npy')

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # set best k number of features chosen by the chi2 method
    K = 1000

    # Create a selector object that will use the chi2 metric to select the best K features
    selector = SelectKBest(score_func=chi2, k=K) 

    # Fit the selector to training data
    selector.fit(X_train, y_train)

    # Transform training data to select only the top k features
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Perform hyperparameter tuning using the keras-tuner library
    tuner = RandomSearch(
        lambda hp: build_model(hp, X_train_selected.shape[1], num_classes),
        objective='val_accuracy',
        max_trials=10, # number of hyperparameter combinations to try
        executions_per_trial=3, # number of models to train per trial
        directory='my_dir', # save the results in this directory
        project_name='nn_tuning')

    # search for the best hyperparameters
    tuner.search(X_train_selected, y_train, epochs=30, validation_split=0.2)
    
    # Get the best hyperparameters and results
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Get the values of the best hyperparameters
    best_units1 = best_hyperparameters.get('units1')
    best_units2 = best_hyperparameters.get('units2')
    best_units3 = best_hyperparameters.get('units3')
    best_learning_rate = best_hyperparameters.get('learning_rate')

    # Print the best hyperparameters and results
    print("Best Hyperparameters")
    print("1. Best number of units for layer 1:", best_units1)
    print("2. Best number of units for layer 2:", best_units2)
    print("3. Best number of units for layer 3:", best_units3)
    print("4. Best learning rate:", best_learning_rate)

    best_accuracy = tuner.oracle.get_best_trials(1)[0].score  
    print("Best Mean Accuracy:", round(best_accuracy, 3))


if __name__ == "__main__":
    main()