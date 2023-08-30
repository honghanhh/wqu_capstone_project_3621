# Import libraries
import os
import argparse
import numpy as np
random_seed = 42
np.random.seed(random_seed)
from utils import *
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BDeuScore, BDsScore, BicScore, HillClimbSearch

import pickle as pkl

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def find_best_model(train_data, scoring_methods, max_iters):
    best_model = None
    best_score = float('-inf')
    best_method = None
    best_iter = None

    # Initialise Hill Climbing Estimator
    hc = HillClimbSearch(train_data)

    for max_iter in max_iters:
        for method_name, scoring_method in scoring_methods.items():
            print(f"Training with {method_name} and max_iter={max_iter}")
            model = hc.estimate(scoring_method=scoring_method, max_iter=max_iter)
            score = scoring_method.score(model)

            # Print the score for each method and iteration
            print(f"{method_name} score with max_iter={max_iter}: {score}")

            if score > best_score:
                best_score = score
                best_model = model
                best_method = method_name
                best_iter = max_iter

    return best_model, best_method, best_score, best_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayesian Network')
    parser.add_argument('--train_data', type=str, default='../data/hmm_data/train_data.csv', help='Path to train data file')
    parser.add_argument('--val_data', type=str, default='../data/hmm_data/validation_data.csv', help='Path to train data file')
    parser.add_argument('--test_data', type=str, default='../data/hmm_data/test_data.csv', help='Path to train data file')
    args = parser.parse_args()

    #####################
    # Start the progress
    #####################
    print("Training Bayesian model...")

    # Retrieve training set
    train_data = pd.read_csv(args.train_data, index_col=0)

    # Define scoring methods
    scoring_methods = {
        "BicScore": BicScore(train_data),
        "BDeuScore": BDeuScore(train_data),
        "BDsScore": BDsScore(train_data)
    }

    # Use the function to find the best model
    best_model, best_method, best_score, best_iter = find_best_model(train_data, scoring_methods, max_iters=[5, 10])

    print(f"\nThe best method is {best_method} with max_iter={best_iter} and score {best_score}")
    print(f"The best model nodes: {sorted(best_model.nodes())}")
    print(f"The best model edges: {best_model.edges()}")

    # Fit the Bayesian Network with the best model
    model_bayesian = BayesianNetwork(ebunch=best_model.edges())
    model_bayesian.fit(train_data)

    # If the folder does not exist, create a new one
    if not os.path.exists('../models'):
        os.makedirs('../models')

    #Save the model into pkl file
    with open('../models/bayesian_model.pkl', 'wb') as f:
        pkl.dump(model_bayesian, f)

    #Create a figure
    fig = plt.figure(figsize=(5, 5))

    #Plot K2_model
    G1 = nx.DiGraph()
    G1.add_edges_from(model_bayesian.edges())
    pos1 = nx.spring_layout(G1, iterations=20)
    nx.draw(G1, node_color='y', with_labels=True, edge_color='b', font_weight=0.5)
    plt.title('Bayesian Network Graph')

    #Show the plot
    # plt.show()

    if not os.path.exists('./plots/bayesian'):
        os.makedirs('./plots/bayesian')
        
    # Save the plot
    fig.savefig('./plots/bayesian/Bayesian Network Graph.png')

    # Create folder if not exists
    model_bayesian.get_markov_blanket('Close')

    ## Discretise the validation dataset and plot
    states_validation = pd.read_csv(args.val_data, index_col=0)
    states_validation.index = pd.to_datetime(states_validation.index)

    # Record real data observation, to be compared with the predicted one
    validation_real = states_validation['Close'].to_numpy()

    prediction_validation_bayesian = predict_value(model_bayesian, states_validation)
    error_vald_bayesian = calculate_error(prediction_validation_bayesian, validation_real)
    print("The error of validation set using Bayesian methods: ", error_vald_bayesian)


    ## Discretise the test dataset and plot
    states_test = pd.read_csv(args.test_data, index_col=0)
    states_test.index = pd.to_datetime(states_test.index)

    # Record real data observation, to be compared with the predicted one
    test_real = states_test['Close'].to_numpy()

    prediction_test_bayesian = predict_value(model_bayesian, states_test)
    error_test_bayesian = calculate_error(prediction_test_bayesian, test_real)
    print("The error of test set using Bayesian methods: ", error_test_bayesian)

    result_df_bayesian = pd.DataFrame({
        'date': states_test.index,
        'forecast': prediction_test_bayesian,
        'close': test_real
    })

    print(result_df_bayesian)
    result_df_bayesian.to_csv('../data/bayesian_results.csv', index=False)
    
    #####################
    # End the progress
    #####################
    print("Training Bayesian model...Done")
