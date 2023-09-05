# Import libraries
import os
import argparse
import numpy as np
random_seed = 42
np.random.seed(random_seed)

import pandas as pd
import networkx as nx
from pgmpy.models import BayesianModel
from pgmpy.inference import BeliefPropagation

import matplotlib.pyplot as plt
import pickle as pkl

import warnings
warnings.filterwarnings("ignore")

def remove_disconnected_nodes(original_model: BayesianModel) -> BayesianModel:
    """
    Function to remove disconnected nodes from a Bayesian Model.
    It returns a new Bayesian Model, leaving the original one untouched.
    """
    model = original_model.copy()

    # Convert the model to an undirected graph
    undirected = model.to_undirected()

    # Get the list of connected components, sorted by size
    connected_components = sorted(list(nx.connected_components(undirected)), key=len, reverse=True)

    # Keep only the largest connected component
    largest_component = connected_components[0]

    # Get a list of nodes in the smaller components
    nodes_to_remove = [node for component in connected_components[1:] for node in component]

    # Remove the nodes in the smaller components from the original directed graph
    for node in nodes_to_remove:
        model.remove_node(node)

    return model

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Bayesian Network')
    argparser.add_argument('--cleaned_data', type=str, default='../data/cleaned_data/test_data.csv', help='Path to the data')
    argparser.add_argument('--hmm_data', type=str, default='../data/hmm_data/test_data.csv', help='Path to the test data')
    args = argparser.parse_args()

    #####################
    # Start the progress
    #####################
    print("Constructing Markov model...")

    # Load pkl model
    with open('../models/bayesian_model.pkl', 'rb') as f:
        model_bayesian = pkl.load(f)
    
    model_bayesian_connected = remove_disconnected_nodes(model_bayesian)

    # Convert Bayesian Network to Markov Network
    markov_model = model_bayesian_connected.to_markov_model()

    # If the folder does not exist, create a new one
    if not os.path.exists('../models'):
        os.makedirs('../models')

    #Save the model into pkl file
    with open('../models/markov_model.pkl', 'wb') as f:
        pkl.dump(markov_model, f)

    # Initialize the Belief Propagation class with the Markov Model
    bp = BeliefPropagation(markov_model)

    for cpd in model_bayesian_connected.get_cpds():
        print("CPD for {0}:".format(cpd.variable))
        print(cpd)

    # Create a new graph from the Markov model's edges
    G = nx.Graph()
    G.add_edges_from(markov_model.edges())

    # Draw the graph
    fig = plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # set layout
    nx.draw(G, pos, with_labels=True, font_weight='bold')

    plt.title('Markov Network Graph')
    # plt.show()

    # Create folder if not exists
    if not os.path.exists('./plots/markov'):
        os.makedirs('./plots/markov')

    # Save the plot
    fig.savefig('./plots/markov/Markov Model.png')


    # discretise test data and plot
    test_data = pd.read_csv(args.cleaned_data, index_col=0)
    states_test = pd.read_csv(args.hmm_data, index_col=0)
    
    # Initialize an empty DataFrame to store the results
    results_df_markov = pd.DataFrame()

    # Get a set of variable names from the model
    model_variables = set(model_bayesian_connected.nodes())
    print(model_variables)

    # Loop over all rows in the DataFrame
    for i in range(len(states_test)):
        # Get the current row and convert it to a dictionary
        evidence_row = test_data.iloc[i]
        evidence = evidence_row.to_dict()

        # # Remove 'Close' and forecast from the evidence
        if 'Close' in evidence:
            del evidence['Close']

        # Only keep variables in the evidence that are in the model
        evidence = {var: value for var, value in evidence.items() if var in model_variables and value in [0, 1, 2]}

        # Perform the inference
        result = bp.query(variables=['Close'], evidence=evidence)

        # Convert the result to a DataFrame and append it to the results DataFrame
        result_df = pd.DataFrame([result.values], columns=['prob_0', 'prob_1', 'prob_2'])
        results_df_markov = pd.concat([results_df_markov, result_df], ignore_index=True)

    results_df_markov['forecast'] = np.argmax(results_df_markov.values, axis=1)
    results_df_markov['date'] = states_test.index
    test_dataset = states_test.reset_index(drop=True)
    results_df_markov['close'] = test_dataset['Close']
    # print(results_df_markov)
    results_df_markov[['date', 'prob_0', 'prob_1', 'prob_2', 'forecast', 'close']].to_csv('../data/markov_results.csv', index=False)
    
    #####################
    # End the progress
    #####################
    print("Training Markov model...Done")
