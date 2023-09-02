# Import libraries
import os
import hmms
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures

import warnings
warnings.filterwarnings("ignore")

# Set up logging
import logging
logging.basicConfig(filename="training.log", level=logging.INFO)

#############################################################################################################################################################################################################
# Discretising data using Hidden Markov Models (HMMs)
#############################################################################################################################################################################################################

def train_and_save_model(series_id, train_data):
    best_score = -np.inf
    best_model = None
    # best_n_init = None
    # best_n_iter = None
    # best_n_states = None
    # best_n_observables = None

    # Define range of n_init and n_iter values to try
    n_init_values = [10, 20, 30, 40, 50]
    n_iter_values = [100, 150, 200, 250, 300]

    # Define range of hidden states and observable states
    n_states_values = [2, 3, 4, 5]
    n_observables_values = [2, 3, 4, 5]

    data_diff = train_data[series_id].diff()[1:]
    bins = np.histogram_bin_edges(data_diff, bins='auto')  # Determine bin edges based on data
    emit_seq = np.digitize(data_diff, bins)  # Use digitize to get bin indices for each value

    for n_states in n_states_values:
        for n_observables in n_observables_values:
            for n_init in n_init_values:
                for n_iter in n_iter_values:
                    for i in range(n_init):
                        dhmm = hmms.DtHMM.random(n_states, n_observables)

                        # Try to fit the model to the data
                        try:
                            dhmm.baum_welch(emit_seq, n_iter)
                            score = dhmm.log_likelihood(emit_seq)

                            # If this model is better than the previous best, update best_score and best_model
                            if score > best_score:
                                best_score = score
                                best_model = dhmm
                                # best_n_init = n_init
                                # best_n_iter = n_iter
                                # best_n_states = n_states
                                # best_n_observables = n_observables
                        except Exception as e:
                            # If the model fails to fit the data, just skip it
                            logging.info(f"Training failed for series {series_id} on initialization {i} with error {str(e)}")
                            pass

    # If no model was successfully trained, create and save a random model
    if best_model is None:
        best_model = hmms.DtHMM.random(3,2)

    path = "./hmms/" + series_id.replace(".", "_")
    best_model.save_params(path)

    # Save log likelihood, best_n_init, best_n_iter, best_n_states, and best_n_observables
    # with open(f"{path}_score.txt", "w") as f:
    #     f.write(f"Best score: {best_score}\nBest n_init: {best_n_init}\nBest n_iter: {best_n_iter}\nBest n_states: {best_n_states}\nBest n_observables: {best_n_observables}")

    
def discretise_data_with_hmm_and_save_csv(data, data_type):
    if not os.path.exists('../data/hmm_data/'):
        os.makedirs('../data/hmm_data/')

    disc_test = pd.DataFrame(index = data[1:].index)

    for series_id in data.columns:
        path = "./hmms/" + series_id.replace(".", "_") + ".npz"

        if series_id == 'forecast':
            dhmm = hmms.DtHMM.from_file('./hmms/Close.npz')
        else:
            dhmm = hmms.DtHMM.from_file(path)

        data_diff = data[series_id].diff()[1:]
        emit_seq = np.array(data_diff.apply(lambda x: 1 if x > 0 else 0).values)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
        # print(type(emit_seq))
        # print(emit_seq)
        _, s_seq = dhmm.viterbi(emit_seq)
        disc_test[series_id] = s_seq

    disc_test.to_csv(f'../data/hmm_data/{data_type}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_val_test_folder", type=str, default='../data/cleaned_data/', help='Path to the input folder')
    args = parser.parse_args()

    #####################
    # Start the progress
    #####################
    print("Discretising the data...")

    train_data = pd.read_csv(args.train_val_test_folder + 'train_data.csv', index_col=0)
    val_data = pd.read_csv(args.train_val_test_folder + 'validation_data.csv', index_col=0)
    test_data = pd.read_csv(args.train_val_test_folder + 'test_data.csv', index_col=0)
    print('The shape of train data is: ', train_data.shape)
    print('The shape of validation data is: ', val_data.shape)
    print('The shape of test data is: ', test_data.shape)

    # if not os.path.exists("./hmms"):
    #     os.makedirs("./hmms")

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     series_ids = [col for col in train_data.columns if col != 'forecast']
    #     futures = {executor.submit(train_and_save_model, series_id, train_data) for series_id in series_ids}

    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Training models"):
    #         pass

    # Discretise the data using the trained models and save to csv
    # discretise_data_with_hmm_and_save_csv(train_data, 'train_data')
    # discretise_data_with_hmm_and_save_csv(val_data, 'validation_data')
    # discretise_data_with_hmm_and_save_csv(test_data, 'test_data')


    #####################
    # End the progress
    #####################
    print("Discretising the data...Done")
