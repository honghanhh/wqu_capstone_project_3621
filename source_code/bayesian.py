# Retrieve training set
train_data = pd.read_csv("./data/train_data.csv", index_col=0)

# Define scoring methods
scoring_methods = {
    "BicScore": BicScore(train_data),
    "BDeuScore": BDeuScore(train_data),
    "BDsScore": BDsScore(train_data)
}

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

# Use the function to find the best model
best_model, best_method, best_score, best_iter = find_best_model(train_data, scoring_methods, max_iters=[5, 10])

print(f"\nThe best method is {best_method} with max_iter={best_iter} and score {best_score}")
print(f"The best model nodes: {sorted(best_model.nodes())}")
print(f"The best model edges: {best_model.edges()}")

# Fit the Bayesian Network with the best model
model_bayesian = BayesianNetwork(ebunch=best_model.edges())
model_bayesian.fit(train_data)

#Create a figure
fig = plt.figure(figsize=(5, 5))

#Plot K2_model
G1 = nx.DiGraph()
G1.add_edges_from(model_bayesian.edges())
pos1 = nx.spring_layout(G1, iterations=20)
nx.draw(G1, node_color='y', with_labels=True, edge_color='b', font_weight=0.5)
plt.title('Bayesian Network Graph')

#Show the plot
plt.show()

model_bayesian.get_markov_blanket('Close')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_value(model_struct, states):
    """
    Predicts the value for a given model and states.

    Parameters:
    model_struct (BayesianModel): The trained model.
    states (pd.DataFrame): The states data.

    Returns:
    pred_value (np.array): The predicted value.
    """
    try:
        column_names_df1 = set(model_struct.nodes())
        column_names_df2 = set(states.columns)
        columns_only_in_df2 = column_names_df2 - column_names_df1

        data_new = states.drop(columns=list(columns_only_in_df2) + ['forecast'], axis=1)

        logging.info(f'data_new columns: {data_new.columns}')
        logging.info(f'model nodes: {model_struct.nodes()}')

        prediction = model_struct.predict(data_new)
        pred_value = prediction['forecast'].to_numpy()
        print(f'Predicted value: {pred_value}')

        return pred_value

    except Exception as e:
        logging.error("Failed to predict values with error: %s", e)

def calculate_error(pred_value, real):
    """
    Calculates the error between predicted and real values.

    Parameters:
    pred_value (np.array): The predicted values.
    real (np.array): The real values.

    Returns:
    error (float): The error value.
    """
    try:
        error = np.mean(real != np.roll(pred_value, 1))
        print(f'\nError: {error * 100}%')
        logging.info(f'Error: {error * 100} %')

        return error

    except Exception as e:
        logging.error("Failed to calculate error with error: %s", e)

## Discretise the validation dataset and plot
discretise_data_with_hmm_and_save_csv(vald_data,'validation_data')
states_validation = pd.read_csv("./data/validation_data.csv", index_col=0)
states_validation.index = pd.to_datetime(states_validation.index)
plot_regime_switch(vald_data, states_validation, 'VALIDATION')

# Record real data observation, to be compared with the predicted one
validation_real = states_validation['Close'].to_numpy()

prediction_validation_bayesian = predict_value(model_bayesian,states_validation)
error_vald_bayesian = calculate_error(prediction_validation_bayesian,validation_real)
