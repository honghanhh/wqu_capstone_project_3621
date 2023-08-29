# Import libraries
import numpy as np
random_seed = 42
np.random.seed(random_seed)
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
