cd source_code

# python crawl_data.py --start_date '2010-01-01' --end_date '2019-12-31' --output_folder '../data/historical_data/' # Unchecked

# python preprocess_data.py --input_folder '../data/historical_data/' --output_file '../data/preprocessed_data/' # DONE
# python clean_data.py --input_data '../data/preprocessed_data/preprocessed_data.csv' --output_folder '../data/cleaned_data/' # DONE

# python model_hmm.py --train_val_test_folder '../data/cleaned_data/' # ERROR

python regime_switch_plot.py --data_version 'train_data.csv' # DONE
python regime_switch_plot.py --data_version 'validation_data.csv' # DONE
python regime_switch_plot.py --data_version 'test_data.csv' # DONE

# python bayesian.py  # Unchecked
# python markov.py  # Unchecked
# python eval.py # Unchecked
