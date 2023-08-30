cd source_code
python crawl_data.py --output_folder '../data/historical_data/' 
python preprocess_data.py --input_folder '../data/historical_data/' --output_file '../data/preprocessed_data/'
python clean_data.py --input_data '../data/preprocessed_data/preprocessed_data.csv' --output_folder '../data/cleaned_data/'
python model_hmm.py --train_val_test_folder '../data/cleaned_data/' 
python regime_switch_plot.py --data_version 'train_data.csv'
python regime_switch_plot.py --data_version 'validation_data.csv'
python regime_switch_plot.py --data_version 'test_data.csv'
python bayesian.py  --train_data '../data/hmm_data/train_data.csv' --val_data '../data/hmm_data/validation_data.csv' --test_data '../data/hmm_data/test_data.csv' 
python markov.py  --cleaned_data '../data/cleaned_data/test_data.csv' --hmm_data '../data/hmm_data/test_data.csv' 
python eval.py 
