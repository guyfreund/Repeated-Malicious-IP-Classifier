from preprocessor import Preprocessor
from constants import FINALIZED_MODEL_FILE
from utils import prepare_data_for_model, str2bool

import argparse
import os
import joblib
from datetime import datetime


def args_handler():
    parser = argparse.ArgumentParser(description='Predict on single example')
    parser.add_argument('-p', '--path-to-json', help='path to json containing one example')
    parser.add_argument('-sd', '--save-data', default=False, type=str2bool, help='Whether to save the processed data or not')
    return parser.parse_args()


def log(msg):
    print(f'{str(datetime.now())} Predict: {msg}')


def run_predictor(path_to_json, save_data):
    preprocessor = Preprocessor(path_to_json=path_to_json, save_pickles=save_data, add_label=False)
    processed_data = preprocessor.process()
    prepare_data_for_model(processed_data)

    if not os.path.exists(FINALIZED_MODEL_FILE):
        raise Exception(f'{FINALIZED_MODEL_FILE} does not exist, could not predict.')
    model = joblib.load(FINALIZED_MODEL_FILE)

    prediction = model.predict(processed_data.values).round()
    log(prediction)


def main():
    args = args_handler()
    path_to_json = args.path_to_json
    save_data = args.save_data

    run_predictor(path_to_json, save_data)


if __name__ == '__main__':
    main()
