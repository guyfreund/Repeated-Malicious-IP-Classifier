from utils import str2bool, prepare_data_for_model
from preprocessor import Preprocessor
from constants import FINALIZED_MODEL_FILE, X_TRAIN_FILE, Y_TRAIN_FILE, X_TEST_FILE, Y_TEST_FILE, PICKLES_DIR

import argparse
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from imblearn.ensemble import BalancedRandomForestClassifier


def args_handler():
    parser = argparse.ArgumentParser(description='Train on new data')
    parser.add_argument('-p', '--path-to-json', help='path to json containing all examples')
    parser.add_argument('-sm', '--save-model', default=True, type=str2bool, help='Whether to save the model or not')
    parser.add_argument('-sd', '--save-data', default=False, type=str2bool, help='Whether to save the train and test data or not')
    return parser.parse_args()


def get_train_test_data(processed_data, save_data):
    """ Processing the data and splitting into test & train sets differentiated by IPs """

    log(processed_data["label"].value_counts())

    # getting all unique ideas to create the data sets
    unique_ips = processed_data.drop_duplicates(subset=["ip"])
    X = unique_ips.loc[:, processed_data.columns != "label"]
    y = unique_ips["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    train_ips = X_train["ip"].unique()
    test_ips = X_test["ip"].unique()

    train = processed_data[processed_data["ip"].isin(train_ips)]  # all rows that has the train ips
    test = processed_data[processed_data["ip"].isin(test_ips)]  # all rows that has the test ips

    X_train = train.loc[:, processed_data.columns != "label"]
    y_train = train["label"]

    X_test = test.loc[:, processed_data.columns != "label"]
    y_test = test["label"]

    # remove ip columns and NaN columns
    prepare_data_for_model(X_train)
    prepare_data_for_model(X_test)

    if save_data:
        if not os.path.isdir(PICKLES_DIR):
            os.mkdir(PICKLES_DIR)

        X_train.to_pickle(X_TRAIN_FILE)
        X_test.to_pickle(X_TEST_FILE)
        y_train.to_pickle(Y_TRAIN_FILE)
        y_test.to_pickle(Y_TEST_FILE)

    log(f' ========== X_train size: {len(X_train)}, y_train size: {len(y_train)} ========== '
        f'X_test size: {len(X_test)}, y_test size: {len(y_test)} ==========')

    return X_train.values, X_test.values, y_train.values.astype(int), y_test.values.astype(int)


def train(X_train, y_train, save_model):
    model = BalancedRandomForestClassifier(random_state=4)
    model.fit(X_train, y_train)

    if save_model:
        joblib.dump(model, open(FINALIZED_MODEL_FILE, 'wb'))
        log(f'Saved model => {FINALIZED_MODEL_FILE}')

    return model


def evaluate_model(model, X_test, y_test):
    test_preds = model.predict(X_test).round()
    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    log(f"precision_score on test: {precision}")
    log(f"recall_score on test: {recall}")
    fps = ((y_test != test_preds) & (test_preds == 1)).sum()
    total_negatives = (y_test == 0).sum()
    fpr = fps / total_negatives
    log(f"num fps: {fps}, fpr: {fpr}")


def log(msg):
    print(f'{str(datetime.now())} Train: {msg}')


def run_trainer(path_to_json, save_model, save_data):
    preprocessor = Preprocessor(path_to_json=path_to_json, save_pickles=save_data)
    processed_data = preprocessor.process()

    X_train, X_test, y_train, y_test = get_train_test_data(processed_data, save_model)
    model = train(X_train, y_train, save_model)
    evaluate_model(model, X_test, y_test)


def main():
    args = args_handler()
    path_to_json = args.path_to_json
    save_model = args.save_model
    save_data = args.save_data

    run_trainer(path_to_json, save_model, save_data)


if __name__ == '__main__':
    main()
