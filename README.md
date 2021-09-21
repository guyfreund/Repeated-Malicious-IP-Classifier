# Repeated-Malicious-IP-Classifier

<!--- ![nucleon_logo](https://user-images.githubusercontent.com/45962209/132684129-7f859f56-2ddd-48ca-9381-4f5c7ffa9540.png)
![reichman_logo](https://user-images.githubusercontent.com/45962209/132684855-f175a197-743e-4f04-991a-0e1b845bce76.png) --->

This repository is the implementation of a final project in Projects with the industry course taken by Guy Freund & Rotem Shalev from The department of Computer Science, Reichman University, Israel.<br>
This project is jointly guided by the Reichman University and the [Nucleon Cyber company](https://nucleoncyber.com). <br>
The project's goal is to create a classifier that will provide a prediction on whether an IP address will attack again or not. <br>
The data that is being used was given by the Nucleon Cyber company.

## Database Format
The structure of the data we use is a json, where each entry in it is part of a "Session" representing an attack. <br>
We aggregate each attack session into a single entry, preprocess it using the preprocessor,
and label the data such that each ip address that attacks more than once, gets a label of 1, or 0 otherwise.

## Files
1. `Pipfile` & `Pipfile.lock` - Virtual environment, optional.
2. `requirements.txt` - The requirements file.
3. `constans.py` - Shared constants file.
4. `utils.py` - Shared functions file.
5. `finalized_model.joblib` - The weights of the final model (after training).
6. `preprocessor.py` - A class that represents a Preprocessor object (used for processing raw data).
7. `train.py` - A Python script used to train the model from scracth (raw data to final model).
8. `predict.py` - A Python script used to give a prediction on a single example (database entry).
9. `EDA.ipynb` - A Jupyter Notebook that contains all the Exploratory Data Analysis and Model Selection (with plots).

## Usage
1. Install Python (version>=3.8.0)
2. Run: `pip install -r requirements.txt` or `pipenv shell`
3. For training run: `python3 train.py -p <path_to_json>` (add -sm, -sd if you want to change the defaults, see train.py for more information)
4. For prediction using existing model run: `python3 predict.py -p <path_to_json>` (add -sd true if you want to save the processed data)
