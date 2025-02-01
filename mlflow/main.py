import mlflow
from mlflow.models import infer_signature

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load iris dataset
X,y = datasets.load_iris(return_X_y=True)

#split the dataset to 2 parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# set hyperparameters of logistic Regression
params = {"solver": "lbfgs", "C": 1.2, "random_state": 64}

mlflow.set_tracking_uri("http://127.0.0.1:8080")

#set the experiment
mlflow.set_experiment("iris_experiment")

with mlflow.start_run():
    # log params
    mlflow.log_params(params)
    
    # train the model
    clf = LogisticRegression(**params)
    clf.fit(X_train, y_train)

    # infer model signature
    signature = infer_signature(X_train, clf.predict(X_train))

    # log model
    mlflow.sklearn.log_model(sk_model=clf, artifact_path = "iris_model",
                             signature=signature, registered_model_name="iris_model",
                             input_example=X_train)

    # make predictions 
    y_pred = clf.predict(X_test)

    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    # log metrics
    mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})