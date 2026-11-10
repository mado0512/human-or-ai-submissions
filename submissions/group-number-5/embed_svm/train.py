import json
import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
import joblib

with open("submissions/group-number-5/embed_svm/embeddings.json") as f:
    data = json.load(f)

train_df = pd.read_csv("submissions/group-number-5/embed_svm/train.csv")
starter_df = pd.read_csv("submissions/group-number-5/embed_svm/train_starter_2000.csv")

starter_ids = set(starter_df["id"])
full_ids = set(train_df["id"])

def train_model(train_ids, model_path, **hyperparameters):
    X = np.array([data[str(i)]["embedding"] for i in train_ids])
    y = np.array([data[str(i)]["label"] for i in train_ids])

    model = SVC(**hyperparameters)

    start = time.time()
    model.fit(X, y)
    print(f"{model_path}: {time.time() - start:.2f}s")

    joblib.dump(model, model_path)


# Full model: train on everything except starter set
# Wait — full model trains on all data, so starter is its dev set.
"""train_model(
    train_df["id"],
    "submissions/group-number-5/embed_svm/SVM_full_train",
    kernel="rbf",
    C=1.0,
    probability=True
)"""

# Starter model: train only on starter set
train_model(
    starter_df["id"],
    "submissions/group-number-5/embed_svm/SVM_starter_train",
    kernel="rbf",
    C=1.0,
    probability=True
)