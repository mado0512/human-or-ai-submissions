import json
import pandas as pd
import numpy as np
import joblib
import ollama
from sklearn.metrics import roc_auc_score

train_df = pd.read_csv("submissions/group-number-5/embed_svm/train.csv")
starter_df = pd.read_csv("submissions/group-number-5/embed_svm/train_starter_2000.csv")

starter_ids = set(starter_df["id"])

# Full model dev set: starter subset
# Starter model dev set: everything outside starter subset
full_dev = starter_df
starter_dev = train_df[~train_df["id"].isin(starter_ids)]


def evaluate(model_path, dev_df):
    model = joblib.load(model_path)

    X = np.array([
        ollama.embed(
            model="embeddinggemma-2",
            input=text
        )["embeddings"][0]
        for text in dev_df["text"]
    ])

    y = dev_df["label"].to_numpy()

    predictions = model.predict_proba(X)[:, 1]

    print(f"{model_path} AUROC: {roc_auc_score(y, predictions):.4f}")


evaluate("submissions/group-number-5/embed_svm/SVM_full_train", full_dev)
evaluate("submissions/group-number-5/embed_svm/SVM_starter_train", starter_dev)