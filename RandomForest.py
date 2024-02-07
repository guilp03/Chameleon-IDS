from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from math import sqrt

def RandomForest(n_features, conjunto, labels, n_estimators):
    rf_model = RandomForestClassifier(n_estimators = 100, max_features = int(sqrt(n_features)), random_state = 44)
    rf_model.fit(conjunto, labels)
    return rf_model

def get_metrics(model, validation, labels):
    preds = model.predict(validation)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels,preds)
    recall = recall_score(labels, preds)
    return accuracy, f1, precision, recall