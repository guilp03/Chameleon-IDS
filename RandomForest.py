from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from math import sqrt

def RandomForest(n_features, conjunto, labels):
    rf_model = RandomForestClassifier(n_estimators = 100, max_features = int(sqrt(n_features)), random_state = 44)
    rf_model.fit(conjunto, labels)
    return rf_model

def get_metrics(model, validation, labels):
    preds = model.predict(validation)
    f1 = f1_score(labels,preds)
    return f1