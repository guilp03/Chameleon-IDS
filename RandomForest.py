from sklearn.ensemble import RandomForestClassifier
from math import sqrt

def RandomForest(n_features, conjunto):
    rf_model = RandomForestClassifier(n = 50, max_features = sqrt(n_features), random_state = 44)
    rf_model.fit(conjunto)
    