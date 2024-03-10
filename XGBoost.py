from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def GradientBoost(conjunto, labels, n_estimators, alpha):
    gb_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=alpha, random_state=42)
    gb_model.fit(conjunto, labels)
    
    return gb_model

def get_metrics(model, validation, labels):
    preds = model.predict(validation)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels,preds)
    recall = recall_score(labels, preds)
    return accuracy, f1, precision, recall
