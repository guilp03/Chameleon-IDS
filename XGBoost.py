from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

def GradientBoost(conjunto, labels):
    gb_model = GradientBoostingClassifier()
    gb_model.fit(conjunto, labels)
    
    return gb_model

def get_metrics(model, validation, labels):
    preds = model.predict(validation)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels,preds)
    recall = recall_score(labels, preds)
    return f1, precision, recall
