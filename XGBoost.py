from sklearn.ensemble import GradientBoostingClassifier

def GradientBoost(conjunto):
    gb_model = GradientBoostingClassifier()
    gb_model.fit(conjunto)