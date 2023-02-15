import joblib

def predict(data):
    clf = joblib.load("/Users/matrikasubedi/AIclass/irisclassificationapp/output_model/rf_model.sav")
    return clf.predict(data)