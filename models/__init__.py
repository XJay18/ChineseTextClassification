from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from .bert import BERT

ML_MODELS = {
    "MultinomialNB": MultinomialNB,
    "RandomForest": RandomForestClassifier,
    "AdaBoostDT": AdaBoostClassifier,
    "GradientBoostDT": GradientBoostingClassifier,
    "SGD": SGDClassifier,
    "MLP": MLPClassifier,
    "SVM": SVC
}

NN = {
    "BERT": BERT
}


def fetch_model(name="SVM"):
    assert name in ML_MODELS, "The model '%s' currently is not supported." % name
    return ML_MODELS[name]


def fetch_nn(name="BERT"):
    assert name in NN, "The model '%s' currently is not supported" % name
    return NN[name]
