import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
from IPython.display import Image
from six import StringIO
from sklearn import metrics, model_selection, tree
from sklearn.tree import DecisionTreeClassifier

credit = pd.read_csv("./input/credit.csv")

col_dicts = {}
cols = ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 'employment_length', 'personal_status',
        'other_debtors', 'property', 'installment_plan', 'housing', 'job', 'telephone', 'foreign_worker']

const_val_dict = {
    "A11": "< 0 DM",
    "A12": "1 - 200 DM",
    "A13": ">= 200 DM",
    "A14": "no checking account",

    "A30": "fully repaid",
    "A31": "fully repaid this bank",
    "A32": "repaid",
    "A33": "delayed",
    "A34": "critical",

    "A40": "car (new)",
    "A41": "car (used)",
    "A42": "furniture",
    "A43": "radio/television",
    "A44": "domestic appliances",
    "A45": "repairs",
    "A46": "education",
    "A47": "vacation",
    "A48": "retraining",
    "A49": "business",
    "A410": "others",
    "A61": "< 100 DM",
    "A62": "100 <= ... < 500 DM",
    "A63": "500 <= ... < 1000 DM",
    "A64": ".. >= 1000 DM",
    "A65": "unknown",

    "A71": "unemployed",
    "A72": "... < 1 year",
    "A73": "1 <= ... < 4 years",
    "A74": "4 <= ... < 7 years",
    "A75": ".. >= 7 years",

    "A91": "male : divorced/separated",
    # Female: "A92": "female : divorced/separated/married",
    "A92": "female",
    "A93": "male : single",
    "A94": "male : married/widowed",
    # A95: "female : single",
    "A95": "female",
    "A101": "none",
    "A102": "co-applicant",
    "A103": "guarantor",
    "A121": "real estate",
    "A122": "building society savings",
    "A123": "other",
    "A124": "unknown / no property",
    "A141": "bank",
    "A142": "stores",
    "A143": "none",
    "A151": "rent",
    "A152": "own",
    "A153": "for free",
    "A171": "unemployed non-resident",
    "A172": "unskilled resident",
    "A173": "skilled employee",
    "A174": "management self-employed",
    "A191": "none",
    "A192": "yes",
    "A201": "yes",
    "A202": "no",
}

col_dicts = {
    'checking_balance': {'1 - 200 DM': 2, '< 0 DM': 1, '>= 200 DM': 3, 'no checking account': 0},
    'credit_history': {'critical': 0, 'delayed': 2, 'fully repaid': 3, 'fully repaid this bank': 4, 'repaid': 1},
    'employment_length': {'... < 1 year': 1, '1 <= ... < 4 years': 2, '4 <= ... < 7 years': 3, '.. >= 7 years': 4, 'unemployed': 0},
    'foreign_worker': {'no': 1, 'yes': 0},
    'housing': {'for free': 1, 'own': 0, 'rent': 2},
    'installment_plan': {'bank': 1, 'none': 0, 'stores': 2},
    'job': {'management self-employed': 3, 'skilled employee': 2, 'unemployed non-resident': 0, 'unskilled resident': 1},
    'other_debtors': {'co-applicant': 2, 'guarantor': 1, 'none': 0},
    'personal_status': {'male : divorced/separated': 2, 'female': 1, 'male : married/widowed': 3, 'male : single': 0},
    'property': {'building society savings': 1, 'other': 3, 'real estate': 0, 'unknown / no property': 2},
    'purpose': {'business': 5, 'car (new)': 3, 'car (used)': 4, 'domestic appliances': 6, 'education': 1, 'furniture': 2, 'others': 8, 'radio/television': 0, 'repairs': 7, 'retraining': 9},
    'savings_balance': {'100 <= ... < 500 DM': 2, '500 <= ... < 1000 DM': 3, '< 100 DM': 1, '.. >= 1000 DM': 4, 'unknown': 0},
    'telephone': {'none': 1, 'yes': 0}
}

for col in cols:
    credit[col] = credit[col].map(const_val_dict).map(col_dicts[col])

y = credit['default']
X = credit.loc[:, 'checking_balance':'foreign_worker']
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=1)

# train model
credit_model = DecisionTreeClassifier(min_samples_leaf=6, random_state=1)
credit_model.fit(X_train, y_train)

# draw decision trees
dot_data = StringIO()
tree.export_graphviz(credit_model, out_file=dot_data, feature_names=X_train.columns,
                     class_names=['no default', 'default'], filled=True, rounded=True, special_characters=True)
(graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

# test model
credit_pred = credit_model.predict(X_test)

print(metrics.classification_report(y_test, credit_pred))
print(metrics.confusion_matrix(y_test, credit_pred))
print(metrics.accuracy_score(y_test, credit_pred))

# optimization model
class_weights = {1: 1, 2: 4}
credit_model_cost = DecisionTreeClassifier(
    max_depth=6, class_weight=class_weights)
credit_model_cost.fit(X_train, y_train)
credit_pred_cost = credit_model_cost.predict(X_test)

print(metrics.classification_report(y_test, credit_pred_cost))
print(metrics.confusion_matrix(y_test, credit_pred_cost))
print(metrics.accuracy_score(y_test, credit_pred_cost))
