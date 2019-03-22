# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:33:39 2017

@author:
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
#import os

# to make the output stable across runs
np.random.seed(42)

# To plot pretty figures
#import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

## Where to save the figures
#PROJECT_ROOT_DIR = "C:/Users/jahan/Documents/SaintMaryCollege/Courses/OPS808/Summer 2018/PythonCodeExamples/figures"
##CHAPTER_ID = "decision_trees"
#
#def image_path(fig_id):
#    #return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)
#    return os.path.join(PROJECT_ROOT_DIR, fig_id)
#
#def save_fig(fig_id, tight_layout=True):
#    print("Saving figure", fig_id)
#    if tight_layout:
#        plt.tight_layout()
#    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)


 
from sklearn.model_selection import train_test_split
#from sklearn.datasets import make_moons
import pandas as pd  

"""
In this section we will predict whether a bank note is authentic or fake 
depending upon the four different attributes of the image of the note. 
The attributes are Variance of wavelet transformed image, 
Kurtosis of the image, entropy, and skewness of the image.
"""
dataset = pd.read_csv("pima-indians-diabetes.data.csv")  # to point to data

dataset.shape

dataset.head(20)

X = dataset.drop(dataset.columns[8], axis=1)  
y = dataset[dataset.columns[8]]  

"""
Preparing the Data
In this section we will divide our data into attributes and labels 
and will then divide the resultant data into both training and test sets. 
By doing this we can train our algorithm on one set of data and then test it out 
on a completely different set of data that the algorithm hasn't seen yet. 
This provides you with a more accurate view of how your trained 
algorithm will actually perform. We now split the data
"""
# split into training and test sets   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

 #   Bagging ensembles
print("Bagging ensembles")
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=1, random_state=42)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# determine accuracy score for the bagging method
print(accuracy_score(y_test, y_pred))

# now use a standard decision tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
# compare bagging method with standard decision tree classifier


# Random Forests
print("Random Forests")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

rnd_clf = RandomForestClassifier(max_depth=2, n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_prob_rf = rnd_clf.predict_proba(X_test)

y_pred_rf = rnd_clf.predict(X_test)

np.sum(y_pred == y_pred_rf) / len(y_pred)  # almost identical predictions

y_score_rf = y_prob_rf[:,1]
fpr_rf,tpr_rf, threshold_rf = roc_curve(y_test, y_score_rf)

# ROC curve plotting
def plot_roc_curve(fpr, tpr, lable=None):
    plt.plot(fpr,tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
plot_roc_curve(fpr_rf,tpr_rf)
plt.legend(loc="bottom right")
plt.show()

# Out-of-Bag evaluation
print("Out-of-Bag")
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, oob_score=True, random_state=40)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_

from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))



# Boosting method, three boosting algorithms
# First Ada boost
print("Ada boost")
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

y_pred_ada = ada_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_ada))

""" 
Question 2:
The bagging method trains individual models in parallel and each model is trained
by a random subset of the sample data. The Random Forest algorithm's models are decision trees
and the models are trained individually. AdaBoost trains its models sequentially. As a result,
the model can learn from its mistakes from the previous model by increasing the weight of
misclassified data points.

Quesiton 3:
The more depth a tree has, the more accurate the result. The variance is increased and there
is less bias. The only tradeoff is that there will be higher cost.
"""
