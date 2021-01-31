def get_acc(y, y_hat):
    return sum(yi == yi_hat for yi, yi_hat in zip(y, y_hat)) / len(y)

## TPR = TP/# Postive
def get_tpr(y, y_hat):
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    actual_positive = sum(y)
    return true_positive / actual_positive

## Precision = TP/# Predicted Postive
def get_precision(y, y_hat):
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    predicted_positive = sum(y_hat)
    return true_positive / predicted_positive

## TNR = TN/# Negative
def get_tnr(y, y_hat):
    true_negative = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(y, y_hat))
    actual_negative = len(y) - sum(y)
    return true_negative / actual_negative

## ROC ~ TPR vs FPR (1-TNR)
def get_roc(y, y_hat_prob):
    thresholds = sorted(set(y_hat_prob), reverse=True)
    ret = [[0, 0]]
    for threshold in thresholds:
        y_hat = [int(yi_hat_prob >= threshold) for yi_hat_prob in y_hat_prob]
        ret.append([get_tpr(y, y_hat), 1 - get_tnr(y, y_hat)])
    return ret

## AUC
def get_auc(y, y_hat_prob):
    roc = iter(get_roc(y, y_hat_prob))
    tpr_pre, fpr_pre = next(roc)
    auc = 0
    for tpr, fpr in roc:
        auc += (tpr + tpr_pre) * (fpr - fpr_pre) / 2
        tpr_pre = tpr
        fpr_pre = fpr
    return auc


## AUC = 0.5
import pandas as pd
import numpy as np
from numpy.random import rand, seed, shuffle, normal
from matplotlib import pyplot as plt
seed(100)
y = np.array([0, 1] * 500)
shuffle(y)
seed(20)
y_pred = rand(1000)
points = get_roc(y, y_pred)
df = pd.DataFrame(points, columns=["tpr", "fpr"])
print("AUC is %.3f." % get_auc(y, y_pred))
df.plot(x="fpr", y="tpr", label="roc")
plt.show()

## AUC = 0.5
y_pred = np.array([0.9] * len(y))
points = get_roc(y, y_pred)
df = pd.DataFrame(points, columns=["tpr", "fpr"])
print("AUC is %.3f." % get_auc(y, y_pred))
df.plot(x="fpr", y="tpr", label="roc")
plt.show()

## AUC = 1
y_pred = np.array(y)
points = get_roc(y, y_pred)
df = pd.DataFrame(points, columns=["tpr", "fpr"])
print("AUC is %.3f." % get_auc(y, y_pred))
df.plot(x="fpr", y="tpr", label="roc")
plt.show()

# AUC = 0.7 equal positive predictions and negative predictions
seed(15)
f = lambda x: rand() / 2 + 0.5 if x else rand() / 2
y_pred = np.array([f(yi) if rand() > 0.3 else f(1 - yi) for yi in y])
points = get_roc(y, y_pred)
df = pd.DataFrame(points, columns=["tpr", "fpr"])
print("AUC is %.3f." % get_auc(y, y_pred))
df.plot(x="fpr", y="tpr", label="roc")
plt.show()

## AUC = 0.8 better positive predictions - 95% positive, 70% negative
seed(200)
def f(x):
    if x == 1:
        if rand() > 0.05:
            return rand() / 2 + 0.5
        else:
            return rand() / 2
    else:
        if rand() > 0.3:
            return rand() / 2
        else:
            return rand() / 2 + 0.5
y_pred = np.array([f(yi) for yi in y])
points = get_roc(y, y_pred)
df = pd.DataFrame(points, columns=["tpr", "fpr"])
print("AUC is %.3f." % get_auc(y, y_pred))
df.plot(x="fpr", y="tpr", label="roc")
plt.show()

## AUC = 0.8 better negative predictions - 70% positive, 95% negative
seed(200)
def f(x):
    if x == 1:
        if rand() > 0.3:
            return rand() / 2 + 0.5
        else:
            return rand() / 2
    else:
        if rand() > 0.05:
            return rand() / 2
        else:
            return rand() / 2 + 0.5
y_pred = np.array([f(yi) for yi in y])
points = get_roc(y, y_pred)
df = pd.DataFrame(points, columns=["tpr", "fpr"])
print("AUC is %.3f." % get_auc(y, y_pred))
df.plot(x="fpr", y="tpr", label="roc")
plt.show()

## AUC = 0.8 80% correct predicted values around 0.5 cutoff
seed(120)
def f(x):
    if x == 1:
        if rand() > 0.3:
            return rand() / 2 + 0.5
        else:
            return rand() / 2
    else:
        if rand() > 0.05:
            return rand() / 2
        else:
            return rand() / 2 + 0.5
y_pred = np.array([f(yi) for yi in y])
points = get_roc(y, y_pred)
df = pd.DataFrame(points, columns=["tpr", "fpr"])
print("AUC is %.3f." % get_auc(y, y_pred))
df.plot(x="fpr", y="tpr", label="roc")
plt.show()

## AUC = 0.8 80% correct predicted values around 0, 1
seed(50)
def f(x):
    if x == 1:
        if rand() > 0.2:
            return 1 - rand() / 10
        else:
            return rand() / 10
    else:
        if rand() > 0.2:
            return rand() / 10
        else:
            return 1 - rand() / 10
y_pred = np.array([f(yi) for yi in y])
points = get_roc(y, y_pred)
df = pd.DataFrame(points, columns=["tpr", "fpr"])
print("AUC is %.3f." % get_auc(y, y_pred))
df.plot(x="fpr", y="tpr", label="roc")
plt.show()

