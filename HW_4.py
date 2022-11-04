#!/usr/bin/env python
# coding: utf-8

# In[147]:


# Anthony Perales
# 801150315
# ================================== Homework 4 ===================================
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import pandas as pd
import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

def plot_elements(x=None,y=None):
    sns.scatterplot(x = x[:,0], y = x[:,1], hue = y, s = 50)

def plot_svc_decision_function(model):
    ax = plt.gca()
    ax.set_xlim(-6,10)
    ax.set_ylim(-6,10)
    w = model.coef_[0]
    b = model.intercept_[0]
    x_points = np.linspace(-10,10)
    y_points = (-w[0]/w[1]) * x_points - (b/w[1])
    plt.plot(x_points, y_points, c ='r')
    sns.scatterplot(model.support_vectors_[:,0], model.support_vectors_[:,1],s = 65, linewidth = 1.5, facecolors = 'black')

    w_hat = model.coef_[0] / (np.sqrt(np.sum(model.coef_[0]**2)))
    margin = 1/ np.sqrt(np.sum(model.coef_[0] **2))
    boundary_points  = np.array(list(zip(x_points, y_points)))
    above_points = boundary_points + margin
    below_points = boundary_points - margin


    ax.plot(above_points[:,0], above_points[:,1],'b--',linewidth=2)
    ax.plot(below_points[:,0], below_points[:,1],'b--',linewidth=2) 

    
cancer = load_breast_cancer()

X = cancer['data']
Y = cancer['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 42)


# In[148]:


# ======================= Problem 1 ========================
pca = PCA()
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

model = SVC(kernel = 'linear', C = 0.001, decision_function_shape = 'ovr')
model.fit(X_train_pca, Y_train)

predicted = model.predict(X_test_pca)
acc = accuracy_score(Y_test, predicted)
pre = precision_score(Y_test, predicted)
rec = recall_score(Y_test, predicted)
print(" Accuracy:", acc*100,'\n',"Precision: ", pre*100 ,'\n', "Recall: ", rec*100)

plt.figure(figsize = (10,8))
plt.title('C = 0.001')
plot_elements(X_train_pca,Y_train)
plot_svc_decision_function(model)


# In[149]:


model = SVC(kernel = 'linear', C = 0.08, decision_function_shape = 'ovo')
model.fit(X_train_pca, Y_train)

predicted = model.predict(X_test_pca)
acc = accuracy_score(Y_test, predicted)
pre = precision_score(Y_test, predicted)
rec = recall_score(Y_test, predicted)
print(" Accuracy:", acc*100,'\n',"Precision: ", pre*100 ,'\n', "Recall: ", rec*100)

plt.figure(figsize = (10,8))
plt.title('C = 0.08')
plot_elements(X_train_pca,Y_train)
plot_svc_decision_function(model)


# In[150]:


model = SVC(kernel = 'linear', C = 1, decision_function_shape = 'ovr')
model.fit(X_train_pca, Y_train)

predicted = model.predict(X_test_pca)
acc = accuracy_score(Y_test, predicted)
pre = precision_score(Y_test, predicted)
rec = recall_score(Y_test, predicted)
print(" Accuracy:", acc*100,'\n',"Precision: ", pre*100 ,'\n', "Recall: ", rec*100)

plt.figure(figsize = (10,8))
plt.title('C = 1')
plot_elements(X_train_pca,Y_train)
plot_svc_decision_function(model)


# In[151]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import pandas as pd
import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.metrics import accuracy_score
#from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ======================= Problem 2 ========================
from sklearn.svm import SVR
test_history = list()
mse_test = list()
mse_train = list()
train_history = list()

df = pd.read_csv('Housing.csv', header = 0)

list_bm = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning','prefarea']

def binary_map(x):
    return x.map({'yes': 1, 'no': 0})

df[list_bm] = df[list_bm].apply(binary_map)

X = df.drop(['price','furnishingstatus'],axis=1)
Y= df.pop('price')

sc = StandardScaler()
X = sc.fit_transform(X)

for n in range(1,12):
    X_df = pd.DataFrame(data = X)
    pca = PCA(n_components = n)
    X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y, train_size = 0.8, test_size = 0.2, random_state = 32)
    X_trn_pca = pca.fit_transform(X_train)
    X_tst_pca = pca.transform(X_test)
    model = SVR(kernel = 'rbf', C = 8000)
    model.fit(X_trn_pca, Y_train)
    Y_pred = model.predict(X_tst_pca)
    Y2_pred = model.predict(X_trn_pca)
    mse_test.append(metrics.mean_squared_error( Y_test,Y_pred))
    mse_train.append(metrics.mean_squared_error( Y_train, Y2_pred))

plt.figure(figsize = (10,8))
plt.title('Model(RBF) vs PCA Components')
plt.xlabel('PCA Components')
plt.plot(mse_test, label = 'Testing Accuracy', color = 'r')
plt.plot(mse_train, label = 'Training Accuracy', color = 'b')
plt.legend()
plt.show()


# In[152]:


poly_mse_test = list()
poly_mse_train = list()

for n in range(1,12):
    X_df = pd.DataFrame(data = X)
    pca = PCA(n_components = n)
    X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y, train_size = 0.8, test_size = 0.2, random_state = 32)
    X_trn_pca = pca.fit_transform(X_train)
    X_tst_pca = pca.transform(X_test)
    model = SVR(kernel = 'poly', C = 500)
    model.fit(X_trn_pca, Y_train)
    Y_pred = model.predict(X_tst_pca)
    Y2_pred = model.predict(X_trn_pca)
    poly_mse_test.append(metrics.mean_squared_error( Y_test,Y_pred))
    poly_mse_train.append(metrics.mean_squared_error( Y_train, Y2_pred))

plt.figure(figsize = (10,8))
plt.title('Model(POLY) vs PCA Components')
plt.xlabel('PCA Components')
plt.plot(poly_mse_test, label = 'Testing Accuracy', color = 'r')
plt.plot(poly_mse_train, label = 'Training Accuracy', color = 'b')
plt.legend()
plt.show()


# In[153]:


sig_mse_test = list()
sig_mse_train = list()

for n in range(1,12):
    X_df = pd.DataFrame(data = X)
    pca = PCA(n_components = n)
    X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y, train_size = 0.8, test_size = 0.2, random_state = 32)
    X_trn_pca = pca.fit_transform(X_train)
    X_tst_pca = pca.transform(X_test)
    model = SVR(kernel = 'sigmoid', C = 90000)
    model.fit(X_trn_pca, Y_train)
    Y_pred = model.predict(X_tst_pca)
    Y2_pred = model.predict(X_trn_pca)
    sig_mse_test.append(metrics.mean_squared_error( Y_test,Y_pred))
    sig_mse_train.append(metrics.mean_squared_error( Y_train, Y2_pred))

plt.figure(figsize = (10,8))
plt.title('Model(SIGMOID) vs PCA Components')
plt.xlabel('PCA Components')
plt.plot(sig_mse_test, label = 'Testing Accuracy', color = 'r')
plt.plot(sig_mse_train, label = 'Training Accuracy', color = 'b')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




