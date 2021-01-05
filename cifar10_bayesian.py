import pickle
import numpy as np
from numpy import cov
import matplotlib.pyplot as plt
from random import random
import pandas as pd
import scipy.stats as stats
from scipy.stats import multivariate_normal
from skimage.transform import rescale, resize, downscale_local_mean

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def class_acc(pred,gt):
    return (np.sum(pred==gt)/len(pred))*100


def cifar10_color(X):
   Xf = np.zeros((len(X),3))
   for i in range(X.shape[0]):
       img = X[i]
       img_1x1 = resize(img, (1, 1))
       r_vals = img_1x1[:,:,0].reshape(1*1)
       g_vals = img_1x1[:,:,1].reshape(1*1)
       b_vals = img_1x1[:,:,2].reshape(1*1)
       mu_r = r_vals.mean()
       mu_g = g_vals.mean()
       mu_b = b_vals.mean()
       Xf[i,:] = (mu_r, mu_g, mu_b)
   return Xf
   
   
def cifar10_2X2_color(X):
   data=np.array([])
   for i in range(X.shape[0]):
       img = X[i]
       img_2x2 = resize(img, (2, 2))
       r_vals = img_2x2[:,:,0].reshape(2*2)
       g_vals = img_2x2[:,:,1].reshape(2*2)
       b_vals = img_2x2[:,:,2].reshape(2*2)
       vector = np.concatenate((np.transpose(r_vals),np.transpose(g_vals),np.transpose(b_vals)))
       data=np.append(data,vector)
   return data
   
   
def cifar10_4X4_color(X):
   data=np.array([])
   for i in range(X.shape[0]):
       img = X[i]
       img_4x4 = resize(img, (4, 4))
       r_vals = img_4x4[:,:,0].reshape(4*4)
       g_vals = img_4x4[:,:,1].reshape(4*4)
       b_vals = img_4x4[:,:,2].reshape(4*4)
       vector = np.concatenate((np.transpose(r_vals),np.transpose(g_vals),np.transpose(b_vals)))
       data=np.append(data,vector)
   return data
   
   
def cifar10_8X8_color(X):
   data=np.array([])
   for i in range(X.shape[0]):
       img = X[i]
       img_8x8 = resize(img, (8, 8))
       r_vals = img_8x8[:,:,0].reshape(8*8)
       g_vals = img_8x8[:,:,1].reshape(8*8)
       b_vals = img_8x8[:,:,2].reshape(8*8)
       vector = np.concatenate((np.transpose(r_vals),np.transpose(g_vals),np.transpose(b_vals)))
       data=np.append(data,vector)
   return data
 
 
def cifar10_16X16_color(X):
   data=np.array([])
   for i in range(X.shape[0]):
       img = X[i]
       img_16x16 = resize(img, (16, 16))
       r_vals = img_16x16[:,:,0].reshape(16*16)
       g_vals = img_16x16[:,:,1].reshape(16*16)
       b_vals = img_16x16[:,:,2].reshape(16*16)
       vector = np.concatenate((np.transpose(r_vals),np.transpose(g_vals),np.transpose(b_vals)))
       data=np.append(data,vector)
   return data

   
   
   
def cifar_10_naivebayes_learn(Xf,Y):
    df_x=pd.DataFrame(data=Xf)
    df_y=pd.DataFrame(data=Y)
    frames=[df_x,df_y]
    data=pd.concat(frames,axis=1)
    data.columns=['r','g','b','class']
    data=data.sort_values(by=['class'],ascending=True)
    mu=np.zeros((10,3))
    std=np.zeros((10,3))
    cov=np.zeros((10,3,3))
    p=np.repeat(0.1,10)
    m=0
    n=5000
    for i in range(0,10):
            df_select=data.iloc[m:n, 0:3]
            mu[i]=np.array(df_select.mean(axis=0))
            std[i]=np.array(df_select.std(axis=0))
            cov[i]=np.array(df_select.cov())
            m=m+5000
            n=n+5000
    return mu,std,p,cov
    


def cifar_10_naivebayes_learn_super_powerful(Xf,Y):
    df_x=pd.DataFrame(data=Xf)
    df_y=pd.DataFrame(data=Y)
    frames=[df_x,df_y]
    data=pd.concat(frames,axis=1)
    print(data.shape)
    t=Xf.shape[1]
    print(t)
    col_val=[]
    for i in range(0,Xf.shape[1]):
        col_val.append(str(i))
    col_val.append('class')
    data.columns=col_val
    data=data.sort_values(by='class',ascending=True)
    mu=np.zeros((10,t))
    cov=np.zeros((10,t,t))
    p=np.repeat(0.1,10)
    m=0
    n=5000
    for i in range(0,10):
            df_select=data.iloc[m:n, 0:t]
            mu[i]=np.array(df_select.mean(axis=0))
            cov[i]=np.array(df_select.cov())
            m=m+5000
            n=n+5000
    return mu,p,cov 
   


    
def cifar10_classifier_naivebayes(x,mu,std,p):
        prob_class_0=stats.norm.pdf(x[0],mu[0][0], std[0][0])*stats.norm.pdf(x[1],mu[0][1], std[0][1])*stats.norm.pdf(x[2],mu[0][2], std[0][2])*p[0]
        prob_class_1=stats.norm.pdf(x[0],mu[1][0], std[1][0])*stats.norm.pdf(x[1],mu[1][1], std[1][1])*stats.norm.pdf(x[2],mu[1][2], std[1][2])*p[1]
        prob_class_2=stats.norm.pdf(x[0],mu[2][0], std[2][0])*stats.norm.pdf(x[1],mu[2][1], std[2][1])*stats.norm.pdf(x[2],mu[2][2], std[2][2])*p[2]
        prob_class_3=stats.norm.pdf(x[0],mu[3][0], std[3][0])*stats.norm.pdf(x[1],mu[3][1], std[3][1])*stats.norm.pdf(x[2],mu[3][2], std[3][2])*p[3]
        prob_class_4=stats.norm.pdf(x[0],mu[4][0], std[4][0])*stats.norm.pdf(x[1],mu[4][1], std[4][1])*stats.norm.pdf(x[2],mu[4][2], std[4][2])*p[4]
        prob_class_5=stats.norm.pdf(x[0],mu[5][0], std[5][0])*stats.norm.pdf(x[1],mu[5][1], std[5][1])*stats.norm.pdf(x[2],mu[5][2], std[5][2])*p[5]
        prob_class_6=stats.norm.pdf(x[0],mu[6][0], std[6][0])*stats.norm.pdf(x[1],mu[6][1], std[6][1])*stats.norm.pdf(x[2],mu[6][2], std[6][2])*p[6]
        prob_class_7=stats.norm.pdf(x[0],mu[7][0], std[7][0])*stats.norm.pdf(x[1],mu[7][1], std[7][1])*stats.norm.pdf(x[2],mu[7][2], std[7][2])*p[7]
        prob_class_8=stats.norm.pdf(x[0],mu[8][0], std[8][0])*stats.norm.pdf(x[1],mu[8][1], std[8][1])*stats.norm.pdf(x[2],mu[8][2], std[8][2])*p[8]
        prob_class_9=stats.norm.pdf(x[0],mu[9][0], std[9][0])*stats.norm.pdf(x[1],mu[9][1], std[9][1])*stats.norm.pdf(x[2],mu[9][2], std[9][2])*p[9]
        class_predicted=np.argmax([prob_class_0,prob_class_1,prob_class_2,prob_class_3,prob_class_4,prob_class_5,prob_class_6,prob_class_7,prob_class_8,prob_class_9])
        return class_predicted
        
def cifar10_classifier_naivebayes_better_super(x,mu,cov,p):
        prob_class_0=multivariate_normal.logpdf(x, mu[0], cov[0])*p[0]
        prob_class_1=multivariate_normal.logpdf(x, mu[1], cov[1])*p[1]
        prob_class_2=multivariate_normal.logpdf(x, mu[2], cov[2])*p[2]
        prob_class_3=multivariate_normal.logpdf(x, mu[3], cov[3])*p[3]
        prob_class_4=multivariate_normal.logpdf(x, mu[4], cov[4])*p[4]
        prob_class_5=multivariate_normal.logpdf(x, mu[5], cov[5])*p[5]
        prob_class_6=multivariate_normal.logpdf(x, mu[6], cov[6])*p[6]
        prob_class_7=multivariate_normal.logpdf(x, mu[7], cov[7])*p[7]
        prob_class_8=multivariate_normal.logpdf(x, mu[8], cov[8])*p[8]
        prob_class_9=multivariate_normal.logpdf(x, mu[9], cov[9])*p[9]
        class_predicted=np.argmax([prob_class_0,prob_class_1,prob_class_2,prob_class_3,prob_class_4,prob_class_5,prob_class_6,prob_class_7,prob_class_8,prob_class_9])
        return class_predicted
     

        
        
datadict = unpickle('cifar-10-batches-py/test_batch')
X_test = datadict["data"]
Y_test = datadict["labels"]
labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
X_test= X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int32")
Y_test = np.array(Y_test)
trainset1=unpickle('cifar-10-batches-py/data_batch_1')
trainset2=unpickle('cifar-10-batches-py/data_batch_2')
trainset3=unpickle('cifar-10-batches-py/data_batch_3')
trainset4=unpickle('cifar-10-batches-py/data_batch_4')
trainset5=unpickle('cifar-10-batches-py/data_batch_5')
X_train1=trainset1["data"]
Y_train1=trainset1["labels"]
X_train2=trainset2["data"]
Y_train2=trainset2["labels"]
X_train3=trainset3["data"]
Y_train3=trainset3["labels"]
X_train4=trainset4["data"]
Y_train4=trainset4["labels"]
X_train5=trainset5["data"]
Y_train5=trainset5["labels"]
X_train = np.concatenate((X_train1,X_train2,X_train3,X_train4,X_train5))
Y_train=np.concatenate((Y_train1,Y_train2,Y_train3,Y_train4,Y_train5))
X_train= X_train.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("int32")

#Task 1
Xf_train=cifar10_color(X_train)
[mu,std,p,cov]=cifar_10_naivebayes_learn(Xf_train,Y_train)
Xf_test=cifar10_color(X_test)
y_pred_a=[]
for i in range(0,len(Xf_test)):
    y_bayesian_simple=cifar10_classifier_naivebayes(Xf_test[i],mu,std,p)
    y_pred_a.append(y_bayesian_simple)
print("Accuracy for simple bayesian classifier:"+str(class_acc(y_pred_a,Y_test))+"%")

#Task 2:
y_pred_b=[]
for i in range(0,len(Xf_test)):
    y_bayesian_better=cifar10_classifier_naivebayes_better(Xf_test[i],mu,cov,p)
    y_pred_b.append(y_bayesian_better)
print("Accuracy for better bayesian classifier:"+str(class_acc(y_pred_b,Y_test))+"%")

#Task 3

#For 2X2
data=cifar10_2X2_color(X_train)
data=data.reshape(50000,12)
[mu,p,cov]=cifar_10_naivebayes_learn_super_powerful(data,Y_train)
data_test=cifar10_2X2_color(X_test)
data_test=data_test.reshape(10000,12)
y_pred_s=[]
for i in range(0,len(data_test)):
    y_bayesian_super=cifar10_classifier_naivebayes_better_super(data_test[i],mu,cov,p)
    y_pred_s.append(y_bayesian_super)
print("Accuracy for super powerful bayesian classifier with image size 2X2:"+str(class_acc(y_pred_s,Y_test))+"%")


#For 4X4
data=cifar10_4X4_color(X_train)
data=data.reshape(50000,48)
[mu,p,cov]=cifar_10_naivebayes_learn_super_powerful(data,Y_train)
data_test=cifar10_4X4_color(X_test)
data_test=data_test.reshape(10000,48)
y_pred_q=[]
for i in range(0,len(data_test)):
    y_bayesian_super=cifar10_classifier_naivebayes_better_super(data_test[i],mu,cov,p)
    y_pred_q.append(y_bayesian_super)
print("Accuracy for super powerful bayesian classifier with image size 4X4:"+str(class_acc(y_pred_q,Y_test))+"%")


#For 8X8
data=cifar10_8X8_color(X_train)
data=data.reshape(50000,192)
[mu,p,cov]=cifar_10_naivebayes_learn_super_powerful(data,Y_train)
data_test=cifar10_8X8_color(X_test)
data_test=data_test.reshape(10000,192)
y_pred_r=[]
for i in range(0,len(data_test)):
    y_bayesian_super=cifar10_classifier_naivebayes_better_super(data_test[i],mu,cov,p)
    y_pred_r.append(y_bayesian_super)
print("Accuracy for super powerful bayesian classifier with image size 8X8:"+str(class_acc(y_pred_r,Y_test))+"%")


#For 16X16
data=cifar10_16X16_color(X_train)
data=data.reshape(50000,768)
[mu,p,cov]=cifar_10_naivebayes_learn_super_powerful(data,Y_train)
data_test=cifar10_16X16_color(X_test)
data_test=data_test.reshape(10000,768)
y_pred_r=[]
for i in range(0,len(data_test)):
    y_bayesian_super=cifar10_classifier_naivebayes_better_super(data_test[i],mu,cov,p)
    y_pred_r.append(y_bayesian_super)
print("Accuracy for super powerful bayesian classifier with image size 16X16:"+str(class_acc(y_pred_r,Y_test))+"%")



#Graph
N=[2,4,8,16]
accuracies=[31.05,40.22,41.73,43.46]        #Accuracies are obtained after running every task seperately
plt.plot(N,accuracies, 'r-')
plt.xlabel('N')
plt.ylabel('Accuracy')
plt.show()
