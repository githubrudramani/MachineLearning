#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
#sns.set() # you can set seaborn plot style
plt.style.use('ggplot') # importing R plot for better plot


# In[3]:


## data import#######################
count = pd.read_csv("voom_count.csv", index_col = 0)
count = count.T
y = pd.read_csv("colData.csv", index_col = 0)
y_df = y
y_df.head()


# In[4]:


count.head()


# In[ ]:


## Does not need scaling
# Feature Scaling
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)


# In[5]:


#####Recursive feature elimination with cross-validation using RFC###################
#Recursive feature elimination (RFE) with random forest
from sklearn.feature_selection import RFE
x = count
## one hot encoder
y = y.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

#Recursive feature elimination with cross-validation using RFC
# The "accuracy" scoring is proportional to the number of correct classifications
rf_new = RandomForestClassifier() 
rfecv = RFECV(estimator=rf_new, step=0.01, cv=10,scoring='accuracy')   #10-fold cross-validation
rfecv = rfecv.fit(x, y)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', count.columns[rfecv.support_])


# In[7]:


## OOB error#####################################
x = count[count.columns[rfecv.support_]]
y = y_df.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
error_rate = {}
for i in range(10,1000,10):
    rfc = RandomForestClassifier(n_estimators=i, random_state=42, warm_start=True, oob_score=True)
    rfc.fit(x_train,y_train)
    oob_error = 1 - rfc.oob_score_
    error_rate[i] = oob_error


# In[10]:


### plotting OOB Error ####################################
# Convert dictionary to a pandas series for easy plotting 
oob_series = pd.Series(error_rate)
fig, ax = plt.subplots(figsize=(8, 5))

ax.set_facecolor('#fafafa')

oob_series.plot(kind='line',
                color = 'red')
plt.axhline(0.02, 
            color='#875FDB',
           linestyle='--')
plt.axhline(0.001, 
            color='#875FDB',
           linestyle='--')
plt.xlabel('n_estimators', fontsize = 20)
plt.ylabel('OOB Error Rate', fontsize = 20)
plt.title('OOB Error Rate Across various Forest sizes \n(From 10 to 1000 trees)', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
#plt.axvline(750) #not wnt 360, all = 500, fourvsrest = 750
#plt.savefig("OOB_error_fourvsRest.png", dpi = 300,  bbox_inches='tight')
plt.show()


# In[50]:


## Applying the Random Forest Finally
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier

#### remove this for fair calculation #####
imp = list(count.columns[rfecv.support_]) ## to enclude circ_63706
imp.append("circ_63706") ## to enclude circ_63706
x = count[imp]
##################

#x = count[count.columns[rfecv.support_]]
y = y_df.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state =42)
rfc= BalancedRandomForestClassifier(n_estimators=500, random_state=42) ## n_estimator given by OOB Error
rfc.fit(x_train,y_train)
accuracy = accuracy_score(rfc.predict(x_test), y_test)
importance = rfc.feature_importances_
print("accuracy_score: ", accuracy)


# In[51]:


importance = rfc.feature_importances_
# summarize feature importance
#for i,v in enumerate(importance):
    #print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[130]:


### Random Forest Important features Identification ####
group = ["WNT", "SHH", "Group3", "Group4"]
Li = []
for i in group:
    c = i
    #X = count[count.columns[rfecv.support_]]
    X = count[imp]
    Y = y_df
    Y= Y[Y.Group==c]
    Y = list(Y.index)
    vip = X.loc[Y].mean()*importance
    vip = vip.sort_values(ascending = False)
    Li.append(vip)
    
vip_df = pd.concat(Li, axis = 1)
vip_df.columns = group
vip_df = vip_df.replace(np.nan, 0)
vip_df2 = vip_df[vip_df >0.1]  # filter the features having coefficient greater than 0.1
vip_df2 = vip_df2.dropna(thresh = 1)
#vip_df = vip_df.replace(np.nan, 0)
vip = list(vip_df2.index)
vip.append("circ_63706")
vip_df = vip_df.loc[vip]

vip_df = np.round(vip_df,2)
#vip_df[vip_df <=0.05] = 0
vip_df.columns
vip_df["sum"] = vip_df.sum(axis =1)


# In[131]:


## Converting to 100 % contribution
vip_df["WNT"] = vip_df["WNT"]/vip_df["sum"]
vip_df["SHH"] = vip_df["SHH"]/vip_df["sum"]
vip_df["Group3"] = vip_df["Group3"]/vip_df["sum"]
vip_df["Group4"] = vip_df["Group4"]/vip_df["sum"]
vip_df = vip_df.drop("sum", axis = 1)
vip_df = vip_df*100


# In[132]:


## Plot the VIP score  #######
plt.figure(figsize = (6,12))
sns.heatmap(vip_df, cmap = "Reds", annot = True, annot_kws = {"fontsize":15, "fontweight": "bold"})
plt.ylabel("Circular RNAs", fontsize = 25, weight = "bold")
plt.xlabel("",fontsize = 25, weight = "bold")
plt.xticks(fontsize = 20, rotation = 45, weight = "bold")
plt.yticks(fontsize = 20,rotation = 0, weight = "bold")
plt.savefig("VIPrandomForest_revised.pdf", dpi = 300,  bbox_inches='tight')
plt.show()


# In[133]:


#### ROC curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
x= count
y = y_df.values

y = label_binarize(y, classes=["WNT","SHH","Group3", "Group4"])
n_classes = 4
## split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state =42)
# classifier
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=500, random_state=42))
y_score = clf.fit(x_train, y_train).predict_proba(x_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
col = ["green", "blue", "orange", "red"]
alpha = [0.1, 0.2,0.3,0.4]
ls = ["--", "--", "-.", "--"]
plt.figure(figsize= (8,5))
for i in range(n_classes):
  fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
  plt.plot(fpr[i], tpr[i], color=col[i], lw=2,  linestyle=ls[i])
  print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i]))) ## reference this for label plot
  roc_auc[i] = auc(fpr[i], tpr[i])

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([-0.03, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.xticks(fontsize = 20, rotation = 0, weight = "bold")
plt.yticks(fontsize = 20,rotation = 0, weight = "bold")
plt.title('Receiver Operating Characteristic Curves', fontsize = 20)
# Replace the values in the label according to the resultin roc_auc dictionary
labels = ['AUC for WNT: 0.97', 'AUC for SHH: 1', 'AUC for Group3: 1', "AUC for Group4: 0.95"]
plt.legend(labels, prop={"size":17})
#plt.savefig("ROCcurve.pdf", dpi = 300,  bbox_inches='tight')
plt.show()


# In[175]:


wnt = vip_df.sort_values("Group4", ascending = False)
top5 = [i for i in wnt.head(5).index]
[print(i) for i in top5]


# In[176]:


## Importing top from Limma
limma = pd.read_csv("limmaTop.csv", sep = "\s+")
for i in top5:
    if i in limma["Group4"].values:
        print(i)



# In[148]:


limma


# In[149]:


vip_df


# In[ ]:




