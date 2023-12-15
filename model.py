#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import all the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[3]:


df = pd.read_csv('Darknet.csv')
df.head()


# In[4]:


#check the label column for types of attacks and count them
df['Label'].value_counts()

#plot the pie chart for the types of attacks
plt.figure(figsize=(10,10))
df['Label'].value_counts().plot(kind='pie', autopct='%.2f%%')
plt.show()


# In[5]:


#plot Non-Tor and NonVPN as normal traffic and Tor and VPN as malicious traffic
df['Label'] = df['Label'].replace(['Non-Tor', 'NonVPN'], 'Normal')
df['Label'] = df['Label'].replace(['Tor', 'VPN'], 'Malicious')

#make a pie chart for the new labels and add labels with values counts not percentage 
plt.figure(figsize=(10,10))
df['Label'].value_counts().plot(kind='pie', autopct='%.2f%%', labels=['Normal', 'Malicious'], labeldistance=1.1)
#plot the pie chart for the types of attacks
plt.show()


# In[6]:


#Do a value count for the new labels
df['Label'].value_counts()


# In[7]:


#drop the second label column
df = df.drop(['Label.1'], axis=1)


# In[8]:


df.info()


# In[9]:


#create a holdout set for testing later with 10% of the data by putting 0.1 in the test_size
from sklearn.model_selection import train_test_split
main_set, hold_out_set = train_test_split(df, test_size=0.1, random_state=42)


# In[10]:


# Select relevant numeric features for the model
numeric_features = main_set.select_dtypes(include=['float64', 'int64']).columns


# In[11]:


# Drop non-relevant columns (identifiers and protocol which is categorical)
X = main_set[numeric_features].drop(['Src Port', 'Dst Port', 'Protocol'], axis=1)


# In[12]:


# Replace infinities and NaNs
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)


# In[13]:


#only select the numeric columns
numeric_cols = X.columns.values.tolist()
numeric_cols


# In[14]:


#get the correlation of the features
corr = X.corr()

#plot the correlation heatmap
plt.figure(figsize=(15,15))
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[15]:


# Encode the labels to binary format
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(main_set['Label'])


# In[16]:


# Normalize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[17]:


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)


# In[18]:


# Initialize the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)


# In[19]:


# Predict on the test set
y_pred = rf_classifier.predict(X_test)


# In[20]:


from sklearn.metrics import classification_report, accuracy_score

# Print classification report and accuracy
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# In[21]:


#plot the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[22]:


#plot the ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[23]:


# Apply SMOTE to the training data
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# In[24]:


# Initialize the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train_smote, y_train_smote)


# In[25]:


# Predict on the test set
y_pred = rf_classifier.predict(X_test)


# In[26]:


# Print classification report and accuracy
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# In[27]:


#plot the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[28]:


#plot the ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[29]:


#use the holdout set for testing
X_holdout = hold_out_set[numeric_features].drop(['Src Port', 'Dst Port', 'Protocol'], axis=1)
X_holdout.replace([np.inf, -np.inf], np.nan, inplace=True)
X_holdout.fillna(X_holdout.mean(), inplace=True)
X_holdout_scaled = scaler.transform(X_holdout)
y_holdout = le.transform(hold_out_set['Label'])

# Predict on the holdout set
y_pred = rf_classifier.predict(X_holdout_scaled)

# Print classification report and accuracy
print(classification_report(y_holdout, y_pred))
print(f"Accuracy: {accuracy_score(y_holdout, y_pred)}")

#plot the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_holdout, y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cmap=plt.cm.Reds)
plt.show()

#plot the ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_holdout, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[30]:


#save the model
import pickle
filename = 'darknet_model.sav'
pickle.dump(rf_classifier, open(filename, 'wb'))

#load the model
loaded_model = pickle.load(open(filename, 'rb'))

# Predict on the holdout set
y_pred = loaded_model.predict(X_holdout_scaled)

# Print classification report and accuracy
print(classification_report(y_holdout, y_pred))
print(f"Accuracy: {accuracy_score(y_holdout, y_pred)}")

