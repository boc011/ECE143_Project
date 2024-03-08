#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fastFM')


# In[ ]:


import pandas as pd
import gzip
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import numpy as np
from fastFM import als, sgd, mcmc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler,label_binarize
from sklearn.metrics import roc_auc_score,accuracy_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,classification_report
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


clean_dataset_path = "/content/drive/MyDrive/AIHW/AS2/caius_data_clean.csv"


# In[ ]:


df = None
if not os.path.exists(clean_dataset_path):
  df = None
  print("Empty Dataset")
else:
  df = pd.read_csv(clean_dataset_path)


# In[ ]:


average_height = df['height'].dropna().mean()
df['height'] = df['height'].fillna(average_height)
df['height_mul'] = df['height']/10
average_rating = df['rating'].dropna().mean()
df['rating'] = df['rating'].fillna(average_rating)
df['size_mul'] = df['size']*2


# In[ ]:


encoder = OneHotEncoder()
categorical_data = encoder.fit_transform(df[['user_id', 'item_id']])
scaler = StandardScaler()
numerical_data = scaler.fit_transform(df[['size','rating','height']])
# Increase all size_mul by 3 times
for i in range(len(numerical_data)):
  numerical_data[i][0] = numerical_data[i][0]*4


# In[ ]:


numerical_data


# In[ ]:


# Optimized data preparation
# Define the resampling strategy
#X = hstack([categorical_data, height_data])
X = hstack([categorical_data,numerical_data])
fit_mapping = {'small': 0, 'fit': 1, 'large': 2}
y = df['fit'].map(fit_mapping).values
over = SMOTE(sampling_strategy={0: int(len(y) * 0.5), 2: int(len(y) * 0.5)}, k_neighbors=3)
under = RandomUnderSampler(sampling_strategy={1: int(len(y) * 0.4)})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Define the pipeline
# pipeline = Pipeline(steps=[('o', over), ('u', under)])
pipeline = Pipeline(steps=[('o', over)])
y_binary = label_binarize(y, classes=np.unique(y),neg_label=-1)
# Apply the pipeline to your data
X_train, y_train = pipeline.fit_resample(X_train, y_train)


# In[ ]:


#X = hstack([categorical_data, height_data])
#Size only
X = categorical_data
# Binarize target variable for One-vs-Rest strategy
fit_mapping = {'small': 0, 'fit': 1, 'large': 2}
y = df['fit'].map(fit_mapping).values
y_binary = label_binarize(y, classes=np.unique(y),neg_label=-1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.1)


# In[ ]:


class FMClassifier(als.FMClassification):
    def fit(self, X, y, *args):
        y = y.copy()
        y[y == 0] = -1
        return super(FMClassifier, self).fit(X, y, *args)

    def predict_proba(self, X):
        probs = super(FMClassifier, self).predict_proba(X)
        return np.tile(probs, 2).reshape(2, probs.shape[0]).T

ovr_classifier = OneVsRestClassifier(FMClassifier(n_iter=150,init_stdev=0.2, rank=4, l2_reg_w=0.3, l2_reg_V=0.3), n_jobs=-1)

# Initialize the factorization machine model
# fm_model = als.FMClassification(n_iter=25, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.1)

# Apply One-vs-Rest strategy for multiclass prediction
# ovr_classifier = OneVsRestClassifier(fm_model)

# Fit the model
ovr_classifier.fit(csr_matrix(X_train), y_train)



# In[ ]:


# Predict the probabilities
y_pred_prob = ovr_classifier.predict_proba(csr_matrix(X_test))
y_pred = ovr_classifier.predict(X_test)
# Calculate AUC for each class and average
auc_scores = roc_auc_score(y_test, y_pred_prob, average='macro', multi_class='ovr')

print(f'AUC Scores: {auc_scores}')


# In[ ]:


small_count=0
fit_count=0
large_count=0
for i in range(len(y_pred_prob)):
  if y_pred_prob[i][0]>y_pred_prob[i][1] and y_pred_prob[i][0]>y_pred_prob[i][2]:
    small_count+=1
  elif y_pred_prob[i][1]>y_pred_prob[i][0] and y_pred_prob[i][1]>y_pred_prob[i][2]:
    fit_count+=1
  else:
    large_count+=1

print(small_count)
print(fit_count)
print(large_count)


# In[ ]:


small_acc = 0
fit_acc = 0
large_acc = 0
for i in range(len(y_pred)):
  if y_pred[i]==0 and y_test[i]==0:
    small_acc+=1
  elif y_pred[i]==1 and y_test[i]==1:
    fit_acc+=1
  elif y_pred[i]==2 and y_test[i]==2:
    large_acc+=1

print(small_acc)
print(fit_acc)
print(large_acc)


# In[ ]:


true_small = 0
true_fit = 0
true_large = 0
for i in range(len(y_pred)):
  if y_test[i]==0:
    true_small+=1
  elif y_test[i]==1:
    true_fit+=1
  elif y_test[i]==2:
    true_large+=1


# In[ ]:


report = classification_report(y_test, y_pred, target_names=['Small', 'Fit', 'Large'])

print(report)


# In[ ]:


# In the top 73% data, how many are with label 1
def getTopNPositive(percentile,label,prob_list,test_data):
  # Convert inputs to numpy arrays if they aren't already
    prob_list = np.array(prob_list)
    test_data = np.array(test_data)

    # Step 1: Sort predictions for the specified label and get indices
    sorted_indices = np.argsort(prob_list[:, label])[::-1]

    # Step 2: Select top N% of the records
    top_n_percent = int(len(prob_list) * (percentile / 100))
    selected_indices = sorted_indices[:top_n_percent]

    # Step 3: Fetch the corresponding true labels
    selected_true_labels = test_data[selected_indices]

    # Step 4: Count how many of these are actually labeled as the specified label
    correct_predictions = np.sum(selected_true_labels == label)

    return correct_predictions/top_n_percent


# In[ ]:


getTopNPositive(73,1,y_pred_prob,y_test)


# In[ ]:


prob_list = np.array(y_pred_prob)
test_data = np.array(y_test)
sorted_indices = np.argsort(prob_list[:,1])[::-1]
top_n_percent = int(len(prob_list) * (73 / 100))
selected_indices = sorted_indices[top_n_percent:]
selected_labels = test_data[selected_indices]
selected_prob  = prob_list[selected_indices]
new_label = []
for i,_ in enumerate(selected_prob):
  large_prob = selected_prob[i][2]
  small_prob = selected_prob[i][0]
  if large_prob >= small_prob:
    new_label.append(2)
  else:
    new_label.append(0)
print(classification_report(selected_labels, new_label, target_names=['Small', 'Fit', 'Large']))


# In[ ]:


# Now, find the indices where the true label is not 'fit' (1) but the predicted label is 'fit' (1)
incorrect_fit_indices = np.where((y_test != 1) & (y_pred == 1))

# Extract the probabilities for these specific cases
# This gives you the probabilities assigned to 'fit' for the wrongly predicted samples
incorrect_fit_probabilities = y_pred_prob[incorrect_fit_indices]
incorrect_labels = y_test[incorrect_fit_indices]



# In[ ]:


incorrect_labels


# In[ ]:


incorrect_fit_probabilities


# In[ ]:


# Assuming y_test is your true labels and y_pred_labels is your predicted labels
conf_matrix = confusion_matrix(y_test, y_pred)

# The diagonal elements of the confusion matrix correspond to correct predictions (true positives)
# For each class, divide the true positive count by the total actual instances of that class (the sum of the corresponding row in the confusion matrix)
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Now class_accuracies[i] will give you the accuracy for class i
for i, accuracy in enumerate(class_accuracies):
    print(f'Accuracy for class {i}: {accuracy:.2f}')


# In[ ]:


def getAcc(y_t,y_p):
  conf_matrix = confusion_matrix(y_t, y_p)
  class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=0)
  return class_accuracies


# In[ ]:


getAcc(y_test,y_pred)


# In[ ]:





# In[ ]:


t = np.arange(0.0, 0.5, 0.005)
fit_thresh = np.arange(0.3,0.8,0.01)
best_t = 1.0
best_fit_t = 0.0
best_acc = 0
for fit_thresh in fit_thresh:
  for thresh in t:
    new_labels = []
    for i in range(len(y_pred_prob)):
      large_prob = y_pred_prob[i][2]
      fit_prob = y_pred_prob[i][1]
      small_prob = y_pred_prob[i][0]
      # Originally fit
      if fit_prob > small_prob and fit_prob > large_prob:
        if fit_prob > fit_thresh:
          new_labels.append(1)
          continue
        if large_prob > small_prob:
          if large_prob + thresh > fit_prob:
            new_labels.append(2)
          else:
            new_labels.append(1)
        else:
          if small_prob + thresh > fit_prob:
            new_labels.append(0)
          else:
            new_labels.append(1)
      else:
        if large_prob > small_prob:
          new_labels.append(2)
        else:
          new_labels.append(0)
    acc = getAcc(y_test,new_labels)
    print(acc)
    print(f'Threshold: {thresh}, Accuracy: {acc}', f'Fit Threshold: {fit_thresh}')


# In[ ]:


for thresh in t:
  new_labels = []
  for i in range(len(incorrect_fit_probabilities)):
    large_prob = incorrect_fit_probabilities[i][2]
    fit_prob = incorrect_fit_probabilities[i][1]
    small_prob = incorrect_fit_probabilities[i][0]
    if large_prob > small_prob:
      if large_prob * thresh > fit_prob:
        new_labels.append(2)
      else:
        new_labels.append(1)
    else:
      if small_prob * thresh > fit_prob:
        new_labels.append(0)
      else:
        new_labels.append(1)
  acc = accuracy_score(incorrect_labels, new_labels)
  if acc > best_acc:
    best_t = thresh
    best_acc = acc
    print(f'Threshold: {thresh}, Accuracy: {acc}')

