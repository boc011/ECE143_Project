#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import pandas as pd
import gzip
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
from scipy.sparse import hstack
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# In[ ]:


clean_dataset_path = "/content/drive/MyDrive/AIHW/AS2/caius_data_clean.csv"
df = None
if not os.path.exists(clean_dataset_path):
  df = None
  print("Empty Dataset")
else:
  df = pd.read_csv(clean_dataset_path)


# In[ ]:


average_height = df['height'].dropna().mean()
df['height'] = df['height'].fillna(average_height)
average_rating = df['rating'].dropna().mean()
df['rating'] = df['rating'].fillna(average_rating)


# In[ ]:


scaler = StandardScaler()
numerical_data = scaler.fit_transform(df[['size','height']])
fit_mapping = {'small': 0, 'fit': 1, 'large': 2}
y = df['fit'].map(fit_mapping).values
X = numerical_data


# In[ ]:


over = SMOTE(sampling_strategy={0: int(len(y) * 0.5), 2: int(len(y) * 0.5)}, k_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Define the pipeline
pipeline = Pipeline(steps=[('o', over)])
# Apply the pipeline to your data
X_train, y_train = pipeline.fit_resample(X_train, y_train)


# In[ ]:


# Create the SVM model
model = svm.SVC(kernel='rbf',verbose = True, max_iter = 1500,C = 2)

# Train the model
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


decision_scores = model.decision_function(X_test)


# In[ ]:


# Binarize the output
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

# Initialize variables to store the cumulative AUC
auc_sum = 0
n_classes = y_test_binarized.shape[1]

# Compute ROC AUC for each class
for i in range(n_classes):
    # Compute ROC curve and AUC for this class
    fpr, tpr, _ = metrics.roc_curve(y_test_binarized[:, i], decision_scores[:, i])
    roc_auc = metrics.auc(fpr, tpr)
    auc_sum += roc_auc
    # Optionally, you can plot the ROC curves here

# Compute the average AUC
average_auc = auc_sum / n_classes
print(f"Average AUC: {average_auc}")


# In[ ]:


decision_scores


# In[ ]:


auc_score =  roc_auc_score(y_test, decision_scores,multi_class = 'ovr')


# In[ ]:


# Compute AUC
probabilities = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, probabilities)


# In[ ]:


report = classification_report(y_test, y_pred, target_names=['Small', 'Fit', 'Large'])

print(report)


# In[ ]:


y_test.shape

