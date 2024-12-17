import streamlit as st

"""# Project 2 - More Extended Heart Study"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
plt.style.use('ggplot')

"""Our research question: Are LDL, HDL and total cholesterol levels linked to getting diagnosed with CVD?"""

"""# Our Dataset"""

data=pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

data

"""We followed with looking at the statistics of the data:"""

st.write(data.describe())


"""# What can we see?
- patients: 111627
- Average age: 55
- Sex: M=5022, F=6605
"""
data.hist(figsize=(20,20))

"""More specifically, for our research questions:"""


selected_variable1 = st.selectbox('Which variable do you want to visualize:', ['LDLC', 'HDLC','TOTCHOL', 'CVD'])
st.write(f'### {selected_variable1} levels')
fig,ax = plt.subplots (figsize= (10,8))
ax.hist(x=data[selected_variable1])
st.pyplot(fig)



"""# Handling missing values
There are a lot of missing values in variables: TOTCHOL, CIGPDAY, BMI, BPMEDS, HEARTRATE, GLUCOSE, educ, HDLC and LDLC.
Since we want to examine HDLC, LDLC and TOTCHOL, we need to handle these missing values before proceeding with the analysis.
"""

data.loc[:, ['LDLC', 'HDLC', 'TOTCHOL']]

data_CHOL = data.dropna(subset = ['LDLC', 'HDLC'] )
data_CHOL.describe()

from sklearn.impute import KNNImputer
TOTCHOL_data = data[['TOTCHOL']]
knn_imputer = KNNImputer(n_neighbors = 5)
TOTCHOL_imputed = knn_imputer.fit_transform(TOTCHOL_data)
data.loc[:, 'TOTCHOL'] = TOTCHOL_imputed

"""In this table you can see that there are quite a few LDLC and HDLC missing values (8600), this is due to the fact that the values were only measured in period 3. (3 different examination cycles) 

The TOTCHOL outliers were only 409 and were not related to any periods, thus we decided to impute using the KNN imputer (5 neighbors)
"""

"""Distribution of patients in the 3 periods: """

st.write(data.PERIOD.value_counts())

"""After handeling the missing values:"""

show_missing_values = ['LDLC','HDLC','TOTCHOL'] 
st.write(data_CHOL[show_missing_values].isnull().sum())

"""# Outliers"""
fig, ax = plt.subplots()
sns.boxplot(data=data_CHOL, orient='v', x='CVD', y='LDLC', ax = ax)
st.pyplot(fig)

"""According to our research, LDLC levels can increase up to 600mg/dL in cases of severe genetic disorders, this can significantly increase the risk for cardiovascular disease. Therefore, we decided to keep our outliers so that we can include patients with these disorders as well."""
fig, ax = plt.subplots()
sns.boxplot(data=data_CHOL, orient='v', x='CVD', y='HDLC', ax = ax)
st.pyplot(fig)
"""According to outside sources, values above 100mg/dl can be considered an outlier. However, since we still have a lot of values within this range, we decided to only remove the values above 125."""
fig, ax = plt.subplots()
sns.boxplot(data=data, orient='v', x='CVD', y='TOTCHOL', ax = ax)
st.pyplot(fig)
"Finally, there are also some outliers in the TOTCHOL group, here similar to LDLC the levels can increase to very high levels in certain disorders. We did not want to exclude these people"

"""There are some outliers in variables: LDLC and HDLC. In order to continue with our data analysis, we need to correct these."""


"""Statistics of our final dataset"""
data_CHOL = data_CHOL.loc[(data_CHOL.HDLC<125), ]
final_dataset = data_CHOL.loc[:, ['HDLC','LDLC', 'TOTCHOL', 'CVD']]
st.write(final_dataset.describe())

"""# Features of our final dataset"""

"""patients: 3023"""

"""Average age: 60"""
"""Sex: M=1303, F=1720"""
"""Averge TOTCHOL = 236"""
"""23% of patients have CVD"""

""" # Correlations"""

selected_variable = st.selectbox('select which variable you want to see correlated to CVD:', ['LDLC', 'HDLC','TOTCHOL'])
st.write(f'### Correlation between {selected_variable} and CVD')
fig,axes = plt.subplots (figsize= (10,8))
sns.regplot(x=final_dataset[selected_variable], y= final_dataset['CVD']);
st.pyplot(fig)

"""Although not very prominent the analysis has shown that there is a small, positive association between LDL-C levels and a CVD diagnosis.

On the other hand the analysis has also shown a slightly larger, negative correlation between HDL-C levels and a CVD diagnosis.

Finally, for the total cholesterol there is a very slight positive correlation.
"""


"""# Models"""

"""Logistic Regression:"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score

# Fill missing values
data = data.fillna(0)

# Define features (X) and target (y)
X = data[['LDLC', 'HDLC']]  # Replace with appropriate feature columns
y = data['CVD']  # Replace with appropriate target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
y_pred_log = logistic_model.predict(X_test)

# Evaluate the model
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# Display confusion matrix
ConfusionMatrixDisplay.from_estimator(logistic_model, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

st.title("Logistic Regression Confusion Matrix")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(logistic_model, X_test, y_test, ax=ax)
ax.set_title("Logistic Regression Confusion Matrix")

st.pyplot(fig)

"""Accuracy = 0.75

AUC-score = 0.54"""


"""Random forest: """

random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

# Evaluate the Random Forest model
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Display confusion matrix for Random Forest
ConfusionMatrixDisplay.from_estimator(random_forest_model, X_test, y_test)
plt.title("Random Forest Confusion Matrix")
plt.show()

st.title("Random Forest Confusion Matrix")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(random_forest_model, X_test, y_test, ax=ax)
ax.set_title("Random Forest Confusion Matrix")

st.pyplot(fig)

"""Accuracy = 0.73

AUC score = 0.51"""

"""Support Vector Machine: """

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
print("\nSupport Vector Machine Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

# Display confusion matrix for SVM
ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test)
plt.title("SVM Confusion Matrix")
plt.show()

st.title("Support Vector Machine Confusion Matrix")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test, ax=ax)
ax.set_title("Support Vector Machine Confusion Matrix")

st.pyplot(fig)

"""Accuracy = 0.75

AUC-score = 0.51"""




"""# Conlcusion"""
"""In conclusion we can see that these variables are not highly correlated to CVD and thus will also not make good features to predict the presence of CVD. From the accuracy scores we see that this is true and that these models can not accuratly predict CVD from LDLC, HDLC and total cholesterol levels. 

Our best model is the random forest one, it has slightly lower accuracy, but the True negatives and False negatives are slightly better then the other models """








