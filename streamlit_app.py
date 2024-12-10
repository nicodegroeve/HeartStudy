import streamlit as st

"""# Project - Heart Study"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
plt.style.use('ggplot')

"""# Our Dataset"""

data=pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

data

st.write(data.describe())


"""# What can we see?
- patients: 111627
- Average age: 55
- Sex: M=5022, F=6605
"""
data.hist(figsize=(20,20))

"""More specifically, for our research questions:"""


selected_variable1 = st.selectbox('Which variable do you want to visualize:', ['LDLC', 'HDLC','TOTCHOL'])
st.write(f'### {selected_variable1} levels')
fig,ax = plt.subplots (figsize= (10,8))
ax.hist(x=data[selected_variable1])
st.pyplot(fig)



"""# Handling missing values
There are a lot of missing values in variables: TOTCHOL, CIGPDAY, BMI, BPMEDS, HEARTRATE, GLUCOSE, educ, HDLC and LDLC
Since we want to examine HDLC, LDLC and TOTCHOL, we need to remove the patients with missing values in order to start our analysis.
"""

data.loc[:, ['LDLC', 'HDLC', 'TOTCHOL']]

data_CHOL = data.dropna(subset = ['LDLC', 'HDLC', 'TOTCHOL'] )
data_CHOL.describe()

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
"Finally, there are also some outliers in the TOTCHOL group, ..."

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






tree_data = final_dataset[['LDLC', 'HDLC', 'CVD']]
X,y = final_dataset[['LDLC', 'HDLC']], final_dataset[['CVD']]




# step 1: train-test split
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size=0.20,
                                                    stratify=y,
                                                    random_state=25)

# step 2: pic the algorithm
from sklearn import tree
clf_1 = tree.DecisionTreeClassifier(random_state=42, max_depth= 4)

clf_1 = clf_1.fit(train_X, train_y)

plt.figure(figsize=(16,10))
tree.plot_tree(clf_1);

# step 3: train the algorithm
prediction = clf_1.predict(test_X)

print("Original Labels", test_y.values)
print("Labels Predicted", prediction)


# step 4: prediction

# Step 5: Evaluate the prediction
from sklearn.metrics import classification_report
print(classification_report(y_true=test_y, y_pred=prediction))

# confusion matrix

# step 1: train-test split
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size=0.20,
                                                    stratify=y,
                                                    random_state=25)

# step 2: pic the algorithm
from sklearn import tree
clf_1 = tree.DecisionTreeClassifier(random_state=42, max_depth= 4)

clf_1 = clf_1.fit(train_X, train_y)

plt.figure(figsize=(16,10))
tree.plot_tree(clf_1);


# step 3: train the algorithm
prediction = clf_1.predict(test_X)

print("Original Labels", test_y.values)
print("Labels Predicted", prediction)


# step 4: prediction

# Step 5: Evaluate the prediction
from sklearn.metrics import classification_report
print(classification_report(y_true=test_y, y_pred=prediction))

# confusion matrix

from sklearn.metrics import accuracy_score

accuracy_score(y_true=test_y, y_pred=prediction)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true=test_y,
                 y_pred=prediction)
cm

from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf_1.classes_)

disp.plot();
