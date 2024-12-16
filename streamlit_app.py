import streamlit as st

"""# Project - Heart Study"""


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


selected_variable1 = st.selectbox('Which variable do you want to visualize:', ['LDLC', 'HDLC','TOTCHOL'])
st.write(f'### {selected_variable1} levels')
fig,ax = plt.subplots (figsize= (10,8))
ax.hist(x=data[selected_variable1])
st.pyplot(fig)



"""# Handling missing values
There are a lot of missing values in variables: TOTCHOL, CIGPDAY, BMI, BPMEDS, HEARTRATE, GLUCOSE, educ, HDLC and LDLC.
Since we want to examine HDLC, LDLC and TOTCHOL, we need to handle these missing values before proceeding with the analysis.
"""

data.loc[:, ['LDLC', 'HDLC', 'TOTCHOL']]

data_CHOL = data.dropna(subset = ['LDLC', 'HDLC', 'TOTCHOL'] )
data_CHOL.describe()

"""In this table you can see that there are quite a few LDLC and HDLC missing values (8600), this is due to the fact that the values were only measured in period 3. (Explain periods) 

The TOTCHOL outliers were only 409 and were not related to any periods, thus we decided to impute using the KNN imputer (5 neighbors)
"""

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


"""# Models"""

"""Which ones we tried: outcomes..."""

"""# Conlcusion"""
"""In conclusion we can see that these variables ..."""








