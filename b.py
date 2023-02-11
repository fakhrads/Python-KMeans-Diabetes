import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

df=pd.read_excel("diabetes.xlsx")
print(df.head())

sns.set_style('whitegrid')
sns.lmplot(x='BMI', y='Age', data=df, hue='ClassOutcome',
           palette='coolwarm', height=60, aspect=1, fit_reg=False)
kmeans=KMeans(n_clusters=2, n_init=10)
print(kmeans)
kmeans.fit(df)

kmeans.cluster_centers_
print(df)
def converter(cluster):
    if cluster=='1':
        return 1
    else:
        return 0
df['Cluster'] = df['ClassOutcome'].apply(converter)
print(df.head())

print("Confusion Matrix: \n" ,confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))