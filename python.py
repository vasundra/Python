
# coding: utf-8

# In[5]:

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[6]:

def get_iris_data():
    if os.path.exists("f11.csv"):
        df = pd.read_csv("f11.csv")  #data from local file irish.csv
        return df
    else:
        print("Data File Not Found!")

df=get_iris_data()
print("*** df.head()", df.head(), sep="\n", end="\n\n")
print("*** df.tail()", df.tail(), sep="\n", end="\n\n")

print("*** iris types:", df["YN"].unique(), sep="\n")


# In[8]:

def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

df_processed, targets = encode_target(df, "YN")
print("*** df.head()", df_processed[["Target", "YN"]].head(),sep="\n", end="\n\n")
print("*** df.tail()", df_processed[["Target", "YN"]].tail(),sep="\n", end="\n\n")
print("*** iris types - targets:", df_processed["YN"].unique(), sep="\n")
print(df_processed["Target"].unique(), sep="\n", end="\n\n")


# In[9]:

features = list(df_processed.columns[:8])
print("*** features:", features,sep="\n")


# In[1]:

y = df_processed["Target"]
X = df_processed[features]

X,X_test,y,y_test=train_test_split(X, y,test_size=0.3, random_state=99)

dt = MultinomialNB()
dt.fit(X, y)
r=count(X_test)
print(r)
dt.predict(X_test)

dt.score(X_test,y_test)
#print(len(y),len(y_test))
#print(len(X),len(X_test))


# In[9]:

dt.score(X_test,y_test)


# In[ ]:



