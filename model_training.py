import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string 
import joblib

#note: axis 0 means rows axis 1 means collums

#initializes data sets for true and fake data 
data_fake = pd.read_csv("data/Fake.csv")
data_true = pd.read_csv("data/True.csv")

data_fake.loc[:, "class"] = 0
data_true.loc[:, "class"] = 1


#Saves last 10 rows fom data to do manual testing after machine learns other rows
#the for loop removes the last 10 frmo the data set
data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis = 0, inplace = True)

data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis = 0, inplace = True)


data_fake_manual_testing.loc[:, "class"] = 0
data_true_manual_testing.loc[:, "class"] = 1


#merges data into one set and uses the class columns we initialized earlier to differentiate real and fake
data_merge = pd.concat([data_fake, data_true], axis = 0)


#drops unneccesary columns only keeps text and class. CHANGE THIS IF U WANT SUBJECT AND TITLE TO E INCLUDED
data = data_merge.drop(["title", "subject", "date"], axis = 1)


#randomly shuffles data
data = data.sample(frac = 1)


#Puts indexes in order after shuffling
data.reset_index(inplace = True)
data.drop(["index"], axis = 1, inplace = True)


#cleans data
def wordopt(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text) 
    text = re.sub(r"\\W", " ", text) 
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  
    text = re.sub(r"<.*?>+", "", text)  
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)  
    text = re.sub(r"\n", "", text) 
    text = re.sub(r"\w*\d\w*", "", text)
    return text

data["text"]= data["text"].apply(wordopt)

#x is independent y is dependent
x = data["text"]
y = data["class"]


#means 25 percent used for testing 75 percent used for training
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.05)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train) #takes vocab of text and uses it to tranfer it into numbers
xv_test = vectorization.transform(x_test)
joblib.dump(vectorization, 'vectorizer.pkl')


from sklearn.linear_model import LogisticRegression
#makes LR model and trains it to use the training data
LR = LogisticRegression()
LR.fit(xv_train, y_train)


from sklearn.tree import DecisionTreeClassifier
#makes DT model and trains it to use the training data
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


from sklearn.ensemble import GradientBoostingClassifier
#makes GB model and trains it to use the training data
GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)

from sklearn.ensemble import RandomForestClassifier
#makes GB model and trains it to use the training data
RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)

joblib.dump(LR, 'logistic_regression_model.pkl')
joblib.dump(DT, 'decision_tree_model.pkl')
joblib.dump(GB, 'gradient_boosting_model.pkl')
joblib.dump(RF, 'random_forest_model.pkl')
print("Dump successful")

