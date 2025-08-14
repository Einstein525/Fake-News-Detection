import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
# Load the CSV file into a DataFrame
df = pd.read_csv('data.csv')

# Fill missing values in the 'Body' column with an empty string
df['Body'] = df['Body'].fillna('')

ps = PorterStemmer()
def wordopt(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    return text
from sklearn.model_selection import train_test_split
X = df['Body']
Y = df['Label']

# Split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
from sklearn.feature_extraction.text import TfidfVectorizer
# Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Create the LogisticRegression model
lr = LogisticRegression()

# Fitting training set to the model
lr.fit(xv_train, y_train)

# Predicting the test set results based on the model
lr_y_pred = lr.predict(xv_test)

# Calculate the accuracy score of this model
score_lr = accuracy_score(y_test, lr_y_pred)
print('Accuracy of LogisticRegression model is', score_lr)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Create the SVM model
lr = LogisticRegression()

# Fitting training set to the model
lr.fit(xv_train, y_train)

# Predicting the test set results based on the model
lr_y_pred = lr.predict(xv_test)

# Calculate the accuracy score of this model
score_lr = accuracy_score(y_test, lr_y_pred)
print('Accuracy of LogisticRegression model is', score_lr)
print('Accuracy of LogisticRegression model in percentage', score_lr*100)
#Accuracy of LogisticRegression model is 0.9680957128614157
#Accuracy of LogisticRegression model in percentage 96.80957128614158

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Create the SVM model
svm_model = SVC()

# Fitting training set to the model
svm_model.fit(xv_train, y_train)

# Predicting the test set results based on the model
svm_y_pred = svm_model.predict(xv_test)

# Calculate the accuracy score of this model
score = accuracy_score(y_test, svm_y_pred)
print('Accuracy of SVM model is', score)
print('Accuracy of SVM model in percentage', score*100)
#Accuracy of SVM model is 0.9880358923230309
#Accuracy of SVM model in percentage 98.8035892323031

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Create the Random model
rfc_model = RandomForestClassifier()

# Fitting training set to the model
rfc_model.fit(xv_train, y_train)

# Predicting the test set results based on the model
rfc_y_pred = rfc_model.predict(xv_test)

# Calculate the accuracy score of this model
score_rfc = accuracy_score(y_test, rfc_y_pred)
print('Accuracy of RandomForestClassifier model is', score_rfc)
print('Accuracy of RandomForestClassifier model in percentage', score_rfc*100)
#Accuracy of RandomForestClassifier model is 0.967098703888335
#Accuracy of RandomForestClassifier model in percentage 96.7098703888335

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
# Create the PassiveAggressiveClassifier  model
Pac_model = PassiveAggressiveClassifier()

# Fitting training set to the model
Pac_model.fit(xv_train, y_train)

# Predicting the test set results based on the model
Pac_y_pred = Pac_model.predict(xv_test)

# Calculate the accuracy score of this model
score_Pac = accuracy_score(y_test, Pac_y_pred)
print('Accuracy of PassiveAggressiveClassifier model is', score_Pac)
print('Accuracy of PassiveAggressiveClassifier model in percentage', score_Pac*100)
#Accuracy of PassiveAggressiveClassifier model is 0.9830508474576272
#Accuracy of PassiveAggressiveClassifier model in percentage 98.30508474576271

def fake_news_det(news):
    input_data = {"text": [news]}
    new_def_test = pd.DataFrame(input_data)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    vectorized_input_data = vectorization.transform(new_x_test)
    prediction = svm_model.predict(vectorized_input_data)
    if prediction == 1:
        return "Not a Fake News"
    else:
        return "Fake News"
root = tk.Tk()
root.title("Fake News Detector")

def check_news():
    news = news_entry.get("1.0", "end-1c")
    if news == "":
        messagebox.showerror("Error", "Please enter a news")
    else:
        result = fake_news_det(news)
        messagebox.showinfo("Result", result)

result_label = tk.Label(root, text="", font=("Helvetica", 12))
result_label.pack()
news_label = tk.Label(root, text="Enter the news:")
news_label.pack()
news_entry = tk.Text(root,height=5, width=50)
news_entry.pack()

check_button = tk.Button(root, text="Check", command=check_news)
check_button.pack()

root.mainloop()
