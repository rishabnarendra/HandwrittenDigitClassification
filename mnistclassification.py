# Import the libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("./datasets/train.csv")
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Split the dataset into train set and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32)

# Train the decision tree model on the dataset 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Plot one of our test set values 
# data = X_test[81]
# data.shape = (28, 28)
# plt.imshow(255 - data, cmap = 'gray')
# plt.show()

# Predict the test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))