from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file containing the dataset
data = pd.read_csv("data.csv", delimiter=',')
print(data.describe())

pre_processed_data = pd.read_csv("data.csv", delimiter=',')

# Encoding  String values from the dataset into numbers
le = preprocessing.LabelEncoder()
num_features = 10
for i in range(num_features) :
    data.iloc[:, i] = le.fit_transform(data.iloc[:, i]) + 1

# Selects Win/Loss/Draw column for labels

y = data['Win/Loss/Draw'].values

# Dropping  irrelevant columns and extracting  values from other
x = data.drop(columns=['date', 'Win/Loss/Draw', 'tournament', 'city', 'country', 'neutral'], axis=1).values

# splitting data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)

# metric= minkowski, euclidean, manhattan, hamming
classifier = KNeighborsClassifier(4, metric='minkowski')
# prints params of the KNN clasifier
print(classifier)
# fitting the model using X and Y training values
classifier.fit(X_train, y_train)

# predicting class labels for the given X_test set
predicted_label = classifier.predict(X_test)

# row number in the dataset used  to see the prediction for the particular row
idx = 612

# FINDING THE OPTIMAL K VALUE
# THIS CODE IS COMMENTED BECAUSE THE OPTIMAL VALUE HAS BEEN FOUND

# error_rate = []
# for i, idx in enumerate(range(1, 20)):
#
#     print(i)
#     print("error: ", np.mean(error_rate))
#     classifier = KNeighborsClassifier(idx)
#     classifier.fit(X_train, y_train)
#     predicted_label = classifier.predict(X_test)
#     error_rate.append(np.mean(predicted_label != y_test))

# plt.plot(range(1, 20), error_rate, color='blue', linestyle='-', markerfacecolor='red', markersize=1)
# plt.title('Error Rate vs K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()
print("1 = Home win, 2 = Away Win, 3 = Draw")
print("teams:", pre_processed_data['home_team'][idx], pre_processed_data['away_team'][idx], "expected label:",
      pre_processed_data['Win/Loss/Draw'][idx], "predicted label: ", predicted_label[idx])

# Evaluation
print(confusion_matrix(y_test, predicted_label))
print(classification_report(y_test, predicted_label))
print("accuracy: {0:.3f}%".format(accuracy_score(y_test, predicted_label) * 100))


