import pandas as pd
from sklearn.metrics import confusion_matrix
data = pd.read_csv("FOOTBALL-RESULTS.csv", delimiter=',')

#Label encoding for Strings
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in range(10):
    data.iloc[:, i] = le.fit_transform(data.iloc[:, i])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit_transform(data)

#Splitting the dataset into  training and validation sets
from sklearn.model_selection import train_test_split
training_set, validation_set = train_test_split(data, test_size = 0.2, random_state = 21)



#classifying the predictors and target variables as X and Y
X_train = training_set.iloc[:,0:-1].values
Y_train = training_set.iloc[:,-1].values
X_val = validation_set.iloc[:,0:-1].values
y_val = validation_set.iloc[:,-1].values


def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


#Importing MLPClassifier
from sklearn.neural_network import MLPClassifier




    # Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(100, 150, 100), max_iter=100, activation='relu', solver='lbfgs',
                               random_state=9)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_val)
cm = confusion_matrix(y_pred, y_val)


#Fitting the training data to the network




#Predicting y for X_val


#Importing Confusion Matrix

#Comparing the predictions against the actual observations in y_val

#Printing the accuracy
print ("Accuracy of MLPClassifier : ", accuracy(cm))
#Evaluating the algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))



# import matplotlib.pyplot as plt
# plt.ylabel('accuracy')
# plt.xlabel('iterations')
# plt.title("Number of iteration graph")
# plt.plot(mi,wetin)
# plt.show()