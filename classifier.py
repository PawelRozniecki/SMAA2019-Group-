from sklearn.model_selection import train_test_split
from PyQt5.QtWidgets import QTableWidget,QTableWidgetItem

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data.csv", delimiter=',')

x = data.iloc[: , :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state = 0)


print(len(data))
print(len(X_train))
print(len(X_test))
