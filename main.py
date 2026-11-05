import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Imports all of the required libraries ^^^ [NOTE : Doesn't work if the libraries aren't already installed]

#Let's start by importing our dataset [https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset]

df=pd.read_csv('diabetes_prediction_dataset.csv')

