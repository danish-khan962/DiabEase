from django .shortcuts import render

#Importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Major and main dependencies
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score



def model(request):
    return render(request, 'model.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    result1 = None  # Start with no result

    if request.method == 'GET':  # Ensure data is being processed after form submission
        # Loading the dataset
        diabetes_dataset = pd.read_csv(r'C:\Users\hp\DiabEase\DiabetesPrediction\assets\diabetes.csv')
        diabetes_df = pd.DataFrame(diabetes_dataset)

        # Separating data and labels
        X = diabetes_df.drop(columns='Outcome', axis=True)
        Y = diabetes_df['Outcome']

        # Standardizing the data
        scaler = StandardScaler()
        scaler.fit(X)  # Fit the scaler on the training data

        # Feeding standardized data to X
        X = scaler.transform(X)  # Scale the features
        
        # Train-test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2, stratify=Y)

        # Training the model
        classifier = svm.SVC(kernel='linear')
        classifier.fit(X_train, Y_train)

        # Extracting form input values
        val1 = float(request.GET['id1'])
        val2 = float(request.GET['id2'])
        val3 = float(request.GET['id3'])
        val4 = float(request.GET['id4'])
        val5 = float(request.GET['id5'])
        val6 = float(request.GET['id6'])
        val7 = float(request.GET['id7'])
        val8 = float(request.GET['id8'])

        # Print values for debugging
        print(val1, val2, val3, val4, val5, val6, val7, val8)

        # Standardizing the input values (ensure same scaling)
        input_values = np.array([[val1, val2, val3, val4, val5, val6, val7, val8]])
        scaled_input = scaler.transform(input_values)

        # Making the prediction
        prediction_result = classifier.predict(scaled_input)

        # Handle prediction result
        if prediction_result == 1:
            result1 = "Result is POSITIVE.."
        elif prediction_result == 0:
            result1 = "Result is NEGATIVE.."

    return render(request, 'predict.html', {"result2": result1})
