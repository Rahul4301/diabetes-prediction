import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('diabetes_extended.csv')

# Define features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Prepare data for the chart
logistic_regression_line = [{'x': x, 'y': model.predict_proba([[x] + [0] * (X.shape[1] - 1)])[0][1]} for x in range(int(X['Glucose'].min()), int(X['Glucose'].max()))]
user_data_points = [{'x': X_test.iloc[i]['Glucose'], 'y': model.predict_proba([X_test.iloc[i]])[0][1]} for i in range(len(y_pred))]

@app.route('/')
def index():
    return render_template('diabetes.html', logistic_regression_line=json.dumps(logistic_regression_line), user_data_points=json.dumps(user_data_points))

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    return render_template('diabetes.html', prediction=prediction, logistic_regression_line=json.dumps(logistic_regression_line), user_data_points=json.dumps(user_data_points))

if __name__ == '__main__':
    app.run(debug=True)
