from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the dataset and train the model
data = pd.read_csv('diabetes_extended.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

@app.route('/')
def index():
    return render_template('diabetes.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    return render_template('diabetes.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
