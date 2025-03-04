from flask import Flask, render_template_string
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

app = Flask(__name__)

# Load data and train model
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Generate metrics
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Define a simple HTML template
template = '''
<!DOCTYPE html>
<html>
<head>
    <title>KNN Model Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h2 { color: #333; }
        pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h2>K-Nearest Neighbors (KNN) Model Results</h2>
    <h3>Confusion Matrix:</h3>
    <pre>{{ conf_matrix }}</pre>
    <h3>Classification Report:</h3>
    <pre>{{ class_report }}</pre>
    <h3>Accuracy Score:</h3>
    <p>{{ accuracy }}</p>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(template, 
                                  conf_matrix=conf_matrix, 
                                  class_report=class_report, 
                                  accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
