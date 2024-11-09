import json
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# NetworkApi c++
API_URL = 'http://localhost:5001/predict'

# root route
@app.route('/', methods = ['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.methods == 'POST':
        feature1 = request.form.get('feature1', type = float)
        feature2 = request.form.get('feature2', type = float)
    
        data = {
            'feature1': feature1,
            'feature2': feature2
        }

        try:
            # requests to Api
            response = requests.post(API_URL, data = json.dump(data), headers= {'Content-Type': 'application/json'})

            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
            else:
                error = f'Error: {response.status_code}'
        except Exception as e:
            error = f'Error Connecting to API: {str(e)}'
    
    return render_template('index.html', prediction = prediction, error = error)

if __name__ == '__main__':
    app.run(debug = True)