from flask import Flask
from flask import jsonify
import numpy as np
import model

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to House price prediction api"


@app.route('/predict/<string:test_data>')
def predict(test_data):
    input_value_list = test_data.split(',')
    input_value_list = list(map(str, input_value_list))
    test_data = [np.array(input_value_list)]
    predicted_price = model.makePrediction(test_data).ravel()
    result = {
        "per capita crime rate by town": input_value_list[0],
        "proportion of residential land zoned for lots over 25,000 sq.ft.": input_value_list[1],
        "proportion of non-retail business acres per town.": input_value_list[2],
        "Charles River dummy variable (1 if tract bounds river; 0 otherwise)": input_value_list[3],
        "nitric oxides concentration (parts per 10 million)": input_value_list[4],
        "average number of rooms per dwelling": input_value_list[5],
        "proportion of owner-occupied units built prior to 1940": input_value_list[6],
        "weighted distances to five Boston employment centres": input_value_list[7],
        "index of accessibility to radial highways": input_value_list[8],
        "full-value property-tax rate per $10,000": input_value_list[9],
        "pupil-teacher ratio by town": input_value_list[10],
        "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town": input_value_list[11],
        "percentage lower status of the population": input_value_list[12],
        "Predicted House Price":str(predicted_price[0])
    }
    print(result)
    return jsonify(result)

app.run(debug=True)