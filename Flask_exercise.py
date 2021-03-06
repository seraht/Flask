"""
Flask exercise:
Python inference server file

Author: Serah

"""

import numpy as np
import pandas as pd
from flask import Flask, render_template
import os
from flask import request
from flask import jsonify, make_response

clf2 = pd.read_pickle("model.pkl")

app = Flask(__name__)


@app.route('/')
def main_page():
    return 'API for CHURNED predictions'


@app.route('/predict')
def predict():
    """ Return the churned predictions for a given customer  """
    to_predict = np.zeros(5).reshape(1, 5)
    features = ['is_male', 'num_interactions_with_cust_service', 'late_on_payment', 'age', 'years_in_contract']
    for i, feat in enumerate(features):
        if request.args.get(feat) is not None:
            to_predict[0][i] = request.args.get(feat)

    response = clf2.predict(to_predict)

    if response:
        return "The customer is likely to churn"
    else:
        return "He is a loyal customer"


@app.route('/predict_multiple', methods=['GET', 'POST'])
def predict_multiple():
    """ Return the churned predictions for multiple customers """
    req = request.json
    values = pd.DataFrame(data=req['data'])
    pred = clf2.predict(values)
    responses = {}
    for i, rep in enumerate(pred):
        if rep:
            responses[f"Customer {i + 1}"] = f"Input: {', '.join([str(values.loc[i, :][j]) for j in range(values.shape[1])])}, Output: is likely to churn"

        else:
            responses[f"Customer {i + 1}"] = f"Input: {', '.join([str(values.loc[i, :][j]) for j in range(values.shape[1])])}, Output: is a loyal customer"

    return make_response(jsonify(responses))

@app.route('/predict_html')
def predict_html():
    to_predict = np.zeros(5).reshape(1, 5)
    features = ['is_male', 'num_interactions_with_cust_service', 'late_on_payment', 'age', 'years_in_contract']
    for i, feat in enumerate(features):
        if request.args.get(feat) is not None:
            to_predict[0][i] = request.args.get(feat)

    response = clf2.predict(to_predict)

    if response:
        pred = "The customer is likely to churn"
    elif np.allclose(to_predict, np.zeros(5).reshape(1, 5)):
        pred = "Please enter values to the path"
    else:
        pred = "He is a loyal customer"

    try:
        return render_template('predict.html', predictions=pred, inputs=to_predict)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()
