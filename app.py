import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
#import render_template

app = Flask(__name__)


ridge_model = pickle.load(open("models /ridge.pkl", "rb"))
standard_scaler = pickle.load(open("models /scaler.pkl", "rb"))


@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        standardized_data = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        prediction = ridge_model.predict(standardized_data)
        return render_template('home.html', prediction_text=f"The FWI prediction is {prediction[0]}")


    else:
        return render_template('home.html', prediction_text="Please provide input data.")

if __name__ == "__main__":
    app.run(port=5001)