from flask import Flask,render_template,url_for,request
import pandas as pd
import math
import pickle
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    rf = open("RF_Reg.pkl", "rb")
    rf_reg = pickle.load(rf)

    if request.method == 'POST':
        hum = request.form['humidity']
        aTemp = request.form['aTemp']
        workDay = request.form['workDay']
        year = request.form['year']
        month = request.form['month']
        hour = request.form['hour']
        weather = request.form['weather']
        season = request.form['season']

        hum = float(hum)
        aTemp = float(aTemp)
        workDay = int(workDay)
        year = int(year)
        month = int(month)
        hour = int(hour)
        weather = int(weather)
        season = int(season)

        data = []
        data = [hum, aTemp, workDay, year, month, hour, weather, season]
        my_prediction = math.floor(rf_reg.predict(pd.DataFrame(data).T))
        my_prediction = int(my_prediction)
        my_prediction = np.abs(my_prediction)

    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(host='0.0.0.0',port=5000)