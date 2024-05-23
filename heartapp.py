import pickle
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import os
app = Flask(__name__)
model = pickle.load(open('D:\MLprojects\HeartAttackapp\heart_model_1.pkl', 'rb'))

#model = joblib.load('heart_model_1.joblib')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/')
def index():
    return render_template('heartindex.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='POST':
         age = int(request.form['age'])
         sex = int(request.form['sex'])
         chest_pain_type = int(request.form['chestPainType'])
         resting_bp = int(request.form['restingBP'])
         cholesterol = int(request.form['cholesterol'])
         fasting_bs = int(request.form['fastingBS'])
         resting_ecg = int(request.form['restingECG'])
         max_hr = int(request.form['maxHR'])
         exercise_angina = int(request.form['exerciseAngina'])
         oldpeak = float(request.form['oldpeak'])
         st_slope = int(request.form['stSlope'])
         input_data = (cholesterol,max_hr,oldpeak)
    


         input_dta_arr=np.asarray(input_data)
         input_data_reshaped=input_dta_arr.reshape(1,-1)
         prediction=model.predict(input_data_reshaped)
    
         res=""
         if prediction[0] == 0:
             res = 'Low risk of heart attack'
         else:
             res = 'High risk of heart attack'
         return render_template('heartresult.html',result=res)
    
if __name__ == '__main__':
    app.run(debug=True)