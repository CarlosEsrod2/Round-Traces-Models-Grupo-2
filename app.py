#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def predict_h3_clasificacion(team_starting_equipment_value, round_kills, survived):
    """H3 Clasificación Model: Predict based on Combined features"""
    model_h3 = pickle.load(open('checkpoints/random_forest_h3_Clasificacion.pkl', 'rb'))
    sample_data = np.array([[team_starting_equipment_value, round_kills, survived]])
    prediction = model_h3.predict(sample_data)
    probability = model_h3.predict_proba(sample_data)
    return prediction[0], probability[0]

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        
        try:
            model_type = form_data.get('model_type')
            
            if model_type == 'h3_clasificacion':
                team_equipment_value_str = form_data.get('team_equipment_value_h3', '').strip()
                round_kills_str = form_data.get('round_kills', '').strip()
                survived_str = form_data.get('survived', '').strip()
                
                if not team_equipment_value_str:
                    raise ValueError("Team Starting Equipment Value is required")
                if not round_kills_str:
                    raise ValueError("Round Kills is required")
                if not survived_str:
                    raise ValueError("Survived field is required")
                
                team_equipment_value = float(team_equipment_value_str)
                round_kills = int(round_kills_str)
                survived = int(survived_str)
                
                # Validate survived value (should be 0 or 1)
                if survived not in [0, 1]:
                    raise ValueError("Survived must be 0 or 1")
                
                prediction, probability = predict_h3_clasificacion(team_equipment_value, round_kills, survived)
                model_name = "H3 (Classification - Combined Features)"
                features_used = f"TeamStartingEquipmentValue: {team_equipment_value}, RoundKills: {round_kills}, Survived: {survived}"
            
            else:
                return render_template("result.html", 
                                     prediction="Error: Invalid model type selected",
                                     model_name="",
                                     features_used="",
                                     probability_won="",
                                     probability_lost="")
            
            # Interpret results: 0 = Round Won, 1 = Round Lost
            if int(prediction) == 0:
                result_text = 'Round Won'
                result_class = 'win'
            elif int(prediction) == 1:
                result_text = 'Round Lost'
                result_class = 'loss'
            else:
                result_text = f'Unknown result: {int(prediction)}'
                result_class = 'unknown'
            
            # Probability formatting
            prob_won = f"{probability[0]:.2%}"
            prob_lost = f"{probability[1]:.2%}"
            
        except (ValueError, KeyError) as e:
            result_text = f'Error in data format: {str(e)}'
            model_name = ""
            features_used = ""
            prob_won = ""
            prob_lost = ""
            result_class = 'error'

        return render_template("result.html", 
                             prediction=result_text,
                             model_name=model_name,
                             features_used=features_used,
                             probability_won=prob_won,
                             probability_lost=prob_lost,
                             result_class=result_class)


if __name__=="__main__":

    app.run(port=5001)