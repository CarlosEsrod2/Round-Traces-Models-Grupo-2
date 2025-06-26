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

def predict_h1_regresion(features_dict):
    """H1 Regresión Model: Predict based on all features"""
    model_h1 = pickle.load(open('checkpoints/H1_rfr_model_random_forest_regressor.pkl', 'rb'))
    
    # Define the expected feature order (excluding target variable)
    feature_order = [
        'InternalTeamId', 'MatchId', 'RoundId', 'RoundWinner', 'MatchWinner',
        'Survived', 'AbnormalMatch', 'TimeAlive', 'TravelledDistance',
        'RLethalGrenadesThrown', 'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle',
        'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol',
        'FirstKillTime', 'RoundKills', 'RoundAssists', 'RoundHeadshots',
        'RoundFlankKills', 'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue',
        'MatchKills', 'MatchFlankKills', 'MatchAssists', 'MatchHeadshots',
        'Map_de_dust2', 'Map_de_inferno', 'Map_de_mirage', 'Map_de_nuke',
        'Team_CounterTerrorist', 'Team_Terrorist'
    ]
    
    # Create feature array in correct order
    sample_data = np.array([[features_dict[feature] for feature in feature_order]])
    prediction = model_h1.predict(sample_data)
    return prediction[0]

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
                    raise ValueError("El Valor del Equipo Inicial del Equipo es requerido")
                if not round_kills_str:
                    raise ValueError("Las Eliminaciones en la Ronda son requeridas")
                if not survived_str:
                    raise ValueError("El campo Supervivencia es requerido")
                
                team_equipment_value = float(team_equipment_value_str)
                round_kills = int(round_kills_str)
                survived = int(survived_str)
                
                # Validate survived value (should be 0 or 1)
                if survived not in [0, 1]:
                    raise ValueError("Supervivencia debe ser 0 o 1")
                
                prediction, probability = predict_h3_clasificacion(team_equipment_value, round_kills, survived)
                model_name = "H3 (Clasificación - Características Combinadas)"
                features_used = f"Valor del Equipo Inicial: {team_equipment_value}, Eliminaciones: {round_kills}, Supervivencia: {survived}"
            
            elif model_type == 'h1_regresion':
                # Collect all required features for H1 regression model
                features = {}
                
                # Required numeric fields
                required_fields = [
                    'InternalTeamId', 'MatchId', 'RoundId', 'RoundWinner', 'MatchWinner',
                    'Survived', 'AbnormalMatch', 'TimeAlive', 'RLethalGrenadesThrown',
                    'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 'PrimarySniperRifle',
                    'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol', 'FirstKillTime',
                    'RoundKills', 'RoundAssists', 'RoundHeadshots', 'RoundFlankKills',
                    'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue',
                    'MatchKills', 'MatchFlankKills', 'MatchAssists', 'MatchHeadshots',
                    'Map_de_dust2', 'Map_de_inferno', 'Map_de_mirage', 'Map_de_nuke',
                    'Team_CounterTerrorist', 'Team_Terrorist'
                ]
                
                # Validate and convert all fields
                for field in required_fields:
                    field_value = form_data.get(field, '').strip()
                    if not field_value:
                        raise ValueError(f"El campo {field} es requerido")
                    
                    if field == 'TravelledDistance':
                        features[field] = float(field_value)
                    else:
                        features[field] = int(field_value)
                
                # Handle TravelledDistance separately as it's float
                travelled_distance_str = form_data.get('TravelledDistance', '').strip()
                if not travelled_distance_str:
                    raise ValueError("TravelledDistance es requerido")
                features['TravelledDistance'] = float(travelled_distance_str)
                
                prediction = predict_h1_regresion(features)
                model_name = "H1 (Regresión - Todas las características)"
                features_used = "Todas las características del modelo H1"
                
                result_text = f'Valor predicho: {prediction:.2f}'
                result_class = 'regression'
                prob_won = ""
                prob_lost = ""
            
            else:
                return render_template("result.html", 
                                     prediction="Error: Tipo de modelo inválido seleccionado",
                                     model_name="",
                                     features_used="",
                                     probability_won="",
                                     probability_lost="")
            
            # Handle classification results (existing code for h3_clasificacion)
            if model_type == 'h3_clasificacion':
                # Interpret results: 0 = Round Won, 1 = Round Lost
                if int(prediction) == 0:
                    result_text = 'Ronda Ganada'
                    result_class = 'win'
                elif int(prediction) == 1:
                    result_text = 'Ronda Perdida'
                    result_class = 'loss'
                else:
                    result_text = f'Resultado desconocido: {int(prediction)}'
                    result_class = 'unknown'
                
                # Probability formatting
                prob_won = f"{probability[0]:.2%}"
                prob_lost = f"{probability[1]:.2%}"
            
        except (ValueError, KeyError) as e:
            result_text = f'Error en formato de datos: {str(e)}'
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