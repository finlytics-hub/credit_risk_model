from flask import Flask
from flask_cors import CORS
import numpy as np
import pandas as pd
import sys
from flask import render_template
from flask import request
import pickle
import os

class ScoreCardBuilder(object):
    def __init__(self, working_dir = "working-dir",pdo = 20, target_score = 800, target_odds=200):
        self.working_dir = working_dir
        self.load_state()
        self.pdo = pdo
        self.target_score = target_score
        self.target_odds = target_odds
        self.factor = pdo/np.log(2)
        self.offset = target_score - self.factor * np.log(target_odds)
        
        
    def load_state(self):
        
        with open(os.path.join(self.working_dir, "attributes_woe.json"), "rb") as input_file:
            self.attributes_woe = pickle.load(input_file)
            
        with open(os.path.join(self.working_dir, "feature-weights.json"), "rb") as input_file:
            self.feature_weights = pickle.load(input_file)
            
            
        
    def compute_score(self, feature_values):
        output = 0.0
        output += self.feature_weights['intercept'] * self.factor + self.offset
        for feature in feature_values:
            attribute = feature_values[feature]
            attribute_woe = self.attributes_woe[feature][attribute]
            feature_weight = self.feature_weights[feature]
            
            factor = self.pdo/np.log(2)
            offset = self.target_score + factor * np.log(self.target_odds)
            
            output += (feature_weight * attribute_woe) * factor
            
        return output
scorecardBuilder = ScoreCardBuilder("../notebooks/working-dir")
def get_feature_names():
    path2bin = "../notebooks/working-dir/data.bin"

    data = pd.read_csv(path2bin)
    
    return list(data.columns[:-1])
def parse_interval(interval):
    '(3.932, 10.8]'
    left, right = list(map(float, interval.replace("(", "").replace("]","") .split(",")))
    return pd._libs.interval.Interval(left, right, closed="right")
column_names = get_feature_names()
interval_columns = ["duration.in.month", "age.in.years", "credit.amount"]
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/get_scorecard")
def get_scorecard():
    params = {}
    for col in column_names:
        if col in interval_columns:
            params[col] = parse_interval(request.args.get(col))
        else:
            p = request.args.get(col)
            if p.isdigit():
                params[col] = int(p)
            else:
                params[col] = p
    score = scorecardBuilder.compute_score(params)
    if score < 350:
        score = 350
    if score > 1000:
        score = 1000
    score = int(score)
    return {"score":score}