import numpy as np
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
            
             
        