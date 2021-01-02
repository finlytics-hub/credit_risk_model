import pandas as pd 
import numpy as np
from collections import defaultdict

class WoeComputer(object):
    def __init__(self, data, target_column = "creditability"):
        self.target_column = target_column
        self.woe = self.__compute_woe(data)
        
    def __compute_woe(self, data):
        counts = self.__get_count_dfs(data)
        for col in counts:
            counts[col]["total_distr"] = counts[col]["count"]/counts[col]["count"].sum()
            counts[col]["good_distr"] = counts[col]['good_counts']/counts[col]['good_counts'].sum() + np.finfo('float').eps
            counts[col]["bad_distr"] = counts[col]['bad_counts']/counts[col]['bad_counts'].sum()+ np.finfo('float').eps
            counts[col]['woe'] = np.log(counts[col]['good_distr']/counts[col]['bad_distr'])
            counts[col]['woe%'] = counts[col]['woe'] * 100
        return counts
    def __get_count_dfs(self, df):
        output = {}
        for col in df.columns[:-1]:
            current_col = {"attribute":[], "count":[], "good_counts":[], "bad_counts":[]}
            for (val, count) in df[col].value_counts().items():
                good_counts = (df[df[col] == val][self.target_column] == "good").sum()
                bad_counts = (df[df[col] == val][self.target_column] == "bad").sum()
                assert good_counts + bad_counts == count
                current_col["attribute"].append(val)
                current_col["good_counts"].append(good_counts)
                current_col["bad_counts"].append(bad_counts)
                current_col["count"].append(count)
            output[col] = pd.DataFrame(current_col)
        return output
class IvComputer(object):
    def __init__(self, data):
        self.weo_computer = WoeComputer(data)
        self.iv = self.__compute_iv(self.weo_computer)
    def __compute_iv(self, woe_computer):
        
        output = {"feature":[], "iv":[]}
        woe_df = self.weo_computer.woe.copy()
        
        for col in woe_df:
            woe_df[col]['good - bad'] = woe_df[col]["good_distr"] - woe_df[col]["bad_distr"]
            woe_df[col]["(good - bad) * woe"] = woe_df[col]['good - bad'] * woe_df[col]["woe"]
            
            iv = woe_df[col]["(good - bad) * woe"].sum()
            output["feature"].append(col)
            output["iv"].append(iv)
        output = pd.DataFrame(output)
        return output
    
