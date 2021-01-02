import numpy as np
import pandas as pd



class Pipeline(object):
    def __init__(self, columns):
        self.columns = columns
        self.selected_columns = set()
    @property
    def selected_columns(self):
        return self.__selected_columns
    @selected_columns.setter
    def selected_columns(self, s):
        self.__selected_columns = s
    def __call__(self, data):
        self.data = self.process(data)
        return self.data

class Binner(Pipeline):
    def __init__(self, columns, num_bins):
        super().__init__(columns)
        self.num_bins = num_bins
    def process(self, data):
        output  = data.copy()
        for col in self.columns:
            if col in self.selected_columns:
                output[col] = pd.cut(data[col], self.num_bins[col])
        return output
    
class OnehotEncoder(Pipeline):
    def __init__(self, columns):
        super().__init__(columns)
    def process(self, data):
        output = data.copy()
        for col in self.columns:
            if col in self.selected_columns:
                dummies = pd.get_dummies(data[col], prefix="", prefix_sep = col+"_", drop_first=True)
                output = pd.concat([dummies, output], axis=1)

                output.drop(col, axis=1, inplace=True)
        
        return output

class StdNormalization(Pipeline):
    def __init__(self, columns):
        super().__init__(columns)
    def process(self, data):
        output = data.copy()
        for col in self.columns:
            if col in  self.selected_columns:
                output[col] = output[col] - output[col].mean()
                output[col] = output[col]/ output[col].std()
        return output

class CompositePipeline(Pipeline):
    def __init__(self, pipelines):
        assert type(pipelines) == list
        self.pipelines = pipelines
    def process(self, data):
        output = data 
        for pipeline in self.pipelines:
            output = pipeline(output)
        return output
    @property
    def selected_columns(self):
        return self.__selected_columns
    @selected_columns.setter
    def selected_columns(self, s):
        
        for pipeline in self.pipelines:
            pipeline.selected_columns = s
        self.__selected_columns = s