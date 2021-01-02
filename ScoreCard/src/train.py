import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from .data import Dataset, split_dataset
from .woe_iv import IvComputer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from collections import defaultdict

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from .preprocess import Binner
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
import os
import pickle

from sklearn.model_selection import cross_val_score

class TrainingPipeline(object):
    def __init__(self, dataframe, preprocess_args, feature_selection_args, train_args = None, evaluation_args = None):
        self.dataframe = dataframe
        self.preprocess_args = preprocess_args
        self.feature_selection_args = feature_selection_args
        self.train_args = train_args
        self.evaluation_args = evaluation_args
        
        self.selected_features = self.feature_selection()
        self.dataframe = self.dataframe[self.selected_features + ["creditability"]]
        self.train_ds, self.test_ds = self.preprocess_data()
        self.model = LogisticRegression(class_weight={"bad":3, "good":2})
        self.train_model()
        
        
        

    def preprocess_data(self):
        preprocess_func = self.preprocess_args["func"]
        preprocess_func.selected_columns = self.selected_features
        df = preprocess_func(self.dataframe)
        dataset = Dataset.from_dataframe(df, "creditability")
        test_size = self.preprocess_args.get("test_size", 0.2)
        return split_dataset(dataset, test_size = test_size)
    
    def feature_selection(self):
        
        iv_computer = IvComputer(self.dataframe)
        
        
        if "num_features" in self.feature_selection_args:
            num_features = self.feature_selection_args.get("num_features")
            output = iv_computer.iv.sort_values("iv", ascending=False)['feature'].values.tolist()[:num_features]
            
        elif "threshold" in self.feature_selection_args:
            threshold = self.feature_selection_args["threshold"]
            output = []
            for index, row in iv_computer.iv.iterrows():
                if row["iv"]>=threshold:
                    output.append(row["feature"])
        elif "fraction" in self.feature_selection_args:
            fraction = self.feature_selection_args["fraction"]
            num_features = int(fraction * len(iv_computer.iv))
            output = iv_computer.iv.sort_values("iv", ascending=False)['feature'].values.tolist()[:num_features]
        else:
            raise ValueError("One of num_features, threshold or fraction should be provided")
        return output
        
    def train_model(self):
        
        return self.model.fit(self.train_ds.X, self.train_ds.y)
    def evaluate(self):
        y_pred = self.model.predict(self.test_ds.X)
        return classification_report(self.test_ds.y, y_pred)
    def show_performance(self):
        plt.figure(figsize=(10, 10))
        plt.title("Logistic Regression ROC curve")
        
            
        lr_probs = self.model.predict_proba(self.test_ds.X)
        lr_probs = lr_probs[:, 1]
        lr_fpr, lr_tpr, _ = roc_curve(self.test_ds.y=="good", lr_probs)
        a = auc(lr_fpr, lr_tpr)
        plt.plot(lr_fpr, lr_tpr, marker='.', label="auc= {:.2f}%".format(a * 100))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
        
        

class WoeLogisticRegressionTrainer(object):
    def __init__(self, dataset_path, working_dir = "working-dir"):
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)
        
        
        self.train_ds, self.test_ds = self.get_train_dataset(dataset_path)
        # self.model = SGDClassifier(loss="log", learning_rate = "adaptive", eta0 = 1e-3, early_stopping = True)
        self.model = LogisticRegression(class_weight={"bad":3, "good":2}, max_iter = 1000)
        
        
    def train_model(self):
    
        X_train = self.train_ds.X
        y_train = self.train_ds.y
        
        all_accuracies = cross_val_score(estimator=self.model, X=X_train, y=y_train, cv=5)
        print("all_accuracies", all_accuracies)
        print("Average cv accuracy: {:.2f}%".format(np.mean(all_accuracies)))
        self.model.fit(X_train, y_train)
        
        with open(os.path.join(self.working_dir, "model.pkl"), "wb") as model_file:
            pickle.dump(self.model, model_file)
        coef_ = self.model.coef_.flatten().tolist()
        features_weights = {}
        for i in range(len(self.feature_cols_order)):
            features_weights[self.feature_cols_order[i]] = coef_[i]
        features_weights["intercept"] =  self.model.intercept_[0]
        
        with open(os.path.join(self.working_dir, "feature-weights.json"), "wb") as output_file:
            pickle.dump(features_weights, output_file)
        
        
        
    def show_performance(self):
        plt.figure()
        plt.title("Logistic Regression ROC curve")
        
            
        lr_probs = self.model.predict_proba(self.test_ds.X)
        lr_probs = lr_probs[:, 1]
        lr_fpr, lr_tpr, _ = roc_curve(self.test_ds.y=="good", lr_probs)
        a = auc(lr_fpr, lr_tpr)
        plt.plot(lr_fpr, lr_tpr, marker='.', label="auc= {:.2f}%".format(a * 100))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.figure()
        plot_confusion_matrix(self.model, self.test_ds.X, self.test_ds.y)
        plt.show()
        
                    
    def get_train_dataset(self, path):
        df = pd.read_csv(path)
        cols_num_beans = {
            "age.in.years":20, 
            "duration.in.month":10,
            "credit.amount":50
            
        }
        cols_todo_bins = list(cols_num_beans)
        binner = Binner(cols_todo_bins, cols_num_beans)
        selected_colums = df.columns[:-1]
        binner.selected_columns = selected_colums
        binner(df)
        binner.data.to_csv(os.path.join(self.working_dir, "data.bin"), index=False)
        iv_computer = IvComputer(binner.data)
        new_df = self.replace_with_woes(binner.data, iv_computer.weo_computer.woe)
        
        self.feature_cols_order = list(new_df.columns)
        self.feature_cols_order.remove("creditability")
        
        dataset = Dataset.from_dataframe(new_df, "creditability")
        
        return split_dataset(dataset, test_size = 0.2)
        
    def get_attributes_woes(self, woe):
        attributes_woe = defaultdict(dict)
        for col in woe:
            for index, row in woe[col].iterrows():
                attributes_woe[col][row['attribute']] = row["woe"]
        return attributes_woe
    def replace_with_woes(self, dataframe, woe_values):
        attributes_woe = self.get_attributes_woes(woe_values)
        with open(os.path.join(self.working_dir, "attributes_woe.json"), "wb") as output_file:
            pickle.dump(attributes_woe, output_file)
        
        output_dict = {}
        for col in woe_values:
            output_dict[col] = []
        for index, row in dataframe.iterrows():
            for col in woe_values:
                attr = row[col]
                
                output_dict[col].append(attributes_woe[col][attr])
        output = pd.DataFrame(output_dict)
        
        output["creditability"] = dataframe["creditability"]
        return output
