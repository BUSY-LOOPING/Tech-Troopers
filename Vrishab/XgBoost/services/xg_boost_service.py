
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

class xgBoostService:
    def __init__(self, train, test, columns):
        self.raw_train = train
        self.raw_test = test
        self.raw_columns = columns
        
    def build_pandas(self):
        function_name = 'build_pandas'
        try:
            self.df_train = pd.DataFrame(self.raw_train, columns = self.raw_columns)
            self.df_test = pd.DataFrame(self.raw_test, columns=self.raw_columns)


        except Exception as err:
            print('{fn}: err: {err}'.format(fn = function_name, err = err))
            raise err
    
    def process(self):
        function_name = 'process'
        try:
            df_train = self.df_train
            df_test = self.df_test

            df_train['class'].replace({'UP': 1, 'DOWN': 0}, inplace=True)
            df_test['class'].replace({'UP': 1, 'DOWN': 0}, inplace=True)

            y_train = df_train['class']
            y_test = df_test['class']

            df_train.drop('class', axis = 1, inplace = True)
            df_test.drop('class', axis = 1, inplace = True)

            for column in df_train.columns:
                df_train[column] = df_train[column].astype(float)
                df_test[column] = df_test[column].astype(float)

            x_train = df_train
            x_test = df_test

            model = XGBClassifier()
            model.fit(x_train, y_train)
            
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print("Accuracy:", accuracy)
        except Exception as err:
            print('{fn}: err: {err}'.format(fn = function_name, err = err))
            raise err


