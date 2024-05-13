
import pandas as pd
import os
import time

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

class xgBoostService:
    def __init__(self, train, test, columns):
        self.raw_train = train
        self.raw_test = test
        self.raw_columns = columns
        
        #create pandas and clean row
        self.build_pandas()
        self.split_to_test_train()

        #create logstream
        self.log_name = 'xgboost_' + str(time.time())
        self.log_stream = '{ts} service model succesfully initialised'.format(ts = time.time())
        self.log_stream += '\n'

    def build_pandas(self):
        function_name = 'build_pandas'
        try:
            self.df_train = pd.DataFrame(self.raw_train, columns = self.raw_columns)
            self.df_test = pd.DataFrame(self.raw_test, columns=self.raw_columns)


        except Exception as err:
            print('{fn}: err: {err}'.format(fn = function_name, err = err))
            raise err
    
    def split_to_test_train(self):
        function_name='split_to_test_train'
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

            self.train_test_obj = {
                'train': {'x': x_train, 'y': y_train},
                'test': {'x': x_test, 'y': y_test}
            }

        except Exception as err:
            print('{fn}: err: {err}'.format(fn = function_name, err = err))
            raise err

    def process(self):
        function_name = 'process'
        try:            
            x_train = self.train_test_obj['train']['x']
            y_train = self.train_test_obj['train']['y']

            x_test = self.train_test_obj['test']['x']
            y_test = self.train_test_obj['test']['y']

            model = XGBClassifier()
            model.fit(x_train, y_train)
            
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print("Accuracy:", accuracy)
            self.log_stream += '{ts} - {fn} accurracy: {acc}'.format(ts = time.time(), fn = function_name, acc = accuracy)
        except Exception as err:
            print('{fn}: err: {err}'.format(fn = function_name, err = err))
            raise err
        
    def process_hyperparams_tuning(self):
        function_name = 'process_hyperparams_tuning'
        try:
            x_train = self.train_test_obj['train']['x']
            y_train = self.train_test_obj['train']['y']

            x_test = self.train_test_obj['test']['x']
            y_test = self.train_test_obj['test']['y']

            model = XGBClassifier() 
            
            #create the params grid
            param_grid = {
                 'learning_rate': [0.01, 0.1, 0.3],  # Step size shrinkage used in update to prevent overfitting
                 'max_depth': [3, 5, 7],  # Maximum depth of a tree
                 'n_estimators': [50, 100, 200],  # Number of boosting rounds
                 'subsample': [0.5, 0.7, 0.9],  # Subsample ratio of the training instances
                 'colsample_bytree': [0.5, 0.7, 0.9]  # Subsample ratio of columns when constructing each tree
            }
            
            # Use GridSearchCV to search for the best hyperparameters
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
            grid_search.fit(x_train, y_train)

            # Print the best hyperparameters found
            print("Best hyperparameters found:")
            print(grid_search.best_params_)
            
            self.log_stream += '{ts} - {fn} best hpyer params: {bp}'.format(ts = time.time(), fn = function_name, bp = grid_search.best_params_)
            self.log_stream += '\n'

            # Evaluate the model with the best hyperparameters on the test set
            best_xgb = grid_search.best_estimator_
            y_pred = best_xgb.predict(x_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy on the test set:", accuracy)
            
            self.log_stream += '{ts} - {fn} accurracy: {acc}'.format(ts = time.time(), fn = function_name, acc = accuracy)
            self.log_stream += '\n'
        except Exception as err:
            print('{fn}: err: {err}'.format(fn = function_name, err = err))
            raise err
    
    def flush_stream_to_file(self):
        function_name = 'flush_stream_to_file'
        try:
            file_name = self.log_name + '.txt'
            file_path = os.path.join(os.getcwd(), '.', 'results', file_name) #this will only work if called from the main file or on that level
            
            f = open(file_path, 'a')
            f.write(self.log_stream)
            
            f.close()
            self.log_stream = ''
        except Exception as err:
            print('{fn}: err: {err}'.format(fn = function_name, err = err))
            raise err