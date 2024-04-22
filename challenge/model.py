import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self.raw_data = pd.read_csv(filepath_or_buffer="./data/data.csv")
        self.top_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    # Preprocess data: get features and target (if not None)
    def preprocess(
        self, 
        data: pd.DataFrame, 
        target_column: str=None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """        
        # Get dummies and concatenate
        features = None
        try:
            features = pd.concat([
                pd.get_dummies(data['OPERA'], prefix='OPERA'),
                pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'), 
                pd.get_dummies(data['MES'], prefix='MES')], 
                axis = 1)
        except:
            print("Input data is missing at least one of the columns: 'OPERA', 'MES', 'TIPOVUELO'")

        # Filter top features
        filtered_features = self.check_filter_top_feat(features)

        # If target_column is None: return features
        if target_column is None:
            return filtered_features

        elif target_column not in data.columns: # elif not in columns: find target ('delay')
            try:
                data_target = self.get_delay(data, threshold=15)
                target = pd.DataFrame(data_target['delay'])
            except:
                print("Input data is missing at least one of the columns: 'Fecha-I', 'Fecha-O'")
            return ([filtered_features, target])
        
        else:   # if in columns
            target = pd.DataFrame(data[target_column])
            return ([filtered_features, target])

    # Fit model
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Reorder columns
        features = features.reindex(columns=self.top_features)

        # Split data
        X_train, _, y_train, _ = train_test_split(features, 
                                                  target, 
                                                  test_size=0.33,
                                                  random_state=42)
        # Get training data scale
        n_y0 = len(y_train[y_train == 0].dropna())
        n_y1 = len(y_train[y_train == 1].dropna())
        scale = n_y0 / n_y1

        # Define model with balanced classes
        self._model = xgb.XGBClassifier(random_state=1, 
                                     learning_rate=0.01,
                                     scale_pos_weight=scale)
        # Train model
        self._model.fit(X_train, y_train)
        print('Model has been trained successfully!')

        return None

    # Predict delay
    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        # Reorder columns
        features = features.reindex(columns=self.top_features)
        
        # Check if self._model is None
        if self._model is None:
            _features, _target = self.preprocess(data=self.raw_data,
                                                 target_column='delay')
            self.fit(features=_features, target=_target)
            y_pred = self._model.predict(features)
            y_pred_list = [int(y) for y in y_pred]
        else:
            # Predict for new flights
            y_pred = self._model.predict(features)
            y_pred_list = [int(y) for y in y_pred]

        return y_pred_list
    
    ###### Preprocessing methods ######
    # Get minute difference
    def get_min_diff(
        self, 
        data:pd.DataFrame
    ) -> float:
        """
        Get minute difference between scheduled and operation dates.

        Args:
            data (pd.DataFrame): raw data.

        Returns:
            float: minute difference.
        """
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        
        return min_diff
    
    # Get delay column (target)
    def get_delay(
        self, 
        data:pd.DataFrame, 
        threshold:int=15
    ) -> pd.DataFrame:
        """Get delay column (target).

        Args:
            data (pd.DataFrame): raw data.
            threshold (int, optional): time in minutes to consider a delay. 
            Defaults to 15.

        Returns:
            pd.DataFrame: data with target column.
        """
        data['min_diff'] = data.apply(self.get_min_diff, axis=1)
        data['delay'] = np.where(data['min_diff'] > threshold, 1, 0)

        return data
    
    ###### API methods ######
    # Save trained model
    def save_model(
        self,
        version:str
    ) -> None:
        try:
            with open(f'delay_model-{version}.pkl', 'wb') as f:
                pickle.dump(self._model, f)
        except:
            print('The model is not trained yet!')

        return None

    # Filter important features
    def check_filter_top_feat(
        self,
        features:pd.DataFrame
    ) -> pd.DataFrame:        
        # Add top features if not in features
        for top_feat in self.top_features:
            if top_feat not in features.columns:
                features[top_feat] = False

        drop_features = [feat for feat in features.columns if feat not in self.top_features]
        features.drop(columns=drop_features, inplace=True)

        return features
        

        