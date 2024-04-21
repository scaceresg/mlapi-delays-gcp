import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
import xgboost as xgb

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self.raw_data = pd.read_csv(filepath_or_buffer="./data/data.csv")

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
        # Important features
        top_features = [
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
        features = features[top_features]

        # If target_column is None: return features
        if target_column is None:
            return features

        elif target_column not in data.columns: # elif not in columns: find target ('delay')
            try:
                data_target = self.get_delay(data, threshold=15)
                target = pd.DataFrame(data_target['delay'])
            except:
                print("Input data is missing at least one of the columns: 'Fecha-I', 'Fecha-O'")
            return ([features, target])
        
        else:   # if in columns
            target = pd.DataFrame(data[target_column])
            return ([features, target])

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
    
    ###### New Methods: Preprocessing methods ######
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
        