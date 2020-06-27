import os
import pickle

import numpy as np
from sklearn.externals import joblib

from fastai_custom.imports import *
from fastai_custom.structured import *

#from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#from IPython.display import display
import pickle

from sklearn.metrics import accuracy_score

from sklearn import metrics

import preprocess

class MyPredictor(object):
    """An example Predictor for an AI Platform custom prediction routine."""

    def __init__(self, model, preprocessor):
        """Stores artifacts for prediction. Only initialized via `from_path`.
        """
        self._model = model
        self._preprocessor = preprocessor

    def predict(self, instances, **kwargs):
        """Performs custom prediction.

        Preprocesses inputs, then performs prediction using the trained
        scikit-learn model.

        Args:
            instances: A list of prediction input instances.
            **kwargs: A dictionary of keyword args provided as additional
                fields on the predict request body.

        Returns:
            A list of outputs containing the prediction results.
        """
        #print(self._model)
        #print(self._preprocessor)
        #inputs = np.asarray(instances)
        df_cat = self._preprocessor.preprocess()
        #print(df_cat)
        cols = df_cat.columns
        cols = pd.Index.drop(cols,'Cover_Type').tolist()
        #print(instances)
        #print(cols)
        df_test = pd.DataFrame([instances],columns=cols)
        #preprocessed_inputs = self._preprocessor.preprocess(inputs)
        apply_cats(df_test,df_cat)
        
        df_test,_,_ = proc_df(df_test)
        #print(df_test)
        
        outputs = self._model.predict(df_test)
        return outputs.tolist()

    @classmethod
    def from_path(cls, model_dir):
        """Creates an instance of MyPredictor using the given path.

        This loads artifacts that have been copied from your model directory in
        Cloud Storage. MyPredictor uses them during prediction.

        Args:
            model_dir: The local directory that contains the trained
                scikit-learn model and the pickled preprocessor instance. These
                are copied from the Cloud Storage model directory you provide
                when you deploy a version resource.

        Returns:
            An instance of `MyPredictor`.
        """
#         model_path = os.path.join(model_dir, 'model.pkl')
#         with open(model_path, 'rb') as f:
#             model = pickle.load(f)
        
        model_path = os.path.join(model_dir, 'model.joblib')
        model = joblib.load(model_path)

        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        return cls(model, preprocessor)