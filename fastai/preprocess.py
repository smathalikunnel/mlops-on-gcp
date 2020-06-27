import numpy as np
import pandas as pd


class TrainedCategories(object):
    """Stores means of each column of a matrix and uses them for preprocessing.
    """

    def __init__(self,df):
        """On initialization, is not tied to any distribution."""
        self._df = df

    #def preprocess(self, data):
    def preprocess(self):
    
        """Transforms a matrix.

        The first time this is called, it stores the means of each column of
        the input. Then it transforms the input so each column has mean 0. For
        subsequent calls, it subtracts the stored means from each column. This
        lets you 'center' data at prediction time based on the distribution of
        the original training data.

        Args:
            data: A NumPy matrix of numerical data.

        Returns:
            A transformed matrix with the same dimensions as the input.
        """
        #if self._df is None:  # during training only
         #   self._means = df_raw
        return self._df