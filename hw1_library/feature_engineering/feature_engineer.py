import pandas as pd
import numpy as np
from typing import Union

class FeatureEngineer:

    """
    This is a class that contains general methods that can be applied to a DataFrame to create new features. Examples of such methods include creating lagged variables, rolling min/mean/max/sum and weighted rolling mean/sum.
    The methods in this class are designed to be used in a pipeline to create new features for a given DataFrame.

    Attributes:
    -----------
    groupby_cols : Union[str, list]
        A str or list of columns to group by

    Methods:
    --------

    lag(input_df:pd.DataFrame, y_col:str, lags:list):
        This is a method that creates lagged variables for a given column in a DataFrame.

    rolling_sum(input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False):
        This is a method that creates the rolling sum of specified windows for a given column in a DataFrame.

    rolling_mean(input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False):
        This is a method that creates the rolling mean of specified windows for a given column in a DataFrame.

    rolling_min(input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False):
        This is a method that creates the rolling min of specified windows for a given column in a DataFrame.

    rolling_max(input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False):
        This is a method that creates the rolling max of specified windows for a given column in a DataFrame.

    rolling_std(input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False):
        This is a method that creates the rolling standard deviation of specified windows for a given column in a DataFrame.

    create_exponential_weights(window_size, alpha=0.8):
        This is a method that enables generating "rolling" exponential weights for a given window size.

    weighted_rolling_sum(input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False, alpha = 0.8):
        This is a method that creates the weighted rolling sum of specified windows for a given column in a DataFrame.

    weighted_rolling_mean(input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False, alpha = 0.8):
        This is a method that creates the weighted rolling mean of specified windows for a given column in a DataFrame.

    count_since(input_df:pd.DataFrame, y_col:str, thresholds:list, shift_knowledge:int = None):
        This is a method that counts the number of periods since a variable has been above a given threshold.

    ongoing(input_df:pd.DataFrame, y_col:str, thresholds:list, shift_knowledge:int = None):
        This is a method that represents a sequential count of the number of periods for which a variable has been above a given threshold.

    Notes:
    -------
    Be very careful with NAs when using the count_since_thresh and ongoing_episode methods.
    The way we are computing things here (i.e. using a > th condition) means they are treated as a 0/False.
    """

    def __init__(self, groupby_cols: Union[str, list]):

        self.groupby_cols = groupby_cols

    def _index_check(self, df:pd.DataFrame):

        """
        This is a method that checks if the index of a DataFrame is sorted correctly.

        Args:
        -----
        :param df: The DataFrame to check.

        Returns:
        --------
        :return: The DataFrame with a sorted index.
        """
        
        assert df.index.is_monotonic_increasing, "The index of the DataFrame should be monotonically increasing."

    def lag(self, input_df:pd.DataFrame, y_col:str, lags:list):

        """
        This is a method that creates lagged variables for a given column in a DataFrame.

        Args:
        -----
        :param input_df: The DataFrame containing the data.
        :param y_col: The name of the column for which to create lagged variables.
        :param lags: A list of lag values to create.

        Returns:
        --------
        :return: The original DataFrame with the lagged variables appended.
        """
        df = input_df.copy()

        self._index_check(df)

        col_names = [y_col + '_basic_lag' + str(lag) for lag in lags]
        for idx, lag in enumerate(lags):
            df[col_names[idx]] = df.groupby(self.groupby_cols)[y_col].shift(lag)
        return df

    def rolling_sum(self, input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False):

        """
        This is a method that creates the rolling sum of specified windows for a given column in a DataFrame.

        Args:
        -----
        :param input_df: The DataFrame containing the data.
        :param y_col: The name of the column for which to create lagged variables.
        :param groupby_cols: A list of columns to group by.
        :param windows: A list of windows to generate a rolling sum for.
        :param closed: A string indicating the side of the window interval to close on. Closed = 'left' omits the current observation.
        :param return_logs: A boolean indicating whether to return the log of the rolling sum.

        Returns:
        --------
        :return: The original DataFrame with the rolling sum variables appended.

        """

        df = input_df.copy()

        self._index_check(df)

        col_names = [y_col + '_rolling_sum' + str(w) for w in windows]

        for idx, w in enumerate(windows):
            df[col_names[idx]] = df.groupby(self.groupby_cols)[y_col].rolling(w, min_periods=1, closed = closed).sum().values
            if return_logs:
                df['ln_' + col_names[idx]] = np.log1p(df[col_names[idx]])
                df = df.drop(col_names[idx], axis = 1)
        return df

    def rolling_mean(self, input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False):

        """
        This is a method that creates the rolling mean of specified windows for a given column in a DataFrame.

        Args:
        -----
        :param input_df: The DataFrame containing the data.
        :param y_col: The name of the column for which to create rolling variables.
        :param windows: A list of windows to generate a rolling mean for.
        :param closed: A string indicating the side of the window interval to close on. Closed = 'left' omits the current observation.
        :param return_logs: A boolean indicating whether to return the log of the rolling mean.

        Returns:
        --------
        :return: The original DataFrame with the rolling mean variables appended.

        """

        df = input_df.copy()

        self._index_check(df)

        col_names = [y_col + '_rolling_mean' + str(w) for w in windows]

        for idx, w in enumerate(windows):
            df[col_names[idx]] = df.groupby(self.groupby_cols)[y_col].rolling(w, min_periods=1, closed = closed).mean().values
            if return_logs:
                df['ln_' + col_names[idx]] = np.log1p(df[col_names[idx]])
                df = df.drop(col_names[idx], axis = 1)
        return df

    def rolling_min(self, input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False):

        """
        This is a method that creates the rolling min of specified windows for a given column in a DataFrame.

        Args:
        -----
        :param input_df: The DataFrame containing the data.
        :param y_col: The name of the column for which to create rolling variables.
        :param windows: A list of windows to generate a rolling min for.
        :param closed: A string indicating the side of the window interval to close on. Closed = 'left' omits the current observation.
        :param return_logs: A boolean indicating whether to return the log of the rolling min.

        Returns:
        --------
        :return: The original DataFrame with the rolling min variables appended.

        """

        df = input_df.copy()

        self._index_check(df)

        col_names = [y_col + '_rolling_min' + str(w) for w in windows]

        for idx, w in enumerate(windows):
            df[col_names[idx]] = df.groupby(self.groupby_cols)[y_col].rolling(w, min_periods=1, closed = closed).min().values
            if return_logs:
                df['ln_' + col_names[idx]] = np.log1p(df[col_names[idx]])
                df = df.drop(col_names[idx], axis = 1)
        return df

    def rolling_max(self, input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False):

        """
        This is a method that creates the rolling max of specified windows for a given column in a DataFrame.

        Args:
        -----
        :param input_df: The DataFrame containing the data.
        :param y_col: The name of the column for which to create rolling variables.
        :param windows: A list of windows to generate a rolling max for.
        :param closed: A string indicating the side of the window interval to close on. Closed = 'left' omits the current observation.
        :param return_logs: A boolean indicating whether to return the log of the rolling max.

        Returns:
        --------
        :return: The original DataFrame with the rolling max variables appended.

        """

        df = input_df.copy()

        self._index_check(df)

        col_names = [y_col + '_rolling_max' + str(w) for w in windows]

        for idx, w in enumerate(windows):
            df[col_names[idx]] = df.groupby(self.groupby_cols)[y_col].rolling(w, min_periods=1, closed = closed).max().values
            if return_logs:
                df['ln_' + col_names[idx]] = np.log1p(df[col_names[idx]])
                df = df.drop(col_names[idx], axis = 1)
        return df

    def rolling_std(self, input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False):

        """
        This is a method that creates the rolling standard deviation of specified windows for a given column in a DataFrame.

        Args:
        -----
        :param input_df: The DataFrame containing the data.
        :param y_col: The name of the column for which to create rolling variables.
        :param windows: A list of windows to generate a rolling standard deviation for.
        :param closed: A string indicating the side of the window interval to close on. Closed = 'left' omits the current observation.
        :param return_logs: A boolean indicating whether to return the log of the rolling standard deviation.

        Returns:
        --------
        :return: The original DataFrame with the rolling standard deviation variables appended.

        """

        df = input_df.copy()

        self._index_check(df)

        col_names = [y_col + '_rolling_std' + str(w) for w in windows]

        for idx, w in enumerate(windows):
            df[col_names[idx]] = df.groupby(self.groupby_cols)[y_col].rolling(w, min_periods=1, closed = closed).std().values
            if return_logs:
                df['ln_' + col_names[idx]] = np.log1p(df[col_names[idx]])
                df = df.drop(col_names[idx], axis = 1)
        return df

    def _create_exponential_weights(self, window_size, alpha=0.8):

        """
        This is a method that enables generating "rolling" exponential weights for a given window size.

        Args:
        -----
        :param window_size: The size of the window for which weights are calculated.
        :param alpha: The decay factor for weights, defaults to 0.5.
                    A higher alpha discounts older observations faster.

        Returns:
        -----
        :return: A numpy array of weights.
        """

        weights = alpha ** np.arange(window_size)
        normalized_weights = weights / weights.sum()
        return normalized_weights[::-1]

    def weighted_rolling_sum(self, input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False, alpha = 0.8):

        """
        This is a method that creates the weighted rolling sum of specified windows for a given column in a DataFrame.

        Args:
        -----
        :param input_df: The DataFrame containing the data.
        :param y_col: The name of the column for which to create weighted rolling variables.
        :param groupby_cols: A list of columns to group by.
        :param windows: A list of windows to generate a weighted rolling sum for.
        :param closed: A string indicating the side of the window interval to close on. Closed = 'left' omits the current observation.
        :param return_logs:  A boolean indicating whether to return the log of the weighted rolling sum.
        :param alpha: The decay factor for weights, defaults to 0.8. A higher alpha discounts older observations faster.

        Returns:
        -----
        :return: The original DataFrame with the weighted rolling sum variables appended.
        """

        df = input_df.copy()

        self._index_check(df)

        col_names = [y_col + '_weighted_rolling_sum' + str(w) for w in windows]

        for idx, w in enumerate(windows):
            df[col_names[idx]] = df.groupby(self.groupby_cols)[y_col].rolling(w, min_periods=1, closed = closed).apply(lambda x: np.sum(self._create_exponential_weights(len(x), alpha) * x), raw = True).values
            if return_logs:
                df['ln_' + col_names[idx]] = np.log1p(df[col_names[idx]])
                df = df.drop(col_names[idx], axis = 1)
        return df

    def weighted_rolling_mean(self, input_df:pd.DataFrame, y_col:str, windows:list, closed = None, return_logs = False, alpha = 0.8):
        """
        This is a method that creates the weighted rolling mean of specified windows for a given column in a DataFrame.

        Args:
        -----
        :param input_df: The DataFrame containing the data.
        :param y_col: The name of the column for which to create weighted rolling variables.
        :param windows: A list of windows to generate a weighted rolling mean for.
        :param closed: A string indicating the side of the window interval to close on. Closed = 'left' omits the current observation.
        :param return_logs:  A boolean indicating whether to return the log of the weighted rolling mean.
        :param alpha: The decay factor for weights, defaults to 0.8. A higher alpha discounts older observations faster.

        Returns:
        --------
        :return: The original DataFrame with the weighted rolling mean variables appended.
        """

        df = input_df.copy()

        self._index_check(df)

        col_names = [y_col + '_weighted_rolling_mean' + str(w) for w in windows]

        for idx, w in enumerate(windows):
            df[col_names[idx]] = df.groupby(self.groupby_cols)[y_col].rolling(w, min_periods=1, closed = closed).apply(lambda x: np.sum(self._create_exponential_weights(len(x), alpha) * x) / len(x), raw = True).values
            if return_logs:
                df['ln_' + col_names[idx]] = np.log1p(df[col_names[idx]])
                df = df.drop(col_names[idx], axis = 1)
        return df

    def _count_since(self, x: pd.Series):
        """
        This is a method that counts the number of periods since a variable has been 1.

        :param x: A pandas Series containing the target variable.

        Returns:
        - y (list): A list containing the number of periods since the target variable has been 1.
        """

        x = list(x)
        y = []
        for n in range(0, len(x)):
            if (x[n] == 0) & (n == 0):
                y.append(1) # if it starts with no flows
            elif x[n] == 1:
                y.append(0) # reset to 0 if flows
            else:
                y.append(y[n-1]+1) # add 1 if no flows
        return y

    def since(self, input_df:pd.DataFrame, y_col:str, thresholds:list, shift_knowledge:int = None):

        """
        This is a method that counts the number of periods since a variable has been above a given threshold.

        Args:
        -----
        :param input_df: The DataFrame containing the data.
        :param y_col: The name of the column for which to create the count since variable.
        :param thresholds: A list of thresholds to count since.
        :param shift_knowledge: An integer defining by how many periods to shift the count since variable.

        Returns:
        --------
        :return: The original DataFrame with the count since variables appended.
        """

        df = input_df.copy()

        self._index_check(df)


        binary_col_names = [y_col + '_above' + str(th) for th in thresholds]
        col_names = [y_col + '_since_' + str(th) for th in thresholds]

        for idx, th in enumerate(thresholds):
            df[binary_col_names[idx]] = (df[y_col] > th).astype(int)
            df[col_names[idx]] = df.groupby(self.groupby_cols)[binary_col_names[idx]].transform(self._count_since)

            if shift_knowledge is None:
                pass
            else:
                #in case we need to shift by one since we don't know the y_col in current period
                df[binary_col_names[idx]] = df.groupby(self.groupby_cols)[[binary_col_names[idx]]].shift(shift_knowledge)
                df[col_names[idx]] = df.groupby(self.groupby_cols)[col_names[idx]].shift(shift_knowledge)
        return df[[y_col, *[x for x in df.columns if 'since' in x]]]

    def _count_ongoing(self, x: pd.Series):
        """
        This is a method that generates a sequential count of the periods for which a variable has been 1.

        :param x: A pandas Series containing the target variable.

        Returns:
        - y (list): A list containing the sequential count of the periods for which the target variable has been 1.
        """

        x = list(x)
        y = []
        episode_counter = 0
        for n in range(0, len(x)):
            if (x[n] == 0) & (n == 0):
                y.append(episode_counter) # if it starts with no flows
            elif x[n] == 1:
                episode_counter += 1
                y.append(episode_counter) # if there are flows
            else:
                y.append(0) # reset to 0 if no flows
                episode_counter = 0
        return y

    def ongoing(self, input_df:pd.DataFrame, y_col:str, thresholds:list, shift_knowledge:int = None):

        """
        This is a method that represents a sequential count of the number of periods for which a variable has been above a given threshold.

        Args:
        -----
        :param input_df: The DataFrame containing the data.
        :param y_col: The name of the column for which to create the count since variable.
        :param thresholds: A list of thresholds to count since.
        :param shift_knowledge: An integer defining by how many periods to shift the count since variable.

        Returns:
        --------
        :return: The original DataFrame with the count since variables appended.
        """

        df = input_df.copy()

        self._index_check(df)

        binary_col_names = [y_col + '_above' + str(th) for th in thresholds]
        col_names = [y_col + '_ongoing_' + str(th) for th in thresholds]

        for idx, th in enumerate(thresholds):
            df[binary_col_names[idx]] = (df[y_col] > th).astype(int)
            df[col_names[idx]] = df.groupby(self.groupby_cols)[binary_col_names[idx]].transform(self._count_ongoing)

            if shift_knowledge is None:
                pass
            else:
                #in case we need to shift by one since we don't know the y_col in current period
                df[binary_col_names[idx]] = df.groupby(self.groupby_cols)[[binary_col_names[idx]]].shift(shift_knowledge)
                df[col_names[idx]] = df.groupby(self.groupby_cols)[col_names[idx]].shift(shift_knowledge)
        return df[[y_col, *[x for x in df.columns if 'ongoing' in x]]]
