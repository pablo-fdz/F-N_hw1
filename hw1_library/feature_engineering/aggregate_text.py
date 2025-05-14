import pandas as pd
import numpy as np

class AggregateText:
    def __init__(self, df, unit_col, time_col):
        self.unit = unit_col
        self.time = time_col
        self.df = df
        self.df_agg = None
    
    def _get_topic_cols(self):
        self.pr_topic_cols = self.df.columns[self.df.columns.str.contains('pr_topic')]

    def GroupbyAgg(self, type='mean', delta=0.8):

        """
        Aggregate text data of individual articles by unit and time.
        
        Parameters:
        -----------
        type : str
            Type of aggregation to perform. Either 'mean' or 'stock'.
        delta : float
            Decay factor for stock computation (default: 0.8).
        
        Returns:
        --------
        pd.DataFrame
            Aggregated data frame.
        """

        self._get_topic_cols()

        if type == 'mean':
            agg_dict = {col:'mean' for col in self.pr_topic_cols}
            agg_dict.update({'tokens':'sum', 'article_id':'count'})
            self.df_agg = self.df.groupby([self.unit, self.time]).agg(agg_dict).rename(columns={'article_id':'article_count'})
        
        elif type == 'stock':
            # First, aggregate by unit and time to get total tokens and topic shares per period
            agg_dict = {col:'mean' for col in self.pr_topic_cols}
            agg_dict.update({'tokens':'sum', 'article_id':'count'})
            agg_df = self.df.groupby([self.unit, self.time]).agg(agg_dict).rename(columns={'article_id':'article_count'})
            
            # Create stock versions of topic columns
            stock_topic_cols = [f'stock_{col}' for col in self.pr_topic_cols]
            
            # Initialize result dataframe
            result_dfs = []
            
            # Process each unit separately to calculate the stock
            for unit_value, unit_df in agg_df.groupby(level=0):  # Group by unit (level 0)
                # Sort by time period
                unit_df = unit_df.sort_index(level=1)
                
                # Initialize columns for token stock and weighted topic shares
                unit_df['token_stock'] = 0.0  # Set W_T to 0.0
                for col in stock_topic_cols:
                    unit_df[col] = 0.0  # Set X_k,T to 0.0
                
                # Calculate token stock and weighted topic shares
                for i, (idx, row) in enumerate(unit_df.iterrows()):
                    # Get current period
                    period = idx[1] if isinstance(idx, tuple) else idx
                    
                    # Initialize accumulators (W_T and X_k,T for each topic)
                    token_stock = 0.0
                    weighted_topics = {col: 0.0 for col in self.pr_topic_cols}
                    
                    # Calculate weighted sum for all periods up to the current one
                    for j, (prev_idx, prev_row) in enumerate(unit_df.iloc[:i+1].iterrows()):
                        time_diff = i - j  # T - t: distance from current period
                        weight = delta ** time_diff
                        
                        # Token stock: W_T = sum(delta^(T-t) * w_t)
                        token_stock += weight * prev_row['tokens']
                        
                        # Weighted topic shares for numerator: sum(delta^(T-t) * w_t * x_k,t)
                        for col in self.pr_topic_cols:
                            weighted_topics[col] += weight * prev_row['tokens'] * prev_row[col]
                    
                    # Store token stock
                    unit_df.at[idx, 'token_stock'] = token_stock
                    
                    # Calculate final topic stocks: X_k,T = weighted_topic_k / token_stock
                    for col, stock_col in zip(self.pr_topic_cols, stock_topic_cols):
                        if token_stock > 0:
                            unit_df.at[idx, stock_col] = weighted_topics[col] / token_stock
                        else:
                            unit_df.at[idx, stock_col] = 0.0
                
                result_dfs.append(unit_df)
            
            # Combine results from all units
            self.df_agg = pd.concat(result_dfs)

            # Drop the original topic columns
            self.df_agg = self.df_agg.drop(columns=self.pr_topic_cols)
        
        else:
            raise ValueError('Invalid type. Use "mean" or "stock".')
        
        return self.df_agg