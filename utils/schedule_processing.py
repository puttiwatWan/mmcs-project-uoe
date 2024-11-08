import pandas as pd
import numpy as np


### This script deal with so call "slot" #####

def consolidate_time_to_30_mins_slot(df):
    return df.resample('30min').mean()


def combine_schedule(df_30min):

    combine_30min = df_30min[['children_baseline_view_count', 
          'adults_baseline_view_count', 
          'retirees_baseline_view_count']] * df_30min[['prime_time_factor']].to_numpy()
    combine_30min.columns = ['children_baseline_primetime', 
                             'adults_baseline_primetime', 
                             'retirees_baseline_primetime']
    
    return combine_30min