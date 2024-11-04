import pandas as pd
import numpy as np


### This script deal with so call "slot" #####

def consolidate_time_to_30_mins_slot(df):
    return df.resample('30min').mean()