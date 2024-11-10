import pandas as pd


### This script deal with so call "slot" #####

def consolidate_time_to_30_mins_slot(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample('30min').mean()


def combine_schedule(df_30min: pd.DataFrame) -> pd.DataFrame:
    combine_30min = df_30min[['children_baseline_view_count',
                              'adults_baseline_view_count',
                              'retirees_baseline_view_count']] * df_30min[['prime_time_factor']].to_numpy()
    combine_30min.columns = ['children_prime_time_view_count',
                             'adults_prime_time_view_count',
                             'retirees_prime_time_view_count']

    return combine_30min
