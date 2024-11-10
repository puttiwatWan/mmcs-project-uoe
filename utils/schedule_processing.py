import pandas as pd


### This script deal with so call "slot" #####

def consolidate_time_to_30_mins_slot(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample('30min').mean().dropna(axis=0, how='all')


def combine_schedule(df_30min: pd.DataFrame) -> pd.DataFrame:
    combine_30min = df_30min[['children_baseline_view_count',
                              'adults_baseline_view_count',
                              'retirees_baseline_view_count']] * df_30min[['prime_time_factor']].to_numpy()
    combine_30min.columns = ['children_prime_time_view_count',
                             'adults_prime_time_view_count',
                             'retirees_prime_time_view_count']

    return combine_30min


def create_competitor_schedule(channel_0_schedule_df, channel_1_schedule_df, channel_2_schedule_df):

    ### To use movie_df[~movie_df['title'].isin(combine_schedule[0])] where week 0
    def create_week_year(schedule, offset=1):

        schedule['week'] = schedule.index - - pd.Timedelta(offset, unit='D').isocalendar().week
        schedule['year'] = schedule.index.isocalendar().year
        return schedule
    
    channel_0_schedule_df = create_week_year(channel_0_schedule_df)
    channel_1_schedule_df = create_week_year(channel_1_schedule_df)
    channel_2_schedule_df = create_week_year(channel_2_schedule_df)
    channel_0_unique_week = channel_0_schedule_df.groupby(['week', 'year'])['content'].agg(['unique'])
    channel_1_unique_week = channel_1_schedule_df.groupby(['week', 'year'])['content'].agg(['unique'])
    channel_2_unique_week = channel_2_schedule_df.groupby(['week', 'year'])['content'].agg(['unique'])

    combine_schedule = []
    for week in range(channel_0_unique_week['unique'].size):
        zero_list = channel_0_unique_week['unique'].to_list()[week].tolist()
        one_list = channel_1_unique_week['unique'].to_list()[week].tolist()
        two_list = channel_2_unique_week['unique'].to_list()[week].tolist()
        all_list = list(set(zero_list + one_list + two_list))
        combine_schedule.append(all_list)

    return combine_schedule
