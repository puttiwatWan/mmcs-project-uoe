import pandas as pd

DEMOGRAPHIC_LIST = ['children', 'adults', 'retirees']
TOTAL_VIEW_COUNT = 1000000

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


def strip_ads_only(df_list):
    ads_ratio = []
    counter = 0
    for df in df_list:
        ads_df = df.loc[df['content'] == 'Advert']
        sum_ad_price = ads_df['ad_slot_price'].sum()
        print(f'Total ads price {sum_ad_price}')
        total_expected_view = 0
        for demographic in DEMOGRAPHIC_LIST:
            total_expected_view = ads_df[[f'{demographic}_expected_view_count']].sum().values[0] + total_expected_view
        print(f'Total expected view {total_expected_view * TOTAL_VIEW_COUNT}')
        ads_ratio.append(sum_ad_price/ (total_expected_view* TOTAL_VIEW_COUNT))
        print(f'The price per view of channel {counter} is {sum_ad_price/ (total_expected_view* TOTAL_VIEW_COUNT)}')
        counter = counter + 1
    return ads_ratio
