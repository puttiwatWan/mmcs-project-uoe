import numpy as np
import pandas as pd

DEMOGRAPHIC_LIST = ['children', 'adults', 'retirees']
TOTAL_VIEW_COUNT = 1000000
DAY_OFFSET = 1
MIN_ADS_PRICE_PER_VIEW = 0.75
LOWER_PRICE = 0.1

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


def create_competitor_schedule(competitor_list):

    ### To use movie_df[~movie_df['title'].isin(combine_schedule[0])] where week 0
    def create_week_year(schedule, offset=DAY_OFFSET):

        modify_index = schedule.index
        schedule['week'] = (modify_index - pd.Timedelta(offset, unit='D')).isocalendar().week
        schedule['year'] = modify_index.isocalendar().year
        return schedule

    unique_film_list = []
    for df in competitor_list:

        df = create_week_year(df)
        unique_film_list.append(df.groupby(['week', 'year'])['content'].agg(['unique']))
        
    combine_schedule = []
    for week in range(unique_film_list[0].size):
        all_unique_list = []
        for channel in range(len(competitor_list)):
            all_unique_list = list(set(all_unique_list + (unique_film_list[channel]['unique'].to_list()[week].tolist())))
        combine_schedule.append(all_unique_list)

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


def return_selected_week(df, week):
    mask = (df.index - pd.Timedelta(DAY_OFFSET, unit='D')).isocalendar().week == week
    return df.loc[mask.values]


def dynamic_pricing(week, competitor_list):
    week_df_list = []
    for df in competitor_list:
        week_df_list.append(return_selected_week(df, week))
    comp_ads_ratio = strip_ads_only(week_df_list)
    return max(0.75, (min(comp_ads_ratio)) * (1-LOWER_PRICE))


def process_current_week(schedule_df, movie_df):
    schedule_df = schedule_df['content'].reset_index().groupby(['content']).first()
    schedule_df.columns = ['aired_datetime']
    schedule_df = schedule_df.reset_index()
    schedule_df = schedule_df.merge(movie_df[['title']], left_on='content', right_on='title', how="right")
    return schedule_df


def update_schedule(schedule_df, past_schedule_df):
    
    past_schedule_df['latest_aired_datetime'] = np.where(~schedule_df['aired_datetime'].isnull(),
                                                         schedule_df['aired_datetime'],
                                                         past_schedule_df['latest_aired_datetime'])
    return past_schedule_df


def decay(lambda_rate, X):
    return np.exp(-lambda_rate * X)


def decay_view_penelty(estimate_view, latest_showing_date, current_date):
    lambda_rate = 1/7
    delta_week = np.ceil((current_date - latest_showing_date).dt.days / 7)
    penalty = decay(lambda_rate, delta_week)
    return penalty * estimate_view


def get_date_from_week(week, year):
    return pd.to_datetime(str(year)+str(week)+f'{DAY_OFFSET+1}',
                   format='%Y%W%w')

# past_schedule_df = movie_df.copy()
# schedule_df = process_current_week(demo_week_1, movie_df)
# past_schedule_df = update_schedule(schedule_df, past_schedule_df)
# schedule_df = process_current_week(demo_week_2, movie_df)
# past_schedule_df = update_schedule(schedule_df, past_schedule_df)