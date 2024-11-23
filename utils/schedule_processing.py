import numpy as np
import pandas as pd
from config.config import (DAY_OFFSET,
                           DEMOGRAPHIC_LIST,
                           LOWER_PRICE,
                           MIN_ADS_PRICE_PER_VIEW,
                           TOTAL_DAYS,
                           TOTAL_SLOTS,
                           TOTAL_VIEW_COUNT)

# This script deal with so called "slot" #


def consolidate_time_to_30_mins_slot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample dataframe to be 30 mins interval
    """
    return df.resample('30min').mean().dropna(axis=0, how='all')


def combine_schedule(df_30min: pd.DataFrame) -> pd.DataFrame:
    """
    Create a combine schedule multiply this with film popularity
    (primetime * baseline)
    to get E(view)
    """
    combine_30min = df_30min[['children_baseline_view_count',
                              'adults_baseline_view_count',
                              'retirees_baseline_view_count']] * df_30min[['prime_time_factor']].to_numpy()
    combine_30min.columns = ['children_prime_time_view_count',
                             'adults_prime_time_view_count',
                             'retirees_prime_time_view_count']

    return combine_30min


def create_competitor_schedule(competitor_list):
    """
    Combine competitors schedule into a list week by week.
    So if n weeks was sent, list of unique film with n lenght will be returned
    """

    # To use movie_df[~movie_df['title'].isin(combine_schedule[0])] where week 0
    def create_week_year(schedule, offset=DAY_OFFSET):
        """
        Create week year in ISO format with optional offset day.
        Example 1 to make week start 1 day later than iso format
        """
        modify_index = schedule.index
        schedule['week'] = (
            modify_index - pd.Timedelta(offset, unit='D')).isocalendar().week
        schedule['year'] = modify_index.isocalendar().year
        return schedule

    unique_film_list = []
    for df in competitor_list:

        df = create_week_year(df)
        unique_film_list.append(df.groupby(['week', 'year'])[
                                'content'].agg(['unique']))

    combined_schedule = []
    for week in range(unique_film_list[0].size):
        all_unique_list = []
        for channel in range(len(competitor_list)):
            all_unique_list = list(set(
                all_unique_list + (unique_film_list[channel]['unique'].to_list()[week].tolist())))
        combined_schedule.append(all_unique_list)

    return combined_schedule


def strip_ads_only(df_list):
    """
    Calculate money per view of the ads of the competitors
    """
    ads_ratio = []
    counter = 0
    for df in df_list:
        ads_df = df.loc[df['content'] == 'Advert']
        sum_ad_price = ads_df['ad_slot_price'].sum()
        # TODO: clean up printing
        # print(f'Total ads price {sum_ad_price}')
        total_expected_view = 0
        for demographic in DEMOGRAPHIC_LIST:
            total_expected_view = ads_df[[f'{demographic}_expected_view_count']].sum(
            ).values[0] + total_expected_view
        # print(f'Total expected view {total_expected_view * TOTAL_VIEW_COUNT}')
        ads_ratio.append(
            sum_ad_price / (total_expected_view * TOTAL_VIEW_COUNT))
        # print(f'The price per view of channel {counter} is {
        #       sum_ad_price / (total_expected_view * TOTAL_VIEW_COUNT)}')
        counter = counter + 1
    return ads_ratio


def return_selected_week(df, week):
    """
    Return sliced DF of a week requested
    """
    mask = (df.index - pd.Timedelta(DAY_OFFSET, unit='D')).isocalendar().week == week
    return df.loc[mask.values]


def dynamic_pricing(week: int, competitor_schedule_list: list[pd.DataFrame]) -> float:
    """
    Dynamically calculate pricing for each week based on competitors
    pricing, if will be LOWER_PRICE cheaper than the cheapest competitor.
    But not lower than MIN PRICE.
    Possible improvement think of probability lowest 1.0 of selling all the ads
    0.8 for 2, 0.7 for 3 as priors than update after we got more informations.
    Maybe can do it by Bayesian search?
    """
    week_df_list = []
    for df in competitor_schedule_list:
        week_df_list.append(return_selected_week(df, week))
    comp_ads_ratio = strip_ads_only(week_df_list)
    return max(MIN_ADS_PRICE_PER_VIEW, (min(comp_ads_ratio)) * (1-LOWER_PRICE))


def process_current_week(schedule_df, movie_df):
    """
    Input current week schedule and process it to a more convenient form.
    """
    schedule_df = schedule_df['content'].reset_index().groupby([
        'content']).first()
    schedule_df.columns = ['aired_datetime']
    schedule_df = schedule_df.reset_index()
    schedule_df = schedule_df.merge(
        movie_df[['title']], left_on='content', right_on='title', how="right")
    return schedule_df


def update_schedule(schedule_df, past_schedule_df):
    """
    Update schedule df with current week schedule for "long term" storage.
    """

    past_schedule_df['latest_aired_datetime'] = np.where(~schedule_df['aired_datetime'].isnull(),
                                                         schedule_df['aired_datetime'],
                                                         past_schedule_df['latest_aired_datetime'])
    return past_schedule_df


def process_competitor_current_week(competitor_list_df, movie_df):
    """
    Input current week schedule and process it to a more convenient form.
    """

    all_competitor_dfs = []  # List to store temporary DataFrames
    for comp in competitor_list_df:
        temp = comp.copy()
        temp = temp['content'].reset_index().groupby(['content']).first()
        temp.columns = ['aired_datetime']
        temp = temp.reset_index()
        all_competitor_dfs.append(temp)  # Add temp DataFrame to the list

    # Vertically stack all DataFrames using pd.concat
    all_competitor_df = pd.concat(all_competitor_dfs, ignore_index=True)

    # Merge with the movie DataFrame
    all_competitor_df = all_competitor_df.merge(
        movie_df[['title']], left_on='content', right_on='title', how="right"
    )

    # Sort by aired_datetime and drop duplicates
    all_competitor_df = all_competitor_df.sort_values(by='aired_datetime', ascending=False)
    all_competitor_df = all_competitor_df.drop_duplicates(subset=['title'], keep='first')

    return all_competitor_df


def update_schedule(schedule_df, competitor_schedule_df, past_schedule_df):
    """
    Update schedule df with current week schedule for "long term" storage.
    """

    past_schedule_df['latest_aired_datetime'] = np.where(~schedule_df['aired_datetime'].isnull(),
                                                         schedule_df['aired_datetime'],
                                                         past_schedule_df['latest_aired_datetime'])
    past_schedule_df['comp_latest_aired_datetime'] = np.where(~competitor_schedule_df['aired_datetime'].isnull(),
                                                        competitor_schedule_df['aired_datetime'],
                                                        past_schedule_df['latest_aired_datetime'])
    
    return past_schedule_df


def decay(lambda_rate, X):
    """Calculate an exponential decay function"""
    return np.exp(-lambda_rate * X)


def decay_view_penelty(estimate_view, latest_showing_date, competitors_latest_showing_date, current_date):
    """Calculate a penalty view for recently view films, currently set to calculate week by week"""
    lambda_rate = 1/4
    competitors_rate = 1/2
    delta_week = np.floor((current_date - latest_showing_date).dt.days / 7).to_numpy()
    competitors_delta_week = np.floor((
        current_date - competitors_latest_showing_date).dt.days / 7).to_numpy()
    penalty = np.maximum(decay(lambda_rate, delta_week), decay(competitors_rate, competitors_delta_week))
    return penalty * estimate_view


def get_date_from_week(week, year):
    """Input week and year, then return date (with offset)"""
    return pd.to_datetime(str(year)+str(week)+f'{DAY_OFFSET+1}',
                          format='%Y%W%w')


def return_ads_30_mins(schedule: pd.DataFrame, compare_index: pd.DataFrame.index) -> \
        (list[list[tuple[int, float]]], list[list[list[float]]]):
    """
    :returns:
        1. list of list of tuple (binary int whether ads slot exists, price for that ad slot)
            dim = n_days x n_time_slots x tuple(str_datetime, 0|1, ad_price) <br>
        2. list of list of list of float
            dim = n_demographic x n_days x n_time_slots
    """
    ad_schedule = schedule.loc[schedule['content_type'] == 'Advert']

    # Resample and sum the ad slot prices
    slot_price = ad_schedule['ad_slot_price'].resample('30min').sum()
    slot_price = slot_price[slot_price.index.isin(compare_index)]

    demo_viewership = []
    for i, _ in enumerate(DEMOGRAPHIC_LIST):
        demo_viewership.append(ad_schedule[f"{DEMOGRAPHIC_LIST[i]}_expected_view_count"].resample('30min').sum())
        demo_viewership[i] = demo_viewership[i][demo_viewership[i].index.isin(compare_index)]
        demo_viewership[i] = np.array(demo_viewership[i]).reshape(TOTAL_DAYS, TOTAL_SLOTS)

    ad_price_list = np.array(slot_price).reshape(TOTAL_DAYS, TOTAL_SLOTS)
    days_list = np.array(slot_price.astype(bool).astype(int)).reshape(TOTAL_DAYS, TOTAL_SLOTS)

    # Combine the lists
    ads = [
        [(x, y) for x, y in zip(sublist1, sublist2)]
        for sublist1, sublist2 in zip(days_list, ad_price_list)
    ]

    return ads, demo_viewership


def resample_to_day_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input a dataframe that has datetime as an index then return a same
    DataFrame with MultiIndex with time and day.
    """
    # Extract 'day' and 'time' from the index
    new_index = df.index
    new_df = pd.DataFrame({
        'day': new_index.date,   # Extract date (YYYY-MM-DD)
        'time': new_index.time   # Extract time (HH:MM:SS)
    })
    
    # Set MultiIndex on the new DataFrame
    new_df = new_df.set_index(['day', 'time'])
    
    # Assign values from the original DataFrame, aligning by index
    new_df = pd.DataFrame(df.values, index=new_df.index, columns=df.columns)
    
    return new_df


def sort_df_by_slot_day(df:pd.DataFrame) -> df:pd.DataFrame:
    """
    Sort DataFrame columns by days and slots then return the sorted DataFrame.
    """

    sorted_columns = (sorted(df.columns[1:], key=lambda x: (int(x.split("_day_")[1]), int(x.split("_")[1]))))
    sorted_columns = ["title"] + sorted_columns
    new = [item for item in sorted_columns if item in df.columns]
    return df[new]


# Corrected function to ensure rows with no active slots are included as None
def de_one_hot_columns_include_empty_slots(
    df: pd.DataFrame, datetime_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    De-one-hot the slot columns, associating times with titles, and including rows with None for inactive slots.

    Args:
    df (pd.DataFrame): The dataframe containing one-hot encoded columns.
    datetime_index (pd.DatetimeIndex): The datetime index to map slots to times.

    Returns:
    pd.DataFrame: A DataFrame with "time" and "title" columns, including rows with None for inactive slots.
    """
    # Adjust datetime index to match the slot columns
    adjusted_index: pd.DatetimeIndex = datetime_index[: df.shape[1] - 1]
    
    # Rename columns using the adjusted datetime index
    df.columns: List[str] = ["title"] + list(adjusted_index)
    
    # Create a list to hold the results
    result: List[dict[str, Optional[pd.Timestamp]]] = []
    
    # Iterate over the datetime index
    for time in adjusted_index:
        # Check each movie for the specific time slot
        empty: bool = True  # Assume the slot is empty
        for _, row in df.iterrows():
            title: str = row["title"]
            if row[time] > 0:  # Active slot
                result.append({"time": time, "title": title})
                empty = False
        if empty:  # If no movies are active for this slot
            result.append({"time": time, "title": None})
    
    # Convert the results to a DataFrame
    return pd.DataFrame(result)





