import numpy as np
import pandas as pd
from itertools import chain
from config.config import (COMPETITORS,
                           FIRST_WEEK,
                           MAX_CONVERSION_RATE,
                           SLOT_DURATION,
                           WEEK_CONSIDERED,
                           YEAR)
from utils.data_processing import (process_table,
                                   DEMOGRAPHIC_LIST,
                                   top_n_viable_film)
from utils.schedule_processing import (combine_schedule,
                                       consolidate_time_to_30_mins_slot,
                                       dynamic_pricing,
                                       return_selected_week,
                                       get_date_from_week,
                                       create_competitor_schedule,
                                       decay_view_penelty,
                                       process_current_week,
                                       process_competitor_current_week,
                                       update_schedule,
                                       return_ads_30_mins)
from datetime import datetime as dt
from advert_conversion_rates import generate_conversion_rates

from solver import SchedulingSolver


def import_data():
    base_path = "data/"

    mov_df = pd.read_csv(base_path + 'movie_database.csv')

    ch_0_conversion_rates_df = pd.read_csv(base_path + 'channel_0_conversion_rates.csv', index_col=[0])
    ch_0_conversion_rates_df.index = pd.to_datetime(ch_0_conversion_rates_df.index)

    ch_1_conversion_rates_df = pd.read_csv(base_path + 'channel_1_conversion_rates.csv', index_col=[0])
    ch_1_conversion_rates_df.index = pd.to_datetime(ch_1_conversion_rates_df.index)

    ch_2_conversion_rates_df = pd.read_csv(base_path + 'channel_2_conversion_rates.csv', index_col=[0])
    ch_2_conversion_rates_df.index = pd.to_datetime(ch_2_conversion_rates_df.index)

    ch_a_schedule_df = pd.read_csv(base_path + 'channel_A_schedule.csv', index_col=[0])
    ch_a_schedule_df.index = pd.to_datetime(ch_a_schedule_df.index)

    ch_0_schedule_df = pd.read_csv(base_path + 'channel_0_schedule.csv', index_col=[0])
    ch_0_schedule_df.index = pd.to_datetime(ch_0_schedule_df.index)

    ch_1_schedule_df = pd.read_csv(base_path + 'channel_1_schedule.csv', index_col=[0])
    ch_1_schedule_df.index = pd.to_datetime(ch_1_schedule_df.index)

    ch_2_schedule_df = pd.read_csv(base_path + 'channel_2_schedule.csv', index_col=[0])
    ch_2_schedule_df.index = pd.to_datetime(ch_2_schedule_df.index)

    return (mov_df, ch_0_conversion_rates_df, ch_1_conversion_rates_df, ch_2_conversion_rates_df, ch_a_schedule_df,
            ch_0_schedule_df, ch_1_schedule_df, ch_2_schedule_df)


print("==== Starting Importing Data ====")
# Import Data
st = dt.now()
(movie_df, channel_0_conversion_rates_df, channel_1_conversion_rates_df, channel_2_conversion_rates_df,
 channel_a_schedule_df, channel_0_schedule_df, channel_1_schedule_df, channel_2_schedule_df) = import_data()
print("===== Total time used to import data: {0} seconds".format((dt.now() - st).total_seconds()))

print(movie_df.columns)
# Process Data

movie_df = process_table(movie_df)
original_movie_df = movie_df.copy()
movie_df = top_n_viable_film(movie_df, p=0.7)
print(movie_df.head())
print(movie_df.columns)

# Create DF needed in the models
competitor_schedules = [channel_0_schedule_df, channel_1_schedule_df, channel_2_schedule_df]
channel_a_30_schedule_df = consolidate_time_to_30_mins_slot(channel_a_schedule_df)
combine_30min_df = combine_schedule(channel_a_30_schedule_df)

# Return Pricing for the week (first week is week 40)
ads_price_per_view = dynamic_pricing(week=40, competitor_schedule_list=competitor_schedules)

# Weekly schedule
# Initialized schedule
all_schedule_df = movie_df.copy()
all_schedule_df['latest_showing_date'] = pd.to_datetime('2000-01-01')
all_schedule_df['latest_aired_datetime'] = pd.NaT
all_schedule_df['comp_latest_aired_datetime'] = pd.NaT

for week in range(FIRST_WEEK, FIRST_WEEK + WEEK_CONSIDERED):

    current_date = get_date_from_week(week, YEAR)
    this_week_competitor_list = [return_selected_week(comp, week) for comp in competitor_schedules]
    # Get competitor schedule
    combine_schedule = create_competitor_schedule(this_week_competitor_list)

    # Create Modify DF for this week
    current_adjusted_df = all_schedule_df.copy()
    # Cut all the same movie as competitor out
    current_adjusted_df = current_adjusted_df[~current_adjusted_df['title'].isin(combine_schedule[0])]
    print(current_adjusted_df['latest_aired_datetime'].values)
    print(current_adjusted_df['comp_latest_aired_datetime'].values)

    # Create Decay for popularity
    for demo in DEMOGRAPHIC_LIST:
        penalty = decay_view_penelty(
            current_adjusted_df[f'{demo}_scaled_popularity'],
            current_adjusted_df['latest_aired_datetime'],
            current_adjusted_df['comp_latest_aired_datetime'],
            current_date).fillna(0)
        current_adjusted_df[f'adjusted_{demo}_scaled_popularity'] = current_adjusted_df[
                                                                        f'{demo}_scaled_popularity'] - penalty

    current_adjusted_df.to_csv(f'out/current_adjusted_df_{week}.csv')

    # RUN XPRESS GET SCHEDULE
    schedule_df = return_selected_week(channel_0_schedule_df, week)

    # Get competitor schedule
    competitor_current_list_df = []
    for comp in competitor_schedules:
        comp_schedule_df = return_selected_week(comp, week)
        competitor_current_list_df.append(comp_schedule_df)

    # Process current week schedule
    schedule_df = process_current_week(schedule_df, movie_df)

    # Process current week competitor schedule
    competitor_schedule_df = process_competitor_current_week(competitor_current_list_df, movie_df)

    # Update Schedule for what has been schedule this time.
    all_schedule_df = update_schedule(schedule_df, competitor_schedule_df, all_schedule_df)

    last_week_schedule_df = schedule_df
    all_schedule_df.to_csv(f'out/all_schedule_df_{week}.csv')

# -----------------------------------------
number_of_movies = len(movie_df)
number_of_competitors = len(COMPETITORS)
number_of_time_slots = int((24 - 7) * 60 / SLOT_DURATION)  # 30 min each
number_of_days = 7

Movies = range(number_of_movies)
Competitors = range(number_of_competitors)
TimeSlots = range(number_of_time_slots)
Days = range(number_of_days)

M = 1  # Max Viewership gains

print("==== Starting return_ads_30_mins ====")
st = dt.now()
# Create ads slots price in 30-min time slot for competitors
comp_ads_slots = []  # np_array of dimension n_comp x n_days x n_time_slots x (0|1, ad_price)
comp_ads_viewership = []  # np_array of dimension n_comp x n_demo x n_days x n_time_slots
for comp in competitor_schedules:
    ads, ad_viewership = return_ads_30_mins(comp, channel_a_30_schedule_df.index)
    comp_ads_slots.append(ads)
    comp_ads_viewership.append(ad_viewership)
comp_ads_slots = np.array(comp_ads_slots)
comp_ads_viewership = np.array(comp_ads_viewership)
print("===== Total time used to return ads 30 min: {0} seconds".format((dt.now() - st).total_seconds()))

# Generate the conversion for each movie and each ad slot.
# The conversion_rates is of dimension (n_movies x n_days x n_ad_slot_30_min)
all_genres = list(set(chain.from_iterable(original_movie_df["genres"])))
conversion_rates = generate_conversion_rates(competitor_schedules[0], movie_df, original_movie_df, all_genres,
                                             MAX_CONVERSION_RATE)


def main():
    solver = SchedulingSolver(original_movie_df=original_movie_df,
                              movie_df=movie_df,
                              channel_a_30_schedule_df=channel_a_30_schedule_df,
                              competitor_schedules=competitor_schedules,
                              number_of_days=number_of_days,
                              )

    solver.run(set_hard_limit=True)


main()
