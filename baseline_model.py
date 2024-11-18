import numpy as np
import xpress as xp
import pandas as pd
from itertools import chain
from config.config import *
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
from advert_conversion_rates import calculate_conversion_rate
from IPython.display import display

# xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

whole_st = dt.now()


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

# Process Data
movie_df = process_table(movie_df)
# movie_df = movie_df.head(100)

# Create DF needed in the models
competitor_schedules = [channel_0_schedule_df, channel_1_schedule_df, channel_2_schedule_df]
channel_a_30_schedule_df = consolidate_time_to_30_mins_slot(channel_a_schedule_df)
combine_30min_df = combine_schedule(channel_a_30_schedule_df)

# Return Pricing for the week (first week is week 40)
ads_price_per_view = dynamic_pricing(week=40, competitor_schedule_list=competitor_schedules)

# Filter movies out
movie_df = top_n_viable_film(movie_df, p=1.0)

# Weekly schedule
## Initialized schedule
all_schedule_df = movie_df.copy()
all_schedule_df['latest_showing_date'] = pd.to_datetime('2000-01-01')
all_schedule_df['latest_aired_datetime'] = pd.NaT
all_schedule_df['comp_latest_aired_datetime'] = pd.NaT

for week in range(FIRST_WEEK, FIRST_WEEK + WEEK_CONSIDERED):

    current_date = get_date_from_week(week, YEAR)
    this_week_competitor_list = [return_selected_week(comp, week) for comp in competitor_schedules]
    ### Get competitor schedule
    combine_schedule = create_competitor_schedule(this_week_competitor_list)

    ### Create Modify DF for this week
    current_adjusted_df = all_schedule_df.copy()
    ### Cut all the same movie as competitor out
    current_adjusted_df = current_adjusted_df[~current_adjusted_df['title'].isin(combine_schedule[0])]
    print(current_adjusted_df['latest_aired_datetime'].values)
    print(current_adjusted_df['comp_latest_aired_datetime'].values)

    # a = np.floor((current_date - current_adjusted_df['latest_aired_datetime']).dt.days / 7).values
    # b = np.floor((current_date - current_adjusted_df['latest_aired_datetime']).dt.days / 7).values
    # print(np.max(a, b))
    ### Create Decay for popularity
    current_adjusted_df['adjusted_popularity'] = current_adjusted_df['popularity'] - decay_view_penelty(
        current_adjusted_df['popularity'], 
        current_adjusted_df['latest_aired_datetime'],
        current_adjusted_df['comp_latest_aired_datetime'], 
        current_date)

    ### RUN XPRESS GET SCHEDULE
    schedule_df = return_selected_week(channel_0_schedule_df, week)

    ### Get competitor schedule
    competitor_list_df = []
    for comp in competitor_schedules:
        comp_schedule_df = return_selected_week(comp, week)
        competitor_list_df.append(comp_schedule_df)
        
    ### Process current week schedule
    schedule_df = process_current_week(schedule_df, movie_df)
    
    ### Process current week competitor schedule
    competitor_schedule_df = process_competitor_current_week(competitor_list_df, movie_df)

    #### Update Schedule for what has been schedule this time.
    all_schedule_df = update_schedule(schedule_df, competitor_schedule_df, all_schedule_df)

    last_week_schedule_df = schedule_df


# -----------------------------------------
number_of_movies = len(movie_df)
number_of_competitors = len(COMPETITORS)
number_of_time_slots = int((24 - 7) * 60 / SLOT_DURATION)  # 30 min each
number_of_days = 1

Movies = range(number_of_movies)
Competitors = range(number_of_competitors)
TimeSlots = range(number_of_time_slots)
Days = range(number_of_days)

print("==== Starting return_ads_30_mins ====")
st = dt.now()
# Create ads slots price in 30-min time slot for competitors
comp_ads_slots = []  # n_comp x n_days x n_time_slots x tuple(datetime, 0|1)
comp_ads_viewership = []  # n_comp x n_demo x n_days x n_time_slots
for comp in competitor_schedules:
    ads, ad_viewership = return_ads_30_mins(comp, channel_a_30_schedule_df.index)
    comp_ads_slots.append(ads)
    comp_ads_viewership.append(ad_viewership)
print("===== Total time used to return ads 30 min: {0} seconds".format((dt.now() - st).total_seconds()))

all_genres = list(set(chain.from_iterable(movie_df["genres"])))

print("==== Starting calc conversion rate ====")
st = dt.now()
calculate_conversion_rate(competitor_schedules[0], movie_df, all_genres,
                          comp_ads_slots[0][0][0][0], movie_df["title"][0],
                          MAX_CONVERSION_RATE)
print(number_of_days * number_of_competitors * number_of_movies * number_of_time_slots * 0.0094)
print("===== Total time used to do conversion rate: {0} seconds".format((dt.now() - st).total_seconds()))

# Declare
print("==== Starting Adding Var ====")
st = dt.now()
scheduling = xp.problem('scheduling')
movie = scheduling.addVariables(number_of_movies, number_of_days, name='m', vartype=xp.binary)
movie_time = scheduling.addVariables(number_of_movies, number_of_time_slots, number_of_days, name="mt",
                                     vartype=xp.binary)
start_time = scheduling.addVariables(number_of_movies, number_of_days, name='s', vartype=xp.integer)
end_time = scheduling.addVariables(number_of_movies, number_of_days, name='e', vartype=xp.integer)

# TODO: Note: Buyers now only include competitors. If our ad slot is used for ourselves, we can check by
#             xp.Sum(sold_ad_slots[i, t, c, d] for c in Buyers) == 0 (meaning no competitor bought that slot)
sold_ad_slots = scheduling.addVariables(number_of_movies, number_of_time_slots, number_of_competitors, number_of_days,
                                        name="sa", vartype=xp.binary)
bought_ad_slots = scheduling.addVariables(number_of_movies, number_of_time_slots, number_of_competitors, number_of_days,
                                          name="ba", vartype=xp.binary)
increased_viewers = scheduling.addVariables(number_of_time_slots, number_of_competitors, number_of_days,
                                            name="iv", vartype=xp.continuous)
print("===== Total time used to add var: {0} seconds".format((dt.now() - st).total_seconds()))

# Objective Function
print("==== Starting Obj Fn ====")
st = dt.now()
# TODO: Multiply sold ad slots by some price
scheduling.setObjective(-xp.Sum(movie_df['license_fee'][i] * xp.Sum(movie[i, d] for d in Days) for i in Movies) +
                        xp.Sum(
                            (
                                xp.Sum(
                                    combine_30min_df[f"{demo}_prime_time_view_count"].iloc[t] *
                                    movie_df[f"{demo}_scaled_popularity"][i] for demo in DEMOGRAPHIC_LIST
                                ) +
                                increased_viewers[t, c, d]
                            ) *
                            sold_ad_slots[i, t, c, d]
                            for i in Movies for t in TimeSlots for c in Competitors for d in Days
                        ) -
                        xp.Sum(
                            xp.Sum(bought_ad_slots[i, t, c, d] for i in Movies) * comp_ads_slots[c][d][t][2]
                            for t in TimeSlots for c in Competitors for d in Days
                        ),
                        sense=xp.maximize)
print("===== Total time used to add obj fn: {0} seconds".format((dt.now() - st).total_seconds()))

print("==== Starting Constraints ====")
# Constraints
scheduling.addConstraint(xp.Sum(movie_time[i, t, d] for i in Movies) == 1 for t in TimeSlots for d in Days)
scheduling.addConstraint(xp.Sum(movie_time[i, t, d] for t in TimeSlots) ==
                         movie[i, d] * movie_df['total_time_slots'][i] for i in Movies for d in Days)

# Schedule constraints
scheduling.addConstraint(end_time[i, d] - start_time[i, d] ==
                         movie[i, d] * movie_df['total_time_slots'][i] for i in Movies for d in Days)
scheduling.addConstraint(end_time[i, d] >= (t + 1) * movie_time[i, t, d]
                         for i in Movies for t in TimeSlots for d in Days)
scheduling.addConstraint(start_time[i, d] <= t * movie_time[i, t, d] + (1 - movie_time[i, t, d]) * TOTAL_SLOTS
                         for i in Movies for t in TimeSlots for d in Days)

# Total slots constraint
scheduling.addConstraint(xp.Sum(
    movie_df['total_time_slots'][i] * movie[i, d] for i in Movies) <= TOTAL_SLOTS for d in Days)

# Sold ads constraints
scheduling.addConstraint(xp.Sum(sold_ad_slots[i, t, c, d] for c in Competitors for t in TimeSlots) <= movie[i, d] *
                         movie_df['n_ad_breaks'][i] for i in Movies for d in Days)
scheduling.addConstraint(xp.Sum(sold_ad_slots[i, t, c, d] for c in Competitors) <= movie_time[i, t, d]
                         for i in Movies for d in Days for t in TimeSlots)

# Bought ads constraints
# Can only buy available ad slots. comp_ad_slots[c, t, d, 1] = 0 means that competitor has no ad in that time slot.
scheduling.addConstraint(bought_ad_slots[i, t, c, d] <= comp_ads_slots[c][d][t][1]
                         for i in Movies for c in Competitors for d in Days for t in TimeSlots)
# A constraint for limiting the bought ad slot to be before the movie start time
scheduling.addConstraint(start_time[i, d] + d * number_of_time_slots >=
                         bought_ad_slots[i, t, c, d] * (t + d * number_of_time_slots + 4)
                         for i in Movies for c in Competitors for d in Days for t in TimeSlots)
# Converted viewers constraint (conversion rate)
scheduling.addConstraint(increased_viewers[t, c, d] ==
                         xp.Sum(bought_ad_slots[i, t, c, d] *
                                xp.Sum(comp_ads_viewership[c][demo][d][t] for demo, _ in enumerate(DEMOGRAPHIC_LIST)) *
                                calculate_conversion_rate(competitor_schedules[c], movie_df, all_genres,
                                                          comp_ads_slots[c][d][t][0], movie_df["title"][i],
                                                          MAX_CONVERSION_RATE)
                                for i in Movies)
                         for t in TimeSlots for d in Days for c in Competitors)

# TODO: If needed, add conversion rate when selling ad slots.

# expect no duplicated movies within the number_of_days
scheduling.addConstraint(xp.Sum(movie[i, d] for d in Days) <= 1 for i in Movies)
print("===== Total time used to add constraints: {0} seconds".format((dt.now() - st).total_seconds()))

xp.setOutputEnabled(False)
print("==== Starting Solving ====")
st = dt.now()
scheduling.solve()
print("===== Total time used to solve: {0} seconds".format((dt.now() - st).total_seconds()))

# Printing
# TODO: Add more printing and saving to files
print(f"Objective value: {scheduling.getObjVal()}")

days_labels = ['day_{0}'.format(d) for d in Days]
time_slots_labels = ['slot_{0}'.format(t) for t in TimeSlots]
st = dt.now()
mdf = pd.DataFrame(data=scheduling.getSolution(movie), index=movie_df['title'], columns=days_labels)
filtered_mdf = mdf[mdf.any(axis='columns')]
filtered_mdf.to_csv('out/movie.csv')
# TODO: clear display
# display(filtered_mdf)

mt_sol = scheduling.getSolution(movie_time)
m, n, r = mt_sol.shape
mt_sol = mt_sol.reshape(m, n * r)
slot_day_labels = ['slot_{0}_day_{1}'.format(t, d) for t in TimeSlots for d in Days]
mt_df = pd.DataFrame(data=mt_sol, index=movie_df['title'], columns=slot_day_labels)
filtered_mt_df = mt_df[mt_df.any(axis='columns')]
filtered_mt_df.to_csv('out/movie_time.csv')
# display(filtered_mt_df)

st_df = pd.DataFrame(data=scheduling.getSolution(start_time), index=movie_df['title'], columns=days_labels)
filtered_st_df = st_df[mdf.any(axis='columns')]
filtered_st_df.to_csv("out/start_time.csv")
# display(st_df[mdf.any(axis='columns')])
et_df = pd.DataFrame(data=scheduling.getSolution(end_time), index=movie_df['title'], columns=days_labels)
filtered_et_df = et_df[mdf.any(axis='columns')]
filtered_et_df.to_csv("out/end_time.csv")
# display(et_df[mdf.any(axis='columns')])

as_sol = scheduling.getSolution(sold_ad_slots)
m, n, p, q = as_sol.shape
as_sol = as_sol.reshape(m, n * q * p)
slot_buyer_day_label = ['slot_{0}_buyer_{1}_day_{2}'.format(t, c, d)
                        for t in TimeSlots for c in Competitors for d in Days]
as_df = pd.DataFrame(data=as_sol, index=movie_df['title'], columns=slot_buyer_day_label)
filtered_as_df = (as_df[mdf.any(axis='columns')])
filtered_as_df.to_csv('out/ad_slots.csv')

print("===== Total time used to get solutions into dataframe: {0} seconds".format((dt.now() - st).total_seconds()))

print("===== Total time used for printing: {0} seconds".format((dt.now() - st).total_seconds()))
print("===== Total time used for everything: {0} seconds".format((dt.now() - whole_st).total_seconds()))
