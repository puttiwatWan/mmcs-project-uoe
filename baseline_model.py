import numpy as np
import xpress as xp
import pandas as pd
from utils.data_processing import process_table, DEMOGRAPHIC_LIST, SLOT_DURATION
from utils.schedule_processing import (combine_schedule,
                                       consolidate_time_to_30_mins_slot,
                                       dynamic_pricing,
                                       return_selected_week,
                                       get_date_from_week,
                                       create_competitor_schedule,
                                       decay_view_penelty,
                                       process_current_week,
                                       update_schedule,
                                       return_ads_30_mins)
from datetime import datetime as dt
from IPython.display import display

xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

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

## Import Data
(movie_df, channel_0_conversion_rates_df, channel_1_conversion_rates_df, channel_2_conversion_rates_df,
 channel_a_schedule_df, channel_0_schedule_df, channel_1_schedule_df, channel_2_schedule_df) = import_data()

## Process Data
movie_df = process_table(movie_df)

## Create DF needed in the models
competitor_list = [channel_0_schedule_df, channel_1_schedule_df, channel_2_schedule_df]
channel_a_30_schedule_df = consolidate_time_to_30_mins_slot(channel_a_schedule_df)
combine_30min_df = combine_schedule(channel_a_30_schedule_df)

## Create ads slots price in 30 mins time slot of competitors
comp_ads_slots = []
for comp in competitor_list:
    comp_ads_slots.appends(return_ads_30_mins(comp))


### Return Pricing for the week (first week is week 40)
ads_price_per_view = dynamic_pricing(week=40, competitor_list=competitor_list)

first_week = 40
week_consider = 2
all_schedule_df = movie_df.copy()
all_schedule_df['latest_showing_date'] = pd.to_datetime('2000-01-01')
last_week_schedule_df = return_selected_week(channel_a_schedule_df, 40) ### Dummy
year = 2024


## Weekly schedule
for week in range(first_week, first_week + week_consider):

    current_date = get_date_from_week(week, year)
    this_week_competitor_list = [return_selected_week(comp, week) for comp in competitor_list]
    ### Get competitor schedule
    combine_schedule = create_competitor_schedule(this_week_competitor_list)

    ### Create Modify DF for this week
    current_adjusted_df = all_schedule_df.copy()
    ### Cut all the same movie as competitor out
    current_adjusted_df = current_adjusted_df[~current_adjusted_df['title'].isin(combine_schedule[0])]
    ### Create Decay for popularity
    current_adjusted_df['adjusted_popularity'] = current_adjusted_df['popularity'] - decay_view_penelty(
        current_adjusted_df['popularity'], current_adjusted_df['latest_showing_date'], current_date)

    print(current_adjusted_df.head())
    ### RUN XPRESS GET SCHEDULE
    schedule_df = return_selected_week(channel_0_schedule_df, week)

    ### Process current week schedule
    schedule_df = process_current_week(schedule_df, movie_df)
    #### Update Schedule for what has been schedule this time.
    all_schedule_df = update_schedule(schedule_df, all_schedule_df)

    last_week_schedule_df = schedule_df

# TODO: Get available ad slots for each competitors
mock_comp_num = 3
mock_time_slots = 34
mock_days = 7
comp_ad_slots = np.array([1 for i in range(mock_comp_num * mock_time_slots * mock_days)]).reshape(
    mock_comp_num, mock_time_slots, mock_days
)

######## --------------- ###################
MAX_RUNTIME_MIN_PER_DAY = 17 * 60
TOTAL_SLOTS = MAX_RUNTIME_MIN_PER_DAY / SLOT_DURATION

number_of_movies = len(movie_df)
Movies = range(number_of_movies)

competitors = ["c1", "c2", "c3"]
number_of_competitors = len(competitors)
Competitors = range(number_of_competitors)

number_of_time_slots = (24 - 7) * 2  # 30 min each
TimeSlots = range(number_of_time_slots)

number_of_days = 1
Days = range(number_of_days)

scheduling = xp.problem('scheduling')

# Declare
movie = scheduling.addVariables(number_of_movies, number_of_days, name='m', vartype=xp.binary)
movie_time = scheduling.addVariables(number_of_movies, number_of_time_slots, number_of_days, name="mt",
                                     vartype=xp.binary)
start_time = scheduling.addVariables(number_of_movies, number_of_days, name='s', vartype=xp.integer)
end_time = scheduling.addVariables(number_of_movies, number_of_days, name='e', vartype=xp.integer)

# TODO: Buyers now only include competitors. If our ad slot is used for ourselves, we can check by
#       xp.Sum(sold_ad_slots[i, t, c, d] for c in Buyers) == 0 (meaning no competitor bought that slot)
sold_ad_slots = scheduling.addVariables(number_of_movies, number_of_time_slots, number_of_competitors, number_of_days,
                                        name="sa", vartype=xp.binary)
bought_ad_slots = scheduling.addVariables(number_of_movies, number_of_time_slots, number_of_competitors, number_of_days,
                                          name="ba", vartype=xp.binary)
# increased_viewers = scheduling.addVariables(number_of_time_slots, number_of_days, name="iv", vartype=xp.continuous)

# Objective Function
# TODO: Add bought ad slots into obj fn
scheduling.setObjective(-xp.Sum(movie_df['license_fee'][i] * xp.Sum(movie[i, d] for d in Days) for i in Movies) +
                        xp.Sum(combine_30min_df[f"{demo}_prime_time_view_count"][t] *
                               movie_df[f"{demo}_scaled_popularity"][i] * sold_ad_slots[i, t, c, d]
                               for i in Movies for t in TimeSlots for c in Competitors for d in Days
                               for demo in DEMOGRAPHIC_LIST),
                        # xp.Sum(xp.Sum(ad_slots[i, t, c, d] for b in Ad_Buyers for d in Days for t in TimeSlots) *
                        #        movie_df['ad_slot_with_viewership'][i] for i in Movies),
                        sense=xp.maximize)

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
# Can only buy available ad slots. comp_ad_slots[c, t, d] = 0 means that competitor has no ad in that time slot.
scheduling.addConstraint(bought_ad_slots[i, t, c, d] <= comp_ad_slots[c, t, d]
                         for i in Movies for c in Competitors for d in Days for t in TimeSlots)
# TODO: Add a constraint for limiting the bought ad slot to be before the movie start time (below still does not work !)
# scheduling.addConstraint(start_time[i, d] - (bought_ad_slots[i, t, c, d] * t) <= 4
#                          for i in Movies for c in Competitors for d in Days for t in TimeSlots)
# TODO: Add converted viewers constraint (conversion rate)
# scheduling.addConstraint(increased_viewers[t, d] == conversion_rate
#                          for t in TimeSlots for d in Days)

# expect no duplicated movies within the number_of_days
scheduling.addConstraint(xp.Sum(movie[i, d] for d in Days) <= 1 for i in Movies)

xp.setOutputEnabled(False)
st = dt.now()
scheduling.solve()
print("===== Total time used to solve: {0} seconds".format((dt.now() - st).total_seconds()))

# Printing
print(f"Objective value: {scheduling.getObjVal()}")

days_labels = ['day_{0}'.format(d) for d in Days]
time_slots_labels = ['slot_{0}'.format(t) for t in TimeSlots]
st = dt.now()
mdf = pd.DataFrame(data=scheduling.getSolution(movie), index=movie_df['title'], columns=days_labels)
filtered_mdf = mdf[mdf.any(axis='columns')]
filtered_mdf.to_csv('out/movie.csv')
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
