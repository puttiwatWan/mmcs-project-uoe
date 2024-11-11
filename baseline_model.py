import xpress as xp
import pandas as pd
from utils.data_processing import process_table, DEMOGRAPHIC_LIST, SLOT_DURATION
from utils.schedule_processing import combine_schedule, consolidate_time_to_30_mins_slot
from datetime import datetime as dt
from IPython.display import display

xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')
ADS_PRICE_PER_VIEW = 0.75 ## Get for analysis from competitors

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


(movie_df, channel_0_conversion_rates_df, channel_1_conversion_rates_df, channel_2_conversion_rates_df,
 channel_a_schedule_df, channel_0_schedule_df, channel_1_schedule_df, channel_2_schedule_df) = import_data()
movie_df = process_table(movie_df)
channel_a_30_schedule_df = consolidate_time_to_30_mins_slot(channel_a_schedule_df)
combine_30min_df = combine_schedule(channel_a_30_schedule_df)


# movie_df = movie_df.head(100)

######## --------------- ###################
MAX_RUNTIME_MIN_PER_DAY = 17 * 60
TOTAL_SLOTS = MAX_RUNTIME_MIN_PER_DAY / SLOT_DURATION

number_of_movies = len(movie_df)
Movies = range(number_of_movies)

buyers = ["self", "c1", "c2", "c3"]
number_of_buyers = len(buyers)
Ad_Buyers = range(number_of_buyers)

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
ad_slots = scheduling.addVariables(number_of_movies, number_of_time_slots, number_of_buyers, number_of_days,
                                   name="abs", vartype=xp.binary)

# Objective Function
scheduling.setObjective(-xp.Sum(movie_df['license_fee'][i] * xp.Sum(movie[i, d] for d in Days) for i in Movies) +
                        xp.Sum(combine_30min_df[f"{demo}_prime_time_view_count"][t] *
                               movie_df[f"{demo}_scaled_popularity"][i] * ad_slots[i, t, b, d]
                               for i in Movies for t in TimeSlots for b in Ad_Buyers for d in Days
                               for demo in DEMOGRAPHIC_LIST),
                        # xp.Sum(xp.Sum(ad_slots[i, t, b, d] for b in Ad_Buyers for d in Days for t in TimeSlots) *
                        #        movie_df['ad_slot_with_viewership'][i] for i in Movies),
                        sense=xp.maximize)

# Constraints
scheduling.addConstraint(xp.Sum(movie_time[i, t, d] for i in Movies) == 1 for t in TimeSlots for d in Days)
scheduling.addConstraint(xp.Sum(movie_time[i, t, d] for t in TimeSlots) ==
                         movie[i, d] * movie_df['total_time_slots'][i] for i in Movies for d in Days)

scheduling.addConstraint(end_time[i, d] - start_time[i, d] + 1 ==
                         movie[i, d] * movie_df['total_time_slots'][i] for i in Movies for d in Days)
scheduling.addConstraint(end_time[i, d] >= t * movie_time[i, t, d]
                         for i in Movies for t in TimeSlots for d in Days)
scheduling.addConstraint(start_time[i, d] <= t * movie_time[i, t, d] + (1 - movie_time[i, t, d]) * TOTAL_SLOTS
                         for i in Movies for t in TimeSlots for d in Days)

scheduling.addConstraint(xp.Sum(
    movie_df['total_time_slots'][i] * movie[i, d] for i in Movies) <= TOTAL_SLOTS for d in Days)
scheduling.addConstraint(xp.Sum(ad_slots[i, t, b, d] for b in Ad_Buyers for t in TimeSlots) <= movie[i, d] *
                         movie_df['n_ad_breaks'][i] for i in Movies for d in Days)
scheduling.addConstraint(xp.Sum(ad_slots[i, t, b, d] for b in Ad_Buyers) <= movie_time[i, t, d]
                         for i in Movies for d in Days for t in TimeSlots)

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

as_sol = scheduling.getSolution(ad_slots)
m, n, p, q = as_sol.shape
as_sol = as_sol.reshape(m, n * q * p)
slot_buyer_day_label = ['slot_{0}_buyer_{1}_day_{2}'.format(t, b, d)
                        for t in TimeSlots for b in Ad_Buyers for d in Days]
as_df = pd.DataFrame(data=as_sol, index=movie_df['title'], columns=slot_buyer_day_label)
filtered_as_df = (as_df[mdf.any(axis='columns')])
filtered_as_df.to_csv('out/ad_slots.csv')

print("===== Total time used to get solutions into dataframe: {0} seconds".format((dt.now() - st).total_seconds()))

print("===== Total time used for printing: {0} seconds".format((dt.now() - st).total_seconds()))
print("===== Total time used for everything: {0} seconds".format((dt.now() - whole_st).total_seconds()))
