import xpress as xp
from advert_conversion_rates import *  # including np and pd
from utils.data_processing import process_table, SLOT_DURATION
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


(movie_df, channel_0_conversion_rates_df, channel_1_conversion_rates_df, channel_2_conversion_rates_df,
 channel_a_schedule_df, channel_0_schedule_df, channel_1_schedule_df, channel_2_schedule_df) = import_data()
movie_df = process_table(movie_df)
movie_df = movie_df.head(100)

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

number_of_days = 2
Days = range(number_of_days)

scheduling = xp.problem('scheduling')

# Declare
movie = scheduling.addVariables(number_of_movies, number_of_days, name='m', vartype=xp.binary)
movie_time = scheduling.addVariables(number_of_movies, number_of_time_slots, number_of_days, name="mt",
                                     vartype=xp.binary)
start_time = scheduling.addVariables(number_of_movies, number_of_days, name='s', vartype=xp.integer)
end_time = scheduling.addVariables(number_of_movies, number_of_days, name='e', vartype=xp.integer)
ad_slots = scheduling.addVariables(number_of_movies, number_of_buyers, number_of_days, name="as",
                                   vartype=xp.integer)

print(movie_df.columns)

# Objective Function
scheduling.setObjective(-xp.Sum(movie_df['license_fee'][i] * xp.Sum(movie[i, d] for d in Days) for i in Movies) +
                        xp.Sum(xp.Sum(ad_slots[i, j, d] for j in Ad_Buyers for d in Days) *
                               movie_df['ad_slot_with_viewership'][i] for i in Movies),
                        sense=xp.maximize)

# Constraints
scheduling.addConstraint(xp.Sum(movie_time[i, j, d] for i in Movies) == 1 for j in TimeSlots for d in Days)
scheduling.addConstraint(xp.Sum(movie_time[i, j, d] for j in TimeSlots) ==
                         movie[i, d] * movie_df['total_time_slots'][i] for i in Movies for d in Days)

scheduling.addConstraint(end_time[i, d] - start_time[i, d] + 1 ==
                         movie[i, d] * movie_df['total_time_slots'][i] for i in Movies for d in Days)
scheduling.addConstraint(end_time[i, d] >= j * movie_time[i, j, d]
                         for i in Movies for j in TimeSlots for d in Days)
scheduling.addConstraint(start_time[i, d] <= j * movie_time[i, j, d] + (1 - movie_time[i, j, d]) * TOTAL_SLOTS
                         for i in Movies for j in TimeSlots for d in Days)

scheduling.addConstraint(xp.Sum(
    movie_df['total_time_slots'][i] * movie[i, d] for i in Movies) <= TOTAL_SLOTS for d in Days)
scheduling.addConstraint(xp.Sum(ad_slots[i, j, d] for j in Ad_Buyers) <= movie[i, d] * movie_df['n_ad_breaks'][i]
                         for i in Movies for d in Days)

# expect no duplicated movies within the number_of_days
scheduling.addConstraint(xp.Sum(movie[i, d] for d in Days) <= 1 for i in Movies)

xp.setOutputEnabled(False)
st = dt.now()
scheduling.solve()
print("===== Total time used to solve: {0} seconds".format((dt.now() - st).total_seconds()))

# Printing
print(f"Objective value: {scheduling.getObjVal()}")

days_list = ['day_{0}'.format(i) for i in Days]
time_slots_list = ['slot_{0}'.format(i) for i in TimeSlots]
st = dt.now()
mdf = pd.DataFrame(data=scheduling.getSolution(movie), index=movie_df['title'], columns=days_list)
display(mdf[mdf.any(axis='columns')])

# mt_df = pd.DataFrame(data=scheduling.getSolution(movie_time), index=movie_df['title'],
#                      columns=pd.MultiIndex.from_tuples(zip(days_list, time_slots_list)))
# display(mt_df)
print("===== Total time used to get solutions into dataframe: {0} seconds".format((dt.now() - st).total_seconds()))

print("===== Total time used for printing: {0} seconds".format((dt.now() - st).total_seconds()))
print("===== Total time used for everything: {0} seconds".format((dt.now() - whole_st).total_seconds()))
