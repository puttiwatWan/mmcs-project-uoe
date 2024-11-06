import xpress as xp
from advert_conversion_rates import *  # including np and pd
from utils.data_processing import process_table, SLOT_DURATION
from datetime import datetime as dt

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
movie = np.array([xp.var(name="m_{0}_{1}".format(i + 1, j + 1), vartype=xp.binary)
                  for i in Movies for j in Days], dtype=xp.npvar).reshape(number_of_movies, number_of_days)
movie_time = (np.array([xp.var(name="mt_{0}_{1}_{2}".format(i + 1, j + 1, k + 1), vartype=xp.binary)
                       for i in Movies for j in TimeSlots for k in Days], dtype=xp.npvar).
              reshape(number_of_movies, number_of_time_slots, number_of_days))
start_time = np.array([xp.var(name="s_{0}_{1}".format(i + 1, d + 1), vartype=xp.integer)
                       for i in Movies for d in Days], dtype=xp.npvar).reshape(number_of_movies, number_of_days)
end_time = np.array([xp.var(name="e_{0}_{1}".format(i + 1, d + 1), vartype=xp.integer)
                     for i in Movies for d in Days], dtype=xp.npvar).reshape(number_of_movies, number_of_days)
ad_slots = (np.array([xp.var(name="ad_slot_{0}_{1}_{2}".format(i + 1, j + 1, d + 1), vartype=xp.integer)
                     for i in Movies for j in Ad_Buyers for d in Days], dtype=xp.npvar).
            reshape(number_of_movies, number_of_buyers, number_of_days))

scheduling.addVariable(movie, movie_time, start_time, end_time, ad_slots)

print(movie_df.columns)

# Objective Function
scheduling.setObjective(-xp.Sum(movie_df['license_fee'][i] * movie[i] for i in Movies) +
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

scheduling.addConstraint(xp.Sum(movie[i, d] for d in Days) <= 1 for i in Movies)

# xp.setOutputEnabled(False)
st = dt.now()
scheduling.solve()
print("===== Total time used to solve: {0} seconds".format((dt.now() - st).total_seconds()))

# Printing
print(f"Objective value: {scheduling.getObjVal()}")

st = dt.now()
selected_index = [i for i in Movies for d in Days if scheduling.getSolution(movie[i, d]) == 1]
print("===== Total time used to get index for printing: {0} seconds".format((dt.now() - st).total_seconds()))

selected_movies = [movie_df['title'][i] for i in selected_index]
selected_n_ad_break = [movie_df['n_ad_breaks'][i] for i in selected_index]
selected_ad_price = [movie_df['ad_slot_with_viewership'][i]
                     for i in selected_index]
selected_runtime_with_ad = [
    movie_df['runtime_with_ads'][i] for i in selected_index]
selected_licensing_fee = [movie_df['license_fee'][i] for i in selected_index]
selected_popularity = [movie_df['popularity'][i] for i in selected_index]

total_runtime = sum(selected_runtime_with_ad)
total_ad_price = sum([selected_n_ad_break[i] * selected_ad_price[i]
                     for i in range(len(selected_index))])
total_licensing_fee = sum(selected_licensing_fee)

print(f"Movies selected: {selected_movies}")
print(f"Popularity: {selected_popularity}")
print(f"Ad break: {selected_n_ad_break}")
print(f"Ad price: {selected_ad_price}")
print(f"Licensing fee: {selected_licensing_fee}")
print(f"Runtime with ad: {selected_runtime_with_ad}")

print("Total runtime: {:,}".format(total_runtime))
print("Total ad price: {:,}".format(total_ad_price))
print("Total licensing fee: {:,}".format(total_licensing_fee))
print("Total profit: {:,}".format(total_ad_price - total_licensing_fee))

print("===== Total time used for printing: {0} seconds".format((dt.now() - st).total_seconds()))
print("===== Total time used for everything: {0} seconds".format((dt.now() - whole_st).total_seconds()))
