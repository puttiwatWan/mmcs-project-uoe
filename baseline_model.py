import pandas as pd
import numpy as np
import xpress as xp
from advert_conversion_rates import *
from utils.data_processing import process_table

xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')


def import_data():
    mov_df = pd.read_csv('data/movie_database.csv')
    ch_a_schedule_df = pd.read_csv('data/channel_A_schedule.csv')
    ch_0_conversion_rates_df = pd.read_csv('data/channel_0_conversion_rates.csv')
    ch_1_conversion_rates_df = pd.read_csv('data/channel_1_conversion_rates.csv')
    ch_2_conversion_rates_df = pd.read_csv('data/channel_2_conversion_rates.csv')
    ch_0_schedule_df = pd.read_csv('data/channel_0_schedule.csv')
    ch_1_schedule_df = pd.read_csv('data/channel_1_schedule.csv')
    ch_2_schedule_df = pd.read_csv('data/channel_2_schedule.csv')

    return mov_df, ch_a_schedule_df, ch_0_schedule_df


movie_df, channel_a_schedule_df, channel_0_schedule_df = import_data()
movie_df = process_table(movie_df)


######## --------------- ###################
number_of_movies = len(movie_df)
Movies = range(number_of_movies)
buyers = ["self", "c1", "c2", "c3"]
number_of_buyers = len(buyers)
Ad_Buyers = range(number_of_buyers)

scheduling = xp.problem('scheduling')

# Declare
m = np.array([xp.var(name="m{0}".format(i+1), vartype=xp.binary)
              for i in Movies], dtype=xp.npvar)
ad_slots = np.array([xp.var(name="ad_slot_{0}_{1}".format(i+1, j+1), vartype=xp.integer)
                    for i in Movies for j in Ad_Buyers], dtype=xp.npvar).reshape(number_of_movies, number_of_buyers)

scheduling.addVariable(m, ad_slots)

print(movie_df.columns)

# Objective Function
scheduling.setObjective(-xp.Sum(movie_df['license_fee'][i] * m[i] for i in Movies) +
                        xp.Sum(xp.Sum(ad_slots[i, j] for j in Ad_Buyers) * movie_df['ad_slot_price'][i]
                               for i in Movies),
                        sense=xp.maximize)

# Constraints
scheduling.addConstraint(xp.Sum(movie_df['runtime_with_ads'][i] * m[i] for i in Movies) <= 17*60)
scheduling.addConstraint(xp.Sum(ad_slots[i, j] for j in Ad_Buyers) <= m[i] * movie_df['n_ad_breaks'][i] for i in Movies)

# xp.setOutputEnabled(False)
scheduling.solve()

print(f"Objective value: {scheduling.getObjVal()}")

selected_index = [i for i in Movies if scheduling.getSolution(m[i]) == 1]
selected_movies = [movie_df.title[i] for i in selected_index]
selected_n_ad_break = [movie_df.n_ad_breaks[i] for i in selected_index]
selected_ad_price = [movie_df.ad_slot_price[i] for i in selected_index]
selected_runtime_with_ad = [movie_df.runtime_with_ads[i] for i in selected_index]
selected_licensing_fee = [movie_df.license_fee[i] for i in selected_index]

total_runtime = sum(selected_runtime_with_ad)
total_ad_price = sum([selected_n_ad_break[i] * selected_ad_price[i] for i in range(len(selected_index))])
total_licensing_fee = sum(selected_licensing_fee)
print(f"Movies selected: {selected_movies}")
print(f"Ad break: {selected_n_ad_break}")
print(f"Ad price: {selected_ad_price}")
print(f"Licensing fee: {selected_licensing_fee}")
print(f"Runtime with ad: {selected_runtime_with_ad}")
print(f"Total runtime: {total_runtime}")
print(f"Total ad price: {total_ad_price}")
print(f"Total licensing fee: {total_licensing_fee}")
print(f"Total profit: {total_ad_price - total_licensing_fee}")



# test_schedule_df = channel_0_schedule_df.head(50)
# demo_df = channel_0_schedule_df.drop_duplicates(['content'])
# demo_content = demo_df['content']
# print(demo_df.head())
# (calculate_ad_slot_price(channel_0_schedule_df, 10000, 0.2, 0.002, 0.001))


# scheduling.addVariable(movie_var)


# adincome = ad_slot_cost * movie_var 
# licensing_fee = licensing_fee * movie_var
# adexpense = 0
# maximize_profit = (adincome) - (licensing_fee) - (adexpense)
