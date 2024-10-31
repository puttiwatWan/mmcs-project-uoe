import pandas as pd
import numpy as np
import xpress as xp
from advert_conversion_rates import *


def import_data():

    movie_df = pd.read_csv('data/movie_database.csv')
    channel_a_schedule_df = pd.read_csv('data/channel_A_schedule.csv')
    channel_0_conversion_rates_df = pd.read_csv('data/channel_0_conversion_rates.csv')
    channel_1_conversion_rates_df = pd.read_csv('data/channel_1_conversion_rates.csv')
    channel_2_conversion_rates_df = pd.read_csv('data/channel_2_conversion_rates.csv')
    channel_0_schedule_df = pd.read_csv('data/channel_0_schedule.csv')
    channel_1_schedule_df = pd.read_csv('data/channel_1_schedule.csv')
    channel_2_schedule_df = pd.read_csv('data/channel_2_schedule.csv')

    return movie_df, channel_a_schedule_df, channel_0_schedule_df

movie_df, channel_a_schedule_df, channel_0_schedule_df = import_data()

print(channel_a_schedule_df.head())





######## --------------- ###################
# Declare
movie_var = [xp.var (vartype = xp.binary) for i in range (len(movie_df))]


test_schedule_df = channel_0_schedule_df.head(50)
demo_df = channel_0_schedule_df.drop_duplicates(['content'])
demo_content = demo_df['content']
print(demo_df.head())
(calculate_ad_slot_price(channel_0_schedule_df, 10000, 0.2, 0.002, 0.001))

## Objective Function

scheduling = xp.problem('scheduling')
scheduling.addVariable(movie_var)


adincome = ad_slot_cost * movie_var 
licensing_fee = licensing_fee * movie_var
adexpense = 0
maximize_profit = (adincome) - (licensing_fee) - (adexpense)
scheduling.setObjective()



## Constraint

