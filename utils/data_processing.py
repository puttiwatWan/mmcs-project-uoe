import pandas as pd
import numpy as np

BASE_FEE = 10000
PROFIT_MARGIN = 0.2
BUDGET_FACTOR = 0.002
BOX_OFFICE_REVENUE_FACTOR = 0.001
DEMOGRAPHIC_LIST = ['children', 'adults', 'retirees']

def create_licence_fee_vector(budget, revenue):
    base_fee = BASE_FEE
    budget_factor = PROFIT_MARGIN
    box_office_revenue_factor = 0.001
    license_fee = (BASE_FEE + (BUDGET_FACTOR * budget)
               + (BOX_OFFICE_REVENUE_FACTOR * revenue)
               ) * (1. + PROFIT_MARGIN)

    return license_fee


def estimated_view_count(demographic_baseline,
                         demographic_popularity):
    

    def estimate_view_count_calculation(baseline, popularity):

        return baseline + popularity
    
    total_view_count = 0
    demographic_view_list = []
    for demographic in range(len(demographic_popularity)):

        demographic_expected_view_count = estimate_view_count_calculation(demographic_baseline[demographic], demographic_popularity[demographic])
        total_view_count = total_view_count + demographic_expected_view_count
        demographic_view_list.append(demographic_expected_view_count)
    
    return demographic_view_list, total_view_count


def ad_slot_price(license_fee, n_ad_breaks):

    return license_fee/ n_ad_breaks


def process_table(movie_df):

    movie_df['license_fee'] = create_licence_fee_vector(movie_df['budget'], movie_df['revenue'])

    demographic_popularity_list = []
    for demographic in DEMOGRAPHIC_LIST:
        demographic_popularity_list.append(movie_df[f'{demographic}_scaled_popularity'])

    baseline = [0, 0, 0]
    demographic_view_list, total_view_count = estimated_view_count(baseline, demographic_popularity_list)

    demo = 0
    for demographic in DEMOGRAPHIC_LIST:
        movie_df[f'{demographic}_expected_view_count'] = demographic_view_list[demo]
        demo = demo + 1

    movie_df[f'total_expected_view_count'] = total_view_count

    movie_df['ad_slot_price'] = ad_slot_price(movie_df['license_fee'], 
                             movie_df['n_ad_breaks'])

    return movie_df
        