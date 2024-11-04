import pandas as pd
import numpy as np

BASE_FEE = 10000
PROFIT_MARGIN = 0.2
BUDGET_FACTOR = 0.002
BOX_OFFICE_REVENUE_FACTOR = 0.001
DEMOGRAPHIC_LIST = ['children', 'adults', 'retirees']
SLOT_DURATION = 30


def create_licence_fee_vector(budget, revenue):
    base_fee = BASE_FEE
    budget_factor = PROFIT_MARGIN
    box_office_revenue_factor = 0.001
    license_fee = (BASE_FEE + (BUDGET_FACTOR * budget)
                   + (BOX_OFFICE_REVENUE_FACTOR * revenue)
                   ) * (1. + PROFIT_MARGIN)

    return license_fee


def estimated_view_count(demographic_baseline,
                         demographic_popularity,
                         latest_showing_date=None,
                         current_date=None):
    def estimate_view_count_calculation(baseline, popularity):

        baseline = 1

        return baseline*popularity

    total_view_count = 0
    demographic_view_list = []
    for demographic in range(len(demographic_popularity)):
        demographic_expected_view_count = estimate_view_count_calculation(demographic_baseline[demographic],
                                                                          demographic_popularity[demographic])
        # penalty = decay_view_penelty(demographic_expected_view_count,
        #                     latest_showing_date,
        #                     current_date)

        # demographic_expected_view_count = demographic_expected_view_count - penalty

        total_view_count = total_view_count + demographic_expected_view_count
        demographic_view_list.append(demographic_expected_view_count)

    return demographic_view_list, total_view_count


def ad_slot_with_viewership(license_fee, n_ad_breaks, total_view_count):
    norm_view_count = total_view_count / (total_view_count.mean())

    ad_slot = norm_view_count * (license_fee / n_ad_breaks)

    return ad_slot


def ad_slot_price(license_fee, n_ad_breaks):
    return license_fee / n_ad_breaks


def decay(lambda_rate, X):
    return np.exp(-lambda_rate * X)


def decay_view_penelty(estimate_view, latest_showing_date, current_date):
    lambda_rate = 1
    delta_day = (current_date - latest_showing_date).days
    penalty = decay(lambda_rate, delta_day)
    return penalty * estimate_view


def process_table(movie_df):
    movie_df['license_fee'] = create_licence_fee_vector(
        movie_df['budget'], movie_df['revenue'])

    demographic_popularity_list = []
    for demographic in DEMOGRAPHIC_LIST:
        demographic_popularity_list.append(
            movie_df[f'{demographic}_scaled_popularity'])

    baseline = [0, 0, 0]
    demographic_view_list, total_view_count = estimated_view_count(
        baseline, demographic_popularity_list)

    demo = 0
    for demographic in DEMOGRAPHIC_LIST:
        movie_df[f'{demographic}_expected_view_count'] = demographic_view_list[demo]
        demo = demo + 1

    movie_df[f'total_expected_view_count'] = total_view_count

    movie_df['ad_slot_price'] = ad_slot_price(movie_df['license_fee'],
                                              movie_df['n_ad_breaks'])

    movie_df['ad_slot_with_viewership'] = ad_slot_with_viewership(movie_df['license_fee'],
                                                                  movie_df['n_ad_breaks'],
                                                                  movie_df['total_expected_view_count'])

    movie_df['total_time_slots'] = movie_df['runtime_with_ads'] / SLOT_DURATION

    return movie_df
