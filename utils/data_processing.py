import ast
import pandas as pd
import numpy as np
import scipy.stats as st

from config.config import *


def create_licence_fee_vector(budget: pd.Series, revenue: pd.Series) -> pd.Series:
    license_fee = (BASE_FEE + (BUDGET_FACTOR * budget)
                   + (BOX_OFFICE_REVENUE_FACTOR * revenue)
                   ) * (1. + PROFIT_MARGIN)

    return license_fee


def estimated_view_count(demographic_baseline: list[float],
                         demographic_popularity: list[pd.Series],
                         latest_showing_date=None,
                         current_date=None) -> (list[pd.Series], int):
    def estimate_view_count_calculation(baseline: float, popularity: pd.Series) -> pd.Series:
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


def process_table(movie_df: pd.DataFrame) -> pd.DataFrame:
    movie_df['license_fee'] = create_licence_fee_vector(
        movie_df['budget'], movie_df['revenue'])

    demographic_popularity_list = []
    for demographic in DEMOGRAPHIC_LIST:
        demographic_popularity_list.append(
            movie_df[f'{demographic}_scaled_popularity'])

    baseline = [1, 1, 1]
    demographic_view_list, total_view_count = estimated_view_count(
        baseline, demographic_popularity_list)

    for demo in range(len(DEMOGRAPHIC_LIST)):
        movie_df[f'{DEMOGRAPHIC_LIST[demo]}_expected_view_count'] = demographic_view_list[demo]

    movie_df = movie_df.drop_duplicates(subset="title")
    movie_df[f'total_expected_view_count'] = total_view_count

    movie_df['ad_slot_price'] = ad_slot_price(movie_df['license_fee'],
                                              movie_df['n_ad_breaks'])

    movie_df['ad_slot_with_viewership'] = ad_slot_with_viewership(movie_df['license_fee'],
                                                                  movie_df['n_ad_breaks'],
                                                                  movie_df['total_expected_view_count'])

    movie_df['total_time_slots'] = movie_df['runtime_with_ads'] / SLOT_DURATION

    movie_df["genres"] = movie_df["genres"].apply(ast.literal_eval)

    return movie_df


def risk_view_functions(expected_view, percentage):
    '''
    Create upper, lowerbound of actual view based on E(view) and percentage
    '''
    z = st.norm.ppf((1+percentage) / 2)
    variance_factor = np.sqrt(1.5 * ((10)**(-1)))
    upper = expected_view + (expected_view * z * variance_factor)
    lower = expected_view - (expected_view * z * variance_factor)
    return upper, lower


def create_viable_meter(movie_df: pd.DataFrame) -> np.array:
    '''
    create a viability metrics for movies
    '''
    demo_columns = [f'{demo}_expected_view_count' for demo in DEMOGRAPHIC_LIST]
    demo_df = movie_df.loc[:,  demo_columns]
    max_view = demo_df.max(axis=1).to_numpy()
    mean_view = movie_df.loc[:, ['total_expected_view_count']].to_numpy().flatten()
    licensing_fee = movie_df.loc[:, ['license_fee']].to_numpy().flatten()
    
    return (((max_view + mean_view))* TOTAL_VIEW_COUNT) - (licensing_fee)


def top_n_viable_film(movie_df: pd.DataFrame, p: float=0.33) -> pd.DataFrame:
    '''
    Select top p * (movies) based on viability score. 
    '''
    movie_df['viable_score'] = create_viable_meter(movie_df)
    movie_df = movie_df.sort_values(['viable_score'], ascending=False)
    return movie_df.head(int(p * len(movie_df)))

