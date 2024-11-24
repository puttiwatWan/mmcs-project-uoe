import ast
import pandas as pd
import numpy as np
import scipy.stats as st

from config.config import (BASE_FEE,
                           BUDGET_FACTOR,
                           BOX_OFFICE_REVENUE_FACTOR,
                           PROFIT_MARGIN,
                           DEMOGRAPHIC_LIST,
                           SLOT_DURATION,
                           TOTAL_VIEW_COUNT)


def create_licence_fee_vector(budget: pd.Series, revenue: pd.Series) -> pd.Series:
    license_fee = (BASE_FEE + (BUDGET_FACTOR * budget)
                   + (BOX_OFFICE_REVENUE_FACTOR * revenue)
                   ) * (1. + PROFIT_MARGIN)

    return license_fee


def estimated_view_count(demographic_baseline: list[float],
                         demographic_popularity: list[pd.Series]
                         ) -> (list[pd.Series], int):
    def estimate_view_count_calculation(baseline: float, popularity: pd.Series) -> pd.Series:
        return baseline*popularity

    total_view_count = 0
    demographic_view_list = []
    for demographic in range(len(demographic_popularity)):
        demographic_expected_view_count = estimate_view_count_calculation(demographic_baseline[demographic],
                                                                          demographic_popularity[demographic])
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
    """
    Create upper, lowerbound of actual view based on E(view) and percentage
    """
    z = st.norm.ppf((1+percentage) / 2)
    variance_factor = np.sqrt(1.5 * ((10)**(-1)))
    upper = expected_view + (expected_view * z * variance_factor)
    lower = expected_view - (expected_view * z * variance_factor)
    return upper, lower


def create_viable_meter(movie_df: pd.DataFrame) -> np.array:
    """
    create a viability metrics for movies
    """
    demo_columns = [f'{demo}_scaled_popularity' for demo in DEMOGRAPHIC_LIST]
    demo_df = movie_df.loc[:,  demo_columns]
    max_view = demo_df.max(axis=1).to_numpy()
    movie_df['total_scaled_popularity'] = movie_df[demo_columns].sum(axis=1)
    mean_view = movie_df.loc[:, ['total_scaled_popularity']].to_numpy().flatten()
    licensing_fee = movie_df.loc[:, ['license_fee']].to_numpy().flatten()
    
    return (((max_view + mean_view))* TOTAL_VIEW_COUNT) - (licensing_fee)


def top_n_viable_film(movie_df: pd.DataFrame, p: float=0.33) -> pd.DataFrame:
    """
    Select top p * (movies) based on viability score.
    """
    movie_df['viable_score'] = create_viable_meter(movie_df)
    movie_df = movie_df.sort_values(['viable_score'], ascending=False)
    return movie_df.head(int(p * len(movie_df)))


def calculate_confidence_bounds_array(
    expected_view: np.ndarray,
    percentage: float = 0.8,
    variance: float = 1*(10**-5)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the upper and lower confidence bounds for an array of expected values.

    Args:
        expected_view (np.ndarray): A NumPy array of expected values.
        percentage (float, optional): The confidence level as a proportion (e.g., 0.8 for 80% confidence). Defaults to 0.8.
        variance (float, optional): The variance factor used in the calculation. Defaults to 1.5.
        scale_factor (float, optional): The scaling factor applied to the variance (e.g., 10^-1). Defaults to 10.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two NumPy arrays:
            - Upper confidence bounds.
            - Lower confidence bounds, adjusted to be no less than 0.
    """
    # Calculate the z-score for the confidence percentage
    z = st.norm.ppf((1 + percentage) / 2)
    
    # Calculate the variance factor
    variance_factor = np.sqrt(variance)
    
    # Calculate upper and lower bounds for the array
    upper = expected_view + (z * variance_factor)
    lower = expected_view - (z * variance_factor)
    
    # Ensure lower bounds are not less than 0
    lower = np.maximum(lower, 0)
    
    return upper, lower
