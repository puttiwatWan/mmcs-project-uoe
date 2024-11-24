import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from functools import cache
from config.config import TOTAL_DAYS, TOTAL_SLOTS


@cache
def create_genre_vector(movie_genres: tuple[str], all_genres: tuple[str]) -> list[int]:
    """
    "One hot encodes" the genres of a movie into the vector space of all
    possible genres.
    """
    
    return [1 if genre in movie_genres else 0 for genre in all_genres]


@cache
def genre_overlap_score(genre_set_A: tuple[str], genre_set_B: tuple[str], all_genres: tuple[str]) -> float:
    """
    Works out how similar two movies A & B are, based on the
    cosine similarity of their genre vectors.
    """
    genre_vec_A = create_genre_vector(movie_genres=genre_set_A, all_genres=all_genres)
    genre_vec_B = create_genre_vector(movie_genres=genre_set_B, all_genres=all_genres)

    return cosine_similarity([genre_vec_A], [genre_vec_B])[0, 0]


def calculate_conversion_rate(schedule_df: pd.DataFrame, movie_df: pd.DataFrame,
                              all_movie_genres: list[str], ad_time_slot: str,
                              movie_title: str, max_conversion_rate: float) -> float:
    """
    This works out how many people that are shown an advert for a movie
    will be likely to watch that movie.  This will depend on the current
    audience's tastes, i.e. is there an overlap between the genres of the
    movie currently being shown and the movie being advertised.

    Takes into account scenarios where an advert straddles the end of one
    movie and the start of another.  Here audience interests is assumed to
    be an average of the two movies.

    This returns a stochastic value between 5-100% of the maximum conversion
    rate, drawn from a normal distribution centred on the overlap score.
    Maximum conversion rate is currently.

    NOTE: THIS ASSUMES AD BREAKS ARE 1 BLOCK SIZE IN DURATION.
    NOTE: Think this function should be used by the students in their codebase somehow.

    :param schedule_df: Time-indexed pandas dataframe containing the tv schedule.
    :param movie_df: Pandas dataframe containing the movie data.
    :param all_movie_genres: List of strings of all unique movie genres in database.
    :param ad_time_slot: string representing timestamp of selected advert spot
                       : in YYYY-MM-DD HH:mm:SS format.
    :param movie_title: Title of the advertised movie.
    :param max_conversion_rate: Maximum expected fraction of audience that would
                              : watch a movie advertised to them.
    """

    if (ad_time_slot not in schedule_df.index or not schedule_df.loc[ad_time_slot].content_type == 'Advert' or
       not movie_df.title.eq(movie_title).any()):
        return 0

    assert ad_time_slot in schedule_df.index, "Selected time slot not found in index."
    assert schedule_df.loc[ad_time_slot].content_type == 'Advert', "Selected time slot is not an advert."
    assert movie_df.title.eq(movie_title).any(), "Selected movie not found in database."

    ad_slot_index = schedule_df.index.get_loc(ad_time_slot)

    # find movies either side of the ad slot, handle edge cases
    previous_content = schedule_df.iloc[ad_slot_index - 1].content if ad_slot_index > 0 else None
    next_content = schedule_df.iloc[ad_slot_index + 1].content if ad_slot_index < (len(schedule_df) - 1) else None

    previous_genres = set(movie_df.query('title == @previous_content').genres.iloc[0]) if previous_content else set()
    next_genres = set(movie_df.query('title == @next_content').genres.iloc[0]) if next_content else set()
    advertised_genres = set(movie_df.query('title == @movie_title').genres.iloc[0])

    overlap_scores = []

    if len(previous_genres) > 0:
        overlap_scores.append(genre_overlap_score(tuple(previous_genres),
                                                  tuple(advertised_genres),
                                                  tuple(all_movie_genres)))
    else:
        pass

    if len(next_genres) > 0:
        overlap_scores.append(genre_overlap_score(tuple(next_genres),
                                                  tuple(advertised_genres),
                                                  tuple(all_movie_genres)))
    else:
        pass

    overlap_score = np.mean(overlap_scores)
    rng = np.random.default_rng()
    stochastic_conversion_rate = np.clip(rng.normal(loc=overlap_score, scale=0.1),
                                         a_min=0.05, a_max=1.)

    return stochastic_conversion_rate * max_conversion_rate


def calculate_genre_conversion_rate(schedule_df: pd.DataFrame, movie_df: pd.DataFrame,
                                    all_movie_genres: list[str], ad_time_slot: str,
                                    advert_genre: str, max_conversion_rate: float) -> float:
    """
    Similar to calculate_conversion_rate, but do individual genres for adverts
    rather than a s

    :param schedule_df: Time-indexed pandas dataframe containing the tv schedule.
    :param movie_df: Pandas dataframe containing the movie data.
    :param all_movie_genres: List of strings of all unique movie genres in database.
    :param ad_time_slot: string representing timestamp of selected advert spot
                       : in YYYY-MM-DD HH:mm:SS format.
    :param advert_genre: Genre of the advertised movie.
    :param max_conversion_rate: Maximum expected fraction of audience that would
                              : watch a movie advertised to them.
    """

    assert ad_time_slot in schedule_df.index, "Selected time slot not found in index."
    assert schedule_df.loc[ad_time_slot].content_type == 'Advert', "Selected time slot is not an advert."
    # assert tmdb_data.title.eq(movie_title).any(), "Selected movie not found in database."

    ad_slot_index = schedule_df.index.get_loc(ad_time_slot)

    # find movies either side of the ad slot, handle edge cases
    previous_content = schedule_df.iloc[ad_slot_index - 1].content if ad_slot_index > 0 else None
    next_content = schedule_df.iloc[ad_slot_index + 1].content if ad_slot_index < (len(schedule_df) - 1) else None

    previous_genres = set(movie_df.query('title == @previous_content').genres.iloc[0]) if previous_content else set()
    next_genres = set(movie_df.query('title == @next_content').genres.iloc[0]) if next_content else set()
    advertised_genres = [advert_genre] # set(movie_df.query('title == @movie_title').genres.iloc[0])

    overlap_scores = []

    if len(previous_genres) > 0:
        overlap_scores.append(genre_overlap_score(tuple(previous_genres),
                                                  tuple(advertised_genres),
                                                  all_movie_genres))
    else:
        pass

    if len(next_genres) > 0:
        overlap_scores.append(genre_overlap_score(tuple(next_genres),
                                                  tuple(advertised_genres),
                                                  all_movie_genres))
    else:
        pass

    overlap_score = np.mean(overlap_scores)
    
    return overlap_score * max_conversion_rate


def calculate_ad_slot_price(schedule_df: pd.DataFrame, base_fee: float,
                            profit_margin: float, budget_factor: float,
                            box_office_factor: float) -> pd.Series:
    """
    Works out the cost required to buy a specific ad slot.  This is based on the time
    of day, and the budget/earnings of the movie being shown before the
    chosen ad slot.

    This function is applied to a schedule dataframe to create a new column
    containing the ad slot prices, returns NaN if the slot is not an ad slot.

    This is also multiplied by the prime time factor, desired profit margin does
    not take into account the effects of prime time factor currently, i.e.
    there'll be a larger profit margin obtained than the one specified for spots
    in prime time.

    Values used in generation of dataset.
    base_fee = 10_000
    profit_margin = 0.2
    budget_factor = 0.002
    box_office_revenue_factor = 0.001

    :param schedule_df: Dataframe containing the populated schedule with movies and
                      : ad breaks.
    :param base_fee: Base fee required for all movies to be licensed to a channel
    :param profit_margin: Percent (in 0-1 scale) of license fee that the channel
                        : wants to make in profit.
    :param budget_factor: What percent (in 0-1 scale) of the movie's budget contributes
                        : to the license fee.
    :param box_office_factor: What percent (in 0-1 scale) of the movie's box office renvenue
                            : contributes to the license fee.
    """

    license_fee = (base_fee
                   + (budget_factor * schedule_df.movie_budget)
                   + (box_office_factor * schedule_df.box_office_revenue)
                   ) * (1. + profit_margin)

    ad_slot_cost = (license_fee / schedule_df.n_ad_breaks) * schedule_df.prime_time_factor

    return np.round(ad_slot_cost, 2)


def vectorize_cosine_sim(A: np.array, B: np.array) -> np.array:
    """
    Create every cosine similarity between row ith of A matrix and row jth of B matrix.
    <br><br>

    For each row, rA and rB, of A and B, the value is calculated using the formula
    cosine_similarity = dot_product_of(rA, rB) / (norm(rA) * norm(rB))
    <br><br>

    :param A: A 2D matrix A containing vectors needed to find the similarity
    :param B: A 2D matrix B containing vectors needed to combine with A to find the similarity
    :returns: An i * j dimension matrix of cosine similarity
    """

    dot_prod = np.dot(A, B.T)
    norms = np.outer(np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1))

    return dot_prod / norms


def generate_conversion_rates(schedule_df: pd.DataFrame, movie_df: pd.DataFrame, original_movie_df: pd.DataFrame,
                              all_movie_genres: list[str], max_conversion_rate: float) -> np.array:
    """
    Calculate all conversion rates for all combinations of movies and each
    ad slot for a given schedule. This takes into account the audience's tastes,
    i.e. how many genres are overlapped between the movies being advertised the
    movies currently being shown. If the ad slot is at the end of a movie, the
    interest is assumed to be an average of the two movies.
    <br><br>

    This returns a stochastic value between 5-100% of the maximum conversion
    rate, drawn from a normal distribution centred on the overlap score.
    <br><br>

    :param schedule_df: Time-indexed pandas dataframe containing the tv schedule.
    :param movie_df: Pandas dataframe containing only the top n selected movie data that
    will be used to solve the problem.
    :param original_movie_df: Pandas dataframe containing all movie data before trimming.
    :param all_movie_genres: List of strings of all unique movie genres in database.
    :param max_conversion_rate: Maximum expected fraction of audience that would watch
    a movie advertised to them.

    :returns: An array of stochastic conversion rate between each movie and each ad slot.
    The dimension is n_movies x n_days x n_ad_slots.
    """

    # Get genres of the movies before and after each ad slot based on the schedule
    prev_genres = schedule_df[(schedule_df['content_type'] == 'Advert').shift(-1).fillna(schedule_df['content'].iloc[1])].merge(
        original_movie_df, left_on='content', right_on='title', how='left').genres.apply(tuple)
    next_genres = schedule_df[(schedule_df['content_type'] == 'Advert').shift(+1).fillna(schedule_df['content'].iloc[-2])].merge(
        original_movie_df, left_on='content', right_on='title', how='left').genres.apply(tuple)

    # Convert genres to genres vectors
    prev_genres_vector = np.stack(
        prev_genres.apply(create_genre_vector, args=(tuple(all_movie_genres),)).values
    )
    next_genres_vector = np.stack(
        next_genres.apply(create_genre_vector, args=(tuple(all_movie_genres),)).values
    )

    # Create all movies' genres vector
    movie_genres_vector = np.stack(
        movie_df.genres.apply(tuple).apply(create_genre_vector, args=(tuple(all_movie_genres),)).values
    )

    # Create similarity between ads' movie and all movies
    prev_genres_score = vectorize_cosine_sim(prev_genres_vector, movie_genres_vector)
    next_genres_score = vectorize_cosine_sim(next_genres_vector, movie_genres_vector)
    
    genres_score = (prev_genres_score + next_genres_score) / 2

    # Generate stochastic rate at the zero mean
    # Firstly, create an array of stochastic rates with mean = 0. Then add these value to each genres score,
    # meaning that the mean is shift to the genres score. Finally, clip the min and max value.
    rng = np.random.default_rng()
    shifter = rng.normal(0, 0.1, size=genres_score.shape)
    stochastic_rate = np.clip(genres_score + shifter, a_min=0.05, a_max=1)

    # Resample the time slot into an interval of 30 minutes and add missing time slots (i.e. time slot that has no ads).
    # First, create a df using the date time on the schedule as an index.
    # Then resample the whole dataframe into a 30-minute interval.
    # Finally, transpose it and convert it into a numpy array of desired dimensions.
    rate_df = pd.DataFrame(stochastic_rate * max_conversion_rate,
                           index=schedule_df[(schedule_df['content_type'] == 'Advert')].index,
                           columns=movie_df.title)
    rates = rate_df.resample('30min').sum().between_time("07:00", "23:30").T.to_numpy().reshape(
        len(movie_df), 7, TOTAL_SLOTS
    )

    return rates
