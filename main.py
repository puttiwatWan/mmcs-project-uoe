import pandas as pd
from config.config import (DAYS_PER_SOLVE,
                           FIRST_WEEK,
                           P,
                           START_FROM_WEEK,
                           WEEK_CONSIDERED,
                           YEAR)
from solver.solver import SchedulingSolver
from utils.data_processing import (process_table,
                                   DEMOGRAPHIC_LIST,
                                   top_n_viable_film)
from utils.schedule_processing import (consolidate_time_to_30_mins_slot,
                                       de_one_hot_columns_include_empty_slots,
                                       return_selected_week,
                                       get_date_from_week,
                                       create_competitor_schedule,
                                       decay_view_penelty,
                                       process_current_week,
                                       process_competitor_current_week,
                                       sort_df_by_slot_day,
                                       update_schedule)
from utils.utils import generate_out_filename, print_title_in_output


def import_data():
    """
    Import all data
    """
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


def main():
    # Import Data
    (movie_df, _, _, _, channel_a_schedule_df, channel_0_schedule_df,
     channel_1_schedule_df, channel_2_schedule_df) = import_data()

    # Process Data
    movie_df = process_table(movie_df)
    original_movie_df = movie_df.copy()

    # Create DF needed in the models
    competitor_schedules = [channel_0_schedule_df, channel_1_schedule_df, channel_2_schedule_df]
    channel_a_30_schedule_df = consolidate_time_to_30_mins_slot(channel_a_schedule_df)

    # Weekly schedule
    # Initialized schedule
    all_schedule_df = movie_df.copy()
    all_schedule_df['latest_showing_date'] = pd.to_datetime('2000-01-01')
    all_schedule_df['latest_aired_datetime'] = pd.NaT
    all_schedule_df['comp_latest_aired_datetime'] = pd.NaT

    # Initialize the solver with related dataframe
    solver = SchedulingSolver(original_movie_df=original_movie_df,
                              movie_df=movie_df,
                              channel_a_30_schedule_df=channel_a_30_schedule_df,
                              competitor_schedules=competitor_schedules,
                              number_of_days=DAYS_PER_SOLVE)

    # Get combined competitor schedule
    combine_schedule = create_competitor_schedule(competitor_schedules)

    # the week to start running the solver
    intended_start_week = START_FROM_WEEK
    for week in range(FIRST_WEEK, FIRST_WEEK + WEEK_CONSIDERED):
        title = f"Starting week {week}"
        print_title_in_output(title, '+')

        current_date = get_date_from_week(week, YEAR)
        week_offset = week - FIRST_WEEK

        # Create Modify DF for this week
        current_adjusted_df = all_schedule_df.copy()
        # Cut all the same movie as competitor out
        current_adjusted_df = current_adjusted_df[~current_adjusted_df['title'].isin(combine_schedule[week_offset])]

        # Create Decay for popularity
        for demo in DEMOGRAPHIC_LIST:
            penalty = decay_view_penelty(
                current_adjusted_df[f'{demo}_scaled_popularity'],
                current_adjusted_df['latest_aired_datetime'],
                current_adjusted_df['comp_latest_aired_datetime'],
                current_date).fillna(0)
            current_adjusted_df[f'{demo}_scaled_popularity'] = current_adjusted_df[
                                                                            f'{demo}_scaled_popularity'] - penalty

        current_adjusted_df = top_n_viable_film(current_adjusted_df, p=P)
        current_adjusted_df.to_csv(generate_out_filename(week, "current_adjusted_df.csv"))

        # RUN XPRESS GET SCHEDULE
        if week >= intended_start_week:
            solver.update_data(movie_df=current_adjusted_df)
            solver.run(week=week, set_soft_limit=True)
            solver.reset_problem()

        schedule_df = pd.read_csv(generate_out_filename(week, "movie_time.csv"))
        schedule_df = sort_df_by_slot_day(schedule_df)
        schedule_df = de_one_hot_columns_include_empty_slots(schedule_df, return_selected_week(channel_a_30_schedule_df, week).index)
        schedule_df = schedule_df.set_index('time')

        # Get competitor schedule
        competitor_current_list_df = []
        for comp in competitor_schedules:
            comp_schedule_df = return_selected_week(comp, week)
            competitor_current_list_df.append(comp_schedule_df)

        # Process current week schedule
        schedule_df = process_current_week(schedule_df, movie_df)

        # Process current week competitor schedule
        competitor_schedule_df = process_competitor_current_week(competitor_current_list_df, movie_df)

        # Update Schedule for what has been schedule this time.
        all_schedule_df = update_schedule(schedule_df, competitor_schedule_df, all_schedule_df)
        all_schedule_df.to_csv(generate_out_filename(week, "all_schedule_df.csv"))


main()
