import pandas as pd
from config.config import (DAYS_PER_SOLVE,
                           FIRST_WEEK,
                           WEEK_CONSIDERED,
                           YEAR)
from solver.solver import SchedulingSolver
from utils.data_processing import (process_table,
                                   DEMOGRAPHIC_LIST,
                                   top_n_viable_film)
from utils.schedule_processing import (consolidate_time_to_30_mins_slot,
                                       return_selected_week,
                                       get_date_from_week,
                                       create_competitor_schedule,
                                       decay_view_penelty,
                                       process_current_week,
                                       process_competitor_current_week,
                                       update_schedule)


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


def main():
    # Import Data
    (movie_df, _, _, _, channel_a_schedule_df, channel_0_schedule_df,
     channel_1_schedule_df, channel_2_schedule_df) = import_data()

    # Process Data
    movie_df = process_table(movie_df)
    original_movie_df = movie_df.copy()
    movie_df = top_n_viable_film(movie_df, p=0.1)

    # Create DF needed in the models
    competitor_schedules = [channel_0_schedule_df, channel_1_schedule_df, channel_2_schedule_df]
    channel_a_30_schedule_df = consolidate_time_to_30_mins_slot(channel_a_schedule_df)

    # Weekly schedule
    # Initialized schedule
    all_schedule_df = movie_df.copy()
    all_schedule_df['latest_showing_date'] = pd.to_datetime('2000-01-01')
    all_schedule_df['latest_aired_datetime'] = pd.NaT
    all_schedule_df['comp_latest_aired_datetime'] = pd.NaT

    for week in range(FIRST_WEEK, FIRST_WEEK + WEEK_CONSIDERED):
        current_date = get_date_from_week(week, YEAR)
        this_week_competitor_list = [return_selected_week(comp, week) for comp in competitor_schedules]
        # Get competitor schedule
        combine_schedule = create_competitor_schedule(this_week_competitor_list)

        # Create Modify DF for this week
        current_adjusted_df = all_schedule_df.copy()
        # Cut all the same movie as competitor out
        current_adjusted_df = current_adjusted_df[~current_adjusted_df['title'].isin(combine_schedule[0])]
        print(current_adjusted_df['latest_aired_datetime'].values)
        print(current_adjusted_df['comp_latest_aired_datetime'].values)

        # Create Decay for popularity
        for demo in DEMOGRAPHIC_LIST:
            penalty = decay_view_penelty(
                current_adjusted_df[f'{demo}_scaled_popularity'],
                current_adjusted_df['latest_aired_datetime'],
                current_adjusted_df['comp_latest_aired_datetime'],
                current_date).fillna(0)
            current_adjusted_df[f'adjusted_{demo}_scaled_popularity'] = current_adjusted_df[
                                                                            f'{demo}_scaled_popularity'] - penalty

        current_adjusted_df.to_csv(f'out/current_adjusted_df_{week}.csv')

        # RUN XPRESS GET SCHEDULE
        schedule_df = return_selected_week(channel_0_schedule_df, week)

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

        last_week_schedule_df = schedule_df
        all_schedule_df.to_csv(f'out/all_schedule_df_{week}.csv')

    solver = SchedulingSolver(original_movie_df=original_movie_df,
                              movie_df=movie_df,
                              channel_a_30_schedule_df=channel_a_30_schedule_df,
                              competitor_schedules=competitor_schedules,
                              number_of_days=DAYS_PER_SOLVE)

    solver.run(set_hard_limit=True)


main()
