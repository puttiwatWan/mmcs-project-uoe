import numpy as np
import pandas as pd
import xpress as xp

from datetime import datetime as dt
from itertools import chain

from advert_conversion_rates import generate_conversion_rates
from config.config import (COMPETITORS,
                           DEMOGRAPHIC_LIST,
                           MAX_CONVERSION_RATE,
                           MAX_HARD_LIMIT_RUNTIME,
                           MAX_SOFT_LIMIT_RUNTIME,
                           SLOT_DURATION,
                           TOTAL_SLOTS)
from utils.schedule_processing import (combine_schedule,
                                       consolidate_time_to_30_mins_slot,
                                       dynamic_pricing,
                                       return_selected_week,
                                       get_date_from_week,
                                       create_competitor_schedule,
                                       decay_view_penelty,
                                       process_current_week,
                                       process_competitor_current_week,
                                       update_schedule,
                                       return_ads_30_mins)


def time_spent_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"====== Starting function {func.__name__} ======")
        st = dt.now()
        func(*args, **kwargs)
        print(f"====== Total time used for function {func.__name__}: {(dt.now() - st).total_seconds()} seconds ======")

    return wrapper


class Solver:
    def __init__(self,
                 original_movie_df: pd.DataFrame,
                 movie_df: pd.DataFrame,
                 competitor_schedules: list[pd.DataFrame],
                 channel_a_30_schedule_df: pd.DataFrame,
                 number_of_days: int,
                 ):
        self.movie = None
        self.movie_time = None
        self.start_time = None
        self.end_time = None
        self.sold_ad_slots = None
        self.bought_ad_slots = None
        self.increased_viewers = None
        self.z = None

        self.original_movie_df = original_movie_df
        self.movie_df = movie_df

        all_genres = list(set(chain.from_iterable(original_movie_df["genres"])))
        self.conversion_rates = generate_conversion_rates(competitor_schedules[0], movie_df, original_movie_df,
                                                          all_genres, MAX_CONVERSION_RATE)

        # Parameters related to competitor_schedules and channel_a_30_schedule_df
        comp_ads_slots = []  # np_array of dimension n_comp x n_days x n_time_slots x (0|1, ad_price)
        comp_ads_viewership = []  # np_array of dimension n_comp x n_demo x n_days x n_time_slots
        for comp in competitor_schedules:
            ads, ad_viewership = return_ads_30_mins(comp, channel_a_30_schedule_df.index)
            comp_ads_slots.append(ads)
            comp_ads_viewership.append(ad_viewership)
        self.comp_ads_slots = np.array(comp_ads_slots)
        self.comp_ads_viewership = np.array(comp_ads_viewership)
        self.ads_price_per_view = dynamic_pricing(week=40, competitor_schedule_list=competitor_schedules)
        self.combine_30min_df = combine_schedule(channel_a_30_schedule_df)

        self.number_of_movies = len(movie_df)
        self.number_of_competitors = len(COMPETITORS)
        self.number_of_time_slots = int((24 - 7) * 60 / SLOT_DURATION)
        self.number_of_days = number_of_days

        self.Movies = range(self.number_of_movies)
        self.Competitors = range(self.number_of_competitors)
        self.TimeSlots = range(self.number_of_time_slots)
        self.Days = range(self.number_of_days)

        self.M = 1  # Maximum viewership possible in one time slot

        self.scheduling = xp.problem('scheduling')

    def update_data(self,
                    original_movie_df: pd.DataFrame = None,
                    movie_df: pd.DataFrame = None,
                    competitor_schedules: list[pd.DataFrame] = None,
                    channel_a_30_schedule_df: pd.DataFrame = None,
                    ):
        if movie_df:
            self.movie_df = movie_df
            self.number_of_movies = len(movie_df)
            self.Movies = range(self.number_of_movies)
        if original_movie_df:
            self.original_movie_df = original_movie_df
        if competitor_schedules or channel_a_30_schedule_df:
            if competitor_schedules:
                self.ads_price_per_view = dynamic_pricing(week=40, competitor_schedule_list=competitor_schedules)
                all_genres = list(set(chain.from_iterable(original_movie_df["genres"])))
                self.conversion_rates = generate_conversion_rates(competitor_schedules[0], movie_df, original_movie_df,
                                                                  all_genres, MAX_CONVERSION_RATE)
            if channel_a_30_schedule_df:
                self.combine_30min_df = combine_schedule(channel_a_30_schedule_df)

            comp_ads_slots = []  # np_array of dimension n_comp x n_days x n_time_slots x (0|1, ad_price)
            comp_ads_viewership = []  # np_array of dimension n_comp x n_demo x n_days x n_time_slots
            for comp in competitor_schedules:
                ads, ad_viewership = return_ads_30_mins(comp, channel_a_30_schedule_df.index)
                comp_ads_slots.append(ads)
                comp_ads_viewership.append(ad_viewership)
            self.comp_ads_slots = np.array(comp_ads_slots)
            self.comp_ads_viewership = np.array(comp_ads_viewership)

    def reset_problem(self):
        del self.scheduling

    @time_spent_decorator
    def add_decision_variables(self):
        self.movie = self.scheduling.addVariables(self.number_of_movies, self.number_of_days, name='m',
                                                  vartype=xp.binary)
        self.movie_time = self.scheduling.addVariables(self.number_of_movies, self.number_of_time_slots,
                                                       self.number_of_days,
                                                       name="mt",
                                                       vartype=xp.binary)
        self.start_time = self.scheduling.addVariables(self.number_of_movies, self.number_of_days, name='s',
                                                       vartype=xp.integer)
        self.end_time = self.scheduling.addVariables(self.number_of_movies, self.number_of_days, name='e',
                                                     vartype=xp.integer)
        self.sold_ad_slots = self.scheduling.addVariables(self.number_of_movies, self.number_of_time_slots,
                                                          self.number_of_competitors, self.number_of_days,
                                                          name="sa", vartype=xp.binary)
        self.bought_ad_slots = self.scheduling.addVariables(self.number_of_movies, self.number_of_time_slots,
                                                            self.number_of_competitors, self.number_of_days,
                                                            name="ba", vartype=xp.binary)
        self.increased_viewers = self.scheduling.addVariables(self.number_of_time_slots, self.number_of_competitors,
                                                              self.number_of_days,
                                                              name="iv", vartype=xp.continuous)
        self.z = self.scheduling.addVariables(self.number_of_movies, self.number_of_time_slots,
                                              self.number_of_competitors,
                                              self.number_of_days,
                                              name="z", vartype=xp.continuous)

    @time_spent_decorator
    def add_objective_function(self):
        # The objective is to maximize the profit gained from showing movies and selling ads.
        self.scheduling.setObjective(
            # Deduct the licensing fee for each movie shown
            - xp.Sum(self.movie_df['license_fee'].iloc[i] * xp.Sum(self.movie[i, d] for d in self.Days)
                     for i in self.Movies)
            # Add the profit from selling ads to the competitor
            + xp.Sum(
                (
                    # Profit from ad slots sold from the channel's calculated expected viewers in that ad slot.
                    # The view count in each time slot is calculated from each demographic expected views,
                    # the prime time factor, and the popularity of the movie shown in that time slot.
                    # The profit is calculated from the expected views in each ad time slot and the price_per_views.
                    # The ads_price_per_view is calculated from the average price of an ad per expected views of
                    # all competitors.
                    xp.Sum(
                        self.combine_30min_df[f"{demo}_prime_time_view_count"].iloc[t] *
                        self.movie_df[f"{demo}_scaled_popularity"].iloc[i] for demo in DEMOGRAPHIC_LIST
                    )
                    * self.sold_ad_slots[i, t, c, d]  # Sold ad slots
                    * 1000000 * self.ads_price_per_view
                )
                + (
                    # Profit from ad slots sold from increased_viewers
                    # This is a profit from gained viewers by buying ads from the competitors. Variable z is just the
                    # linearization of the quadratic sold_ad_slots * increased_viewers
                    # The increased viewers is calculated from the conversion rates of each competitor view count
                    # on that ad slot.
                    self.z[i, t, c, d]  # Auxiliary variable for increased viewers * sold ad slots.
                    * 1000000 * self.ads_price_per_view
                )
                for i in self.Movies for t in self.TimeSlots for c in self.Competitors for d in self.Days
            )
            # Subtract the cost for buying ads on the competitor's channels.
            # comp_ads_slots[c, d, t, 1] is the price for that ad slot on each competitor.
            - xp.Sum(
                xp.Sum(self.bought_ad_slots[i, t, c, d] for i in self.Movies) * self.comp_ads_slots[c, d, t, 1]
                for t in self.TimeSlots for c in self.Competitors for d in self.Days
            ),
            sense=xp.maximize
        )

    @time_spent_decorator
    def add_constraints(self):
        # ====== Constraints for selecting movies to be shown ======
        # At most one movie can be shown at a time
        # noinspection PyArgumentList
        self.scheduling.addConstraint(
            xp.Sum(self.movie_time[i, t, d] for i in self.Movies) <= 1 for t in self.TimeSlots for d in self.Days)
        # Ensure that each movie shown on a day covers the exact amount of time slots needed.
        # Note: movie_df['total_time_slots'] tells how many slots that movie takes when being shown.
        self.scheduling.addConstraint(xp.Sum(self.movie_time[i, t, d] for t in self.TimeSlots) ==
                                      self.movie[i, d] * self.movie_df['total_time_slots'].iloc[i] for i in self.Movies
                                      for d in self.Days)
        # The sum of all time slots used to show movies on each day should not exceed the
        # total time slots available on each day.
        self.scheduling.addConstraint(
            xp.Sum(self.movie_df['total_time_slots'].iloc[i] * self.movie[i, d] for i in self.Movies) <=
            TOTAL_SLOTS for d in self.Days)
        # No duplicated movies within a number of days
        self.scheduling.addConstraint(xp.Sum(self.movie[i, d] for d in self.Days) <= 1 for i in self.Movies)

        # ====== Constraints for linearize increased_viewership
        self.scheduling.addConstraint(
            self.z[i, t, c, d] <= self.increased_viewers[t, c, d] for i in self.Movies for t in self.TimeSlots for c in
            self.Competitors for d in
            self.Days)
        self.scheduling.addConstraint(
            self.z[i, t, c, d] <= self.M * self.sold_ad_slots[i, t, c, d] for i in self.Movies for t in self.TimeSlots
            for c in self.Competitors for d
            in self.Days)
        self.scheduling.addConstraint(
            self.z[i, t, c, d] >= self.increased_viewers[t, c, d] - self.M * (1 - self.sold_ad_slots[i, t, c, d]) for i
            in self.Movies for t in
            self.TimeSlots for c in self.Competitors for d in self.Days)
        self.scheduling.addConstraint(
            self.z[i, t, c, d] >= 0 for i in self.Movies for t in self.TimeSlots for c in self.Competitors for d in
            self.Days)

        # ====== Constraints for the start time and end time of a movie ======
        # The end time and start time of the movie should be the same as the time slots needed to show that movie.
        # Note: movie_df['total_time_slots'] tells how many slots that movie takes when being shown.
        self.scheduling.addConstraint(self.end_time[i, d] - self.start_time[i, d] ==
                                      self.movie[i, d] * self.movie_df['total_time_slots'].iloc[i] for i in self.Movies
                                      for d in self.Days)
        # This constraint calculates the end time of a movie.
        self.scheduling.addConstraint(self.end_time[i, d] >= (t + 1) * self.movie_time[i, t, d]
                                      for i in self.Movies for t in self.TimeSlots for d in self.Days)
        # This constraint calculates the start time of a movie.
        self.scheduling.addConstraint(
            self.start_time[i, d] <= t * self.movie_time[i, t, d] + (1 - self.movie_time[i, t, d]) * TOTAL_SLOTS
            for i in self.Movies for t in self.TimeSlots for d in self.Days)
        # ====== Constraints for selling ads ======
        # Each movie can sell the ad slots depending on how many ads are in the movie.
        # I.e. 10 ads can be sold to competitors during a movie if the movie has 10 ad breaks.
        # Note: movie_df["n_ad_breaks"] shows how many ad breaks are in each movie.
        self.scheduling.addConstraint(
            xp.Sum(self.sold_ad_slots[i, t, c, d] for c in self.Competitors for t in self.TimeSlots) <= self.movie[
                i, d] *
            self.movie_df['n_ad_breaks'].iloc[i] for i in self.Movies for d in self.Days)
        # One ad break can be sold to only one competitor.
        self.scheduling.addConstraint(
            xp.Sum(self.sold_ad_slots[i, t, c, d] for c in self.Competitors) <= self.movie_time[i, t, d]
            for i in self.Movies for d in self.Days for t in self.TimeSlots)

        # ====== Constraints for buying ads ======
        # Can only buy available ad slots and only if the movie is going to be shown on the channel.
        # Note: comp_ad_slots[c, t, d, 0] = 1 means that competitor has an ad in that time slot, otherwise, 0.
        self.scheduling.addConstraint(
            self.bought_ad_slots[i, t, c, d] <= self.comp_ads_slots[c, d, t, 0] * self.movie[i, d]
            for i in self.Movies for c in self.Competitors for d in self.Days for t in
            self.TimeSlots)
        # The bought ad slot needs to be before the movie start time at least 4 time slots
        self.scheduling.addConstraint(self.start_time[i, d] + (d * self.number_of_time_slots) >=
                                      self.bought_ad_slots[i, t, c, d] * (t + (d * self.number_of_time_slots) + 4)
                                      for i in self.Movies for c in self.Competitors for d in self.Days for t in
                                      self.TimeSlots)
        # Calculate the gained viewers from buying an ad.
        # The calculation is simply expected_viewership x conversion_rate.
        self.scheduling.addConstraint(self.increased_viewers[t, c, d] ==
                                      xp.Sum(self.bought_ad_slots[i, t, c, d] *
                                             xp.Sum(self.comp_ads_viewership[c, demo, d, t] for demo, _ in
                                                    enumerate(DEMOGRAPHIC_LIST)) *
                                             self.conversion_rates[i, d, t] for i in self.Movies)
                                      for t in self.TimeSlots for d in self.Days for c in self.Competitors)
        # Each time slot can only be bought once
        self.scheduling.addConstraint(xp.Sum(self.bought_ad_slots[i, t, c, d] for i in self.Movies) <= 1
                                      for t in self.TimeSlots for c in self.Competitors for d in self.Days)
        # Each movie can be advertised only once per each competitor
        self.scheduling.addConstraint(
            xp.Sum(self.bought_ad_slots[i, t, c, d] for t in self.TimeSlots for d in self.Days) <= 1
            for i in self.Movies for c in self.Competitors)

    @time_spent_decorator
    def solve(self, set_soft_limit: bool = False, set_hard_limit: bool = False):
        if set_soft_limit:
            self.scheduling.setControl('soltimelimit', MAX_HARD_LIMIT_RUNTIME)
        if set_hard_limit:
            self.scheduling.setControl('timelimit', MAX_SOFT_LIMIT_RUNTIME)

        self.scheduling.solve()

        if set_soft_limit or set_hard_limit:
            obj_val = self.scheduling.attributes.objval
            best_bound = self.scheduling.getAttrib('bestbound')
            mip_gap = 100 * ((obj_val - best_bound) / obj_val)

            print(f"MIP GAP: {mip_gap}")

    @time_spent_decorator
    def save_results(self, subfolder: str = ""):
        def generate_path_name(filename: str) -> str:
            return "out/" + subfolder + filename

        days_labels = ['day_{0}'.format(d) for d in self.Days]
        mdf = pd.DataFrame(data=self.scheduling.getSolution(self.movie), index=self.movie_df['title'],
                           columns=days_labels)
        filtered_mdf = mdf[mdf.any(axis='columns')]
        filtered_mdf.to_csv(generate_path_name('movie.csv'))

        mt_sol = self.scheduling.getSolution(self.movie_time)
        m, n, r = mt_sol.shape
        mt_sol = mt_sol.reshape(m, n * r)
        slot_day_labels = ['slot_{0}_day_{1}'.format(t, d) for t in self.TimeSlots for d in self.Days]
        mt_df = pd.DataFrame(data=mt_sol, index=self.movie_df['title'], columns=slot_day_labels)
        filtered_mt_df = mt_df[mt_df.any(axis='columns')]
        filtered_mt_df.to_csv(generate_path_name('movie_time.csv'))

        st_df = pd.DataFrame(data=self.scheduling.getSolution(self.start_time), index=self.movie_df['title'],
                             columns=days_labels)
        filtered_st_df = st_df[mdf.any(axis='columns')]
        filtered_st_df.to_csv(generate_path_name("start_time.csv"))

        et_df = pd.DataFrame(data=self.scheduling.getSolution(self.end_time), index=self.movie_df['title'],
                             columns=days_labels)
        filtered_et_df = et_df[mdf.any(axis='columns')]
        filtered_et_df.to_csv(generate_path_name("end_time.csv"))

        as_sol = self.scheduling.getSolution(self.sold_ad_slots)
        m, n, p, q = as_sol.shape
        as_sol = as_sol.reshape(m, n * q * p)
        slot_comp_day_label = ['slot_{0}_comp_{1}_day_{2}'.format(t, c, d)
                               for t in self.TimeSlots for c in self.Competitors for d in self.Days]
        as_df = pd.DataFrame(data=as_sol, index=self.movie_df['title'], columns=slot_comp_day_label)
        filtered_as_df = (as_df[as_df.any(axis='columns')])
        filtered_as_df.to_csv(generate_path_name('sold_ad_slots.csv'))

        bs_sol = self.scheduling.getSolution(self.bought_ad_slots)
        m, n, p, q = bs_sol.shape
        bs_sol = bs_sol.reshape(m, n * q * p)
        slot_comp_day_label = ['slot_{0}_comp_{1}_day_{2}'.format(t, c, d)
                               for t in self.TimeSlots for c in self.Competitors for d in self.Days]
        bs_df = pd.DataFrame(data=bs_sol, index=self.movie_df['title'], columns=slot_comp_day_label)
        filtered_bs_df = (bs_df[bs_df.any(axis='columns')])
        filtered_bs_df.to_csv(generate_path_name('bought_ad_slots.csv'))

        iv_sol = self.scheduling.getSolution(self.increased_viewers)
        m, n, p = iv_sol.shape
        iv_sol = iv_sol.reshape(m * n, p)
        comp_slot_label = ['comp_{0}_slot_{1}'.format(c, t) for c in self.Competitors for t in self.TimeSlots]
        iv_df = pd.DataFrame(data=iv_sol, index=comp_slot_label, columns=[f"Days_{i}" for i in self.Days])
        filtered_iv_df = (iv_df[iv_df.any(axis='columns')])
        filtered_iv_df.to_csv(generate_path_name('increase_viewers.csv'))

    @time_spent_decorator
    def run(self,
            set_soft_limit: bool = False,
            set_hard_limit: bool = False,
            out_subfolder: str = "",
            xp_output: bool = True
            ):
        xp.setOutputEnabled(xp_output)
        self.add_decision_variables()
        self.add_constraints()
        self.add_objective_function()
        self.solve(set_soft_limit=set_soft_limit, set_hard_limit=set_hard_limit)
        self.save_results(subfolder=out_subfolder)
