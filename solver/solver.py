import os
import sys

import numpy as np
import pandas as pd
import xpress as xp

from itertools import chain

from config.config import (AD_BEFORE_SHOWING_SLOT,
                           COMPETITORS,
                           DEMOGRAPHIC_LIST,
                           FIRST_WEEK,
                           MAX_CONVERSION_RATE,
                           MAX_HARD_LIMIT_RUNTIME,
                           MAX_SOFT_LIMIT_RUNTIME,
                           OUT_FOLDER,
                           OUT_SUBFOLDER,
                           SLOT_DURATION,
                           TOTAL_SLOTS,
                           TOTAL_VIEW_COUNT)
from utils.advert_conversion_rates import generate_conversion_rates
from utils.schedule_processing import (combine_schedule,
                                       dynamic_pricing,
                                       return_ads_30_mins)
from utils.utils import time_spent_decorator


class SchedulingSolver:
    """
    A Solver class is for handling solving the problem. It receives necessary parameters
    and used to pre-process into data needed for the solver.

    Methods
    ---------
    run(set_soft_limit: bool = False, set_hard_limit: bool = False, out_subfolder: str = "", xp_output: bool = True):
    ==> Set up the problem and run the solver.

    update_data(original_movie_df: pd.DataFrame = None,
                movie_df: pd.DataFrame = None,
                competitor_schedules: list[pd.DataFrame] = None,
                channel_a_30_schedule_df: pd.DataFrame = None):
    ==> Update the data.

    reset_problem():
    ==> Reset the problem.
    """
    def __init__(self,
                 original_movie_df: pd.DataFrame,
                 movie_df: pd.DataFrame,
                 competitor_schedules: list[pd.DataFrame],
                 channel_a_30_schedule_df: pd.DataFrame,
                 number_of_days: int,
                 ):
        """
        A constructor of a Solver class.
        <br><br>

        :param original_movie_df: a movie dataframe with all movies
        :param movie_df: the filtered movie dataframe. Contains movies picked from the
        original_movie_df by some conditions
        :param competitor_schedules: schedules of all competitors
        :param channel_a_30_schedule_df: our channel schedule, pre-processed into a
        time slot of 30 minutes
        :param number_of_days: how many days should the result schedule be
        """

        # If needed to activate the license file.
        # xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

        # These variables are decision variables. It will be explained in the
        # add_decision_variables() method.
        self.movie = None
        self.movie_time = None
        self.start_time = None
        self.end_time = None
        self.sold_ad_slots = None
        self.bought_ad_slots = None
        self.increased_viewers = None
        self.z = None

        # out_subfolder is the subfolder to store the output file
        self.out_subfolder = ""
        # MIP_GAP will be printed when the soft or hard limit is set.
        self.mip_gap = None

        # original_movie_df is the movie dataframe with all movies.
        self.original_movie_df = original_movie_df
        # movie_df is the filtered movie dataframe. It contains movies picked from the
        # original_movie_df by some conditions
        self.movie_df = movie_df.copy()

        # all_genres contains all unique genres of all movies.
        all_genres = list(set(chain.from_iterable(original_movie_df["genres"])))
        # all_conversion_rates contains all conversion rates for each movie in each ad slot.
        # The dimension is (n_competitors x n_movies x n_days x n_ad_slot_30_min)
        all_conversion_rates = []
        for s in competitor_schedules:
            conv_rates = generate_conversion_rates(s, movie_df, original_movie_df, all_genres, MAX_CONVERSION_RATE)
            all_conversion_rates.append(conv_rates)
        self.all_conversion_rates = np.array(all_conversion_rates)

        # === Parameters related to competitor_schedules and channel_a_30_schedule_df ===
        # comp_ads_slots contains two data. 
        # One is whether that time slot in a competitor has an ad or not (value as 1 or 0)
        # Another data is the price for each ad slot on each competitor. 
        # The dimension is n_comp x n_days x n_time_slots x (0|1, ad_price).
        comp_ads_slots = []  # np_array of dimension n_comp x n_days x n_time_slots x (0|1, ad_price)
        # comp_ads_viewership contains the number of viewership of each demographic in each ad slot of each competitor.
        # The dimension is n_comp x n_demo x n_days x n_time_slots.
        comp_ads_viewership = []
        for comp in competitor_schedules:
            ads, ad_viewership = return_ads_30_mins(comp, channel_a_30_schedule_df.index)
            comp_ads_slots.append(ads)
            comp_ads_viewership.append(ad_viewership)
        self.all_comp_ads_slots = np.array(comp_ads_slots)
        self.all_comp_ads_viewership = np.array(comp_ads_viewership)

        # week is which week the schedule belongs to
        self.week = FIRST_WEEK
        # competitor_schedules contains the schedules of all competitors
        self.competitor_schedules = competitor_schedules
        # ads_price_per_view is a price per view for each ad sold. It is calculated from the average price per 
        # view among all competitors.
        self.ads_price_per_view = dynamic_pricing(week=self.week, competitor_schedule_list=competitor_schedules)
        # based_view_count is the baseline view count in each time slot that takes a prime time factor into account.
        self.all_based_view_count = combine_schedule(channel_a_30_schedule_df)

        self.conversion_rates = None
        self.comp_ads_slots = None
        self.comp_ads_viewership = None
        self.based_view_count = None

        self.__set_offset_data(FIRST_WEEK)

        # number_of_movies is the total of number of movies used to solve.
        self.number_of_movies = len(movie_df)
        # number_of_competitors is the total number of competitors
        self.number_of_competitors = len(COMPETITORS)
        # number_of_time_slots is the number of total time slot. It is calculated by the total available
        # show time per day in minutes divided by the duration in each slot.
        self.number_of_time_slots = int((24 - 7) * 60 / SLOT_DURATION)
        # number_of_days is how many days the result schedule should have.
        self.number_of_days = number_of_days

        # These parameters are the range of movies, competitors, time slots, and days.
        self.Movies = range(self.number_of_movies)
        self.Competitors = range(self.number_of_competitors)
        self.TimeSlots = range(self.number_of_time_slots)
        self.Days = range(self.number_of_days)

        # M is the maximum viewership in one time slot.
        self.M = 1

        # scheduling is an initiation of the problem.
        self.scheduling = xp.problem('scheduling')

    def __set_offset_data(self, week: int):
        """
        Set the offset of data to determine which rows of the data to be used

        :param week: shows which week the solver is working on
        """
        week_offset = week - FIRST_WEEK
        days_offset = week_offset * 7
        time_slot_offset = days_offset * TOTAL_SLOTS

        self.conversion_rates = self.all_conversion_rates[:, :, days_offset:, :]
        self.comp_ads_slots = self.all_comp_ads_slots[:, days_offset:, :, :]
        self.comp_ads_viewership = self.all_comp_ads_viewership[:, :, days_offset:, :]
        self.based_view_count = self.all_based_view_count.iloc[time_slot_offset:]

    def __generate_out_filename(self, filename: str) -> str:
        """
        Generate the full path for the output file and ensure the directory exists.
        """

        folder_path = os.path.join(OUT_FOLDER, self.out_subfolder)

        # Ensure the directory exists
        os.makedirs(folder_path, exist_ok=True)

        # Return the full file path
        return os.path.join(folder_path, filename)

    @time_spent_decorator
    def __add_decision_variables(self):
        """
        Create decision variables.
        """

        # movie shows whether the movie i is shown on day d.
        self.movie = self.scheduling.addVariables(self.number_of_movies, self.number_of_days, name='m',
                                                  vartype=xp.binary)
        # movie_time shows which movie is being shown on the time slot t of day d. If the movie i is being shown,
        # the value is 1, otherwise, 0.
        self.movie_time = self.scheduling.addVariables(self.number_of_movies, self.number_of_time_slots,
                                                       self.number_of_days, name="mt", vartype=xp.binary)
        # start_time is which time slot the movie i starts showing on day d.
        self.start_time = self.scheduling.addVariables(self.number_of_movies, self.number_of_days, name='s',
                                                       vartype=xp.integer)
        # end_time is which time slot the movie i ends on day d.
        self.end_time = self.scheduling.addVariables(self.number_of_movies, self.number_of_days, name='e',
                                                     vartype=xp.integer)
        # sold_ad_slots tells whether the ad in time slot t of the shown movie i is sold to the competitor c on day d.
        # If the ad is sold, the value is 1, otherwise, 0.
        self.sold_ad_slots = self.scheduling.addVariables(self.number_of_movies, self.number_of_time_slots,
                                                          self.number_of_competitors, self.number_of_days,
                                                          name="sa", vartype=xp.binary)
        # bought_ad_slots tells whether movie i is being advertised on a time slot t of competitor c on day d.
        # If it is being advertised, the value is 1, otherwise, 0.
        self.bought_ad_slots = self.scheduling.addVariables(self.number_of_movies, self.number_of_time_slots,
                                                            self.number_of_competitors, self.number_of_days,
                                                            name="ba", vartype=xp.binary)
        # increased_viewers is the converted view count from advertising our own movies on the competitors' channels.
        self.increased_viewers = self.scheduling.addVariables(self.number_of_movies, self.number_of_competitors,
                                                              name="iv", vartype=xp.continuous)
        # z is for the linearization of the calculation of sold_ad_slots * increased_viewers to avoid the
        # quadratic objective function.
        self.z = self.scheduling.addVariables(self.number_of_movies, self.number_of_time_slots,
                                              self.number_of_competitors,
                                              self.number_of_days,
                                              name="z", vartype=xp.continuous)

    @time_spent_decorator
    def __add_objective_function(self):
        """
        Define an objective function.
        """

        # The objective is to maximize the profit gained from showing movies, buying ads, and selling ads.
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
                        self.based_view_count[f"{demo}_prime_time_view_count"].iloc[t + d * TOTAL_SLOTS] *
                        self.movie_df[f"{demo}_scaled_popularity"].iloc[i] for demo in DEMOGRAPHIC_LIST
                    )
                    * self.sold_ad_slots[i, t, c, d]  # Sold ad slots
                    * TOTAL_VIEW_COUNT * self.ads_price_per_view
                )
                + (
                    # Profit from ad slots sold from increased_viewers
                    # This is a profit from gained viewers by buying ads from the competitors. Variable z is just the
                    # linearization of the quadratic sold_ad_slots * increased_viewers
                    # The increased viewers is calculated from the conversion rates of each competitor view count
                    # on that ad slot.
                    self.z[i, t, c, d]  # Auxiliary variable for increased viewers * sold ad slots.
                    * TOTAL_VIEW_COUNT * self.ads_price_per_view
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
    def __add_constraints(self):
        """
        Add constraints to the problem.
        """

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

        # ====== Constraints for linearizing increased_viewership
        # z cannot go beyond increased_viewers. This is to cap z to when the sold_ad_slot is 1, then the value is
        # actually sold_ad_slots * increased_viewers
        self.scheduling.addConstraint(
            self.z[i, t, c, d] <= self.increased_viewers[i, c] for i in self.Movies for t in self.TimeSlots
            for c in self.Competitors for d in self.Days)
        # If the ad is not sold in that time slot t, then z is 0. This means the gained view count is not taken into
        # account when calculating the profit from selling ads since the ad itself is not sold.
        self.scheduling.addConstraint(
            self.z[i, t, c, d] <= self.M * self.sold_ad_slots[i, t, c, d] for i in self.Movies for t in self.TimeSlots
            for c in self.Competitors for d
            in self.Days)
        # Set a lower bound for z to be the value of increased_viewers if sold_ad_slots is 1. Combining with the first
        # constraint of z above, z will be guaranteed to be the same as increased_viewers whenever sold_ad_slots is 1,
        # or to say whenever the ad is sold in that time slot t.
        self.scheduling.addConstraint(
            self.z[i, t, c, d] >= self.increased_viewers[i, c] - self.M * (1 - self.sold_ad_slots[i, t, c, d])
            for i in self.Movies for t in self.TimeSlots for c in self.Competitors for d in self.Days)
        # Cap the lower bound of z to not be negative.
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
                                      self.bought_ad_slots[i, t, c, d] * (t + (d * self.number_of_time_slots) +
                                                                          AD_BEFORE_SHOWING_SLOT)
                                      for i in self.Movies for c in self.Competitors for d in self.Days for t in
                                      self.TimeSlots)
        # Calculate the gained viewers from buying an ad.
        # The calculation is simply competitor_expected_viewership x conversion_rate.
        self.scheduling.addConstraint(self.increased_viewers[i, c] ==
                                      xp.Sum(self.bought_ad_slots[i, t, c, d] *
                                             xp.Sum(self.comp_ads_viewership[c, demo, d, t]
                                                    for demo, _ in enumerate(DEMOGRAPHIC_LIST)) *
                                             self.conversion_rates[c, i, d, t]
                                             for t in self.TimeSlots for d in self.Days)
                                      for i in self.Movies for c in self.Competitors)
        # Each time slot can only be bought once
        self.scheduling.addConstraint(xp.Sum(self.bought_ad_slots[i, t, c, d] for i in self.Movies) <= 1
                                      for t in self.TimeSlots for c in self.Competitors for d in self.Days)
        # Each movie can be advertised only once per each competitor
        self.scheduling.addConstraint(
            xp.Sum(self.bought_ad_slots[i, t, c, d] for t in self.TimeSlots for d in self.Days) <= 1
            for i in self.Movies for c in self.Competitors)

    @time_spent_decorator
    def __solve(self, set_soft_limit: bool = False, set_hard_limit: bool = False):
        """
        This function sets the config for the solver and solves the problem.
        <br><br>

        :param set_soft_limit: is used to check whether the soft limit of the solver should be set
        :param set_hard_limit: is used to check whether the hard limit of the solver should be set
        """
        if set_soft_limit:
            self.scheduling.setControl('soltimelimit', MAX_SOFT_LIMIT_RUNTIME)
        if set_hard_limit:
            self.scheduling.setControl('timelimit', MAX_HARD_LIMIT_RUNTIME)
        self.scheduling.setControl("heurfreq", 10)

        original_stdout = sys.stdout
        with open(self.__generate_out_filename('console_log.txt'), 'w') as f:
            sys.stdout = f
            self.scheduling.solve()

            if set_soft_limit or set_hard_limit:
                # If the limit is set, print out the MIP_GAP
                obj_val = self.scheduling.attributes.objval
                best_bound = self.scheduling.getAttrib('bestbound')
                self.mip_gap = 100 * ((obj_val - best_bound) / obj_val)

            print(f"Objective Value: {self.scheduling.attributes.objval}")

        sys.stdout = original_stdout

    @time_spent_decorator
    def __save_results(self):
        """
        Save results from the solver to files.
        """

        # Retrieve solutions
        movie_solution = self.scheduling.getSolution(self.movie)
        movie_time_solution = self.scheduling.getSolution(self.movie_time)
        sold_ad_slots_solution = self.scheduling.getSolution(self.sold_ad_slots)  # ndarray
        z_solution = self.scheduling.getSolution(self.z)  # ndarray
        bought_ad_slots_solution = self.scheduling.getSolution(self.bought_ad_slots)  # ndarray
        increased_viewership_solution = self.scheduling.getSolution(self.increased_viewers)  # ndarray

        # Calculate total licensing fee
        total_licensing_fee = sum(
            movie_solution[i, d] * self.movie_df['license_fee'].iloc[i]
            for i in self.Movies for d in self.Days
        )
        
        # Initialize variables to store aggregated results
        total_revenue_no_ads = 0
        total_revenue_with_ads = 0
        movie_revenues = []

        # Calculate viewership and revenues
        max_total_views = 0  # max views including converted viewers from bought ads
        max_base_views = 0  # max views excluding converted viewers from bought ads
        min_total_views = 0  # min views including converted viewers from bought ads
        min_base_views = 0  # min views excluding converted viewers from bought ads
        sum_total_views = 0  # sum of views over time slots including converted viewers from bought ads
        sum_base_views = 0  # sum of views over time slots excluding converted viewers from bought ads
        total_slots_shown = 0  # total slots in which the movies are shown

        for i in self.Movies:
            movie_revenue_no_ads = 0  # Revenue without ads for the movie
            movie_revenue_with_ads = 0  # Revenue with ads for the movie

            for d in self.Days:
                if movie_solution[i, d] == 0:
                    continue

                for t in self.TimeSlots:
                    if movie_time_solution[i, t, d] == 0:
                        continue

                    total_slots_shown += 1

                    # Calculate base viewership (independent of ads bought)
                    base_viewership = sum(
                        self.based_view_count[f"{demo}_prime_time_view_count"].iloc[t + d * TOTAL_SLOTS] *
                        self.movie_df[f"{demo}_scaled_popularity"].iloc[i]
                        for demo in DEMOGRAPHIC_LIST
                    ) * TOTAL_VIEW_COUNT

                    # Add increased viewership from buying ads
                    increased_viewership = np.sum(increased_viewership_solution, axis=1)[i] * TOTAL_VIEW_COUNT
                    total_viewership = base_viewership + increased_viewership

                    # Calculate the revenue without taking into account the gained viewers from ads
                    base_revenue = (sum(sold_ad_slots_solution[i, t, c, d] for c in self.Competitors) * base_viewership *
                                    self.ads_price_per_view)
                    # Calculate the revenue from the gained viewers from ads
                    increased_revenue = (sum(z_solution[i, t, c, d] for c in self.Competitors) *
                                         TOTAL_VIEW_COUNT * self.ads_price_per_view)

                    # Increment movie revenue
                    movie_revenue_no_ads += base_revenue
                    movie_revenue_with_ads += base_revenue + increased_revenue

                    # Update the sum values
                    sum_base_views += base_viewership
                    sum_total_views += total_viewership

                    # Update max and min values
                    if max_total_views < total_viewership:
                        max_total_views = total_viewership
                    if min_total_views > total_viewership:
                        min_total_views = total_viewership

                    if max_base_views < base_viewership:
                        max_base_views = base_viewership
                    if min_base_views > base_viewership:
                        min_base_views = base_viewership

                # Store total revenue
                total_revenue_no_ads += movie_revenue_no_ads
                total_revenue_with_ads += movie_revenue_with_ads

                # Store the movie revenue
                movie_revenues.append({
                    "movie_title": self.movie_df['title'].iloc[i],
                    "revenue_no_ads": movie_revenue_no_ads,
                    "revenue_with_ads": movie_revenue_with_ads
                })

                # The movie is shown only on a single day and will not be duplicated so
                # we can break out of the loop and continue with the next movie.
                break

        # Calculate average viewership (with and without ads)
        average_base_views = sum_base_views / total_slots_shown
        average_total_views = sum_total_views / total_slots_shown

        total_ads_slots_sold = np.sum(sold_ad_slots_solution)

        movie_revenue_df = pd.DataFrame(movie_revenues)
        movie_revenue_df.to_csv(self.__generate_out_filename('movie_revenues.csv'), index=False)

        # Calculate overall statistics
        number_of_movies = len(movie_revenue_df[movie_revenue_df['revenue_no_ads'] > 0])
        average_revenue_per_movie_no_ads = total_revenue_no_ads / number_of_movies if number_of_movies > 0 else 0
        average_revenue_per_movie_with_ads = total_revenue_with_ads / number_of_movies if number_of_movies > 0 else 0

        # Initialize variables for aggregation
        total_bought_cost = 0
        total_ads_bought = 0  # Total number of ads bought
        movie_costs = {i: 0 for i in self.Movies}  # Track cost per movie

        # Calculate the cost for buying ads
        for c in self.Competitors:
            for d in self.Days:
                for t in self.TimeSlots:
                    # Get the cost of the ad slot for the competitor
                    ad_slot_cost = self.comp_ads_slots[c, d, t, 1]

                    # Distribute cost across movies
                    for i in self.Movies:
                        if bought_ad_slots_solution[i, t, c, d] == 0:
                            continue

                        total_bought_cost += ad_slot_cost
                        total_ads_bought += 1
                        movie_costs[i] += ad_slot_cost

        # Calculate number of movies with non-zero costs
        non_zero_movies = [cost for cost in movie_costs.values() if cost > 0]
        num_movie_ad_bought = len(non_zero_movies)

        # Calculate average cost per movie
        average_cost_per_movie = total_bought_cost / num_movie_ad_bought if num_movie_ad_bought > 0 else 0

        # Calculate average cost per ad bought
        average_ad_cost = total_bought_cost / total_ads_bought if total_ads_bought > 0 else 0
        
        # Add these metrics to general_stats DataFrame
        general_stats = pd.DataFrame({
            "week": [self.week],
            "Total Gross": [self.scheduling.attributes.objval],
            "MIP Gaps": [self.mip_gap],
            "ad_price_per_view": [self.ads_price_per_view],
            "total_licensing_fee": [total_licensing_fee],
            "total_revenue_no_ads": [total_revenue_no_ads],  # Revenue excluding ad-based revenue
            "total_revenue_with_ads": [total_revenue_with_ads],  # Revenue including ads
            "total_ads_sold": total_ads_slots_sold,
            "average_revenue_with_ads": total_revenue_with_ads / total_ads_slots_sold,
            "viewership_without_ads": [sum_base_views],
            "viewership_with_ads": [sum_total_views],
            "viewership_gains_from_ads": [np.mean(increased_viewership_solution) * TOTAL_VIEW_COUNT],
            "average_viewership_per_movie_without_ads": [average_base_views],
            "average_viewership_per_movie_with_ads": [average_total_views],
            "maximum_viewership_per_movie_without_ads": [max_base_views],
            "maximum_viewership_per_movie_with_ads": [max_total_views],
            "minimum_viewership_per_movie_without_ads": [min_base_views],
            "minimum_viewership_per_movie_with_ads": [min_total_views],
            "number_of_movies": [number_of_movies],
            # Average revenue per movie without ads
            "average_revenue_per_movie_no_ads": [average_revenue_per_movie_no_ads],
            # Average revenue per movie with ads
            "average_revenue_per_movie_with_ads": [average_revenue_per_movie_with_ads],
            "total_bought_cost": [total_bought_cost],  # Total cost of ads bought
            "num_movie_ad_bought": [num_movie_ad_bought],  # Number of movies with at least one ad bought
            "average_cost_per_movie": [average_cost_per_movie],  # Average cost per movie with ads
            "number_of_ads_bought": [total_ads_bought],  # Total number of ads bought
            "average_ad_cost": [average_ad_cost]  # Average cost per ad bought
        })

        # Save results to CSV files
        general_stats.to_csv(self.__generate_out_filename('general_stats.csv'), index=False)
        movie_revenue_df.to_csv(self.__generate_out_filename('movie_revenues.csv'), index=False)

        days_labels = ['day_{0}'.format(d) for d in self.Days]
        # Save results for movie
        mdf = pd.DataFrame(data=movie_solution, index=self.movie_df['title'],
                           columns=days_labels)
        filtered_mdf = mdf[mdf.any(axis='columns')]
        filtered_mdf.to_csv(self.__generate_out_filename('movie.csv'))

        # Save results for movie_time
        m, n, r = movie_time_solution.shape
        mt_sol = movie_time_solution.reshape(m, n * r)
        slot_day_labels = ['slot_{0}_day_{1}'.format(t, d) for t in self.TimeSlots for d in self.Days]
        mt_df = pd.DataFrame(data=mt_sol, index=self.movie_df['title'], columns=slot_day_labels)
        filtered_mt_df = mt_df[mt_df.any(axis='columns')]
        filtered_mt_df.to_csv(self.__generate_out_filename('movie_time.csv'))

        # Save results for start_time
        st_df = pd.DataFrame(data=self.scheduling.getSolution(self.start_time), index=self.movie_df['title'],
                             columns=days_labels)
        filtered_st_df = st_df[mdf.any(axis='columns')]
        filtered_st_df.to_csv(self.__generate_out_filename("start_time.csv"))

        # Save results for end_time
        et_df = pd.DataFrame(data=self.scheduling.getSolution(self.end_time), index=self.movie_df['title'],
                             columns=days_labels)
        filtered_et_df = et_df[mdf.any(axis='columns')]
        filtered_et_df.to_csv(self.__generate_out_filename("end_time.csv"))

        # Save results for sold_ad_slots
        m, n, p, q = sold_ad_slots_solution.shape
        as_sol = sold_ad_slots_solution.reshape(m, n * q * p)
        slot_comp_day_label = ['slot_{0}_comp_{1}_day_{2}'.format(t, c, d)
                               for t in self.TimeSlots for c in self.Competitors for d in self.Days]
        as_df = pd.DataFrame(data=as_sol, index=self.movie_df['title'], columns=slot_comp_day_label)
        filtered_as_df = (as_df[as_df.any(axis='columns')])
        filtered_as_df.to_csv(self.__generate_out_filename('sold_ad_slots.csv'))

        # Save results for bought_ad_slots
        m, n, p, q = bought_ad_slots_solution.shape
        bs_sol = bought_ad_slots_solution.reshape(m, n * q * p)
        slot_comp_day_label = ['slot_{0}_comp_{1}_day_{2}'.format(t, c, d)
                               for t in self.TimeSlots for c in self.Competitors for d in self.Days]
        bs_df = pd.DataFrame(data=bs_sol, index=self.movie_df['title'], columns=slot_comp_day_label)
        filtered_bs_df = (bs_df[bs_df.any(axis='columns')])
        filtered_bs_df.to_csv(self.__generate_out_filename('bought_ad_slots.csv'))

        # Save results for increased_viewers
        iv_df = pd.DataFrame(data=increased_viewership_solution, index=self.movie_df['title'], columns=[f"Competitor_{c}" for c in self.Competitors])
        filtered_iv_df = (iv_df[iv_df.any(axis='columns')])
        filtered_iv_df.to_csv(self.__generate_out_filename('increase_viewers.csv'))

    def __update_week(self, week: int):
        """
        Update parameters related to week once the week gets updated.
        """
        self.week = week
        self.out_subfolder = OUT_SUBFOLDER.format(self.week)
        self.ads_price_per_view = dynamic_pricing(week=self.week, competitor_schedule_list=self.competitor_schedules)

    @time_spent_decorator
    def run(self,
            set_soft_limit: bool = False,
            set_hard_limit: bool = False,
            out_subfolder: str = "",
            xp_output: bool = True,
            week: int = FIRST_WEEK
            ):
        """
        Set up the problem and run the solver.
        <br><br>

        :param set_soft_limit: is used to check whether the soft limit of the solver should be set
        :param set_hard_limit: is used to check whether the hard limit of the solver should be set
        :param out_subfolder: is a subfolder for output files
        :param xp_output: is whether to enable the output of the solver
        :param week: is which week the problem is working on
        means the first week
        """

        self.__update_week(week)
        self.__set_offset_data(week)

        if out_subfolder:
            # set the subfolder of the output files
            self.out_subfolder = out_subfolder

        xp.setOutputEnabled(xp_output)

        self.__add_decision_variables()
        self.__add_constraints()
        self.__add_objective_function()
        self.__solve(set_soft_limit=set_soft_limit, set_hard_limit=set_hard_limit)
        self.__save_results()

    def update_data(self,
                    original_movie_df: pd.DataFrame = None,
                    movie_df: pd.DataFrame = None,
                    competitor_schedules: list[pd.DataFrame] = None,
                    channel_a_30_schedule_df: pd.DataFrame = None,
                    ):
        """
        Update the values. Each value is optional.
        <br><br>

        :param original_movie_df: a movie dataframe with all movies
        :param movie_df: the filtered movie dataframe. Contains movies picked from the
        original_movie_df by some conditions
        :param competitor_schedules: schedules of all competitors
        :param channel_a_30_schedule_df: our channel schedule, pre-processed into a
        time slot of 30 minutes
        """
        if movie_df is not None:
            self.movie_df = movie_df.copy()
            self.number_of_movies = len(movie_df)
            self.Movies = range(self.number_of_movies)
        if original_movie_df is not None:
            self.original_movie_df = original_movie_df.copy()
        if competitor_schedules or channel_a_30_schedule_df is not None:
            if competitor_schedules:
                self.competitor_schedules = competitor_schedules.copy()
                self.ads_price_per_view = dynamic_pricing(week=self.week, competitor_schedule_list=competitor_schedules)

                all_genres = list(set(chain.from_iterable(original_movie_df["genres"])))
                all_conversion_rates = []
                for s in competitor_schedules:
                    conv_rates = generate_conversion_rates(s, movie_df, original_movie_df, all_genres,
                                                           MAX_CONVERSION_RATE)
                    all_conversion_rates.append(conv_rates)
                self.all_conversion_rates = np.array(all_conversion_rates)

            if channel_a_30_schedule_df is not None:
                self.all_based_view_count = combine_schedule(channel_a_30_schedule_df)

            comp_ads_slots = []  # np_array of dimension n_comp x n_days x n_time_slots x (0|1, ad_price)
            comp_ads_viewership = []  # np_array of dimension n_comp x n_demo x n_days x n_time_slots
            for comp in competitor_schedules:
                ads, ad_viewership = return_ads_30_mins(comp, channel_a_30_schedule_df.index)
                comp_ads_slots.append(ads)
                comp_ads_viewership.append(ad_viewership)
            self.all_comp_ads_slots = np.array(comp_ads_slots)
            self.all_comp_ads_viewership = np.array(comp_ads_viewership)

        self.__set_offset_data(self.week)

    def reset_problem(self):
        """
        Reset the problem.
        """
        del self.scheduling
        self.scheduling = xp.problem('scheduling')
