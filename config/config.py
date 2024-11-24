MAX_RUNTIME_MIN_PER_DAY = 17 * 60  # maximum movie runtime combined in one day
SLOT_DURATION = 30  # duration in minutes per a time slot
TOTAL_DAYS = 84  # total days for the problem
TOTAL_WEEKS = int(TOTAL_DAYS / 7)  # total weeks of the problem
TOTAL_SLOTS = int(MAX_RUNTIME_MIN_PER_DAY / SLOT_DURATION)  # total time slots in one day
DAYS_PER_SOLVE = 7  # a number of days used to solve in one iteration
MAX_CONVERSION_RATE = 0.3  # maximum conversion rate
COMPETITORS = ["c1", "c2", "c3"]  # a list of competitors
DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"  # a datetime format in string used in the dataframe index "2024-10-01 07:05:00"

# Base values used to calculate license fee for each movie
BASE_FEE = 10000
PROFIT_MARGIN = 0.2
BUDGET_FACTOR = 0.002
BOX_OFFICE_REVENUE_FACTOR = 0.001
DEMOGRAPHIC_LIST = ['children', 'adults', 'retirees']

TOTAL_VIEW_COUNT = 1000000  # maximum view count per a time slot
DAY_OFFSET = 1  # an offset used to pre-process data
MIN_ADS_PRICE_PER_VIEW = 0.75  # minimum price per view for selling an ad
LOWER_PRICE = 0.1  # a discount rate from the competitors' price

FIRST_WEEK = 40  # first week of the year corresponding to the data given
WEEK_CONSIDERED = 12  # a number of weeks needed to find the schedules
YEAR = 2024  # year of the data

MAX_HARD_LIMIT_RUNTIME = 30  # maximum runtime in seconds
MAX_SOFT_LIMIT_RUNTIME = 600  # maximum  runtime in seconds

P = 0.05  # percentage of movies used in the problem in each week
OUT_FOLDER = "out"
OUT_SUBFOLDER = "week_{0}"
