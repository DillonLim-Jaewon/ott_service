# System
import os
import warnings
from datetime import timedelta

# Data Analysis
import duckdb
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import missingno as msno
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Personal package
from utils import db_utils, eda_utils


""" SETTINGS FOR EDA """
load_dotenv()
# for korean in plots
koreanize_matplotlib.koreanize()
# for plotting style
plt.style.use("seaborn-v0_8-whitegrid")
# to show all columns
pd.set_option("display.max_columns", None)
# to show all rows
pd.set_option("display.max_rows", 1000)
# display the float format rounded to the seconds decimal place
pd.options.display.float_format = "{:,.2f}".format
# ignore all warnings
warnings.filterwarnings("ignore")
# ignore specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# ignore temporary warnings
with warnings.catch_warnings():
    # Code that may produce warnings
    warnings.filterwarnings("ignore")


# Get the names of all tables
def get_table_names(data_path):
    datafiles = os.listdir(data_path)
    table_names = [
        name.replace(".csv", "").replace(".", "_")
        for name in datafiles
        if name.endswith(".csv")
    ]
    return table_names


def query_data(table_name, db_path):
    con_duckdb = duckdb.connect(db_path)
    df = con_duckdb.execute(f"SELECT * FROM {table_name}").df()
    return df


def drop_unnecessary_rows(df):
    """Drop rows that do not meet the specified conditions."""
    # Save only South Korea
    korean_countries = df["country"] == "South Korea"
    # Save only Korean
    korean_languages = df["language"] == "Korean"
    # Save only Dates from 2022-01-01 and onwards
    from_2022_01_onwards = df["client_event_time"] >= pd.Timestamp("2022-01-01")

    # Drop rows that do not meet the conditions
    df.drop(
        index=df.loc[
            ~korean_countries | ~korean_languages | ~from_2022_01_onwards
        ].index,
        errors="ignore",
        inplace=True,
    )

    return df


def drop_unnecessary_columns(df):
    """Remove columns that are not needed."""
    columns_to_remove = [
        "device_carrier",
        "os_version",
        "platform",
        "genre_name",
        "country",
        "language",
        "event_type",
        "city",
    ]

    df.drop(columns=columns_to_remove, errors="ignore", inplace=True)

    return df


def remove_unnecessary_rows_and_columns_and_save_csv(data_path, to_data_path, db_path):
    table_names = get_table_names(data_path)
    for name in table_names:
        df = query_data(name, db_path)
        drop_unnecessary_rows(df)
        print(f"Table: {name} unnecessary rows dropped.")
        drop_unnecessary_columns(df)
        print(f"Table: {name} unnecessary columns dropped.")
        file = f"{name}.csv"
        new_data_path = os.path.join(to_data_path, file)
        df.to_csv(new_data_path, header=True, index=False)
        print(f"Table: {name} added to {to_data_path} directory successfully.")


# Get medatadata of all tables and concat into a dataframe
def get_metadata_of_all_tables_and_save_csv(data_path, to_data_path, db_path):
    # Parameters for get_table_info()
    table_names = get_table_names(data_path)
    engine = duckdb.connect(db_path)

    # Apply to all tables and merge the results
    df_all_stats = pd.concat(
        [eda_utils.get_table_info(table, engine) for table in table_names]
    ).reset_index(drop=True)

    # Close the connection when done
    engine.close()

    # Save df_all_stats metadata to_data_path
    df_all_stats.to_csv(to_data_path, header=True, index=False)


# Read Metadata and color code it
def read_metadata_color_coded(metadata_path):
    df = pd.read_csv(metadata_path)
    df = eda_utils.color_table_by_column_unique_values(df, "dataset_names", "dark")
    return df


# make files dictionary
def make_key_table_value_path_dictionary(data_path):
    dict_files = dict()
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            name = file.replace(".csv", "").replace(".", "_")
            file_path = os.path.join(data_path, file)
            dict_files[name] = file_path
    return dict_files


def get_users_action_journey(dict_files, db_path):
    # Initialize a list to store the filtered DataFrames
    df_list = []

    # open duckdb connection
    con_duckdb = duckdb.connect(db_path)
    # Loop through the CSV files and process them
    for df_name, file_path in dict_files.items():
        if os.path.exists(file_path):  # Check if the file exists
            df = query_data(df_name, db_path)
            # df = pd.read_csv(file_path)

            # Drop rows with missing user_id and select relevant columns
            df = df.dropna(subset=["user_id"])
            df_filtered = df[["user_id", "client_event_time"]].copy()
            # df_filtered = df[['user_id', 'client_event_time', 'plan.type', 'type', 'device_type', 'city']].copy()
            # df_filtered = df[['user_id', 'client_event_time']].copy()

            # Add an 'action' column with the name of the action
            df_filtered["action"] = df_name

            # Append the processed DataFrame to the list
            df_list.append(df_filtered)

    # Concatenate all DataFrames into one
    df_all = pd.concat(df_list, ignore_index=True)

    # Close connection to duckdb
    con_duckdb.close()
    return df_all


def plot_categorical_barplots(df, columns):
    df_regions = df[columns]

    # Visualize Categorical Data
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 15))

    axes = axes.flatten()

    for i, cat in enumerate(df_regions.columns):
        value_counts = df_regions[cat].value_counts().head(20)
        axes[i].bar(value_counts.index, value_counts.values)
        axes[i].set_title(f"{cat}")
        axes[i].tick_params(axis="x", labelrotation=75)

    for i in range(len(df_regions.columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def read_query_from_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def get_time_frame_windows(df, date_column, window_name, date_interval):
    # Convert the date_column to datetime format if it isn't already
    df[date_column] = pd.to_datetime(df[date_column])
    start_date = pd.to_datetime("2022-01-03")
    df = df.assign(
        **{
            window_name: lambda x: (
                start_date
                + pd.to_timedelta(
                    ((x[date_column] - start_date).dt.days // date_interval)
                    * date_interval,
                    unit="D",
                )
            )
        }
    )
    return df


def get_time_frame_windows_month():
    # # Calculate the date ranges for Months
    # df_retention_chart["start_date"] = pd.to_datetime(
    #     df_retention_chart["start_timestamp"].dt.strftime("%Y-%m")
    # )
    # df_retention_chart["complete_episode_date"] = pd.to_datetime(
    #     df_retention_chart["complete_episode_timestamp"].dt.strftime("%Y-%m")
    # )

    # # Truncate duplicates because only one completion per timeframe is enough (month)
    # user_trans_trunc = (
    #     df_retention_chart[["user_id", "start_date", "complete_episode_date"]]
    #     .drop_duplicates()
    #     .reset_index()
    #     .drop(["index"], axis=1)
    # )

    # # Calculate time difference between complete_episode_month and start_date
    # user_trans_trunc["date_diff"] = user_trans_trunc[
    #     "complete_episode_date"
    # ].dt.to_period("M").astype(int) - user_trans_trunc["start_date"].dt.to_period(
    #     "M"
    # ).astype(int)
    return


""" DATA ENGINEERING: make interim data """

# Import raw data to duckdb
data_path = "../data/raw"
db_path = "../databases/raw_db.duckdb"
db_utils.import_data_directory_to_duckdb(data_path, db_path)

# Read from raw duckdb, transform data and save data to interim
data_path = "../data/raw"
db_path = "../databases/raw_db.duckdb"
to_data_path = "../data/interim"
remove_unnecessary_rows_and_columns_and_save_csv(data_path, to_data_path, db_path)

# Import interim data to duckdb
data_path = "../data/interim"
db_path = "../databases/interim_db.duckdb"
db_utils.import_data_directory_to_duckdb(data_path, db_path)

# Import interim data to postgres
data_path = "../data/interim"
db_params = {
    "host": "127.0.0.1",
    "database": "ott_db",
    "user": "postgres",
    "password": os.environ.get("PASSWORD"),
    "port": "5432",
}
engine = create_engine(
    f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}/{db_params["database"]}'
)
db_utils.import_data_directory_to_postgres(data_path, engine)

""" DATA ENGINEERING: optimize querying through metadata """
# Dataframe of raw Metadata and save it as a csv
data_path = "../data/raw"
to_data_path = "../data/metadata/raw_metadata.csv"
db_path = "../databases/raw_db.duckdb"
df_metadata = get_metadata_of_all_tables_and_save_csv(data_path, to_data_path, db_path)

# Dataframe of interim Metadata and save it as a csv
data_path = "../data/interim"
to_data_path = "../data/metadata/interim_metadata.csv"
db_path = "../databases/interim_db.duckdb"
df_metadata = get_metadata_of_all_tables_and_save_csv(data_path, to_data_path, db_path)

""" EDA: AARRRR """

""" Part 1: Before removing rows and columns """
metadata_path = "../data/metadata/raw_metadata.csv"
read_metadata_color_coded(metadata_path)

# Acquisition (check for demographics / personal settings)
db_path = "../databases/raw_db.duckdb"
# Enter main page
df_enter_main_page = query_data("enter_main_page", db_path)
df_enter_main_page.info()
df_enter_main_page.loc[
    ~(df_enter_main_page["country"] == "South Korea"), "city"
].value_counts()
df_enter_main_page.loc[
    (df_enter_main_page["country"] == "South Korea"), "city"
].value_counts()
df_enter_main_page["event_type"].value_counts()
df_enter_main_page["country"].value_counts()
df_enter_main_page["country"].value_counts(normalize=True)
df_enter_main_page["language"].value_counts()
df_enter_main_page["language"].value_counts(normalize=True)

plot_categorical_barplots(
    df_enter_main_page,
    [col for col in df_enter_main_page.columns if col != "client_event_time"],
)
# User distribution
plot_categorical_barplots(df_enter_main_page, ["user_id"])
# User region
plot_categorical_barplots(df_enter_main_page, ["country", "city", "language"])
# User tech environment
plot_categorical_barplots(
    df_enter_main_page, ["device_family", "device_type", "os_name", "os_version"]
)

# Complete signup
df_complete_signup = query_data("complete_signup", db_path)
df_complete_signup.info()
plot_categorical_barplots(
    df_complete_signup,
    [col for col in df_complete_signup.columns if col != "client_event_time"],
)
# User distribution
plot_categorical_barplots(df_complete_signup, ["user_id"])
# User region
plot_categorical_barplots(df_complete_signup, ["country", "city", "language"])
# User region
plot_categorical_barplots(
    df_complete_signup,
    ["device_family", "device_type", "os_name", "os_version", "type"],
)

""" Part 2: with interim data """

"""
ACQUISITION
"""
metadata_path = "../data/metadata/interim_metadata.csv"
read_metadata_color_coded(metadata_path)
# Set db path to the interim duckdb
db_path = "../databases/interim_db.duckdb"
# Query enter_main_page
df_enter_main_page = query_data("enter_main_page", db_path)
# Check the distribution of the columns
df_enter_main_page.columns
plot_categorical_barplots(df_enter_main_page, ["device_family", "user_id"])
# Check for the null values
msno.matrix(df_enter_main_page)

# Make a groupby to count the main page traffic over time
date_counts = (
    df_enter_main_page.groupby(df_enter_main_page["client_event_time"].dt.date)
    .agg({"user_id": "size"})
    .reset_index(names=["client_event_time", "count"])
)
# Plot the Main Page traffic over time
plt.figure(figsize=(10, 6))
plt.plot(
    date_counts["client_event_time"],
    date_counts["user_id"],
    marker="o",
    color="b",
    markersize=3,
)
plt.title("Main Page traffic over time")
plt.xlabel("Date")
plt.ylabel("User (null & non-null) Count")
plt.grid(True)
plt.show()


""" 
EDA: User Journey 
"""

data_path = "../data/interim"
db_path = "../databases/interim_db.duckdb"
df_dict = make_key_table_value_path_dictionary(data_path)

df_all = get_users_action_journey(df_dict, db_path)
df_all.info()
df_all





""" 
CHECK USER JOURNEY 
"""

# Filter and sort by user_id and client_event_time
user_journey = df_all.loc[
    df_all["user_id"] == "f833cca4c382ac8c502c6f99bc432725"
].sort_values(by="client_event_time")

user_journey

db_path = "../databases/interim_db.duckdb"
df_complete_episode = query_data("complete_episode", db_path)
df_complete_episode.groupby(["user_id", "client_event_time"]).agg({"city": "sum"})

# Convert client_event_time to date only
df_complete_episode["client_event_date"] = df_complete_episode[
    "client_event_time"
].dt.date

# Group by user_id and the new client_event_date column, then aggregate on city
result = (
    df_complete_episode.groupby(["user_id", "client_event_date"])
    .agg({"city": "size"})
    .reset_index()
    .rename({"city": "count"}, axis=1)
)
result.loc[result["count"] >= 1]["user_id"].value_counts()
result["user_id"].value_counts()


""" 
RETENTION CHART BY 7 DAYS TIME FRAME 
"""

# SQL params
db_params = {
    "host": "127.0.0.1",
    "database": "ott_db",
    "user": "postgres",
    "password": os.environ.get("PASSWORD"),
    "port": "5432",
}
engine = create_engine(
    f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}/{db_params["database"]}'
)

# Query
query_retention_chart = read_query_from_file(
    "../databases/sql_queries/retention_chart.sql"
)

# Query Dataframe
df_retention_chart = pd.read_sql_query(query_retention_chart, con=engine)

# Change Timestamps to datetime
df_retention_chart["start_timestamp"] = pd.to_datetime(
    df_retention_chart["start_timestamp"]
)
df_retention_chart["complete_episode_timestamp"] = pd.to_datetime(
    df_retention_chart["complete_episode_timestamp"]
)

df_retention_chart.loc[
    df_retention_chart["start_timestamp"]
    > df_retention_chart["complete_episode_timestamp"]
]


df_retention_chart = get_time_frame_windows(
    df_retention_chart, "start_timestamp", "cohort_date", 7
)
df_retention_chart = get_time_frame_windows(
    df_retention_chart, "complete_episode_timestamp", "complete_episode_date", 7
)

# Calculate the Cohort Size
cohort_size = (
    df_retention_chart
    .groupby(["cohort_date"])["user_id"]
    .nunique()
    .reset_index()
)
cohort_size.rename(columns={"user_id": "cohort_size"}, inplace=True)
cohort_size


# Truncate duplicates because only one completion per timeframe is enough (4 days)
user_trans_trunc = (
    df_retention_chart[["user_id", "cohort_date", "complete_episode_date"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
user_trans_trunc

# Calculate the time difference in 7-day intervals
user_trans_trunc["date_diff"] = (
    user_trans_trunc["complete_episode_date"] - user_trans_trunc["cohort_date"]
).dt.days // 7
user_trans_trunc

# Let’s pivot the data with date_diff as columns and cohort_date as rows. The value of each cell will be the count unique of retained users.
date_pivot_count = pd.DataFrame(
    pd.pivot_table(
        user_trans_trunc,
        values="user_id",
        index="cohort_date",
        columns="date_diff",
        aggfunc=lambda x: len(x.unique()),
    ).to_records()
)
date_pivot_count


# Join with the cohort size dataframe
date_pivot_join = pd.merge(cohort_size, date_pivot_count, how="inner", on="cohort_date")
date_pivot_join

# Get the Retention Rate
date_pivot_pct = date_pivot_join[["cohort_date", "cohort_size"]]
for i in range(0, len(date_pivot_join.columns) - 2):
    date_pivot_pct[str(i)] = (
        np.round(
            (date_pivot_join[str(i)] / date_pivot_join["cohort_size"]) * 100,
            0,
        )
        .fillna(0)
        .astype(int)
    )

# Transform to percentage (number of retained users in each month/cohort size)
date_pivot_pct = date_pivot_pct.fillna(0)
date_pivot_pct["cohort_date"] = date_pivot_pct["cohort_date"].astype(str)
date_pivot_pct["cohort_size"] = date_pivot_pct["cohort_size"].astype(str)
date_pivot_pct

# Color dataframe
date_pivot_pct.style.background_gradient(cmap="Blues", vmin=0, vmax=100)

# Save retention chart to csv
date_pivot_pct.to_csv("../data/retention_chart_7days.csv", header=True, index=False)



"""
잔존률 그래프
"""
df_weekly_retention_chart = pd.DataFrame(
    date_pivot_pct.iloc[:, 2:].T.mean(axis=1, skipna=True)
)
# df_weekly_retention_chart.plot(kind='line')
# plt.xlabel('Week')
# plt.ylabel('Retention Rate')
# plt.show()

plt.plot(
    df_weekly_retention_chart.index,
    df_weekly_retention_chart[0],
    marker="o",
    color="b",
    markersize=3,
)
plt.xticks(
    np.arange(
        int(df_weekly_retention_chart.index.min()),
        int(df_weekly_retention_chart.index.max()) + 1,
        5,
    )
)
plt.show()


""" 
RETENTION RATE BY USER IN 3 MONTHS ON 7 DAYS TIME FRAME 
"""

# Initialize the dictionary to store the retention data
dict_retention_users_3months = {"user_id": [], "cohort_group": [], "retention_rate": []}

# Get unique, sorted time frames for cohort dates
time_frames = list(df_retention_chart["cohort_date"].sort_values().unique())

# Precompute the `complete_episode_date`s for each user
user_dates = (
    user_trans_trunc.groupby("user_id")["complete_episode_date"].apply(set).to_dict()
)

# Iterate over unique users and their respective cohort groups
for user, cohort_group in (
    user_trans_trunc[["user_id", "cohort_date"]].drop_duplicates().values
):
    # Initialize retention rate and append user and cohort group to results dictionary
    retention_rate = 0
    dict_retention_users_3months["user_id"].append(user)
    dict_retention_users_3months["cohort_group"].append(cohort_group)

    # Get the start and end indices for the time frame within 1 month (12 x 7-day intervals)
    start_time_frame = time_frames.index(cohort_group)
    end_time_frame = start_time_frame + 12

    # Check if each time frame within 1 month has a corresponding complete_episode_date
    for frame in range(start_time_frame, end_time_frame):
        if frame < len(time_frames) and time_frames[frame] in user_dates.get(
            user, set()
        ):
            retention_rate += 1

    # Append the calculated retention rate for this user and cohort group
    dict_retention_users_3months["retention_rate"].append(retention_rate)

# Convert the dictionary to a DataFrame
df_retention_users_3months = pd.DataFrame(dict_retention_users_3months)
df_retention_users_3months = df_retention_users_3months.loc[
    df_retention_users_3months["cohort_group"] <= "2023-10-09"
]
df_retention_users_3months["retention_rate"] = (
    df_retention_users_3months["retention_rate"]
    / df_retention_users_3months["retention_rate"].max()
)
df_retention_rate_3months = (
    df_retention_users_3months["retention_rate"]
    .value_counts()
    .sort_index(ascending=False)
    .reset_index()
)
# Calculate the cumulative sum
df_retention_rate_3months["cumulative_count"] = df_retention_rate_3months[
    "count"
].cumsum()
df_retention_rate_3months["cumulative_percentage"] = (
    df_retention_rate_3months["cumulative_count"]
    / df_retention_rate_3months["count"].sum()
    * 100
)
df_retention_rate_3months

df_retention_users_3months.to_csv(
    "../data/retention_users_3month.csv", header=True, index=False
)


"""
USER SEGMENTATION
RETENTION OVER 
.75 (핵심)
.25 (일시적)
0 (냉담)
"""
bins = [0, 0.25, 0.75, 1.01]
labels = ["냉담", "일시적", "핵심"]
df_retention_users_3months["segmentation"] = pd.cut(
    df_retention_users_3months["retention_rate"],
    bins=bins,
    labels=labels,
    right=False,
    include_lowest=True,
)

df_user_segmentation = df_retention_users_3months[["user_id", "segmentation"]]
df_user_segmentation.isna().sum()
df_user_segmentation.to_csv("../data/user_segmentation.csv", header=True, index=False)


"""
CALCULATE AVG COMPLETE EPISODE INTERVAL
"""
db_params = {
    "host": "127.0.0.1",
    "database": "ott_db",
    "user": "postgres",
    "password": os.environ.get("PASSWORD"),
    "port": "5432",
}
engine = create_engine(
    f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}/{db_params["database"]}'
)

# Query
query_avg_complete_episode_interval = read_query_from_file(
    "../databases/sql_queries/avg_complete_episode_interval.sql"
)

# Query Dataframe
df_avg_complete_episode_interval = pd.read_sql_query(
    query_avg_complete_episode_interval, con=engine
)

# merge user_segmentation with avg_complete_episode_interval
df_user_segmentation_avg_complete_episode_interval = pd.merge(
    df_user_segmentation,
    df_avg_complete_episode_interval,
    how="inner",
    left_on="user_id",
    right_on="user_id",
)
# check that the null values are only for 냉담 users
df_user_segmentation_avg_complete_episode_interval.loc[
    df_user_segmentation_avg_complete_episode_interval[
        "avg_complete_episode_intervals"
    ].isna()
]["segmentation"].value_counts()
# drop the null value rows from the user_segmentation_avg_complete_episode_interval
df_user_segmentation_avg_complete_episode_interval.dropna(
    subset=["avg_complete_episode_intervals"], inplace=True
)

# table for average complete episode intervals per segmentation
df_segmentation_avg_complete_episode_interval = (
    df_user_segmentation_avg_complete_episode_interval[ 
        ["segmentation", "avg_complete_episode_intervals"]
    ]
    .groupby("segmentation")
    .agg({"avg_complete_episode_intervals": "mean"})
    .sort_values(by="avg_complete_episode_intervals")
    .reset_index()
)
df_segmentation_avg_complete_episode_interval

df_user_segmentation_avg_complete_episode_interval.to_csv(
    "../data/user_segment_complete_episode_interval.csv", header=True, index=False
)







""" 
GET ENTER EPISODE AND COMPLETE EPISODE PAIRS FOR AVG LEARNING TIME
"""

# SQL params
db_params = {
    "host": "127.0.0.1",
    "database": "ott_db",
    "user": "postgres",
    "password": os.environ.get("PASSWORD"),
    "port": "5432",
}
engine = create_engine(
    f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}/{db_params["database"]}'
)

# Query
query_avg_learning_time = read_query_from_file(
    "../databases/sql_queries/avg_learning_time.sql"
)

# Query Dataframe
df_avg_learning_time = pd.read_sql_query(query_avg_learning_time, con=engine)


df_avg_learning_time["client_event_time"] = pd.to_datetime(
    df_avg_learning_time["client_event_time"]
)

# Sort by `user_id`, `episode_id`, and `client_event_time` in ascending order
df_avg_learning_time = df_avg_learning_time.sort_values(
    by=["user_id", "episode_id", "client_event_time"]
)

results = []

for (user_id, episode_id), group in df_avg_learning_time.groupby(
    ["user_id", "episode_id"]
):
    # Separate enter and complete events
    enter_events = group[group["status"] == "enter_episode"]
    complete_events = group[group["status"] == "complete_episode"]

    # If there are both enter and complete events, proceed with pairing
    if not enter_events.empty and not complete_events.empty:
        for _, complete_row in complete_events.iterrows():
            # Find the farthest enter_event before each complete_event within 30 minutes
            potential_enters = enter_events[
                (enter_events["client_event_time"] < complete_row["client_event_time"])
                & (
                    complete_row["client_event_time"]
                    - enter_events["client_event_time"]
                    <= timedelta(minutes=30)
                )
            ]

            if not potential_enters.empty:
                # Select the farthest enter_event within the 30-minute window
                enter_event = potential_enters.iloc[0]
                
                # Append both the enter and complete event rows to results
                results.append(enter_event.to_dict())
                results.append(complete_row.to_dict())

# Convert results to DataFrame
df_avg_learning_time_paired = pd.DataFrame(results)
df_avg_learning_time_paired

"""TEST THE CODE ABOVE"""
df_results = df_avg_learning_time_paired
# Ensure `client_event_time` is in datetime format
df_results['client_event_time'] = pd.to_datetime(df_results['client_event_time'])

# Create a new column to store only the date part of `client_event_time`
df_results['event_date'] = df_results['client_event_time'].dt.date

# Separate enter and complete episodes within each group of `user_id` and `episode_id`
# Then, we use loc to filter only rows where dates differ
df_mismatched_dates = (
    df_results
    .groupby(['user_id', 'episode_id'])
    .filter(lambda x: (
        x.loc[x['status'] == 'enter_episode', 'event_date'].values[0] != 
        x.loc[x['status'] == 'complete_episode', 'event_date'].values[0]
    ))
)
# Display the mismatched dates
df_mismatched_dates.head(1000)

# Save dataframe
df_avg_learning_time_paired.to_csv(
    "../data/avg_learning_time_paired.csv", header=True, index=False
)





"""
CALCULATE AVG LEARNING TIME BY USER
"""
df_avg_learning_time = pd.read_csv('../data/avg_learning_time_paired.csv')


# Load your data and ensure proper columns are formatted as datetime
df_avg_learning_time['client_event_time'] = pd.to_datetime(df_avg_learning_time['client_event_time'])
enter_data = df_avg_learning_time[df_avg_learning_time['status'] == 'enter_episode'].rename(columns={'client_event_time': 'enter_time'})
complete_data = df_avg_learning_time[df_avg_learning_time['status'] == 'complete_episode'].rename(columns={'client_event_time': 'complete_time'})

# Merge on 'user_id' and 'episode_id' to align enters and completes
merged_data = pd.merge(enter_data, complete_data, on=['user_id', 'episode_id'])

# Calculate watch time for each episode in seconds and convert it to minutes
merged_data['watch_time'] = (merged_data['complete_time'] - merged_data['enter_time']).dt.total_seconds() / 60

# Aggregate watch time by person, still in minutes
total_watch_time_per_person = merged_data.groupby('user_id')['watch_time'].sum().reset_index()

enter_data['date'] = enter_data['enter_time'].dt.date
days_per_user = enter_data.groupby('user_id')['date'].nunique().reset_index(name='total_days_entered')

# Merge the unique days data with the total watch time data
result_data = pd.merge(total_watch_time_per_person, days_per_user, on='user_id')

result_data['avg_watch_time_per_day'] = result_data['watch_time'] / result_data['total_days_entered']

result_data

result_data.to_csv('../data/avg_learning_time.csv', header=True, index=False)

df_user_segmentation = pd.read_csv('../data/user_segmentation.csv')

df_user_segementation_avg_learning_time = pd.merge(df_user_segmentation, result_data, on='user_id')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_user_segementation_avg_learning_time, x='watch_time', y='total_days_entered', hue='segmentation')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df_user_segementation_avg_learning_time, x='segmentation', y='avg_watch_time_per_day')
plt.show()

# Save dataframe






"""
USER ACTION JOURNEY FOR ASSOCIATION ANALYSIS
"""

data_path = "../data/interim"
db_path = "../databases/interim_db.duckdb"
df_dict = make_key_table_value_path_dictionary(data_path)


df_user_action_journey = get_users_action_journey(df_dict, db_path)
df_user_action_journey


df = pd.read_csv('../data/raw/enter.episode_page.csv')
df = pd.read_csv('../data/raw/complete.episode.csv')
df.head()