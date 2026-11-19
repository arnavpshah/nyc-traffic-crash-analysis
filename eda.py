"""
CSE 163 - Exploratory Data Analysis

Analyze injury patterns in reported New York City
motor vehicle crashes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


DATA_FILE = "Motor_Vehicle_Collisions_-_Crashes_20261118.csv"
INJURY_COLUMN = "NUMBER OF PERSONS INJURED"
FACTOR_COLUMN = "CONTRIBUTING FACTOR VEHICLE 1"

BOROUGHS = [
    "BRONX",
    "BROOKLYN",
    "MANHATTAN",
    "QUEENS",
    "STATEN ISLAND"
]


def get_hour(time):
    """Return the hour from a crash time."""
    if pd.isnull(time):
        return None

    hour = int(time.split(":")[0])

    if 0 <= hour < 24:
        return hour

    return None


def calculate_rates(data, column, categories):
    """Calculate crash counts and injury percentages."""
    crash_counts = data.groupby(column).size()
    injury_counts = data.groupby(column)["injury_crash"].sum()

    results = []
    table = []

    for category in categories:
        if category in crash_counts.index:
            crashes = crash_counts[category]
            injuries = injury_counts[category]
            percentage = injuries / crashes * 100

            results.append({
                column: category,
                "crashes": crashes,
                "injury_crashes": injuries,
                "injury_percent": percentage
            })

            table.append([crashes - injuries, injuries])

    return pd.DataFrame(results), table


def check_chi_square(table, name):
    """Test injury status against a crash category."""
    if len(table) < 2:
        print(name, "- Not enough groups for testing")
        return

    injured = 0
    not_injured = 0

    for row in table:
        not_injured += row[0]
        injured += row[1]

    if injured == 0 or not_injured == 0:
        print(name, "- Not enough injury variation")
        return

    chi2, p_value, _, expected = chi2_contingency(table)

    below_five = 0
    minimum_expected = injured + not_injured

    for row in expected:
        for count in row:
            if count < 5:
                below_five += 1
            if count < minimum_expected:
                minimum_expected = count

    total = injured + not_injured
    cramers_v = (chi2 / total) ** 0.5

    print("\nChi-square test:", name)
    print("Chi-square:", chi2)
    print("P-value:", p_value)
    print("Cramer's V:", cramers_v)
    print("Minimum expected count:", minimum_expected)
    print("Expected counts below 5:", below_five)

    return chi2, p_value, cramers_v, minimum_expected, below_five


def make_graphs(hour_rates, borough_rates, factor_rates):
    """Create and save the three EDA visualizations."""
    sns.set_theme()

    sns.relplot(
        data=hour_rates,
        x="hour",
        y="injury_percent",
        kind="line"
    )

    plt.title("NYC Crash Injury Percentage by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Crashes With Injuries (%)")
    plt.savefig("injuries_by_hour.png", bbox_inches="tight")

    sns.catplot(
        data=borough_rates,
        x="BOROUGH",
        y="injury_percent",
        kind="bar"
    )

    plt.title("NYC Crash Injury Percentage by Borough")
    plt.xlabel("Borough")
    plt.ylabel("Crashes With Injuries (%)")
    plt.xticks(rotation=20)
    plt.savefig("injuries_by_borough.png", bbox_inches="tight")

    sns.catplot(
        data=factor_rates,
        y=FACTOR_COLUMN,
        x="injury_percent",
        kind="bar"
    )

    plt.title("Injury Percentage by Contributing Factor")
    plt.xlabel("Crashes With Injuries (%)")
    plt.ylabel("Contributing Factor")
    plt.savefig("injuries_by_factor.png", bbox_inches="tight")


def main():
    """Load the crash data and perform the EDA."""
    data = pd.read_csv(DATA_FILE)

    print("DATASET INFORMATION")
    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    print("\nMISSING VALUES")
    print(data.isnull().sum())

    print("\nINJURIES SUMMARY")
    print(data[INJURY_COLUMN].describe())

    print("\nBOROUGH COUNTS")
    print(data["BOROUGH"].value_counts())

    print("\nCONTRIBUTING FACTOR COUNTS")
    print(data[FACTOR_COLUMN].value_counts(dropna=False).to_string())

    # Prepare injury status and hour of day.
    data["hour"] = data["CRASH TIME"].apply(get_hour)
    data["injury_crash"] = data[INJURY_COLUMN] > 0

    valid = data.dropna(subset=[INJURY_COLUMN])
    valid = valid[valid[INJURY_COLUMN] >= 0]

    print("\nRows with valid injury counts:", len(valid))
    print("\nINJURY STATUS COUNTS")
    print(valid["injury_crash"].value_counts().to_string())

    print("\nHOUR SUMMARY")
    print(valid["hour"].describe())
    print("\nCRASH COUNTS BY HOUR")
    print(valid["hour"].value_counts().sort_index().to_string())

    # Research Question 1: Time of day
    hour_data = valid.dropna(subset=["hour"])

    hour_rates, hour_table = calculate_rates(
        hour_data, "hour", range(24)
    )

    print("\nINJURY PERCENTAGE BY HOUR")
    print(hour_rates)
    check_chi_square(hour_table, "Hour of Day")

    # Research Question 2: Borough
    borough_data = valid.dropna(subset=["BOROUGH"])

    borough_rates, borough_table = calculate_rates(
        borough_data, "BOROUGH", BOROUGHS
    )

    print("\nINJURY PERCENTAGE BY BOROUGH")
    print(borough_rates)
    print("Crashes excluded from borough comparison:",
          len(valid) - sum(borough_rates["crashes"]))
    check_chi_square(borough_table, "Borough")

    # Research Question 3: Contributing factors
    factor_data = valid.dropna(subset=[FACTOR_COLUMN])
    factor_data = factor_data[
        factor_data[FACTOR_COLUMN] != "Unspecified"
    ]
    factor_data = factor_data[
        factor_data[FACTOR_COLUMN] != ""
    ]

    top_factors = (
        factor_data[FACTOR_COLUMN].value_counts().head(10).index
    )

    factor_rates, factor_table = calculate_rates(
        factor_data, FACTOR_COLUMN, top_factors
    )

    print("\nINJURY PERCENTAGE BY FACTOR")
    print(factor_rates)
    check_chi_square(factor_table, "Contributing Factor")

    make_graphs(hour_rates, borough_rates, factor_rates)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
