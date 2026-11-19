"""Test NYC crash EDA calculations using small, hand-checked examples."""

import pandas as pd

from eda import calculate_rates, check_chi_square, get_hour


def test_get_hour():
    """Check valid hours, missing times, and hours outside the valid range."""
    assert get_hour("00:00") == 0
    assert get_hour("7:05") == 7
    assert get_hour("12:30") == 12
    assert get_hour("23:59") == 23
    assert get_hour(None) is None
    assert get_hour(float("nan")) is None
    assert get_hour("24:00") is None
    assert get_hour("-1:00") is None


def test_calculate_rates():
    """Check counts, percentages, category order, and table cells."""
    data = pd.DataFrame({
        "BOROUGH": [
            "BRONX", "BRONX", "BRONX", "QUEENS", "QUEENS", "BROOKLYN"
        ],
        "NUMBER OF PERSONS INJURED": [0, 2, 1, 0, 0, 3],
        "injury_crash": [False, True, True, False, False, True]
    })
    categories = ["QUEENS", "BRONX", "BROOKLYN", "STATEN ISLAND"]

    rates, table = calculate_rates(data, "BOROUGH", categories)

    assert rates["BOROUGH"].tolist() == ["QUEENS", "BRONX", "BROOKLYN"]
    assert rates["crashes"].tolist() == [2, 3, 1]
    assert rates["injury_crashes"].tolist() == [0, 2, 1]

    assert rates.loc[0, "injury_percent"] == 0
    assert abs(rates.loc[1, "injury_percent"] - 200 / 3) < 0.000001
    assert rates.loc[2, "injury_percent"] == 100

    # Columns are [crashes without injuries, crashes with injuries].
    assert table == [[2, 0], [1, 2], [0, 1]]


def test_calculate_rates_empty():
    """Check empty data and a requested category that has no crashes."""
    empty_data = pd.DataFrame({"BOROUGH": [], "injury_crash": []})
    rates, table = calculate_rates(empty_data, "BOROUGH", ["BRONX"])
    assert rates.empty
    assert table == []

    data = pd.DataFrame({"BOROUGH": ["BRONX"], "injury_crash": [True]})
    rates, table = calculate_rates(data, "BOROUGH", ["QUEENS"])
    assert rates.empty
    assert table == []


def test_check_chi_square():
    """Check statistics and expected-count diagnostics for known tables."""
    table = [[10, 20], [10, 20], [20, 20]]
    chi2, p_value, cramers_v, minimum, below_five = check_chi_square(
        table, "Known example"
    )

    # This table has chi-square 25/9, 2 degrees of freedom, and V = 1/6.
    assert abs(chi2 - 25 / 9) < 0.000001
    assert abs(p_value - 0.2493522088) < 0.000001
    assert abs(cramers_v - 1 / 6) < 0.000001
    assert abs(minimum - 12) < 0.000001
    assert below_five == 0

    table = [[1, 1], [1, 1], [1, 1]]
    chi2, p_value, cramers_v, minimum, below_five = check_chi_square(
        table, "Small expected counts"
    )

    assert abs(chi2) < 0.000001
    assert abs(p_value - 1) < 0.000001
    assert abs(cramers_v) < 0.000001
    assert abs(minimum - 1) < 0.000001
    assert below_five == 6


def test_check_chi_square_guards():
    """Check cases with too few groups or no variation in injury status."""
    assert check_chi_square([], "Empty table") is None
    assert check_chi_square([[5, 5]], "One group") is None
    assert check_chi_square([[5, 0], [10, 0]], "No injury crashes") is None
    assert check_chi_square([[0, 5], [0, 10]], "All injury crashes") is None


def main():
    """Run the EDA tests."""
    test_get_hour()
    test_calculate_rates()
    test_calculate_rates_empty()
    test_check_chi_square()
    test_check_chi_square_guards()
    print("All tests passed!")


if __name__ == "__main__":
    main()
