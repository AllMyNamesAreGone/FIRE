# Standard library imports

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ======================
# Page Configuration
# ======================
st.set_page_config(
    layout="wide",  # Set the page layout to wide mode
    initial_sidebar_state="collapsed",  # Hide the sidebar by default
    page_title="Stock Growth Forecaster",
    page_icon="📈",
)


# ======================
# User Input Functions
# ======================
def user_inputs():
    """
    Collects user inputs from the Streamlit sidebar, including financial details, portfolio distribution,
    and inflation rate. Returns the user inputs as a tuple.

    Returns:
        tuple: User inputs including initial wealth, income, expenses, portfolio distribution, and more.
    """
    with st.expander("User Inputs", expanded=True):
        col1, col2, col3 = st.columns(3)  # Create columns inside the expander
        with col1:
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                initial_wealth = st.number_input(
                    "Net Worth [$]",
                    min_value=0,
                    value=50000,
                    step=1000,
                    help="Enter your current total net worth.",
                )
                initial_superannuation = st.number_input(
                    "Super [$]",
                    min_value=0,
                    value=10000,
                    step=1000,
                    help="Enter your current superannuation balance.",
                )
            with subcol2:
                annual_income = st.number_input(
                    "Income [$ p.a]",
                    min_value=0,
                    value=135000,
                    step=1000,
                    help="Enter your annual income after tax.",
                )
                annual_expenses = st.number_input(
                    "Expenses [$ p.a]",
                    min_value=0,
                    value=50000,
                    step=1000,
                    help="Enter your annual living expenses.",
                )

        with col2:
            stock_bond_split = st.slider(
                "Stock / Bond / Cash Split [%]",
                min_value=0,
                max_value=100,
                value=(33, 67),
                help="Slide to adjust the percentage of your portfolio allocated to stocks and bonds.",
            )
            stock_percent = stock_bond_split[0]
            bond_percent = stock_bond_split[1] - stock_bond_split[0]
            cash_percent = 100 - stock_bond_split[1]
            st.write(
                f"**Stocks:** {stock_percent}% | **Bonds:** {bond_percent}% | **Cash:** {cash_percent}%"
            )

        with col3:
            current_age = st.number_input(
                "Age [years]",
                min_value=0,
                value=25,
                step=1,
                help="Enter your current age.",
            )
            sex = st.radio(
                "Sex",
                ["Male", "Female"],
                help="Used in Probability of Sufficiency plot to determine probability of mortality.",
            )

    return (
        initial_wealth,
        annual_income,
        initial_superannuation,
        annual_expenses,
        stock_percent / 100,
        bond_percent / 100,
        cash_percent / 100,
        current_age,
        sex,
    )


# ============================
# Retirement Calculation
# ============================
def identify_optimal_retirement(
    initial_wealth,
    annual_income,
    annual_expenses,
    stock_percent,
    bond_percent,
    cash_percent,
    inflation_rate,
):
    """
    Identifies the optimal retirement age by simulating wealth accumulation and withdrawal until age 60.

    Args:
        initial_wealth (float): The initial net worth of the user.
        annual_income (float): The user's annual income.
        annual_expenses (float): The user's annual expenses.
        stock_percent (float): The percentage of the portfolio allocated to stocks.
        bond_percent (float): The percentage of the portfolio allocated to bonds.
        cash_percent (float): The percentage of the portfolio allocated to cash.
        inflation_rate (float): The annual inflation rate.

    Returns:
        int: The optimal retirement age.
    """
    wealth = initial_wealth
    optimal_retirement_age = None

    for age in range(1, 100):
        # Inflate income and expenses over time
        adjusted_income = annual_income * (1 + inflation_rate) ** age
        adjusted_expenses = annual_expenses * (1 + inflation_rate) ** age

        # Calculate the total return for this year
        total_return = (
            1
            + stock_return * stock_percent
            + bond_return * bond_percent
            + cash_return * cash_percent
        )

        # Update wealth by compounding it and adding net income (adjusted_income - adjusted_expenses)
        wealth = wealth * total_return + (adjusted_income - adjusted_expenses)

        # Check if wealth will last until 60 if retired now
        temp_wealth = wealth
        for future_age in range(age, 60):
            temp_expenses = annual_expenses * (1 + inflation_rate) ** future_age
            temp_wealth = temp_wealth * total_return - temp_expenses
            if temp_wealth <= 0:
                break

        # If temp_wealth remains positive until age 60, set this as retirement age
        if temp_wealth > 0 and optimal_retirement_age is None:
            optimal_retirement_age = age

    return optimal_retirement_age


# ============================
# Forecasting
# ============================
def expected_forecast(
    initial_wealth,
    initial_superannuation,
    annual_income,
    annual_expenses,
    stock_percent,
    bond_percent,
    cash_percent,
    inflation_rate,
    optimal_retirement_age,
):
    """
    Generates and plots the expected forecast of wealth and superannuation growth over time.

    Args:
        initial_wealth (float): The initial net worth of the user.
        initial_superannuation (float): The initial superannuation balance.
        annual_income (float): The user's annual income.
        annual_expenses (float): The user's annual expenses.
        stock_percent (float): The percentage of the portfolio allocated to stocks.
        bond_percent (float): The percentage of the portfolio allocated to bonds.
        cash_percent (float): The percentage of the portfolio allocated to cash.
        inflation_rate (float): The annual inflation rate.
        optimal_retirement_age (int): The calculated optimal retirement age.
    """
    wealth = initial_wealth
    superannuation = initial_superannuation

    wealth_history = []
    superannuation_history = []
    wealth_upper = []
    wealth_lower = []
    superannuation_upper = []
    superannuation_lower = []

    for year in range(1, 100):
        # Adjust income to 0 after the optimal retirement age
        if year >= optimal_retirement_age:
            annual_income = 0

        # Inflate expenses over time
        adjusted_expenses = annual_expenses * (1 + inflation_rate) ** year

        # Calculate the total return for this year
        total_return = (
            1
            + stock_return * stock_percent
            + bond_return * bond_percent
            + cash_return * cash_percent
        )
        total_error = (
            1
            + stock_error * stock_percent
            + bond_error * bond_percent
            + cash_error * cash_percent
        )

        # Update wealth by compounding it and adding net income (annual_income - adjusted_expenses)
        wealth = wealth * total_return + (annual_income - adjusted_expenses)
        if wealth < 0:
            adjusted_expenses = -wealth
            wealth = 0

        wealth_history.append([year, wealth])

        # Calculate upper and lower bounds for wealth
        wealth_upper_bound = wealth * total_error  # +1 standard deviation
        wealth_lower_bound = wealth / total_error  # -1 standard deviation
        wealth_upper.append([year, wealth_upper_bound])
        wealth_lower.append([year, wealth_lower_bound])

        # Update superannuation by compounding it and adding 11% of the income
        superannuation = superannuation * total_return + annual_income * 0.11
        if wealth == 0:
            superannuation -= adjusted_expenses

        superannuation_history.append([year, superannuation])

        # Calculate upper and lower bounds for superannuation
        superannuation_upper_bound = (
            superannuation * total_error
        )  # +1 standard deviation
        superannuation_lower_bound = (
            superannuation / total_error
        )  # -1 standard deviation
        superannuation_upper.append([year, superannuation_upper_bound])
        superannuation_lower.append([year, superannuation_lower_bound])

    # Convert to DataFrames
    df_wealth = pd.DataFrame(wealth_history, columns=["Years", "Wealth"])
    df_superannuation = pd.DataFrame(
        superannuation_history, columns=["Years", "Superannuation"]
    )
    df_wealth_upper = pd.DataFrame(wealth_upper, columns=["Years", "Wealth Upper"])
    df_wealth_lower = pd.DataFrame(wealth_lower, columns=["Years", "Wealth Lower"])
    df_superannuation_upper = pd.DataFrame(
        superannuation_upper, columns=["Years", "Super Upper"]
    )
    df_superannuation_lower = pd.DataFrame(
        superannuation_lower, columns=["Years", "Super Lower"]
    )

    # Plot Wealth and Superannuation on the same chart
    plt.figure(figsize=(10, 6))
    plt.plot(df_wealth["Years"], df_wealth["Wealth"], label="Wealth")
    plt.fill_between(
        df_wealth["Years"],
        df_wealth_lower["Wealth Lower"],
        df_wealth_upper["Wealth Upper"],
        color="blue",
        alpha=0.2,
    )
    plt.plot(
        df_superannuation["Years"],
        df_superannuation["Superannuation"],
        label="Superannuation",
        color="orange",
    )
    plt.fill_between(
        df_superannuation["Years"],
        df_superannuation_lower["Super Lower"],
        df_superannuation_upper["Super Upper"],
        color="orange",
        alpha=0.2,
    )

    # Improve y-axis labels and formatting
    plt.xlabel("Years")
    plt.ylabel("Wealth & Superannuation ($)")
    plt.title("Wealth and Superannuation Forecast with Error Bands")
    plt.legend()

    # Format y-axis as currency
    plt.gca().get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "${:,.0f}".format(x))
    )

    st.pyplot(plt)


# ============================
# MAIN
# ============================
def main():
    """
    Main function to control the app flow. It collects user inputs, identifies the optimal retirement age,
    and generates the expected forecast based on the user's inputs.
    """
    (
        initial_wealth,
        annual_income,
        initial_superannuation,
        annual_expenses,
        stock_percent,
        bond_percent,
        cash_percent,
        current_age,
        sex,
    ) = user_inputs()

    # Calculate the optimal retirement age before running the forecast
    optimal_retirement_age = identify_optimal_retirement(
        initial_wealth,
        annual_income,
        annual_expenses,
        stock_percent,
        bond_percent,
        cash_percent,
        inflation_rate,
    )

    st.title("Stock Growth Forecaster")
    option = st.selectbox(
        "Select Option",
        ["Expected Wealth and Superannuation", "Probability of Sufficiency"],
        label_visibility="collapsed",
    )

    if option == "Expected Wealth and Superannuation":
        expected_forecast(
            initial_wealth,
            initial_superannuation,
            annual_income,
            annual_expenses,
            stock_percent,
            bond_percent,
            cash_percent,
            inflation_rate,
            optimal_retirement_age,
        )
        if optimal_retirement_age:
            st.write(f"The optimal retirement age is: {optimal_retirement_age} years")


if __name__ == "__main__":
    # Set return and error rates for stocks, bonds, and cash
    stock_return = 0.07
    stock_error = 0.17
    bond_return = 0.04
    bond_error = 0.06
    cash_return = 0.03
    cash_error = 0.02
    inflation_rate = 0.02
    main()
