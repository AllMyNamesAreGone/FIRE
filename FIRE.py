# Standard library imports
import math

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ======================
# Page Configuration
# ======================
def page_config():
    st.set_page_config(
        layout="wide",  # Set the page layout to wide mode
        initial_sidebar_state="collapsed",  # Hide the sidebar by default
        page_title="Financial Independence Retire Early Stochastic Forecaster",
        page_icon="🔥",
    )


# ======================
# User Input Functions
# ======================
def user_inputs():
    with st.expander("User Inputs", expanded=True):
        finances, split, age = st.columns(3)  # Create columns inside the expander

        with finances:
            time_period, tax_status = st.columns(2)
            with time_period:
                period = st.selectbox(
                    "Time Period",
                    ["Annual", "Monthly", "Fortnightly", "Weekly"],
                    label_visibility="collapsed",
                )
            with tax_status:
                tax = st.checkbox("Pre-tax?", value=True)
            income = st.number_input(
                "Income [$]",
                min_value=0,
                value=100000,
                step=1000,
                help="Enter your income.",
            )
            if period == "Monthly":
                income *= 12
            elif period == "Fortnightly":
                income *= 26
            elif period == "Weekly":
                income *= 52

            if tax:
                if income <= 45000:
                    income -= (income - 18200) * 0.16
                elif income <= 135000:
                    income -= 4288 + (income - 45000) * 0.30
                elif income <= 190000:
                    income -= 31288 + (income - 135000) * 0.37
                else:
                    income -= 51638 + (income - 190000) * 0.45

            expenses = st.number_input(
                "Annual Expenses [$]",
                min_value=0,
                value=50000,
                step=1000,
                help="Enter your annual living expenses.",
            )

            w, s = st.columns(2)
            with w:
                current_wealth = st.number_input(
                    "Net Worth [$]",
                    min_value=0,
                    value=0,
                    step=1000,
                    help="Enter your current wealth outside of super.",
                )
            with s:
                current_superannuation = st.number_input(
                    "Super [$]",
                    min_value=0,
                    value=0,
                    step=1000,
                    help="Enter your current superannuation balance.",
                )
        with split:
            html_string = """  <div class="wrapper">
  <div class="container">

    <div class="slider-wrapper">
      <div id="slider-range"></div>

      <div class="range-wrapper">
        <div class="range"></div>
        <div class="range-alert">+</div>

        <div class="gear-wrapper">
          <div class="gear-large gear-one">
            <div class="gear-tooth"></div>
            <div class="gear-tooth"></div>
            <div class="gear-tooth"></div>
            <div class="gear-tooth"></div>
          </div>
          <div class="gear-large gear-two">
            <div class="gear-tooth"></div>
            <div class="gear-tooth"></div>
            <div class="gear-tooth"></div>
            <div class="gear-tooth"></div>
          </div>
        </div>

      </div>

      <div class="marker marker-0"><sup>$</sup>10,000</div>
      <div class="marker marker-25"><sup>$</sup>35,000</div>
      <div class="marker marker-50"><sup>$</sup>60,000</div>
      <div class="marker marker-75"><sup>$</sup>85,000</div>
      <div class="marker marker-100"><sup>$</sup>110,000+</div>
    </div>

  </div>
</div>

"""
            # st.components.v1.html(html_string, height=700)
            profile = st.selectbox(
                "Select Option",
                ["Conservative", "Balanced", "Growth", "High Growth"],
                label_visibility="collapsed",
            )
            if profile == "Conservative":
                stock_bond_split = (25, 85)
            elif profile == "Balanced":
                stock_bond_split = (50, 90)
            elif profile == "Growth":
                stock_bond_split = (80, 95)
            elif profile == "High Growth":
                stock_bond_split = (90, 95)
            stock_bond_split = st.slider(
                "Stock / Bond / Cash Split [%]",
                min_value=0,
                max_value=100,
                value=(stock_bond_split),
                help="Slide to adjust the percentage of your portfolio allocated to stocks and bonds.",
            )
            stock_percent = stock_bond_split[0] / 100
            bond_percent = (stock_bond_split[1] - stock_bond_split[0]) / 100
            cash_percent = (100 - stock_bond_split[1]) / 100
            st.write(
                "**Stocks:**",
                stock_bond_split[0],
                "% | **Bonds:**",
                stock_bond_split[1] - stock_bond_split[0],
                "% | **Cash:**",
                100 - stock_bond_split[1],
                "%",
            )

            real_annual_return = (
                stock_return * stock_percent
                + bond_return * bond_percent
                + cash_return * cash_percent
                - 0.02  # inflation rate
            )

        with age:
            life_table = pd.read_csv("ABS 2020 Life Table.csv")
            age_now = st.number_input(
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
            if sex == "Male":
                # Get the expected remaining life for males
                ex = life_table.loc[life_table["Age"] == age_now, "exm"].values[0]
            else:
                # Get the expected remaining life for females
                ex = life_table.loc[life_table["Age"] == age_now, "exf"].values[0]

            # Expected lifespan is current age plus the remaining life expectancy
            expected_lifespan = math.ceil(age_now + ex)

    return (
        current_wealth,
        income,
        current_superannuation,
        expenses,
        real_annual_return,
        age_now,
        sex,
        expected_lifespan,
        stock_percent,
        bond_percent,
        cash_percent,
    )


# ============================
# Optimisation Function
# ============================
def optimise_retirement_age_and_super_contribution(
    current_wealth,
    income,
    current_superannuation,
    expenses,
    real_annual_return,
    age_now,
    expected_lifespan,
):
    optimal_retirement_age = None
    optimal_contribution = None
    optimal_wealth_history = None
    optimal_super_history = None

    # Iterate over possible retirement ages
    for retirement_age in range(age_now, expected_lifespan):  # Include age 60
        # Iterate over super_contribution_rate from 11% to 80% in 1% steps
        for super_contribution_rate in reversed(
            np.arange(0.11, 0.99, 0.01)
        ):  # 11% to 80%
            # Initialize wealth and superannuation
            wealth = current_wealth
            superannuation = current_superannuation
            wealth_history = []
            super_history = []
            bankrupt = False  # Flag to track bankruptcy

            # Simulate from current age to expected lifespan
            for age in range(age_now, expected_lifespan):
                # TODO: Age needs to be dynamic, based off of expected life span from Life Table
                if age < retirement_age:
                    # Working age
                    wealth = (
                        wealth * (1 + real_annual_return)
                        + income * (1 - super_contribution_rate)
                        - expenses
                    )
                    superannuation = (
                        superannuation * (1 + real_annual_return)
                        + income * super_contribution_rate
                    )
                elif age < 60:
                    # Retirement but pre-preservation age (before 60)
                    wealth = wealth * (1 + real_annual_return) - expenses
                    superannuation = superannuation * (1 + real_annual_return)
                else:
                    if wealth >= expenses:
                        # Case A: Withdraw expenses from wealth
                        wealth = wealth * (1 + real_annual_return) - expenses
                        superannuation = superannuation * (1 + real_annual_return)
                    elif wealth < expenses:
                        # Case B: Withdraw all wealth and the remaining from superannuation
                        superannuation = (
                            superannuation * (1 + real_annual_return)
                            + wealth
                            - expenses
                        )
                        wealth = 0

                # Check if bankrupt (wealth or superannuation go negative)
                if wealth < 0 or superannuation < 0:
                    bankrupt = True
                    break

                # Append to history
                wealth_history.append(wealth)
                super_history.append(superannuation)

            # Check if we never went bankrupt and found a solution
            if not bankrupt:
                optimal_retirement_age = retirement_age
                optimal_contribution = super_contribution_rate
                optimal_wealth_history = wealth_history
                optimal_super_history = super_history
                break  # Found a valid solution for this retirement age; no need to check further SCRs

        if optimal_retirement_age is not None:
            break  # Found a valid solution; break outer loop

    # If no valid solution was found, set optimal values to None or handle appropriately
    if optimal_retirement_age is None:
        # Handle the scenario where no combination prevents bankruptcy
        # For example, set to maximum retirement_age and highest contribution rate
        # Or raise an exception, or return specific indicators
        # Here, we'll set to age 60 and super_contribution_rate of 80%
        optimal_retirement_age = 60
        optimal_contribution = 0.80
        wealth_history = []
        super_history = []
        bankrupt = False

        # Simulate with retirement_age=60 and super_contribution_rate=80%
        for age in range(age_now, 121):
            if age < optimal_retirement_age:
                # Working age
                wealth = (
                    current_wealth * (1 + real_annual_return)
                    + income * (1 - optimal_contribution)
                    - expenses
                )
                superannuation = (
                    current_superannuation * (1 + real_annual_return)
                    + income * optimal_contribution
                )
            elif age < 60:
                # Retirement but pre-preservation age (before 60)
                wealth = wealth * (1 + real_annual_return) - expenses
                superannuation = superannuation * (1 + real_annual_return)
            else:
                # Preservation age (60 and above)
                W_new = wealth * (1 + real_annual_return)
                if W_new >= expenses:
                    # Case A: Withdraw expenses from wealth
                    wealth = W_new - expenses
                    superannuation = superannuation * (1 + real_annual_return)
                elif W_new > 0 and W_new < expenses:
                    # Case B: Withdraw all wealth and the remaining from superannuation
                    amount_from_wealth = W_new
                    wealth = 0
                    amount_from_super = expenses - amount_from_wealth
                    superannuation = (
                        superannuation * (1 + real_annual_return) - amount_from_super
                    )
                else:
                    # Case C: Withdraw expenses entirely from superannuation
                    wealth = 0
                    superannuation = (
                        superannuation * (1 + real_annual_return) - expenses
                    )

            # Check if bankrupt
            if wealth < 0 or superannuation < 0:
                bankrupt = True
                break

            # Append to history
            wealth_history.append(wealth)
            super_history.append(superannuation)

        if not bankrupt:
            optimal_wealth_history = wealth_history
            optimal_super_history = super_history
        else:
            # No valid solution even with maximum contribution rate
            # Handle accordingly; for simplicity, set histories to None
            optimal_wealth_history = None
            optimal_super_history = None

    # Handle the scenario where optimal_wealth_history might still be None
    if optimal_wealth_history is not None and optimal_super_history is not None:
        df_history = pd.DataFrame(
            {
                "Age": range(age_now, age_now + len(optimal_wealth_history)),
                "Wealth": optimal_wealth_history,
                "Superannuation": optimal_super_history,
            }
        )
    else:
        # If no valid history found, return an empty DataFrame or handle appropriately
        df_history = pd.DataFrame(
            {
                "Age": [],
                "Wealth": [],
                "Superannuation": [],
            }
        )
    return optimal_retirement_age, optimal_contribution, df_history


def simulate_probabilities(age_now, df_history):
    np.random.seed(42)  # For reproducibility
    forecast_years = 120 - age_now
    ages = range(age_now, 121)

    # Assuming wealth forecast from df_history is accurate to some degree, use the last known forecast
    wealth_forecast = df_history[df_history["Age"] >= age_now]["Wealth"].values
    wealth_forecast = np.concatenate(
        [wealth_forecast, np.zeros(forecast_years + 1 - len(wealth_forecast))]
    )

    # Simulate probabilities of different outcomes for each age
    # Use an exponential model for mortality increasing with age
    base_mortality_rate = 0.0001  # Start with a low base mortality rate
    prob_dead = np.clip(
        1 - np.exp(-base_mortality_rate * (np.arange(forecast_years + 1) ** 2)), 0, 1
    )

    # Probability of going broke if wealth is below a certain threshold (e.g., 0)
    prob_broke = np.where(
        wealth_forecast <= 0, np.linspace(0.1, 0.9, len(wealth_forecast)), 0
    )

    # Normalize probabilities to ensure they sum to 1 or less
    remaining_prob = 1 - prob_broke - prob_dead
    prob_less_start = np.clip(
        remaining_prob * np.random.uniform(0.3, 0.4, forecast_years + 1),
        0,
        remaining_prob,
    )
    remaining_prob -= prob_less_start
    prob_greater_start = np.clip(
        remaining_prob * np.random.uniform(0.3, 0.4, forecast_years + 1),
        0,
        remaining_prob,
    )
    remaining_prob -= prob_greater_start
    prob_more_than_double = np.clip(remaining_prob, 0, 1)

    # Convert probabilities to percentages
    data = {
        "Age": ages,
        "Bankruptcy": prob_broke * 100,
        "Dead": prob_dead * 100,
        "Bal < start": prob_less_start * 100,
        "Bal > start": prob_greater_start * 100,
        "Bal > 2x start": prob_more_than_double * 100,
    }

    df_probabilities = pd.DataFrame(data)

    return df_probabilities


def plot_probabilities(df_probabilities):
    fig = go.Figure()

    # Stacked area chart
    fig.add_trace(
        go.Scatter(
            x=df_probabilities["Age"],
            y=df_probabilities["Bankruptcy"],
            mode="lines",
            name="Bankruptcy",
            stackgroup="one",  # define stack group
            line=dict(width=0.5, color="red"),
            fillcolor="rgba(255, 0, 0, 0.5)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_probabilities["Age"],
            y=df_probabilities["Dead"],
            mode="lines",
            name="Dead",
            stackgroup="one",  # define stack group
            line=dict(width=0.5, color="grey"),
            fillcolor="rgba(100, 100, 100, 0.5)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_probabilities["Age"],
            y=df_probabilities["Bal < start"],
            mode="lines",
            name="Bal < start",
            stackgroup="one",
            line=dict(width=0.5, color="lightgreen"),
            fillcolor="rgba(144, 238, 144, 0.5)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_probabilities["Age"],
            y=df_probabilities["Bal > start"],
            mode="lines",
            name="Bal > start",
            stackgroup="one",
            line=dict(width=0.5, color="green"),
            fillcolor="rgba(0, 128, 0, 0.5)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_probabilities["Age"],
            y=df_probabilities["Bal > 2x start"],
            mode="lines",
            name="Bal > 2x start",
            stackgroup="one",
            line=dict(width=0.5, color="darkgreen"),
            fillcolor="rgba(0, 100, 0, 0.5)",
        )
    )

    fig.update_layout(
        title="Probability of Sufficiency Over Time",
        xaxis_title="Age",
        yaxis_title="Frequency (%)",
        yaxis=dict(type="linear", range=[0, 100], ticksuffix="%"),
        showlegend=True,
    )

    st.plotly_chart(fig)


# ============================
# Main Function
# ============================
def main():
    page_config()
    st.title("Financial Independence Retire Early Stochastic Forecaster")
    (
        current_wealth,
        income,
        current_superannuation,
        expenses,
        real_annual_return,
        age_now,
        sex,
        expected_lifespan,
        stock_percent,
        bond_percent,
        cash_percent,
    ) = user_inputs()
    optimal_age, optimal_contribution, df_history = (
        optimise_retirement_age_and_super_contribution(
            current_wealth,
            income,
            current_superannuation,
            expenses,
            real_annual_return,
            age_now,
            expected_lifespan,
        )
    )

    # Display the final results in a clean, formatted way
    st.markdown("### 🎯 **Optimisation Results**")
    age, rate, balance = st.columns(3)
    with age:
        st.success(f"**Optimal Retirement Age:** {optimal_age}")
    with rate:
        st.success(
            f"**Optimal Superannuation Contribution Rate:** {optimal_contribution:.0%}"
        )
    with balance:
        st.success(
            f"**Max Super Balance Required:** ${df_history['Superannuation'].max():,.0f}"
        )

    option = st.selectbox(
        "Select Option",
        ["Expected Wealth and Superannuation", "Probability of Sufficiency"],
        label_visibility="collapsed",
    )

    if option == "Expected Wealth and Superannuation":
        if not df_history.empty:
            # Plot using Plotly
            fig = go.Figure()

            # Plot wealth
            fig.add_trace(
                go.Scatter(
                    x=df_history["Age"],
                    y=df_history["Wealth"],
                    mode="lines",
                    name="Wealth",
                    line=dict(color="blue", width=2),  # Increase line thickness
                )
            )

            # Plot superannuation
            fig.add_trace(
                go.Scatter(
                    x=df_history["Age"],
                    y=df_history["Superannuation"],
                    mode="lines",
                    name="Superannuation",
                    line=dict(color="green", width=2),  # Increase line thickness
                )
            )

            fig.update_layout(
                title="Expected Wealth and Superannuation with Rapidly Increasing Uncertainty Bounds",
                xaxis_title="Age",
                yaxis_title="Value [$]",
                legend=dict(x=0.01, y=0.99),
                xaxis=dict(showgrid=True),  # Add gridlines
                yaxis=dict(showgrid=True),  # Add gridlines
            )

            st.plotly_chart(fig)
        else:
            st.error("No valid data to plot. Check calculations or inputs.")

    elif option == "Probability of Sufficiency":
        df_probabilities = simulate_probabilities(age_now, df_history)
        plot_probabilities(df_probabilities)


if __name__ == "__main__":
    # Set return and error rates for stocks, bonds, and cash
    stock_return = 0.07
    stock_error = 0.17
    bond_return = 0.04
    bond_error = 0.06
    cash_return = 0.03
    cash_error = 0.02
    main()
# TODO: Retirement Date Probability Distribution
# TODO: Savings Rate Sensitivity
# TODO: Nominal vs real
