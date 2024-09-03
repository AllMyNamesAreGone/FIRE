# Standard library imports

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
        col1, col2, col3 = st.columns(3)  # Create columns inside the expander

        with col1:
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                current_wealth = st.number_input(
                    "Net Worth [$]",
                    min_value=0,
                    value=50000,
                    step=1000,
                    help="Enter your current total net worth.",
                )
                current_superannuation = st.number_input(
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

        with col3:
            age_now = st.number_input(
                "Age [years]",
                min_value=0,
                value=25,
                step=1,
                help="Enter your current age.",
            )
            # sex = st.radio(
            #     "Sex",
            #     ["Male", "Female"],
            #     help="Used in Probability of Sufficiency plot to determine probability of mortality.",
            # )

    return (
        current_wealth,
        annual_income,
        current_superannuation,
        annual_expenses,
        real_annual_return,
        age_now,
        # sex,
        stock_percent,
        bond_percent,
        cash_percent,
    )


# ============================
# Optimisation Function
# ============================
def optimise_retirement_age_and_super_contribution(
    current_wealth,
    annual_income,
    current_superannuation,
    annual_expenses,
    real_annual_return,
    age_now,
):
    optimal_retirement_age = None
    optimal_contribution = None
    optimal_wealth_history = None
    optimal_super_history = None

    # Iterate over possible retirement ages
    for retirement_age in range(age_now, 61):  # Include age 60
        for super_contribution_rate in range(11, 81):  # Range from 11% to 80%
            super_contribution_rate /= 100  # Convert to decimal

            wealth = current_wealth
            superannuation = current_superannuation
            wealth_history = []
            super_history = []

            # Simulate from current age to age 120
            for age in range(age_now, 121):
                if age < retirement_age:
                    wealth = (
                        wealth * (1 + real_annual_return)
                        + annual_income * (1 - super_contribution_rate)
                        - annual_expenses
                    )
                    superannuation = (
                        superannuation * (1 + real_annual_return)
                        + annual_income * super_contribution_rate
                    )
                elif age < 60:
                    wealth = wealth * (1 + real_annual_return) - annual_expenses
                    superannuation = superannuation * (1 + real_annual_return)
                else:
                    superannuation = (
                        superannuation * (1 + real_annual_return) - annual_expenses
                    )

                wealth_history.append(wealth)
                super_history.append(superannuation)

            # Check if both wealth and superannuation are non-negative until age 120
            if wealth >= 0 and superannuation >= 0:
                optimal_retirement_age = retirement_age
                optimal_contribution = super_contribution_rate
                optimal_wealth_history = wealth_history
                optimal_super_history = super_history
                break  # Found a valid solution for this retirement age; no need to check further SCRs

        if optimal_retirement_age is not None:
            break  # Found a valid solution; break outer loop

    # Return the optimal results and DataFrame for further use
    df_history = pd.DataFrame(
        {
            "Age": range(age_now, 121),
            "Wealth": optimal_wealth_history,
            "Superannuation": optimal_super_history,
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
        annual_income,
        current_superannuation,
        annual_expenses,
        real_annual_return,
        age_now,
        # sex,
        stock_percent,
        bond_percent,
        cash_percent,
    ) = user_inputs()

    optimal_age, optimal_contribution, df_history = (
        optimise_retirement_age_and_super_contribution(
            current_wealth,
            annual_income,
            current_superannuation,
            annual_expenses,
            real_annual_return,
            age_now,
        )
    )
    # Display the final results in a clean, formatted way
    st.markdown("### 🎯 **Optimization Results**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"**Optimal Retirement Age:** {optimal_age}")
    with col2:
        st.success(
            f"**Optimal Superannuation Contribution Rate:** {optimal_contribution:.0%}"
        )
    with col3:
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
            # Assuming stock_percent, bond_percent, and cash_percent are the percentage allocations
            stock_allocation = stock_percent
            bond_allocation = bond_percent
            cash_allocation = cash_percent

            # Define an exponential scaling factor for the standard deviation
            age_factor = np.exp(
                np.linspace(0, 1.5, len(df_history))
            )  # Exponential increase to create a "blooming" effect

            # Compute the portfolio variance for each year
            portfolio_variance = (
                (stock_error * 3 * stock_allocation * df_history["Wealth"]) ** 2
                + (bond_error * 3 * bond_allocation * df_history["Wealth"]) ** 2
                + (cash_error * 3 * cash_allocation * df_history["Wealth"]) ** 2
            )

            # The total portfolio standard deviation that increases exponentially with age
            total_std_dev = (portfolio_variance**0.5) * age_factor

            upper_bound_wealth = df_history["Wealth"] + total_std_dev
            lower_bound_wealth = df_history["Wealth"] - total_std_dev

            # Repeat for superannuation
            upper_bound_super = df_history["Superannuation"] + total_std_dev
            lower_bound_super = df_history["Superannuation"] - total_std_dev

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

            # Add wealth bounds with more transparency
            fig.add_trace(
                go.Scatter(
                    x=df_history["Age"],
                    y=upper_bound_wealth,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_history["Age"],
                    y=lower_bound_wealth,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(0, 0, 255, 0.2)",  # Increase transparency
                    line=dict(width=0),
                    showlegend=False,
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

            # Add superannuation bounds with more transparency
            fig.add_trace(
                go.Scatter(
                    x=df_history["Age"],
                    y=upper_bound_super,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_history["Age"],
                    y=lower_bound_super,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(0, 255, 0, 0.2)",  # Increase transparency
                    line=dict(width=0),
                    showlegend=False,
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
