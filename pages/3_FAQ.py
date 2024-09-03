import streamlit as st

# Page Title
st.title("FIRE FAQ")

st.markdown("""
Welcome to the Frequently Asked Questions (FAQ) page for our FIRE (Financial Independence, Retire Early) tool. Here, you’ll find answers to common questions about the FIRE movement and how to use our modeling tool effectively.
""")

# General FIRE Questions
st.header("General FIRE Questions")

# Question 1
with st.expander("What is the FIRE movement?"):
    st.write("""
    The FIRE (Financial Independence, Retire Early) movement is a financial strategy and lifestyle choice aimed at achieving financial independence and retiring earlier than the traditional retirement age. The core principles include maximizing savings, minimizing expenses, and investing strategically to grow wealth over time.
    """)

# Question 2
with st.expander("How much money do I need to retire early?"):
    st.write("""
    The amount needed to retire early depends on various factors, including your annual expenses, lifestyle, and the safe withdrawal rate you choose. A common rule of thumb is to save 25 times your annual expenses, which is based on the 4% safe withdrawal rate.
    """)

# Model-Specific Questions
st.header("Model-Specific Questions")

# Question 1
with st.expander("What models are used in the FIRE tool?"):
    st.write("""
    The FIRE tool uses a range of models to simulate different financial scenarios, including:
    - **Global Stock Market Forecasting**: Utilizes an ARIMA(0,2,2) model to predict stock market trends.
    - **Australian 10-Year Bond Rates**: Uses a random walk model to forecast bond rates.
    - **Cash Rate Targets**: Forecasts changes in the Reserve Bank of Australia’s cash rate targets using a random walk model.
    """)

# Question 2
with st.expander("How does the tool account for inflation?"):
    st.write("""
    The tool incorporates inflation by adjusting the real returns of investments. This ensures that the forecasts account for the changing value of money over time, providing a more accurate picture of purchasing power in the future.
    """)

# Financial Planning Questions
st.header("Financial Planning Questions")

# Question 1
with st.expander("What is a safe withdrawal rate?"):
    st.write("""
    A safe withdrawal rate is the percentage of your investment portfolio that you can withdraw annually without running out of money in retirement. The 4% rule is a popular guideline, suggesting that if you withdraw 4% of your savings in the first year of retirement and adjust for inflation each year thereafter, your savings should last for 30 years.
    """)

# Question 2
with st.expander("How should I allocate my assets in retirement?"):
    st.write("""
    Asset allocation in retirement should be based on your risk tolerance, income needs, and market conditions. A diversified portfolio that includes a mix of stocks, bonds, and cash is generally recommended to balance growth potential with risk management.
    """)

# Technical Questions
st.header("Technical Questions")

# Question 1
with st.expander("Can I customize the assumptions used in the models?"):
    st.write("""
    Yes, the tool allows you to customize various assumptions, including return rates, inflation rates, and withdrawal strategies, to better reflect your personal financial situation and expectations.
    """)

# Question 2
with st.expander("How often is the data updated in the FIRE tool?"):
    st.write("""
    The data in the FIRE tool is updated regularly to reflect current market conditions and economic indicators. Users are encouraged to periodically check for updates to ensure their planning reflects the most recent data.
    """)

# Closing note
st.markdown("""
Have more questions? Feel free to reach out to me via Github.
""")
