import streamlit as st

# Page title
st.title("Modelling")

# Introduction paragraph
st.markdown("""
The economic modelling framework employs a diverse suite of models designed to capture the intricacies of financial markets and macroeconomic indicators. Key components of our model portfolio include:
""")

# Section for Global Stock Market
st.subheader("Global Stock Market")
st.markdown("""
Proxied by the **FTSE All World index**, an **ARIMA(0,2,2) model** is utilised to forecast trends and fluctuations in global stock markets.
""")

# Placeholder for an image related to Global Stock Market model (replace 'image_path' with your actual image file)
st.image("Stock ACF.png", use_column_width=True)
st.image("Stock QQ.png", use_column_width=True)

# Section for Australian 10-Year Bond Rates
st.subheader("Australian 10-Year Bond Rates")
st.markdown("""
Serving as a proxy for the **Australian risk-free rate**, a **random walk model** is utilised to analyse and predict movements in long-term bond rates.
""")

# Placeholder for an image related to Australian 10-Year Bond Rates model
st.image("Bond ACF.png", use_column_width=True)
st.image("Bond QQ.png", use_column_width=True)

# Section for Cash Rate Targets
st.subheader("Cash Rate Targets")
st.markdown("""
Extracted from the **Reserve Bank of Australia (RBA)**, a **random walk model** is employed to forecast changes in the RBA’s cash rate targets.
""")

# Placeholder for an image related to Cash Rate Targets model
st.image("Cash ACF.png", use_column_width=True)
st.image("Cash QQ.png", use_column_width=True)

st.code("""
Column: FTSE All World
Series: ts_data
ARIMA(0,2,2)

Coefficients:
          ma1     ma2
      -1.1032  0.5984
s.e.   0.2182  0.2422

sigma^2 = 1793:  log likelihood = -87.5
AIC=181   AICc=182.85   BIC=183.5

Training set error measures:
                   ME      RMSE        MAE      MPE      MAPE         MASE        ACF1
Training set 3.757369   37.6205   30.00522  2.50892  11.08228    0.8513959  0.03718014


Column: AU 10-Yr Bond Yield
Series: ts_data
ARIMA(0,1,0)

sigma^2 = 0.9374:  log likelihood = -33.28
AIC=68.56   AICc=68.74   BIC=69.73

Training set error measures:
                     ME      RMSE         MAE        MPE        MAPE         MASE         ACF1
Training set -0.1231536 0.9486122   0.7223664   -7.28807    23.39129    0.9603808   -0.1843379


Column: RBA Cash Rate Target
Series: ts_data
ARIMA(0,1,0)

sigma^2 = 1.164:  log likelihood = -35.87
AIC=73.75   AICc=73.93   BIC=74.93

Training set error measures:
                  ME     RMSE       MAE       MPE       MAPE      MASE      ACF1
Training set -0.0258 1.057001 0.7449609 -31.52632   48.44513 0.9602578 0.0847086
""")
