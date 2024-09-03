import streamlit as st

st.write("""
    ## Intro to FIRE and Limitations of Traditional Models
    The Financial Independence Retire Early (FIRE) movement encourages people to rethink traditional personal finance, focusing on achieving financial independence and retiring early. This project aims to enhance the FIRE toolkit by introducing a probability-based tool that accounts for the complexities of real-world economic conditions, moving beyond conventional deterministic models to provide a more nuanced understanding of financial planning.
""")

st.image(
    "FIRE.jpg",
    caption="Core concept of draw-down. Speed up Build-up, and minimise Withdrawal.",
    use_column_width=True,
)

st.write("""Traditional FIRE models often oversimplify financial forecasting by assuming constant growth rates, ignoring the economic market's inherent fluctuations. This oversimplification can lead to inaccurate financial projections. By leveraging historical data, this project generates stochastic outcome probabilities, offering a more comprehensive view of potential economic trajectories and explicitly addressing risk.
""")

st.image(
    "deterministic vs stochastic.png",
    caption="Difference between deterministic and stochastic models.",
    use_column_width=True,
)
