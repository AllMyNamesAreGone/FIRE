import time
import numpy as np
import pandas as pd
import streamlit as st

_LOREM_IPSUM = """
    The FIRE (Financial Independence Retire Early) movement has sparked a wave of interest in reimagining traditional approaches to personal finance, inspiring individuals to pursue financial independence & early retirement. Recognising the limitations of conventional FIRE tools, this project seeks to introduce a probability-based tool to the FIRE movement.

    Traditional FIRE models often oversimplify financial projections by assuming deterministic rates over time, neglecting the nuanced fluctuations inherent in economic markets. This project breaks away from this paradigm by harnessing historical data to generate stochastic outcome probabilities, offering a more comprehensive view of economic trajectories with risk explicitly expressed. Tailored specifically to the Australian context, our model integrates key factors such as local inflation, cash, bond rates, and global stock rates, catering to the unique intricacies of the Australian financial landscape.

    Another cornerstone of the project lies in its ability to dissect net worth into distinct categories of assets & liabilities, enabling a granular analysis of financial health. By offering tailored asset allocation strategies for both Super & non-Super funds, our tool ensures alignment with individual risk profiles, investment timeline, taxation frameworks, and return objectives. Additionally, the modeling framework incorporates dynamic spending patterns, anticipated wage growth, and the potential income-generating capacity of non-productive assets, providing users with a holistic understanding of their financial well-being.

    As the models are continually to refined & optimised, the hope is that this tool enhances accuracy & relevance in a dynamic financial & economic landscape. Through ongoing improvements & future directions, the hope is for the project to empower individuals within the FIRE movement to make informed decisions & strategic plans, ultimately paving the way towards financial freedom & early retirement.

    * Traditional FIRE models often oversimplify financial forecasting by assuming constant rates over time.
    * This tool uses historical data to generate the probability various outcomes, providing a more comprehensive view with risk explicitly expressed.
    * Tailored towards Australian-specific factors like inflation, cash, bond rates, and global stock rates.
    * Caters to the two-tier Superannuation retirement setup in Australia.
    * Integrates a mortality component, predicting the likelihood of being liquid, broke, or dead.
    * Stretch goals include implementing Optimal Portfolio Distribution and integrating with monthly income/expenses/savings summaries.
"""


def stream_data():
    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.02)
st.write_stream(stream_data)
