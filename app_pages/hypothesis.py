'''
This file and the contents have been taken from the
Churnometer walk through Project 2 and customized for
this project
'''
import streamlit as st


def hypothesis_body():

    st.success(
        "### Project Hypothesis and Validation\n\n"
        "The following are the hypotheses that I have made for this project:\n\n"
        "1. I suspect that a house with high OverallQual sells for a higher price.\n"
        "A correlation analysis between OverallQual and SalePrice can show this relationship.\n\n"
        "2. I suspect that a house with a big garage sells for a higher price.\n"
        "A correlation analysis between GarageArea and SalePrice can show this relationship.\n\n"
    )