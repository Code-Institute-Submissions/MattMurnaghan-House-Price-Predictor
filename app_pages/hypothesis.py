'''
This file and the contents have been taken from the
Churnometer walk through Project 2 and customized for
this project
'''
import streamlit as st


def hypothesis_body():

    st.info(
        "### Project Hypothesis\n\n"
        "The following are the hypotheses that I have made for this project:\n\n"
        "1. I suspect that a house with high OverallQual sells for a higher price.\n"
        "A correlation analysis between OverallQual and SalePrice can show this relationship.\n\n"
        "2. I suspect that a house with a big garage sells for a higher price.\n"
        "A correlation analysis between GarageArea and SalePrice can show this relationship.\n\n"
    )
    st.write("---")
    st.write("### Hypothesis Validation")
    st.write("The following are the results of the hypothesis validation:\n\n"
            "1. The correlation analysis between OverallQual and SalePrice shows a strong positive correlation.\n"
            "* This means that the higher the OverallQual, the higher the SalePrice.\n\n"
            "* We can see this from the scatter plot below, the relationship is linear.\n\n")
    st.image("images/VariableStudy/OverallQual-vs-SalePrice.png")

    st.write("\n\n2. The correlation analysis between GarageArea and SalePrice shows a moderate positive correlation.\n")
    st.write("* This means that the higher the GarageArea, the higher the SalePrice.\n\n")
    st.write("* We can see this from the scatter plot below, the relationship is linear.\n\n")
    st.write("* There are definitely some outliers but the relationship has a relatively small errror margin.\n\n")
    st.image("images/VariableStudy/GarageArea-vs-SalePrice.png")