import streamlit as st

from src.manage_files import load_housing_price_data
from src.corr_and_pps import plot_corr_pearson_pps_st

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def price_study_body():

    # load data
    df = load_housing_price_data()

    # 7 variables that correlate to Sale Price, uncovered in the Variable Study notebook
    corr_vars = ['1stFlrSF',
    'GarageArea',
    'GarageYrBlt',
    'GrLivArea',
    'OverallQual',
    'TotalBsmtSF',
    'YearBuilt']

    st.write("### House Sale Price Study")
    st.info(
        "* The Client is interested in discovering how the house attributes "
        "correlate with the sale price.\n"
        "* Data visualizations of the "
        "correlated variables against the sale price can help show that."
        )

    # inspect data
    if st.checkbox("Inspect house price data:"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            "The first 5 rows are displayed below.")

        st.write(df.head(5))

    st.write("---")

    # Correlation Study Summary
    st.success(
        "* Correlation studies were conducted, using the Pearson and Spearman methods "
        "to better understand the correlative relationship between the dataset variables the sale price.\n"
        "* By selecting the top 6 correlated features from both the Spearman and Pearson methods, we can \n"
        " conclude that the most correlated variables are: \n"
        "**1stFlrSF, GarageArea, GrLivArea, GarageYrBlt, OverallQual, "
        "TotalBsmtSF, YearBuilt**"
    )

    if st.checkbox("Inspect Spearman and Pearson Correlation"):
        plot_corr_pearson_pps_st(df)

    # Text based on "sale price study" notebook
    # "Conclusions and Next steps" section
    st.info(
        "### The correlations and plots interpretation converge.\n"
        "The following are the variables isolated in the"
        " correlation study:\n"
        "* 1stFlrSF: First Floor square feet.\n"
        "* GarageArea: Size of garage in square feet.\n"
        "* GrLivArea: Above grade (ground) living area square feet.\n"
        "* OverallQual: Rates the overall quality of the material "
        "and finish of the house when constructed / refurbished.\n"
        "* TotalBsmtSF: Total square feet of basement area.\n"
        "* YearBuilt: Original construction date (1872 to 2010).\n\n"
        "The plots show that the variables isolated in the "
        "correlation study, do indeed have a strong correlation and "
        "possibly strong predictive power for Sale Price for these houses.\n"
    )

    # Code copied from "sale price study" notebook
    # "EDA on the Correlated Variable List" section
    df_eda = df.filter(corr_vars + ['SalePrice'])

    # Individual plots per variable
    if st.checkbox("Variable correlation to Sale Price"):
        variable_correlation_to_sale_price(df_eda, corr_vars)


def variable_correlation_to_sale_price(df_eda, corr_vars):
    # function created using "sale price study" notebook
    # "Visualize variable correlation to Sale Price" section
    target_var = 'SalePrice'
    for col in corr_vars:
        plot_numerical(df_eda, col, target_var)
        st.write("\n\n")


def plot_numerical(df, col, target_var):
    # function created using "sale price study" notebook
    # "Visualize variable correlation to Sale Price" section

    fig, axes = plt.subplots(figsize=(15, 8))
    sns.regplot(data=df, x=col, y=target_var)
    plt.title(f"{col}", fontsize=20)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()

