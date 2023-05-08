'''
This file and the contents have been taken from the
Churnometer walk through Project 2 and customized for
this project
'''
import streamlit as st


def summary_body():    
    st.info(
        "## Introduction\n\n"
        "This is the final project that I am undertaking as part of the\n"
        "Code Institute Full Stack Developer program. This project deals\n"
        "with the topics of Machine Learning, Deep Learning, Python\n"
        "development, business case analysis, future API integration and dashboard\n"
        "development using Streamlit.\n\n   "

        "## Project Objective\n\n"
        "The objective of this project is to create a Machine Learning\n"
        "model that can predict the sale price of a house in Ames, Iowa.\n\n"

        "## Project Dataset\n"
        "We will use a public dataset of house prices for Ames, Iowa,\n"
        "sourced from Kaggle. The dataset consists of 1,460 observations\n"
        "and 24 variables that describe (almost) every aspect of residential\n"
        "homes in Ames, Iowa.\n\n"
        "The dataset contains information on the house's features, such as\n"
        "the number of bedrooms, bathrooms, and square footage, as well as\n"
        "information on the lot, such as the size and shape of the lot, and\n"
        "other important details like the age of the house, condition, and\n"
        "location.\n\n"
        "This dataset will enable us to build a model that predicts the sale\n"
        "price of a house in Ames, Iowa, based on its attributes. The dataset\n"
        "is provided by\
            [Kaggle.com](https://www.kaggle.com/datasets/codeinstitute/housing-prices-data)."
    )
    st.success(
        "## Business Requirements\n\n"
        "The business requirements are outlined below:\n"
        "* Perform a correlation and/or PPS study to investigate the most relevant variables correlated to the sale price.\n"
        "* Deliver an ML system that can predict the summed sale price of Lydia's four inherited properties, as well as any other house in Ames, Iowa.\n"
        "* Deliver either a conventional ML or Neural network based system.variables against the sale price.\n"
        "* Develop a dashboard that allows Lydia to explore how the house attributes correlated with the sale price using data visualizations.\n"
        "* Consider changing from regression to classification if suitable/required.\n"
        "* Perform an extensive hyperparameter search for a given algorithm.\n\n"
        )

    st.write(
        "* For additional information, please visit and **read** the "
        "[Project README file.]"
        "(https://github.com/MattMurnaghan/House-Price-Predictor#hypothesis-and-test)"
        )

    st.info(
        "**Dataset Description Table**\n\n"
        " --- \n"
        "|Variable|Meaning|Units|\n"
        "|:----|:----|:----|\n"
        "|1stFlrSF|First Floor square feet|(Min - Max > Sq. ft.) "
        "334 - 4692|\n"
        "|2ndFlrSF|Second floor square feet|(Min - Max > Sq. ft.) "
        "0 - 2065|\n"
        "|BedroomAbvGr|Bedrooms above grade (does NOT include "
        "basement bedrooms)|(Min - Max > Bedrooms) 0 - 8|\n"
        "|BsmtExposure|Refers to walkout or garden level walls|Gd: "
        "Good Exposure; Av: Average Exposure; Mn: Mimimum Exposure; "
        "No: No Exposure; None: No Basement|\n"
        "|BsmtFinType1|Rating of basement finished area|GLQ: Good "
        "Living Quarters; ALQ: Average Living Quarters; BLQ: Below "
        "Average Living Quarters; Rec: Average Rec Room; LwQ: Low "
        "Quality; Unf: Unfinshed; None: No Basement|\n"
        "|BsmtFinSF1|Type 1 finished square feet|(Min - Max > Sq. ft.) "
        "0 - 5644|\n"
        "|BsmtUnfSF|Unfinished square feet of basement area|(Min - "
        "Max > Sq. ft.) 0 - 2336|\n"
        "|TotalBsmtSF|Total square feet of basement area|(Min - "
        "Max > Sq. ft.) 0 - 6110|\n"
        "|GarageArea|Size of garage in square feet|(Min - Max > "
        "Sq. ft.) 0 - 1418|\n"
        "|GarageFinish|Interior finish of the garage|Fin: Finished; "
        "RFn: Rough Finished; Unf: Unfinished; None: No Garage|\n"
        "|GarageYrBlt|Year garage was built|(Min - Max > Year) "
        "1900 - 2010|\n"
        "|GrLivArea|Above grade (ground) living area square feet|"
        "(Min - Max > Sq. ft.) 334 - 5642|\n"
        "|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: "
        "Typical/Average; Fa: Fair; Po: Poor|\n"
        "|LotArea| Lot size in square feet|(Min - Max > Sq. ft.) "
        "1300 - 215245|\n"
        "|LotFrontage| Linear feet of street connected to property|"
        "(Min - Max > Lin. ft.) 21 - 313|\n"
        "|MasVnrArea|Masonry veneer area in square feet|(Min - Max "
        "> Sq. ft.) 0 - 1600|\n"
        "|EnclosedPorch|Enclosed porch area in square feet|"
        "(Min - Max > Sq. ft.) 0 - 286|\n"
        "|OpenPorchSF|Open porch area in square feet|(Min - "
        "Max > Sq. ft.) 0 - 547|\n"
        "|OverallCond|Rates the overall condition of the house|"
        "10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; "
        "6: Above Average; 5: Average; 4: Below Average; 3: Fair; "
        "2: Poor; 1: Very Poor|\n"
        "|OverallQual|Rates the overall material and finish of the "
        "house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: "
        "Good; 6: Above Average; 5: Average; 4: Below Average; 3: "
        "Fair; 2: Poor; 1: Very Poor|\n"
        "|WoodDeckSF|Wood deck area in square feet|(Min - Max > "
        "Sq. ft.) 0 - 736|\n"
        "|YearBuilt|Original construction date|(Min - Max > Year) "
        "1872 - 2010|\n"
        "|YearRemodAdd|Remodel date (same as construction date "
        "if no remodeling or additions)|(Min - Max > Year) 1950 - 2010|\n"
        "|SalePrice|Sale Price|(Min - Max > Price in $) 34900 - 755000|\n\n"
        " ---"
        )