'''
This file and the contents have been taken from the
 Churnometer Walkthrough Project 2 and customised for
 this project
'''
import streamlit as st

from app_pages.multi_page import MultiPage

# load pages scripts
from app_pages.summary import summary_body
from app_pages.price_study import price_study_body
from app_pages.price_predictor import price_predictor_body
from app_pages.hypothesis import hypothesis_body
from app_pages.predict_sale_price import predict_sale_price_body

# Create an instance of the app
app = MultiPage(app_name="Price Predictor - Houses")

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", summary_body)
app.add_page("House Sale Price Study", price_study_body)
app.add_page("Price Predictor", price_predictor_body)
app.add_page("Project Hypothesis and Validation", hypothesis_body)
app.add_page("ML: House Sale Price Prediction", predict_sale_price_body)

app.run()