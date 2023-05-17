import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.manage_files import load_clean_data, load_pkl_file
from src.pipeline_performance import regression_performance
from src.pipeline_performance import regression_plots
from graphviz import Digraph

def predict_sale_price_body():

    # load sale price pipeline files
    version = 'v1'
    path = f"outputs/pipelines/predict_saleprice/{version}"
    sale_price_pipeline = load_pkl_file(f"{path}/best_regressor_pipeline.pkl")
    feature_importance = pd.read_csv(f"{path}/feature_importance.csv")
    feature_importance_plot = plt.imread(f"{path}/feature_importance.png")
    X_train = pd.read_csv(f"{path}/X_train.csv")
    X_test = pd.read_csv(f"{path}/X_test.csv")
    y_train = pd.read_csv(f"{path}/y_train.csv")
    y_test = pd.read_csv(f"{path}/y_test.csv")

    st.write("### ML Pipeline: Predict House Sale Price")
    # display pipeline training summary conclusions
    st.write(
        f"The Regressor Model employed in this prediction is the RandomForrest Regressor.\n"
        f"* Both feature selection and PCA produced similar results and meet "
        f"business requirement 1.\n"
        f"* Feature selection achieved an R2 Score: 0.861 on the train set and "
        f"0.836 on the test set.\n"
        f"* The Client has required an R2 Score of 0.75+.\n"
        )
    st.write("---")

    # show pipeline steps
    st.write("### ML pipeline to predict sale price")
    pipeline_steps = sale_price_pipeline.steps

    # Create a Graphviz graph
    dot = Digraph()
    dot.node('A', f'{pipeline_steps[0][0]}\n{pipeline_steps[1][0]}\n{pipeline_steps[2][0]}')
    dot.node('B', pipeline_steps[3][0])
    dot.node('C', pipeline_steps[4][0])
    dot.edges(['AB', 'BC'])

    st.write("The ML pipeline to predict the sale price of a house is shown below:")
    # Display the graph in Streamlit
    st.graphviz_chart(dot.source)
    st.write("The final pipeline only required 3 transformers before scaling and\
                modeling as we are only assessing 4 features, shown below:")
    # show best features
    st.write("### The features used to train the model and their importance:")
    feature_importance = feature_importance['Feature'].sort_values().to_list()
    for feat in feature_importance:
        st.write(f"* {feat}")

    st.image(feature_importance_plot)
    st.write("---")

    # evaluate pipeline performance
    st.write("### Evaluating the Pipeline Performance.")
    regression_performance(X_train, y_train, X_test, y_test, sale_price_pipeline)
    st.write("---")
    regression_plots(X_train, y_train, X_test, y_test, sale_price_pipeline)
    st.write("---")