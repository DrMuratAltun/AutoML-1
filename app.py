import streamlit as st
from pycaret.classification import setup, compare_models, pull, save_model
import pandas as pd
import os

def main():
    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)

    with st.sidebar:
        st.image('https://leilaabdel.com/img/deep_learning_course_pic.png')
        st.title('AutoML Classification')
        choice = st.radio('Navigation', ['Upload', 'EDA', 'Modelling', 'Download'])

    if choice == 'Upload':
        file_uploader_ui()

    elif choice == 'EDA' and 'df' in locals():
        eda_ui(df)

    elif choice == 'Modelling' and 'df' in locals():
        modelling_ui(df)

    elif choice == 'Download':
        download_ui()

def file_uploader_ui():
    st.title('Upload your data file')
    file = st.file_uploader('Upload your data')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df.head())

def eda_ui(df):
    st.title('Exploratory Data Analysis')
    from streamlit_pandas_profiling import st_profile_report
    profile_df = df.profile_report()
    st_profile_report(profile_df)

def modelling_ui(df):
    target_col = st.selectbox('Choose the target column', df.columns)
    if st.button('Train model'):
        setup(data=df, target=target_col)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model.pkl')

def download_ui():
    try:
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download the best model', f, 'best_model.pkl')
    except Exception as e:
        st.error(f"Error downloading the model: {str(e)}")

if __name__ == "__main__":
    main()
