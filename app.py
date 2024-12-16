import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Model and System
main_path = os.path.abspath(os.getcwd())
path_model = os.path.join(main_path, 'xgboost_model.sav')
#with open(path_model, 'rb') as file:
#    xgboost = pickle.load(file)

st.set_page_config(page_title = 'Ecommerce Customer Churn Predictor', initial_sidebar_state="expanded")
st.title('Ecommerce Customer Churn Predictor')
st.markdown("### Application of Machine Learning")
st.image('./image/Churn-Prediction-scaled.jpg')


# Sidebar ===============================================================
st.sidebar.title('Hi ! I am Huwaida 👋')
st.sidebar.write('Ini adalah proyek Data Science-ku 📊')

# Sidebar tabs
tab = st.sidebar.radio(
    "Pilih Tab: ",
    ("📚 Background", "📈 Model Performance", "🔮 Predictor"))

if tab == "📚 Background":
    st.sidebar.header("📚 Background")
    st.sidebar.write("""
        Menjelaskan apa itu *customer churn* dan mengapa hal ini penting untuk bisnis e-commerce. 💡
    """)

elif tab == "📈 Model Performance":
    st.sidebar.header("📈 Model Performance")
    st.sidebar.write("""
        Menampilkan bagaimana model menangani permasalahan *churn* dan mengevaluasi kinerjanya.
    """)

elif tab == "🔮 Predictor":
    st.sidebar.header("🔮 Predictor")
    st.sidebar.write("""
        Menunjukkan hasil prediksi untuk pelanggan dan bagaimana hasil tersebut dapat digunakan untuk tindakan preventif.
    """)


tab1, tab2, tab3 = st.tabs(["📚 Background", "📈 Model Performance", "🔮 Predictor"])

# TAB 1 ==================================================================
with tab1:
    st.title('Background')

    st.subheader("Context")
    st.write("""
Suatu perusahaan E-commerce menawarkan berbagai layanan di platform mereka. Agar customer terus menggunakan layanan tersebut, berbagai cara dilakukan perusahaan untuk mempertahankan customer mereka menjadi customer yang loyal. 
Dengan kehilangan customer, perusahaan juga mengalami penurunan revenue. Selain itu, biaya untuk mempertahankan customer (**retention cost**) cenderung lebih rendah dibandingkan biaya mendapatkan customer baru (**acquisition cost**). 
Untuk itu, perusahaan mengandalkan **Data Scientist** untuk memprediksi customer yang akan churn dan mengidentifikasi berbagai indikator customer berisiko churn, sehingga kerugian yang akan dialami perusahaan dapat diminimalkan.
""")

    st.subheader("Problem Statement")
    st.write("""
Bagaimana cara memprediksi **apakah seorang pelanggan akan churn** sehingga:
1. Perusahaan dapat secara proaktif mempertahankan pelanggan yang bernilai tinggi (**valuable customers**).
2. Perusahaan dapat meningkatkan **revenue** bisnis secara keseluruhan.
""")

    st.subheader("Machine Learning Objective")
    st.write("""
Tujuan dari pembangunan model **Machine Learning** adalah:
1. Meminimalkan kehilangan revenue akibat customer churn dengan mengidentifikasi **high-risk customers**.
2. Meningkatkan strategi retensi (**retention strategies**) yang lebih efektif.
""")

    st.subheader("Analytic Approach")
    st.write("""
Dalam project ini, akan dilakukan beberapa langkah berikut:
- Analisis data untuk mencari pola (pattern) churn.
- Mengembangkan model prediktif menggunakan algoritma **supervised learning (classification)**.
- Memprediksi probabilitas seorang customer akan churn atau tidak.
""")

    st.subheader("Metric Evaluation")
    st.markdown("""
- **Type 2 Error (False Negative)**:  
  Gagal memprediksi customer yang akan churn. Dampaknya, perusahaan kehilangan revenue dan harus mengeluarkan **acquisition cost** untuk mendapatkan customer baru.

- **Type 1 Error (False Positive)**:  
  Salah memprediksi customer yang tidak churn menjadi churn. Dampaknya, perusahaan tidak kehilangan revenue tetapi harus mengeluarkan **retention cost** yang tidak diperlukan.
""")
    
    st.markdown("---")
    st.markdown("© 2024 Huwaida Nur Asysyifa. All rights reserved.")

# TAB 2 ==================================================================
with tab2 :
    st.write("""
    Model yang digunakan pada predictor app ini adalah model Extreme Gradient Boosting Machine, dengan metode resampling SMOTE.
    berikut akan ditunjukkan performa model hasil training 
             """) 
    st.subheader('Brenchamrking Models using K-Fold and Test Data')
    df_cv_combined = pd.read_csv('./table/df_cv_combined.csv').drop(columns=['Unnamed: 0'], errors='ignore')
    st.dataframe(df_cv_combined.style.apply(lambda x: ['background: yellow' if isinstance(val, (int, float)) else '' for val in x], axis=0).highlight_max(axis=0))

    # .style.apply(lambda x: ['background: yellow' if isinstance(val, (int, float)) else '' for val in x], axis=0).highlight_max(axis=0))
    st.subheader('Benchmarking Models with The 3 Best Models and Resampling Methods.')
    df_resampling = pd.read_csv('./table/df_resampling.csv').drop(columns=['Unnamed: 0'], errors='ignore')
    st.dataframe(df_resampling.style.apply(lambda x: ['background: yellow' if isinstance(val, (int, float)) else '' for val in x], axis=0).highlight_max(axis=0))

    st.subheader('XGBoost Model with Resampling Methods')
    dfEvaluate_combined = pd.read_csv('./table/dfEvaluate_combined.csv').rename(columns = {'Unnamed: 0' : 'resampling'}).drop(columns=['mean'], errors='ignore')
    st.dataframe(dfEvaluate_combined.style.apply(lambda x: ['background: yellow' if isinstance(val, (int, float)) else '' for val in x], axis=0).highlight_max(axis=0))

    st.subheader('Tuning XGBoost Model')
    params = {
    'modeling__subsample': [0.6, 0.8, 1.0],             # Subsampling data untuk setiap tree
    'modeling__scale_pos_weight':[3, 5, 10],            # Mengatasi class imbalance
    'modeling__n_estimators': [150, 200],      # Jumlah pohon
    'modeling__min_child_weight': [1, 3, 5],            # Mencegah overfitting
    'modeling__max_depth': [7, 10, 15],               # Kedalaman pohon
    'modeling__max_delta_step': [0, 1, 3],              # Stabilitas langkah
    'modeling__max_bin': [128, 256, 512],               # Jumlah bin histogram
    'modeling__gamma': [0, 0.1, 0.5],                # Regularisasi
    'modeling__eta': [0.01, 0.05, 0.1],            # Learning rate
    'modeling__colsample_bytree': [0.6, 0.8, 1.0],       # Fitur subset untuk setiap tree
    'modeling__reg_alpha': [0, 0.1, 0.5], 
    'modeling__reg_lambda': [1, 1.5, 2]
    }
    df_params = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in params.items()]))
    st.dataframe(df_params)

    st.subheader('Randomized Search')
    st.image('./image/randomsearch.png')



# Classification report data for both models
    data = {
    'Model': ['Default XGBoost', 'Tuned XGBoost'],
    'Class 0 Precision': [0.94, 0.96],
    'Class 1 Precision': [0.80, 0.74],
    'Class 0 Recall': [0.97, 0.95],
    'Class 1 Recall': [0.67, 0.76],
    'Class 0 F1-Score': [0.96, 0.95],
    'Class 1 F1-Score': [0.73, 0.75],
    'Accuracy': [0.92, 0.92],
    'Macro Avg F1-Score': [0.84, 0.85],
    'Weighted Avg F1-Score': [0.92, 0.92],
    'PR AUC Score': [0.80, 0.80]
}

# Convert data to DataFrame
    class_report = pd.DataFrame(data)

# Display the DataFrame
    st.write("### Classification Report & PR AUC Scores")
    st.dataframe(class_report)

    data2 = {
    'Metric': ['Best Model (RandomizedSearchCV)', 'Benchmark (Train Set)', 'Tuned Model (Train Set)', 
               'Benchmark (Test Set)', 'Tuned Model (Test Set)'],
    'Score': [0.7513, 0.6111, 0.7336, 0.7303, 0.7513]
}

# Convert the data into a DataFrame
    df_performance = pd.DataFrame(data2)

# Display the DataFrame in Streamlit
    st.write("### Model Performance Scores")
    st.dataframe(df_performance)

    st.write('### Optimized Thresholds')
    df_thresholds = pd.read_csv('./table/df_thresholds.csv').drop(columns=['Unnamed: 0'], errors='ignore')
    st.dataframe(df_thresholds.drop(columns=['Unnamed: 0'], errors='ignore').iloc[50:56])
    st.write('Setelah dilakukan optimisasi threshold, diperoleh **thresholds 0.55 dengan f1 score terbaik 0.768**')
# Create a figure
    st.line_chart(df_thresholds.set_index('threshold')['f1'])

    optimal_threshold = df_thresholds.loc[54, 'threshold']
    optimal_f1 = df_thresholds.loc[54, 'f1']

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df_thresholds['threshold'], df_thresholds['f1'], label='F1 Score', color='blue', lw=2)
    ax.scatter(optimal_threshold, optimal_f1, color='black', label='Optimal Threshold', zorder=5)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs Threshold")
    ax.legend()
    st.image('./image/thresholds-f1.png')

    st.write('### Confusion Matrix')
    st.image('./image/confusion_matrix.png')    

    st.write('### Precision - Recall Curve')
    df_pr_curve = pd.read_csv('./table/df_pr_curve.csv')
    def highlight_row(row):
        if row['precision'] == 0.802326:
            return ['background-color: yellow'] * len(row) 
        return [''] * len(row)
    st.dataframe( df_pr_curve.drop(columns=['Unnamed: 0']).iloc[538:544, 0:3])
    st.image('./image/PR-curve.png')

    st.write('### Feature Importances')
    st.image('./image/feature_importances.png')

    st.write('### SHAP Value')
    st.image('./image/SHAP.png')

    st.markdown("---")
    st.markdown("© 2024 Huwaida Nur Asysyifa. All rights reserved.")

# TAB 3 ==================================================================
with tab3 : 
    st.subheader("Please input customer's features")

    def user_input_feature():
    # min_value, max_value, value, step
        Tenure = st.number_input('Tenure', 0, 61, 5,1 )
        WarehouseToHome = st.number_input('WarehouseToHome', 5, 127, 50, 1 )
        NumberOfDeviceRegistered = st.number_input('NumberOfDeviceRegistered', 0, 20, 1, 1 )
        PreferedOrderCat = st.selectbox('PreferedOrderCat', ('Laptop & Accessory', 'Fashion', 'Others', 'Mobile Phone', 'Grocery'))
        SatisfactionScore = st.selectbox('SatisfactionScore', (1, 2, 3, 4, 5))
        MaritalStatus = st.selectbox('MaritalStatus', ('Single', 'Married', 'Divorced'))
        NumberOfAddress = st.number_input('NumberOfAddress', 1, 22, 1, 1)
        Complain = st.selectbox('Complain',(1, 2))
        DaySinceLastOrder = st.number_input('DaySinceLastOrder', 0, 46, 20, 1) 
        CashbackAmount = st.number_input('CashbackAmount', 0, 325, 100, 1)

        df = pd.DataFrame({
            'Tenure': [Tenure],
            'WarehouseToHome': [WarehouseToHome],
            'NumberOfDeviceRegistered': [NumberOfDeviceRegistered],
            'PreferedOrderCat': [PreferedOrderCat],
            'SatisfactionScore': [SatisfactionScore],
            'MaritalStatus': [MaritalStatus],
            'NumberOfAddress': [NumberOfAddress],
            'Complain': [Complain],
            'DaySinceLastOrder': [DaySinceLastOrder],
            'CashbackAmount': [CashbackAmount]
        })
        return df

    user_df = user_input_feature()
    user_df.index = ['value']

    # model_loaded = pickle.load(open('.sav', 'rb'))
    # kelas = model_loaded.predict


    loaded_model = joblib.load('xgboost_model.pkl')

    if st.button('Model Predict'):
        data = user_df.values.reshape(1, -1)
        column_names = ['Tenure', 'WarehouseToHome', 'NumberOfDeviceRegistered',
       'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus',
       'NumberOfAddress', 'Complain', 'DaySinceLastOrder', 'CashbackAmount']

        data_df = pd.DataFrame(data, columns=column_names)
        result = loaded_model.predict(data_df)

        # Make prediction
        prob = loaded_model.predict_proba(data_df)
        churn_prob = prob[0][1]

        churn_probability = churn_prob 
        non_churn_probability = 1 - churn_probability

# Gauge chart
        fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value = churn_probability * 100,
        title={'text': "Churn Probability (%)"},
        gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "white"},
        'steps': [
            {'range': [0, 50], 'color': "#000000"}, 
            {'range': [50, 100], 'color': "#FFFF00"}  
        ],
        'threshold': {
            'line': {'color': "yellow", 'width': 4},
            'thickness': 0.67,
            'value': churn_probability * 100}}))

# Donut chart
        labels = ['Churn', 'Non-Churn']
        values = [churn_probability, non_churn_probability]

        fig_donut = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=0.6,
        marker=dict(colors=['yellow', 'black']),
        textinfo='label+percent'
)])
        fig_donut.update_layout(title_text="Probability Distribution")

# Display the charts in Streamlit
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.write(f'''
        Ada probabilitas sebesar {(churn_probability * 100):.2f}% customer tersebut akan churn.
                 ''')
        st.markdown("---")
        st.markdown("© 2024 Huwaida Nur Asysyifa. All rights reserved.")

# ======================================================
