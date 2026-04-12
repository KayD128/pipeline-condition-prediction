import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import plotly.express as px
import sklearn

st.set_page_config(page_title="Pipeline Maintenance Detection App", page_icon="🎈", layout="wide")

pipeline_path = 'pipeline_model2.pkl'
model = joblib.load(pipeline_path)

col1, col2, col3 = st.columns([0.5, 1, 0.5])
with col2:
    image_banner = Image.open('pipeline.jpeg').resize((700, 300))
    st.image(image_banner)

st.markdown("<h1 style='text-align: center;'>Pipeline Maintenance Detection App</h1>", unsafe_allow_html=True)

def load_feature_importance(file_path):
    return pd.read_excel(file_path)

final_fi2 = load_feature_importance('feature_importance_pipe.xlsx')

left_col, space, right_col = st.columns([1, 0.2, 1])

with left_col:
    st.subheader("Importance of each column")

    final_fi_sorted = final_fi2.sort_values('Importance Score', ascending=True)

    fig = px.bar(
        final_fi_sorted,
        x='Importance Score',
        y='Feature',
        orientation='h',
        # title = 'Feature Importance Scores from Random Forest Model',
        labels={'Importance Score': 'Importance', 'Feature': 'Feature'},
        text='Importance Score',
        color_discrete_sequence=['#1f77b4']
    )

    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        template='plotly_white',
        height=1000,
        yaxis=dict(tickmode='linear'),
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("Predict Pipeline Condition")

    def get_user_input():
        pipe_size = st.number_input("Enter Pipe Size (mm): ", min_value=0, value=1000)
        thickness = st.number_input("Enter Thickness (mm): ", min_value=0.0, value=20.0)
        max_pressure = st.number_input("Enter Maximum Pressure (psi): ", min_value=0, value=1500)
        temperature = st.number_input("Enter Temperature (°C): ", min_value=0.0, value=60.0)
        corrosion_impact = st.slider("Enter Number of Corrosion Impact (%): ", min_value=0.0, max_value=100.0, value=60.0)
        thickness_loss = st.number_input("Enter Thickness Loss (mm): ", min_value=0.0, value=5.0)
        material_loss = st.slider("Enter Material Loss (%): ", min_value=0.0, max_value = 100.0, value=60.0)
    
        time = st.number_input("Enter Time (years): ", min_value=0, value=10)

        material = st.selectbox('Enter Material',['Carbon Steel', 'PVC', 'HDPE', 'Fiberglass', 'Stainless Steel'])
        grade = st.selectbox('Enter Grade',['ASTM A333 Grade 6', 'ASTM A106 Grade B', 'API 5L X52',
            'API 5L X42', 'API 5L X65'])

        user_data = {
            'Pipe_Size_mm': pipe_size,
            'Thickness_mm': thickness,
            f'Material_{material}': 1,
            f'Grade_{grade}': 1,
            'Max_Pressure_psi': max_pressure,
            'Temperature_C': temperature,
            'Corrosion_Impact_Percent': corrosion_impact,
            'Thickness_Loss_mm': thickness_loss,
            'Material_Loss_Percent': material_loss,
            'Time_Years': time
        }

        return user_data

    user_data = get_user_input()

    def prepare_input(data, feature_list):
        input_data = {feature: data.get(feature, 0) for feature in feature_list}
        return pd.DataFrame([input_data])
        # np.array([list(input_data.values())])

    features = [
        'Pipe_Size_mm', 
        'Thickness_mm',
        'Max_Pressure_psi',
        'Temperature_C',
        'Corrosion_Impact_Percent',
        'Thickness_Loss_mm',
        'Material_Loss_Percent',
        'Time_Years',
        'Grade_API 5L X42',
        'Grade_API 5L X52',
        'Grade_API 5L X65',
        'Grade_ASTM A106 Grade B',
        'Grade_ASTM A333 Grade 6',
        'Material_Carbon Steel',
        'Material_Fiberglass',
        'Material_HDPE',
        'Material_PVC',
        'Material_Stainless Steel'
    ]
        # user_data = get_user_input()
        # user_data

    if st.button("Predict pipe Condition"):
        input_array = prepare_input(user_data, features)
        prediction = model.predict(input_array)
        st.subheader("Predicted Condition")
        value = prediction[0]
        st.write(f"{prediction[0]}")
        if value == 'Normal':
            st.success("Pipeline is in Good Condition")
        elif value == 'Moderate':
            st.warning("Pipeline is in Fair Condition")
        else:
            st.error("Pipeline is in Poor Condition")

