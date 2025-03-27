from pyexpat import model
import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np



def get_clean_data():
    # Read the CSV file into a DataFrame
        data = pd.read_csv("data/data.csv")

    # Drop the 'Unnamed: 32' and id columns
        data = data.drop(['Unnamed: 32', 'id'], axis=1)

        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
        label,
        min_value = float(0),
        max_value = float(data[key].max()),
        value = float(data[key].mean())
        )
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop('diagnosis', axis=1)

    scaled_dict = {}

    # scale the data
    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min()
        scaled_values = (value - min_value) / (max_value - min_value)
        # store the scaled values in a dictionary
        scaled_dict[key] = scaled_values
        
    
    return scaled_dict

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ["Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity", 
                  "Concave points", "Symmetry", "Fractal dimension"]
   
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=[
                input_data["radius_mean"],
                input_data["texture_mean"],
                input_data["perimeter_mean"],
                input_data["area_mean"],
                input_data["smoothness_mean"],
                input_data["compactness_mean"],
                input_data["concavity_mean"],
                input_data["concave points_mean"],
                input_data["symmetry_mean"],
                input_data["fractal_dimension_mean"]
            ],
            theta=categories,
            fill='toself',
            name='Mean Values'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
                input_data["radius_se"],
                input_data["texture_se"],
                input_data["perimeter_se"],
                input_data["area_se"],
                input_data["smoothness_se"],
                input_data["compactness_se"],
                input_data["concavity_se"],
                input_data["concave points_se"],
                input_data["symmetry_se"],
                input_data["fractal_dimension_se"]
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
))
    fig.add_trace(go.Scatterpolar(
            r=[
                input_data["radius_worst"],
                input_data["texture_worst"],
                input_data["perimeter_worst"],
                input_data["area_worst"],
                input_data["smoothness_worst"],
                input_data["compactness_worst"],
                input_data["concavity_worst"],
                input_data["concave points_worst"],
                input_data["symmetry_worst"],
                input_data["fractal_dimension_worst"]
            ],
            theta=categories,
            fill='toself',
            name='Worst Values'
))

    fig.update_layout(
    polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
    showlegend=True,
    autosize=True
    )
    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)  # Convert the input data to a numpy array
    input_array_scaled = scaler.transform(input_array)  # Scale the input data
    prediction = model.predict(input_array_scaled)  # Make the prediction
    probabilities = model.predict_proba(input_array_scaled)[0]  # Get prediction probabilities

    st.header("Cell Cluster Prediction")

    # Display the prediction with color-coded text
    if prediction[0] == 0:
        st.markdown(
            "<h3 style='color: green;'>The cell cluster is: Benign</h3>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<h3 style='color: red;'>The cell cluster is: Malignant</h3>",
            unsafe_allow_html=True,
        )

    # Display probabilities with a progress bar
    st.write("Prediction Probabilities:")
    st.markdown(
        f"<p style='color: green;'>Probability of being Benign: {probabilities[0] * 100:.2f}%</p>",
        unsafe_allow_html=True,
    )
    st.progress(probabilities[0])  # Progress bar for benign probability

    st.markdown(
        f"<p style='color: red;'>Probability of being Malignant: {probabilities[1] * 100:.2f}%</p>",
        unsafe_allow_html=True,
    )
    st.progress(probabilities[1])  # Progress bar for malignant probability

    # Add a disclaimer
    st.markdown(
        "<p style='font-size: 12px; color: gray;'>This app can be used to help professionals diagnose breast cancer! It is not a substitute for a professional diagnosis.</p>",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction", 
        page_icon="👩🏻", 
        layout="wide",
        initial_sidebar_state="expanded"
        )
    
    # add the sidebar
    input_data = add_sidebar()

    # add the main content
    with st.container():
        st.title("Breast Cancer Prediction")
        st.write("This web app predicts whether a patient has breast cancer or not.")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer. This app predicts whether a breast mass is benign or malicious using measuremnets from your cytosis lab. You can also update the measurements by hand using the siders in the sidebar.")
        
        col1, col2 = st.columns([4,1])

        with col1:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart, use_container_width=True)
        
        with col2:
            add_predictions(input_data)
        



if __name__ == "__main__":
    main()