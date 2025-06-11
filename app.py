import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler




model = pickle.load(open(r'C:\UAO\Aprendizaje Automático\modelo.pkl', 'rb'))
scaler = pickle.load(open(r'C:\UAO\Aprendizaje Automático\escalado.pkl', 'rb'))
st.set_page_config(page_title="Predicción Género Musical", layout="wide", page_icon=r'C:\UAO\Aprendizaje Automático\icons\Spotify_icon.png')
st.title("APP Predicción Género Musical")
data = pd.read_csv(r'C:\UAO\Aprendizaje Automático\data\train.csv')

st.write("""
## Esta aplicación predice el género musical de una canción basada en sus características.
""")

atristas = data['Artist Name'].unique().tolist()

def histogram(df, column, value,):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[column],
        marker_color='#1DB954',
        opacity=0.75,
        nbinsx=30,
    ))

    fig. add_vline(
        x=value.values[0],
        line_width=3,
        line_dash="dash",
        line_color="#1ED760",
        annotation_font_color="#FFFFFF",
        annotation_text=f"Valor {column}: {value.values[0]}",
    )

    fig.update_layout(
    paper_bgcolor="#121212",
    plot_bgcolor="#121212",
    font=dict(color="#FFFFFF"),
    xaxis=dict(title=column, color="#FFFFFF", gridcolor="#333333"),
    yaxis=dict(title="Frecuencia", color="#FFFFFF", gridcolor="#333333"),
    bargap=0.1,
    showlegend=False,
    title=dict(
        text=f"Distribución de {column}",
        font=dict(size=20, color="#FFFFFF"),
    ),
    width=800,
    height=400,
    )

    return fig


# Seleccione el artista
col1, col2 = st.columns(2)
with col1:
    artista = st.selectbox('Seleccione el artista', atristas)

    # Filtrar el DataFrame por el artista seleccionado
    df_artista = data[data['Artist Name'] == artista]

    # Seleccione la canción
    canciones = df_artista['Track Name'].unique().tolist()
    cancion = st.selectbox('Seleccione la canción', canciones)

    # Filtrar el DataFrame por la canción seleccionada

    df_cancion = df_artista[df_artista['Track Name'] == cancion]
    
    st.dataframe(df_cancion, hide_index=True, use_container_width=True)

    if 'predecir' not in st.session_state:
        st.session_state.predecir = False

    if st.button('Predecir Género Musical'):
        st.session_state.predecir = True


    st.markdown("### ")
    
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">

        <style>
        .button-container {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
        }

        .button-container button {
            background-color: transparent;
            color: white;
            font-size: 24px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .button-container button:hover {
            background-color: white;
            color: black;
        }
        </style>

        <div class="button-container">
            <button title="Anterior"><i class="fas fa-backward"></i></button>
            <button title="Reproducir / Pausar"><i class="fas fa-play"></i></button>
            <button title="Siguiente"><i class="fas fa-forward"></i></button>
        </div>
        """, unsafe_allow_html=True)

    # Línea de tiempo simulada
    st.markdown("### ")

    st.markdown("""
    <div style="width:100%; background-color:#333; height:8px; border-radius:5px; margin-top:10px; position:relative;">
        <div style="width:40%; background-color:#FFFFFF; height:100%; border-radius:5px;"></div>
    </div>
    <div style="display:flex; justify-content:space-between; color:#888; font-size:12px;">
        <span>1:20</span><span>3:45</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.session_state.predecir:
        features = df_cancion[['Popularity', 'danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'duration_in min/ms']]

        col_features = features.columns.tolist()
        # Escalar las características
        features = scaler.transform(features)
        prediction = model.predict(features)
        

        mapa = {
            0: 'Folk',
            1: 'Alt_Music',
            2: 'Blues',
            3: 'Bollywood',
            4: 'Country',
            5: 'HipHop',
            6: 'Indie_Alt',
            7: 'Instrumental',
            8: 'Metal',
            9: 'Pop',
            10: 'Rock'
        }
        prediction = [mapa[p] for p in prediction]

        # Mostrar la predicción

        st.write(f"**Género musical predicho:** {prediction[0]}")

        # Mostrar las características de la canción seleccionada
        st.write("**Características de la canción seleccionada:**")
        st.write('Link de la canción:')

        feature = st.selectbox('Seleccione una característica para ver su distribución', col_features)
        st.plotly_chart(histogram(data, feature, df_cancion[feature]), use_container_width=True)


