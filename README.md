# Music Genre Classification App

Este proyecto consiste en una aplicación web construida con Streamlit que permite predecir el género musical de una canción basándose en sus características acústicas. Fue desarrollado como parte de un proyecto de aprendizaje automático en el marco académico de la Universidad Autónoma de Occidente.

## Descripción

La aplicación permite al usuario seleccionar un artista y una canción de un conjunto de datos. A partir de variables como danceability, energy, loudness, valence, acousticness, entre otras, el modelo predice el género musical más probable. El modelo fue entrenado previamente y se descarga automáticamente al ejecutar la aplicación.

## Enlace a la aplicación

Puedes acceder a la aplicación desplegada en Streamlit Cloud en el siguiente enlace:

https://musicgenreclassificationuao.streamlit.app

## Tecnologías utilizadas

- Python 3.11  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Plotly  
- Pickle (para serialización de modelos)

## Estructura del proyecto

.
├── app.py # Aplicación principal en Streamlit
├── modelo.pkl # Modelo entrenado (descargado desde Hugging Face)
├── escalado.pkl # Escalador de características
├── data/
│ └── train.csv # Conjunto de datos base
├── icons/
│ └── Spotify_icon.png # Ícono de la aplicación
├── requirements.txt # Dependencias necesarias
└── README.md

markdown
Copiar
Editar

## Link despliegue
[https://musicgenreclassificationuao.streamlit.app](https://musicgenreclassificationuao.streamlit.app)

## link modelo
[https://huggingface.co/jmcajigas/Music_Genre_Model/tree/main](https://huggingface.co/jmcajigas/Music_Genre_Model/tree/main)

## Instrucciones para ejecución local

1. Clonar este repositorio:

git clone https://github.com/tu-usuario/aprendizaje-automatico.git
cd aprendizaje-automatico

markdown
Copiar
Editar

2. Instalar las dependencias:

pip install -r requirements.txt

markdown
Copiar
Editar

3. Ejecutar la aplicación:

streamlit run app.py

shell
Copiar
Editar

## Autores

Juan M. Cajigas  
Levi E. Arias 
Herman Borrero
Proyecto académico - Universidad Autónoma de Occidente  