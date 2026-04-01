#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 Sistema de Recomendación de Películas - Interfaz Streamlit
Interface visual para obtener recomendaciones personalizadas
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from typing import List, Dict
import plotly.express as px

# ========== Configuración de Streamlit ==========
st.set_page_config(
    page_title="🎬 Sistema de Recomendación",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS Personalizado ==========
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== Configuración ==========
API_URL = "http://127.0.0.1:8080"

# ========== Funciones para conectar con la API ==========
@st.cache_data
def load_movies():
    """Cargar datos de películas"""
    try:
        with open('data/movies.item', 'r', encoding='latin-1') as f:
            movies = {}
            for line in f:
                parts = line.split('|')
                if len(parts) >= 2:
                    movie_id = int(parts[0])
                    title = parts[1]
                    movies[movie_id] = title
            return movies
    except Exception as e:
        st.warning(f"⚠️ No se pudieron cargar los datos de películas: {e}")
        return {}

def get_recommendations(user_id: int, n_recommendations: int = 5) -> Dict:
    """Obtener recomendaciones de la API"""
    try:
        url = f"{API_URL}/recommend/{user_id}?n_recommendations={n_recommendations}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Error en la API: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar a la API. ¿Está ejecutándose en localhost:8080?")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def get_users() -> List[int]:
    """Obtener lista de usuarios disponibles"""
    try:
        response = requests.get(f"{API_URL}/users", timeout=10)
        if response.status_code == 200:
            return response.json().get('users', [])
        else:
            return []
    except Exception:
        return []

# ========== HEADER ==========
st.markdown("""
# 🎬 Sistema de Recomendación de Películas

*Sistema inteligente que predice qué películas te encantarán usando Machine Learning*
""")

st.divider()

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Cargar datos de películas
    movies_dict = load_movies()
    st.info(f"📽️ Total de películas disponibles: {len(movies_dict)}")
    
    # Obtener lista de usuarios
    available_users = get_users()
    st.info(f"👥 Usuarios en el sistema: {len(available_users)}")
    
    st.divider()
    
    st.subheader("Acerca del Sistema")
    st.write("""
    - **Algoritmo**: k-Nearest Neighbors (k-NN)
    - **Métrica**: Similitud coseno
    - **Tipo**: Recomendación basada en usuarios
    - **Dataset**: MovieLens 100K
    """)

# ========== MAIN CONTENT ==========
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("👤 Selecciona un Usuario")
    
    # Input del usuario
    user_input = st.number_input(
        "ID de Usuario (1-943):",
        min_value=1,
        max_value=943,
        value=1,
        step=1,
        help="Ingresa el ID del usuario para obtener recomendaciones personalizadas"
    )

with col2:
    st.subheader("🎯 Número de Recomendaciones")
    n_recs = st.slider(
        "¿Cuántas películas deseas?",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Selecciona cuántas películas quieres que te recomiende"
    )

st.divider()

# ========== BOTÓN DE GENERAR RECOMENDACIONES ==========
col_button = st.columns([1, 3, 1])

with col_button[1]:
    if st.button("🚀 Generar Recomendaciones", use_container_width=True):
        with st.spinner(f"⏳ Obteniendo recomendaciones para usuario {user_input}..."):
            result = get_recommendations(user_input, n_recs)
        
        if result:
            st.success(result.get('message', '✅ Recomendaciones generadas'))
            
            # ========== MOSTRAR RECOMENDACIONES ==========
            st.divider()
            st.subheader(f"🎬 Películas Recomendadas para Usuario {user_input}")
            
            recommendations = result.get('recommendations', [])
            
            if recommendations:
                # Crear tabla
                rec_data = []
                for i, rec in enumerate(recommendations, 1):
                    item_id = rec.get('item_id')
                    score = rec.get('predicted_score', 0)
                    confidence = rec.get('confidence', 'unknown')
                    
                    movie_title = movies_dict.get(item_id, f"Película {item_id}")
                    
                    rec_data.append({
                        '🎬': i,
                        'Película': movie_title,
                        '⭐ Score': f"{score:.2f}",
                        '📊 Confianza': confidence.capitalize()
                    })
                
                # Mostrar tabla
                df_recs = pd.DataFrame(rec_data)
                st.dataframe(df_recs, use_container_width=True, hide_index=True)
                
                # ========== GRÁFICO DE SCORES ==========
                st.divider()
                st.subheader("📈 Visualización de Scores")
                
                chart_data = pd.DataFrame([
                    {
                        'Película': movies_dict.get(rec.get('item_id'), f"Filme {rec.get('item_id')}")[:30],
                        'Score': rec.get('predicted_score', 0)
                    }
                    for rec in recommendations
                ])
                
                fig = px.bar(
                    chart_data,
                    x='Película',
                    y='Score',
                    title="Scores de Predicción",
                    color='Score',
                    color_continuous_scale='Viridis',
                    height=400,
                    labels={'Score': 'Puntuación Predicha'}
                )
                
                fig.update_layout(
                    xaxis_tickangle=-45,
                    showlegend=False,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ========== DESCARGAR RESULTADOS ==========
                st.divider()
                csv = df_recs.to_csv(index=False)
                st.download_button(
                    label="⬇️ Descargar Recomendaciones (CSV)",
                    data=csv,
                    file_name=f"recomendaciones_usuario_{user_input}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No se obtuvieron recomendaciones")
        else:
            st.error("❌ No se pudo obtener las recomendaciones")

# ========== FOOTER ==========
st.divider()

with st.expander("ℹ️ Más Información"):
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        ### 🔧 Cómo Funciona
        
        1. **Selecciona un usuario** por su ID
        2. **Elige cuántas películas** deseas
        3. **Genera recomendaciones** con un clic
        4. **Visualiza los resultados** con scores y gráficos
        
        El sistema analiza usuarios similares y sus \
        películas favoritas para recomendarte.
        """)
    
    with col_info2:
        st.markdown("""
        ### 📊 Datos del Modelo
        
        - **Usuarios**: 943
        - **Películas**: 1,682
        - **Ratings**: 100,000
        - **RMSE**: 1.40
        - **MAE**: 1.05
        - **Periodo**: Años 90
        """)

st.markdown("""
---
*Sistema de Recomendación | Modelo ML com FastAPI + Streamlit*
""")
