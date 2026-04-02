# -----------------------------------------------------------
# 1. Instalación y Configuración
# -----------------------------------------------------------
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
from sklearn.metrics import mean_squared_error

# Configuramos un experimento para empezar a registrar
mlflow.set_experiment("Sistema_Recomendacion_Demo")

# -----------------------------------------------------------
# 2. Carga de Datos Reales y Entrenamiento de un Modelo kNN
# -----------------------------------------------------------

def load_data():
    """Cargar datos de ratings desde el archivo CSV o .data"""
    try:
        # Intentar cargar ratings.csv primero (preferido)
        ratings_path = os.path.join('data', 'ratings.csv')
        df = pd.read_csv(ratings_path, 
                        names=['user_id', 'item_id', 'rating', 'timestamp'],
                        sep='\t')
        print(f"✅ Datos cargados desde ratings.csv: {len(df):,} ratings")
    except FileNotFoundError:
        try:
            # Si no existe CSV, cargar desde .data
            ratings_path = os.path.join('data', 'ratings.data')
            df = pd.read_csv(ratings_path, 
                            names=['user_id', 'item_id', 'rating', 'timestamp'],
                            sep='\t')
            print(f"✅ Datos cargados desde ratings.data: {len(df):,} ratings")
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    
    print(f"   - Usuarios únicos: {df['user_id'].nunique():,}")
    print(f"   - Items únicos: {df['item_id'].nunique():,}")
    print(f"   - Rating promedio: {df['rating'].mean():.2f}")
    
    return df

# Cargar los datos reales
df = load_data()
if df is None:
    print("No se pudieron cargar los datos. Usando datos de ejemplo...")
    # Fallback a datos simulados
    data = {
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'item_id': [101, 102, 101, 103, 102, 104, 103, 104],
        'rating': [5, 4, 4, 5, 5, 4, 3, 5]
    }
    df = pd.DataFrame(data)

# Filtrar datos para reducir la matriz (opcional - para datasets grandes)
# Tomar solo usuarios e items con suficientes interacciones
min_user_ratings = 5
min_item_ratings = 5

user_counts = df['user_id'].value_counts()
item_counts = df['item_id'].value_counts()

active_users = user_counts[user_counts >= min_user_ratings].index
popular_items = item_counts[item_counts >= min_item_ratings].index

# Filtrar el dataframe
df_filtered = df[df['user_id'].isin(active_users) & df['item_id'].isin(popular_items)]
print(f"   - Datos después del filtrado: {len(df_filtered):,} ratings")
print(f"   - Usuarios activos: {df_filtered['user_id'].nunique():,}")
print(f"   - Items populares: {df_filtered['item_id'].nunique():,}")

# Crear la matriz de interacciones (pivote)
R_df = df_filtered.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
R_matrix = R_df.values

print(f"   - Matriz usuario-item: {R_matrix.shape[0]} x {R_matrix.shape[1]}")

# Entrenar el modelo de Vecinos Cercanos (kNN) para las recomendaciones
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model_knn.fit(R_matrix)

# -----------------------------------------------------------
# 3. Evaluación y Registro del Modelo con MLflow
# -----------------------------------------------------------

def calculate_metrics(df_filtered, R_df, model_knn):
    """Calcular métricas del modelo"""
    # Dividir datos para evaluación (simple split)
    test_ratings = []
    predictions = []
    
    # Tomar una muestra para evaluar
    sample_size = min(1000, len(df_filtered))
    test_sample = df_filtered.sample(n=sample_size, random_state=42)
    
    for _, row in test_sample.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        true_rating = row['rating']
        
        if user_id in R_df.index and item_id in R_df.columns:
            # Predecir usando vecinos más cercanos
            user_idx = R_df.index.get_loc(user_id)
            user_vector = R_df.iloc[user_idx:user_idx+1].values
            
            distances, indices = model_knn.kneighbors(user_vector, n_neighbors=6)
            
            # Calcular predicción como promedio de ratings de vecinos
            neighbor_ratings = []
            for idx in indices[0][1:]:  # Excluir el propio usuario
                neighbor_user_id = R_df.index[idx]
                if R_df.loc[neighbor_user_id, item_id] > 0:
                    neighbor_ratings.append(R_df.loc[neighbor_user_id, item_id])
            
            if neighbor_ratings:
                predicted_rating = np.mean(neighbor_ratings)
            else:
                predicted_rating = R_df[item_id].mean()  # Promedio global del item
            
            test_ratings.append(true_rating)
            predictions.append(predicted_rating)
    
    if len(test_ratings) > 0:
        rmse = np.sqrt(mean_squared_error(test_ratings, predictions))
        mae = np.mean(np.abs(np.array(test_ratings) - np.array(predictions)))
    else:
        rmse, mae = 0.0, 0.0
    
    return float(rmse), float(mae)

# Calcular métricas reales
rmse, mae = calculate_metrics(df_filtered, R_df, model_knn)

# Iniciamos la "corrida" (run) de MLflow
with mlflow.start_run() as run:
    # 1. Registramos Parámetros
    mlflow.log_param("algoritmo", "kNN_UserBased_Cosine")
    mlflow.log_param("k_neighbors", 5)
    mlflow.log_param("min_user_ratings", min_user_ratings)
    mlflow.log_param("min_item_ratings", min_item_ratings)
    mlflow.log_param("matrix_shape", f"{R_matrix.shape[0]}x{R_matrix.shape[1]}")
    mlflow.log_param("total_ratings", len(df_filtered))

    # 2. Registramos Métricas reales
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("sparsity", float(1 - np.count_nonzero(R_matrix) / R_matrix.size))

    # 3. Guardamos el modelo
    import mlflow.sklearn as mlflow_sklearn
    mlflow_sklearn.log_model(
        sk_model=model_knn,
        artifact_path="recomendador_knn",
        registered_model_name="ModeloRecomendacion_v1"
    )

    # Guardar los mapeos necesarios en la carpeta data
    item_cols = R_df.columns.tolist()
    user_cols = R_df.index.tolist()
    
    # Crear directorio data si no existe
    os.makedirs('data', exist_ok=True)
    
    # Guardar mapeos
    np.save('data/item_cols.npy', item_cols)
    np.save('data/user_cols.npy', user_cols)
    
    # Registrar como artefactos en MLflow
    mlflow.log_artifact('data/item_cols.npy')
    mlflow.log_artifact('data/user_cols.npy')

    print(f"\n✅ MLflow Run ID: {run.info.run_id}")
    print(f"📊 Métricas del modelo:")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - Sparsity: {(1 - np.count_nonzero(R_matrix) / R_matrix.size):.4f}")
    print(f"🔧 Modelo y artefactos guardados: ModeloRecomendacion_v1")

# -----------------------------------------------------------
# 4. Función de Recomendación de Ejemplo
# -----------------------------------------------------------

def get_recommendations(user_id, R_df, model_knn, n_recommendations=5):
    """Obtener recomendaciones para un usuario específico"""
    if user_id not in R_df.index:
        print(f"Usuario {user_id} no encontrado en los datos")
        return []
    
    # Obtener el vector del usuario
    user_idx = R_df.index.get_loc(user_id)
    user_vector = R_df.iloc[user_idx:user_idx+1].values
    
    # Encontrar usuarios similares
    distances, indices = model_knn.kneighbors(user_vector, n_neighbors=6)
    
    # Items que el usuario ya ha calificado
    user_items = set(R_df.columns[R_df.iloc[user_idx] > 0])
    
    # Recopilar recomendaciones de usuarios similares
    recommendations = {}
    
    for neighbor_idx in indices[0][1:]:  # Excluir el propio usuario
        neighbor_user_id = R_df.index[neighbor_idx]
        neighbor_items = R_df.iloc[neighbor_idx]
        
        # Items que el vecino ha calificado bien (rating >= 4) y el usuario no ha visto
        for item_id, rating in neighbor_items.items():
            if rating >= 4 and item_id not in user_items:
                if item_id not in recommendations:
                    recommendations[item_id] = []
                recommendations[item_id].append(rating)
    
    # Calcular score promedio para cada item recomendado
    item_scores = {}
    for item_id, ratings in recommendations.items():
        item_scores[item_id] = np.mean(ratings)
    
    # Ordenar por score y retornar top N
    sorted_recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_recommendations[:n_recommendations]

# Ejemplo de uso
print(f"\n🎯 Ejemplo de recomendación:")
if len(R_df) > 0:
    sample_user = R_df.index[0]
    recommendations = get_recommendations(sample_user, R_df, model_knn)
    print(f"Recomendaciones para usuario {sample_user}:")
    for item_id, score in recommendations:
        print(f"   - Item {item_id}: Score {score:.2f}")

print(f"\n🎉 Sistema de recomendación entrenado y listo!")
print(f"📁 Archivos generados:")
print(f"   - data/item_cols.npy (mapeo de items)")
print(f"   - data/user_cols.npy (mapeo de usuarios)")
print(f"   - MLflow artifacts en carpeta mlruns/")

# Función principal para ejecutar todo el proceso
def main():
    """Función principal para entrenar el sistema de recomendación"""
    print("="*60)
    print("🚀 SISTEMA DE RECOMENDACIÓN CON MLFLOW")
    print("="*60)
    
    # El código ya se ejecutó arriba, pero aquí podrías agregar
    # lógica adicional de validación, testing, etc.
    
    return model_knn, R_df, df_filtered

if __name__ == "__main__":
    model, matrix_df, data_filtered = main()