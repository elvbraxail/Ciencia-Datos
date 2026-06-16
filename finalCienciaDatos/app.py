import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================

st.set_page_config(
    page_title="Proyecto Ciencia de Datos",
    page_icon="🌸",
    layout="wide"
)

# =====================================================
# CARGA DE DATOS
# =====================================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

species_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

df["species"] = df["species"].map(species_names)

# =====================================================
# TÍTULO
# =====================================================

st.title("🌸 Clasificación Inteligente de Flores Iris")
st.markdown("## Microproyecto de Ciencia de Datos")
st.markdown("---")

# =====================================================
# FASE 1
# =====================================================

st.header("1️⃣ Comprensión del Negocio")

st.info("""
Objetivo:
Construir un modelo predictivo capaz de clasificar automáticamente una flor Iris
según sus características físicas.
""")

# =====================================================
# FASE 2
# =====================================================

st.header("2️⃣ Comprensión de los Datos")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Registros", df.shape[0])
c2.metric("Variables", df.shape[1])
c3.metric("Clases", df["species"].nunique())
c4.metric("Valores Nulos", int(df.isnull().sum().sum()))

st.dataframe(df.head())

# =====================================================
# FASE 3
# =====================================================

st.header("3️⃣ Preparación de Datos")

st.subheader("Valores nulos")

st.dataframe(df.isnull().sum().to_frame("Cantidad"))

st.success("No existen valores faltantes en el dataset.")

# =====================================================
# FASE 4 - EDA
# =====================================================

st.header("4️⃣ Análisis Exploratorio de Datos")

# -----------------------------------------------------
# DISTRIBUCIÓN DE ESPECIES
# -----------------------------------------------------

st.subheader("Distribución de especies")

species_count = (
    df["species"]
    .value_counts()
    .reset_index()
)

species_count.columns = ["Especie", "Cantidad"]

fig_species = px.bar(
    species_count,
    x="Especie",
    y="Cantidad",
    color="Especie",
    text="Cantidad",
    title="Cantidad de muestras por especie"
)

st.plotly_chart(fig_species, width="stretch")

# -----------------------------------------------------
# HISTOGRAMA
# -----------------------------------------------------

st.subheader("Distribución de Variables")

selected_variable = st.selectbox(
    "Seleccione una variable",
    iris.feature_names
)

fig_hist = px.histogram(
    df,
    x=selected_variable,
    color="species",
    marginal="box",
    title=f"Distribución de {selected_variable}"
)

st.plotly_chart(fig_hist, width="stretch")

# -----------------------------------------------------
# SCATTER
# -----------------------------------------------------

st.subheader("Relación entre variables")

fig_scatter = px.scatter(
    df,
    x="sepal length (cm)",
    y="petal length (cm)",
    color="species",
    size="petal width (cm)",
    hover_data=df.columns,
    title="Relación entre longitud de sépalo y pétalo"
)

st.plotly_chart(fig_scatter, width="stretch")

# -----------------------------------------------------
# BOXPLOT
# -----------------------------------------------------

st.subheader("Boxplot de longitud del pétalo")

fig_box = px.box(
    df,
    x="species",
    y="petal length (cm)",
    color="species",
    title="Distribución del tamaño del pétalo"
)

st.plotly_chart(fig_box, width="stretch")

# -----------------------------------------------------
# PROMEDIO PETALO
# -----------------------------------------------------

st.subheader("Promedio de longitud del pétalo")

avg_petal = (
    df.groupby("species")["petal length (cm)"]
    .mean()
    .reset_index()
)

fig_avg_petal = px.bar(
    avg_petal,
    x="species",
    y="petal length (cm)",
    color="species",
    text_auto=".2f",
    title="Promedio de longitud del pétalo"
)

st.plotly_chart(fig_avg_petal, width="stretch")

# -----------------------------------------------------
# PROMEDIO SÉPALO
# -----------------------------------------------------

st.subheader("Promedio de longitud del sépalo")

avg_sepal = (
    df.groupby("species")["sepal length (cm)"]
    .mean()
    .reset_index()
)

fig_avg_sepal = px.bar(
    avg_sepal,
    x="species",
    y="sepal length (cm)",
    color="species",
    text_auto=".2f",
    title="Promedio de longitud del sépalo"
)

st.plotly_chart(fig_avg_sepal, width="stretch")

# -----------------------------------------------------
# CORRELACIÓN
# -----------------------------------------------------

st.subheader("Mapa de correlación")

corr = df.drop(columns=["species"]).corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Correlación entre variables"
)

st.plotly_chart(fig_corr, width="stretch")

# -----------------------------------------------------
# ESTADÍSTICAS
# -----------------------------------------------------

st.subheader("Estadísticas descriptivas")

st.dataframe(df.describe())

# =====================================================
# FASE 5
# =====================================================

st.header("5️⃣ Construcción del Modelo")

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

st.success("Modelo Random Forest entrenado correctamente.")

# =====================================================
# FASE 6
# =====================================================

st.header("6️⃣ Evaluación del Modelo")

accuracy = accuracy_score(
    y_test,
    predictions
)

st.metric(
    "Precisión del Modelo",
    f"{accuracy*100:.2f}%"
)

st.subheader("Matriz de Confusión")

cm = confusion_matrix(
    y_test,
    predictions
)

cm_df = pd.DataFrame(
    cm,
    index=["Setosa", "Versicolor", "Virginica"],
    columns=["Setosa", "Versicolor", "Virginica"]
)

st.dataframe(cm_df)

st.subheader("Reporte de Clasificación")

st.code(
    classification_report(
        y_test,
        predictions
    )
)

# =====================================================
# FASE 7
# =====================================================

st.header("7️⃣ Interpretación del Modelo")

importance = pd.DataFrame({
    "Variable": iris.feature_names,
    "Importancia": model.feature_importances_
})

importance = importance.sort_values(
    by="Importancia",
    ascending=False
)

fig_importance = px.bar(
    importance,
    x="Variable",
    y="Importancia",
    color="Importancia",
    text_auto=".3f",
    title="Importancia de Variables"
)

st.plotly_chart(fig_importance, width="stretch")

# =====================================================
# FASE 8
# =====================================================

st.header("8️⃣ Dashboard Predictivo")

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.slider(
        "Sepal Length",
        4.0,
        8.0,
        5.5
    )

    sepal_width = st.slider(
        "Sepal Width",
        2.0,
        5.0,
        3.0
    )

with col2:

    petal_length = st.slider(
        "Petal Length",
        1.0,
        7.0,
        4.0
    )

    petal_width = st.slider(
        "Petal Width",
        0.1,
        3.0,
        1.0
    )

if st.button("🔍 Realizar Predicción"):

    prediction = model.predict([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    flower = species_names[int(prediction[0])]

    st.success(
        f"🌸 La flor predicha es: {flower}"
    )

# =====================================================
# CONCLUSIONES
# =====================================================

st.header("9️⃣ Conclusiones")

st.write("""
✅ Se aplicó el proceso CRISP-DM.

✅ Se analizaron 150 registros del dataset Iris.

✅ Se realizó un análisis exploratorio completo.

✅ Se entrenó un modelo Random Forest.

✅ Se obtuvo una precisión elevada.

✅ Se desarrolló un dashboard interactivo para predicciones.
""")

# =====================================================
# RECOMENDACIONES
# =====================================================

st.header("🔟 Recomendaciones")

st.write("""
• Utilizar datasets más grandes.

• Comparar múltiples algoritmos.

• Aplicar validación cruzada.

• Desplegar el proyecto en Streamlit Cloud.

• Integrar nuevas variables para mejorar el rendimiento.
""")