# PLAN MAESTRO - PROYECTO FINAL MACHINE LEARNING
## Marketing Campaign Response Prediction

---

## 📋 INFORMACIÓN GENERAL DEL PROYECTO

### Contexto del Negocio
**Objetivo Principal**: Predecir quién responderá a una oferta de producto/servicio en una campaña de marketing para maximizar la eficiencia y rentabilidad de futuras campañas.

**Dataset**: Marketing Campaign (marketing_campaign.csv)
- **Variable Objetivo**: `Response` (1 si aceptó la oferta en la última campaña, 0 si no)
- **Tipo de Problema**: Clasificación Binaria Supervisada
- **Aplicación**: Modelo de respuesta para optimizar campañas de marketing

### Estructura de Entregas
- **Punto 1**: 29.10.2025 - Estructura y EDA
- **Entrega Final**: 10.11.2025 23:59
- **Repositorio**: GitHub público compartido con `juanseparracourses`
- **Ramas requeridas**: `developer`, `certification`, `master`

---

## 📊 VARIABLES DEL DATASET

### Variables de Campañas Anteriores
- `AcceptedCmp1` a `AcceptedCmp5`: Aceptación en campañas 1-5 (binarias)
- `Response`: **VARIABLE OBJETIVO** - Aceptación en última campaña (binaria)
- `Complain`: Quejas en últimos 2 años (binaria)

### Variables Demográficas
- `DtCustomer`: Fecha de inscripción del cliente
- `Education`: Nivel educativo
- `Marital`: Estado civil
- `Kidhome`: Número de niños pequeños en el hogar
- `Teenhome`: Número de adolescentes en el hogar
- `Income`: Ingreso anual del hogar

### Variables de Comportamiento de Compra (Últimos 2 años)
- `MntFishProducts`: Gasto en pescado
- `MntMeatProducts`: Gasto en carne
- `MntFruits`: Gasto en frutas
- `MntSweetProducts`: Gasto en dulces
- `MntWines`: Gasto en vinos
- `MntGoldProds`: Gasto en productos gold

### Variables de Canales de Compra
- `NumDealsPurchases`: Compras con descuento
- `NumCatalogPurchases`: Compras por catálogo
- `NumStorePurchases`: Compras en tienda física
- `NumWebPurchases`: Compras por web
- `NumWebVisitsMonth`: Visitas al sitio web (último mes)
- `Recency`: Días desde última compra

---

## 🏗️ ESTRUCTURA DEL REPOSITORIO (OBLIGATORIA)

```
final-project-ml_Alejo/
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb          [COMPLETADO]
│       ├── comprension_eda.ipynb       [PENDIENTE]
│       ├── ft_engineering.py           [ESQUELETO CREADO]
│       ├── model_training_evaluation.py [PENDIENTE]
│       ├── model_deploy.py             [PENDIENTE]
│       └── model_monitoring.py         [PENDIENTE]
├── Base_de_datos.csv                   [PENDIENTE - Copiar marketing_campaign.csv]
├── requirements.txt                    [BÁSICO - NECESITA ACTUALIZACIÓN]
├── .gitignore                          [COMPLETADO]
├── setup.bat                           [COMPLETADO]
└── README.md                           [BÁSICO - NECESITA DESARROLLO]
```

---

## 🎯 FASES DE IMPLEMENTACIÓN DEL PROYECTO

---

### **FASE 0: PREPARACIÓN INICIAL** ✅
**Estado**: COMPLETADO
**Responsable**: Manual + Asistente

#### Tareas:
- [x] Crear repositorio en GitHub
- [x] Configurar estructura de carpetas
- [x] Crear archivo requirements.txt básico
- [x] Configurar .gitignore
- [x] Crear setup.bat para entorno virtual
- [ ] **MANUAL**: Copiar marketing_campaign.csv a Base_de_datos.csv
- [ ] **MANUAL**: Crear ramas developer, certification, master
- [ ] **MANUAL**: Compartir repo con juanseparracourses

#### Entregables:
- Repositorio con estructura correcta
- Entorno virtual configurado

---

### **FASE 1: EXPLORACIÓN Y ANÁLISIS DE DATOS (EDA)**
**Archivo**: `mlops_pipeline/src/comprension_eda.ipynb`
**Peso en Evaluación**: 0.7 puntos
**Estado**: PENDIENTE

#### 1.1 Exploración Inicial de Datos
**Checklist de Evaluación**:
- [ ] Descripción general del dataset
- [ ] Caracterización de variables (categóricas, numéricas, ordinales, nominales, dicotómicas, politómicas)
- [ ] Revisión de valores nulos
- [ ] Unificación de representación de nulos
- [ ] Eliminación de variables irrelevantes
- [ ] Conversión de datos a tipos correctos (numéricos, categóricos, booleanos, fechas)
- [ ] Corrección de inconsistencias

**Análisis Específico para Marketing Campaign**:
- Identificar tipos de variables:
  - **Binarias**: AcceptedCmp1-5, Response, Complain
  - **Numéricas continuas**: Income, Mnt* (gastos), Recency
  - **Numéricas discretas**: Kidhome, Teenhome, Num* (conteos)
  - **Categóricas**: Education, Marital
  - **Fecha**: DtCustomer
- Detectar nulos especialmente en Income (común en datasets de marketing)
- Validar rangos lógicos (Income > 0, Recency >= 0, etc.)

#### 1.2 Análisis Univariable
**Checklist de Evaluación**:
- [ ] Ejecutar describe() después de ajustar tipos
- [ ] Histogramas y boxplots para variables numéricas
- [ ] Countplot, value_counts() y tablas pivote para categóricas
- [ ] Medidas estadísticas: media, mediana, moda, max, min
- [ ] Medidas de dispersión: rango, IQR, cuartiles, varianza, desviación estándar
- [ ] Skewness y kurtosis
- [ ] Identificar tipo de distribución

**Análisis Específico**:
- **Variable Objetivo (Response)**: Verificar balance de clases
- **Gastos (Mnt*)**: Analizar distribución (probablemente sesgada)
- **Income**: Detectar outliers y distribución
- **Recency**: Patrón de recencia de compras
- **Campañas anteriores**: Tasa de aceptación histórica

#### 1.3 Análisis Bivariable
**Checklist de Evaluación**:
- [ ] Gráficos y tablas con respecto a variable objetivo (Response)
- [ ] Comentarios e interpretaciones

**Análisis Específico**:
- Response vs AcceptedCmp1-5 (correlación entre campañas)
- Response vs Income (poder adquisitivo)
- Response vs gastos totales (engagement)
- Response vs Education/Marital (demografía)
- Response vs canales de compra (preferencias)
- Response vs Recency (actividad reciente)

#### 1.4 Análisis Multivariable
**Checklist de Evaluación**:
- [ ] Pairplot de variables clave
- [ ] Matriz de correlación
- [ ] Gráficos de dispersión entre numéricas
- [ ] Uso de parámetro hue para categóricas
- [ ] Identificar reglas de validación de datos
- [ ] Identificar transformaciones aplicables
- [ ] Sugerir atributos derivados/calculados

**Análisis Específico**:
- Correlación entre AcceptedCmp1-5 y Response
- Correlación entre diferentes tipos de gastos
- Relación Income vs gastos totales
- Segmentación por Education + Marital + Response
- Interacción Kidhome + Teenhome vs patrones de compra

**Atributos Derivados Sugeridos**:
- `TotalSpent`: Suma de todos los Mnt*
- `TotalPurchases`: Suma de todos los Num*Purchases
- `TotalAcceptedCampaigns`: Suma de AcceptedCmp1-5
- `HasChildren`: Kidhome + Teenhome > 0
- `CustomerAge`: Días desde DtCustomer
- `AvgPurchaseValue`: TotalSpent / TotalPurchases
- `WebEngagement`: NumWebPurchases / NumWebVisitsMonth

#### Entregables Fase 1:
- Notebook comprension_eda.ipynb completamente documentado
- Insights clave sobre el comportamiento de clientes
- Lista de transformaciones necesarias
- Propuesta de features derivados

---

### **FASE 2: INGENIERÍA DE CARACTERÍSTICAS**
**Archivo**: `mlops_pipeline/src/ft_engineering.py`
**Peso en Evaluación**: 0.5 puntos
**Estado**: ESQUELETO CREADO

#### 2.1 Desarrollo del Pipeline de Features
**Checklist de Evaluación**:
- [ ] Genera correctamente features desde dataset base
- [ ] Flujo de transformación documentado
- [ ] Pipelines de sklearn creados
- [ ] Separación correcta train/test
- [ ] Retorna dataset limpio para modelado
- [ ] Transformaciones: escalado, codificación, imputación
- [ ] Decisiones documentadas

**Componentes del Pipeline**:

1. **Limpieza de Datos**:
   - Manejo de nulos en Income (imputación por mediana o eliminación)
   - Unificación de categorías en Education/Marital
   - Eliminación de outliers extremos

2. **Feature Engineering**:
   - Crear features derivados (TotalSpent, TotalPurchases, etc.)
   - Extraer features de DtCustomer (antigüedad, mes/año registro)
   - Binning de variables continuas si necesario
   - Interacciones relevantes

3. **Transformaciones**:
   - **Numéricas**: StandardScaler o RobustScaler (por outliers)
   - **Categóricas**: OneHotEncoder o LabelEncoder
   - **Fechas**: Convertir a features numéricas

4. **Pipeline de sklearn**:
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import StandardScaler, OneHotEncoder
   from sklearn.impute import SimpleImputer
   ```

5. **Split de Datos**:
   - train_test_split con test_size=0.2, random_state=42
   - Estratificación por Response (por desbalance de clases)

#### Entregables Fase 2:
- ft_engineering.py con funciones completas
- Pipelines de transformación reutilizables
- X_train, X_test, y_train, y_test guardados
- Documentación de decisiones

---

### **FASE 3: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS**
**Archivo**: `mlops_pipeline/src/model_training_evaluation.py`
**Peso en Evaluación**: 1.0 punto
**Estado**: PENDIENTE

#### 3.1 Desarrollo de Funciones Reutilizables
**Checklist de Evaluación**:
- [ ] Función build_model() para entrenamiento estructurado
- [ ] Función summarize_classification() para métricas

**Funciones Requeridas**:
```python
def build_model(model, X_train, y_train, X_test, y_test):
    """Entrena y evalúa un modelo"""
    # Entrenamiento
    # Predicción
    # Métricas
    # Retornar resultados
    
def summarize_classification(y_true, y_pred, model_name):
    """Resume métricas de clasificación"""
    # Accuracy, Precision, Recall, F1-Score
    # Matriz de confusión
    # ROC-AUC
    # Retornar dict de métricas
```

#### 3.2 Entrenamiento de Modelos
**Checklist de Evaluación**:
- [ ] Múltiples modelos supervisados entrenados
- [ ] Validación cruzada aplicada
- [ ] Modelo seleccionado guardado

**Modelos a Entrenar** (mínimo 5):
1. **Logistic Regression** (baseline)
2. **Random Forest Classifier**
3. **XGBoost Classifier**
4. **LightGBM Classifier**
5. **Support Vector Machine (SVM)**
6. **Gradient Boosting Classifier**
7. **Extra Trees Classifier** (opcional)

**Técnicas de Validación**:
- Cross-validation (5-fold o 10-fold)
- Stratified K-Fold (por desbalance)
- GridSearchCV o RandomizedSearchCV para hiperparámetros

#### 3.3 Evaluación y Comparación
**Checklist de Evaluación**:
- [ ] Métricas: accuracy, precision, recall, F1-score, ROC-AUC
- [ ] Gráficos comparativos (curvas ROC, matriz confusión)
- [ ] Justificación de selección del mejor modelo

**Métricas Clave para Marketing**:
- **Recall**: Capturar máximo de clientes que responderán
- **Precision**: Evitar gastar en clientes que no responderán
- **F1-Score**: Balance entre ambos
- **ROC-AUC**: Capacidad de discriminación
- **Profit Curve**: Maximizar beneficio de campaña

**Visualizaciones Requeridas**:
- Tabla comparativa de todos los modelos
- Curvas ROC superpuestas
- Matrices de confusión
- Feature importance del mejor modelo
- Gráfico de barras con métricas comparativas

#### 3.4 Selección del Modelo Final
**Criterios**:
- **Performance**: Mejores métricas en test set
- **Consistency**: Bajo overfitting (train vs test)
- **Scalability**: Tiempo de entrenamiento/predicción
- **Interpretability**: Importancia para negocio

**Guardar Modelo**:
```python
import joblib
joblib.dump(best_model, 'best_model.pkl')
```

#### Entregables Fase 3:
- model_training_evaluation.py completo
- Modelo final guardado (.pkl o .joblib)
- Reporte comparativo de modelos
- Justificación técnica de selección

---

### **FASE 4: MONITOREO Y DETECCIÓN DE DATA DRIFT**
**Archivo**: `mlops_pipeline/src/model_monitoring.py`
**Peso en Evaluación**: 1.0 punto
**Estado**: PENDIENTE

#### 4.1 Implementación de Métricas de Drift
**Checklist de Evaluación**:
- [ ] Test de Drift calculado (KS, PSI, JS, Chi-cuadrado)

**Métricas a Implementar**:

1. **Kolmogorov-Smirnov Test** (variables numéricas):
   ```python
   from scipy.stats import ks_2samp
   ```

2. **Population Stability Index (PSI)**:
   - PSI < 0.1: Sin cambio significativo
   - 0.1 ≤ PSI < 0.2: Cambio moderado
   - PSI ≥ 0.2: Cambio significativo

3. **Jensen-Shannon Divergence**:
   ```python
   from scipy.spatial.distance import jensenshannon
   ```

4. **Chi-cuadrado** (variables categóricas):
   ```python
   from scipy.stats import chi2_contingency
   ```

**Implementación**:
- Muestreo periódico de datos
- Comparación distribución histórica vs actual
- Cálculo de métricas por variable
- Generación de alertas por umbrales

#### 4.2 Aplicación en Streamlit
**Checklist de Evaluación**:
- [ ] Interfaz funcional en Streamlit
- [ ] Gráficos comparativos distribución histórica vs actual
- [ ] Indicadores visuales de alerta (semáforo, barras)
- [ ] Alertas por desviaciones significativas

**Componentes de la App**:

1. **Dashboard Principal**:
   - Resumen de estado de drift
   - Semáforos por variable (verde/amarillo/rojo)
   - Última actualización

2. **Visualización de Métricas**:
   - Tabla con métricas de drift por variable
   - Gráficos de distribución histórica vs actual
   - Histogramas superpuestos
   - Box plots comparativos

3. **Análisis Temporal**:
   - Evolución del drift en el tiempo
   - Detección de tendencias
   - Cambios abruptos

4. **Recomendaciones**:
   - Mensajes automáticos si umbral crítico
   - Sugerencias de retraining
   - Variables a revisar

**Estructura de la App**:
```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🔍 Monitoreo de Data Drift - Marketing Campaign")

# Sidebar con configuración
# Sección de métricas generales
# Sección de análisis por variable
# Sección de alertas y recomendaciones
```

#### Entregables Fase 4:
- model_monitoring.py con funciones de drift
- Aplicación Streamlit funcional
- Documentación de umbrales y alertas

---

### **FASE 5: DESPLIEGUE DEL MODELO**
**Archivo**: `mlops_pipeline/src/model_deploy.py`
**Peso en Evaluación**: 1.0 punto
**Estado**: PENDIENTE

#### 5.1 Desarrollo de API con FastAPI
**Checklist de Evaluación**:
- [ ] Framework adecuado (FastAPI o Flask)
- [ ] Endpoint /predict definido
- [ ] Acepta JSON y/o CSV
- [ ] Soporta predicción por lotes
- [ ] Retorna predicción en formato estructurado

**Estructura de la API**:

```python
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Marketing Campaign Prediction API")

# Cargar modelo
model = joblib.load('best_model.pkl')

# Modelo de datos
class CustomerData(BaseModel):
    Income: float
    Recency: int
    # ... todas las features
    
# Endpoint de predicción individual
@app.post("/predict")
def predict_single(data: CustomerData):
    # Transformar a DataFrame
    # Aplicar pipeline
    # Predecir
    # Retornar resultado
    
# Endpoint de predicción por lotes
@app.post("/predict_batch")
def predict_batch(file: UploadFile):
    # Leer CSV
    # Predecir
    # Retornar resultados
    
# Endpoint de salud
@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

**Endpoints Requeridos**:
- `GET /`: Información de la API
- `GET /health`: Health check
- `POST /predict`: Predicción individual (JSON)
- `POST /predict_batch`: Predicción por lotes (CSV/JSON)

#### 5.2 Dockerización
**Checklist de Evaluación**:
- [ ] Dockerfile funcional con instrucciones claras

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mlops_pipeline/src/ ./src/
COPY best_model.pkl .

EXPOSE 8000

CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Archivos Necesarios**:
- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml` (opcional)

**Comandos Docker**:
```bash
# Build
docker build -t marketing-campaign-api .

# Run
docker run -p 8000:8000 marketing-campaign-api

# Test
curl http://localhost:8000/health
```

#### Entregables Fase 5:
- model_deploy.py con API completa
- Dockerfile funcional
- Documentación de endpoints
- Ejemplos de uso de la API

---

### **FASE 6: INTEGRACIÓN CON SONARCLOUD**
**Peso en Evaluación**: 0.5 puntos
**Estado**: PENDIENTE
**Responsable**: MANUAL

#### 6.1 Configuración de SonarCloud
**Checklist de Evaluación**:
- [ ] Repositorio vinculado a SonarCloud
- [ ] Configuración creada y pruebas generadas

**Pasos Manuales**:
1. Ir a https://sonarcloud.io
2. Registrarse con cuenta de GitHub
3. Importar repositorio final-project-ml_Alejo
4. Configurar análisis automático
5. Crear archivo `sonar-project.properties`

**Archivo sonar-project.properties**:
```properties
sonar.projectKey=tu-usuario_final-project-ml_Alejo
sonar.organization=tu-organizacion

sonar.sources=mlops_pipeline/src
sonar.python.version=3.9

sonar.exclusions=**/*.ipynb,**/__pycache__/**,**/venv/**
```

#### 6.2 Validaciones de SonarCloud

**1. Calidad del Código**:
- Código duplicado
- Complejidad ciclomática
- Funciones largas
- Malas prácticas

**2. Seguridad**:
- Exposición de datos sensibles
- Uso inseguro de librerías

**3. Cobertura de Pruebas**:
- Líneas ejecutadas en tests
- Métodos validados

**4. Integridad y Estilo**:
- Nombres de variables/funciones
- Indentación y espacios
- Consistencia

**Acciones Correctivas**:
- Refactorizar código duplicado
- Simplificar funciones complejas
- Agregar docstrings
- Seguir PEP 8

#### Entregables Fase 6:
- Badge de SonarCloud en README
- Reporte de calidad del código
- Capturas de pantalla de análisis

---

### **FASE 7: DOCUMENTACIÓN FINAL**
**Archivo**: `README.md`
**Estado**: BÁSICO - NECESITA DESARROLLO

#### 7.1 Contenido del README

**Estructura Requerida**:

```markdown
# Marketing Campaign Response Prediction

## 📊 Contexto del Negocio
[Descripción del problema y objetivo]

## 🎯 Objetivo del Proyecto
[Objetivo específico del modelo]

## 📁 Estructura del Proyecto
[Árbol de carpetas con descripción]

## 📈 Dataset
[Descripción de variables y fuente]

## 🔍 Principales Hallazgos del EDA
[Insights clave del análisis exploratorio]

## 🛠️ Proceso de Desarrollo

### 1. Exploración de Datos
[Resumen de EDA]

### 2. Ingeniería de Características
[Features creados y transformaciones]

### 3. Entrenamiento de Modelos
[Modelos probados y resultados]

### 4. Modelo Seleccionado
[Justificación y métricas]

### 5. Monitoreo
[Estrategia de drift detection]

### 6. Despliegue
[API y Docker]

## 🚀 Instalación y Uso

### Requisitos Previos
[Python version, etc.]

### Instalación
```bash
git clone [repo]
cd final-project-ml_Alejo
setup.bat
```

### Ejecución
[Comandos para correr notebooks, API, Streamlit]

## 📊 Resultados
[Tabla con métricas finales]

## 🔧 Tecnologías Utilizadas
[Lista de librerías y herramientas]

## 👥 Autor
[Tu nombre]

## 📄 Licencia
[Si aplica]

## 🏆 SonarCloud
[Badge de calidad]
```

#### Entregables Fase 7:
- README.md completo y profesional
- Documentación clara y concisa
- Badges de calidad y estado

---

## 📋 CHECKLIST COMPLETO DE EVALUACIÓN

### Estructura y Configuraciones (0.3 puntos)
- [ ] Estructura mínima respetada
- [ ] requirements.txt con dependencias
- [ ] Entorno virtual configurado y documentado

### Análisis de Datos (0.7 puntos)
- [ ] Descripción general del dataset
- [ ] Tipos de variables identificados
- [ ] Valores nulos revisados y unificados
- [ ] Variables irrelevantes eliminadas
- [ ] Datos convertidos a tipos correctos
- [ ] describe() ejecutado
- [ ] Histogramas y boxplots para numéricas
- [ ] Countplot y value_counts para categóricas
- [ ] Medidas estadísticas completas
- [ ] Tipo de distribución identificado
- [ ] Análisis bivariable con variable objetivo
- [ ] Análisis multivariable (pairplot, correlación)
- [ ] Reglas de validación identificadas
- [ ] Atributos derivados sugeridos

### Ingeniería de Características (0.5 puntos)
- [ ] Features generados correctamente
- [ ] Flujo documentado
- [ ] Pipelines de sklearn creados
- [ ] Train/test separados correctamente
- [ ] Dataset limpio retornado
- [ ] Transformaciones aplicadas
- [ ] Decisiones documentadas

### Entrenamiento y Evaluación (1.0 punto)
- [ ] Múltiples modelos entrenados
- [ ] Función build_model() implementada
- [ ] Validación cruzada aplicada
- [ ] Modelo guardado
- [ ] Función summarize_classification() implementada
- [ ] Métricas completas calculadas
- [ ] Gráficos comparativos generados
- [ ] Selección justificada

### Monitoreo (1.0 punto)
- [ ] Test de drift calculado
- [ ] Interfaz Streamlit funcional
- [ ] Gráficos comparativos de distribución
- [ ] Indicadores visuales de alerta
- [ ] Alertas por desviaciones

### Despliegue (1.0 punto)
- [ ] Framework adecuado usado
- [ ] Endpoint /predict definido
- [ ] JSON y/o CSV aceptado
- [ ] Predicción por lotes soportada
- [ ] Formato estructurado retornado
- [ ] Dockerfile funcional

### SonarCloud (0.5 puntos)
- [ ] Repositorio vinculado
- [ ] Configuración y pruebas generadas

**TOTAL: 5.0 puntos**

---

## 🔧 DEPENDENCIAS ACTUALIZADAS

### requirements.txt Completo
```
# Data manipulation
pandas==1.5.3
numpy==1.24.3

# Machine Learning
scikit-learn==1.2.2
xgboost==1.7.5
lightgbm==3.3.5

# Visualization
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.1

# Notebooks
jupyter==1.0.0
ipykernel==6.22.0

# API
fastapi==0.95.1
uvicorn==0.22.0
pydantic==1.10.7
python-multipart==0.0.6

# Streamlit
streamlit==1.22.0

# Data Drift
scipy==1.10.1

# Model persistence
joblib==1.2.0

# Utilities
python-dotenv==1.0.0
```

---

## 📝 TAREAS MANUALES REQUERIDAS

### Antes de Empezar:
1. ✅ **Copiar dataset**: `marketing_campaign.csv` → `Base_de_datos.csv`
2. ✅ **Crear ramas en GitHub**:
   ```bash
   git checkout -b developer
   git push origin developer
   git checkout -b certification
   git push origin certification
   git checkout main
   ```
3. ✅ **Compartir repositorio**: Agregar a `juanseparracourses` como colaborador

### Durante el Desarrollo:
4. ⚠️ **Ejecutar notebooks**: Los .ipynb deben ejecutarse manualmente
5. ⚠️ **Revisar visualizaciones**: Validar que gráficos sean correctos
6. ⚠️ **Probar API**: Testear endpoints con Postman o curl
7. ⚠️ **Ejecutar Streamlit**: Validar interfaz de monitoreo

### Al Final:
8. ⚠️ **Configurar SonarCloud**: Registro y vinculación manual
9. ⚠️ **Revisar calidad de código**: Corregir issues de SonarCloud
10. ⚠️ **Hacer commits**: Usar mensajes descriptivos
11. ⚠️ **Merge a master**: Desde developer → certification → master
12. ⚠️ **Verificar entrega**: Revisar checklist completo

---

## 🎯 ESTRATEGIA DE TRABAJO

### Orden Recomendado de Ejecución:
1. **Fase 1**: EDA completo (2-3 días)
2. **Fase 2**: Feature Engineering (1 día)
3. **Fase 3**: Entrenamiento de modelos (2 días)
4. **Fase 4**: Monitoreo (1 día)
5. **Fase 5**: Despliegue (1 día)
6. **Fase 6**: SonarCloud (0.5 día)
7. **Fase 7**: Documentación (0.5 día)

### Uso de este Documento:
- **Cuando se llene el contexto**: Referencia este archivo
- **Para retomar trabajo**: Indica "Estoy en Fase X, sección Y"
- **Para validar progreso**: Marca checkboxes completados
- **Para consultar requisitos**: Busca en checklist de evaluación

---

## 📞 CONTACTO Y SOPORTE

**Docente**: Juan Sebastián Parra Sánchez
**Usuario GitHub**: juanseparracourses
**Fecha límite**: 10 de noviembre de 2025, 23:59

---

**Última actualización**: 11 de noviembre de 2025
**Versión**: 1.0
**Estado del Proyecto**: FASE 0 COMPLETADA - INICIANDO FASE 1

