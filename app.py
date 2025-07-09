import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import cv2
import os
import time
from datetime import datetime
import warnings
import base64
import json
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report
from scipy.stats import chi2
from scipy import stats
import itertools
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="🏥 Comparación de 3 Arquitecturas CNN + Estadísticas",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ThreeArchitecturesApp:
    """Aplicación para comparar 3 arquitecturas CNN con análisis estadístico avanzado"""
    
    def __init__(self):
        # Tu código existente de __init__ aquí...
        self.class_info = {
            'Central Serous Chorioretinopathy [Color Fundus]': {
                'nombre': 'Corioretinopatía Serosa Central',
                'descripcion': 'Acumulación de líquido bajo la retina',
                'gravedad': 'Moderada',
                'color': '#FFA07A',
                'tratamiento': 'Observación, láser focal en casos persistentes',
                'pronostico': 'Bueno, resolución espontánea en 80% casos'
            },
            'Diabetic Retinopathy': {
                'nombre': 'Retinopatía Diabética', 
                'descripcion': 'Daño vascular por diabetes',
                'gravedad': 'Alta',
                'color': '#FF6B6B',
                'tratamiento': 'Control glucémico, inyecciones intravítreas, láser',
                'pronostico': 'Manejo temprano previene ceguera'
            },
            'Disc Edema': {
                'nombre': 'Edema del Disco Óptico',
                'descripcion': 'Hinchazón por presión intracraneal',
                'gravedad': 'Alta',
                'color': '#FF4444',
                'tratamiento': 'Urgente: reducir presión intracraneal',
                'pronostico': 'Depende de causa subyacente'
            },
            'Glaucoma': {
                'nombre': 'Glaucoma',
                'descripcion': 'Daño del nervio óptico',
                'gravedad': 'Alta',
                'color': '#DC143C',
                'tratamiento': 'Gotas hipotensoras, láser, cirugía',
                'pronostico': 'Progresión lenta con tratamiento'
            },
            'Healthy': {
                'nombre': 'Ojo Sano',
                'descripcion': 'Sin patologías detectadas',
                'gravedad': 'Normal',
                'color': '#32CD32',
                'tratamiento': 'Exámenes preventivos anuales',
                'pronostico': 'Excelente'
            },
            'Macular Scar': {
                'nombre': 'Cicatriz Macular',
                'descripcion': 'Tejido cicatricial en mácula',
                'gravedad': 'Moderada',
                'color': '#DAA520',
                'tratamiento': 'Rehabilitación visual, ayudas ópticas',
                'pronostico': 'Estable, visión central afectada'
            },
            'Myopia': {
                'nombre': 'Miopía',
                'descripcion': 'Error refractivo',
                'gravedad': 'Leve',
                'color': '#87CEEB',
                'tratamiento': 'Lentes correctivos, cirugía refractiva',
                'pronostico': 'Excelente con corrección'
            },
            'Pterygium': {
                'nombre': 'Pterigión',
                'descripcion': 'Crecimiento anormal en córnea',
                'gravedad': 'Leve',
                'color': '#DDA0DD',
                'tratamiento': 'Observación, cirugía si afecta visión',
                'pronostico': 'Bueno, puede recurrir post-cirugía'
            },
            'Retinal Detachment': {
                'nombre': 'Desprendimiento de Retina',
                'descripcion': 'Emergencia: separación retinal',
                'gravedad': 'Crítica',
                'color': '#B22222',
                'tratamiento': 'URGENTE: cirugía inmediata',
                'pronostico': 'Bueno si se trata en <24-48h'
            },
            'Retinitis Pigmentosa': {
                'nombre': 'Retinitis Pigmentosa',
                'descripcion': 'Degeneración progresiva',
                'gravedad': 'Alta',
                'color': '#8B0000',
                'tratamiento': 'Suplementos, implantes retinales',
                'pronostico': 'Progresivo, investigación activa'
            }
        }
        
        # Tu código existente de architecture_info...
        self.architecture_info = {
            'CNN_Original': {
                'nombre_completo': 'CNN MobileNetV2 Original',
                'descripcion': 'Tu modelo inicial entrenado (70.44% accuracy)',
                'color': '#E91E63',
                'icon': '🧠',
                'ventajas': ['Tu modelo base', 'Conocido', 'Optimizado móvil'],
                'caracteristicas': {
                    'Tipo': 'Depthwise Separable Convolutions',
                    'Parámetros': '~3.5M',
                    'Ventaja principal': 'Eficiencia computacional',
                    'Año': '2018'
                }
            },
            'EfficientNetB0': {
                'nombre_completo': 'EfficientNet-B0',
                'descripcion': 'Arquitectura con compound scaling balanceado',
                'color': '#2196F3',
                'icon': '⚡',
                'ventajas': ['Compound scaling', 'Balance accuracy/params', 'Estado del arte'],
                'caracteristicas': {
                    'Tipo': 'Compound Scaling CNN',
                    'Parámetros': '~5.3M',
                    'Ventaja principal': 'Balance óptimo accuracy/eficiencia',
                    'Año': '2019'
                }
            },
            'ResNet50V2': {
                'nombre_completo': 'ResNet-50 V2',
                'descripcion': 'Red residual profunda con conexiones skip',
                'color': '#FF9800',
                'icon': '🔗',
                'ventajas': ['Conexiones residuales', 'Red profunda', 'Estable'],
                'caracteristicas': {
                    'Tipo': 'Residual Network',
                    'Parámetros': '~25.6M',
                    'Ventaja principal': 'Capacidad de representación profunda',
                    'Año': '2016'
                }
            }
        }
        
        self.models = None
        self.class_names = None
        self.individual_class_names = None
        self.current_analysis = None
        self.statistical_results = None  # Para almacenar resultados estadísticos
    
    @st.cache_resource
    def load_models(_self):
        """Carga las 3 arquitecturas para comparar"""
        try:
            models = {}
            
            # Mapeo de archivos a arquitecturas
            model_files = {
                'CNN_Original': 'eye_disease_model.h5',
                'EfficientNetB0': 'ensemble_efficientnet_model.h5', 
                'ResNet50V2': 'ensemble_resnet_model.h5'
            }
            
            for arch_name, filename in model_files.items():
                if os.path.exists(filename):
                    models[arch_name] = tf.keras.models.load_model(filename)
                    st.success(f"✅ {arch_name} cargado correctamente")
                else:
                    st.warning(f"⚠️ No se encontró {filename}")
            
            # Cargar nombres de clases
            ensemble_class_names = {}
            if os.path.exists('ensemble_class_indices.npy'):
                class_indices = np.load('ensemble_class_indices.npy', allow_pickle=True).item()
                ensemble_class_names = {v: k for k, v in class_indices.items()}
            
            individual_class_names = {}
            if os.path.exists('class_indices.npy'):
                class_indices = np.load('class_indices.npy', allow_pickle=True).item()
                individual_class_names = {v: k for k, v in class_indices.items()}
            
            # Nombres por defecto si no hay archivos
            if not ensemble_class_names:
                ensemble_class_names = {i: f"Clase_{i}" for i in range(10)}
            if not individual_class_names:
                individual_class_names = {i: f"Clase_{i}" for i in range(10)}
            
            return models, ensemble_class_names, individual_class_names
            
        except Exception as e:
            st.error(f"Error cargando modelos: {str(e)}")
            return {}, {}, {}
    
    def preprocess_image(self, image):
        """Preprocesa imagen para predicción"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = image.resize((224, 224))
            img_array = np.array(image)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            st.error(f"Error procesando imagen: {str(e)}")
            return None

    def preprocess_image_from_path(self, image_path):
        """Preprocesa imagen desde ruta para evaluación estadística"""
        try:
            image = Image.open(image_path)
            return self.preprocess_image(image)
        except Exception as e:
            st.error(f"Error procesando imagen {image_path}: {str(e)}")
            return None
    
    def predict_with_timing(self, model, img_array, arch_name):
        """Realiza predicción midiendo tiempo y métricas"""
        try:
            # Medir tiempo de predicción
            start_time = time.time()
            predictions = model.predict(img_array, verbose=0)
            end_time = time.time()
            
            prediction_time = end_time - start_time
            
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Usar nombres de clases correctos
            if arch_name == 'CNN_Original':
                predicted_class = self.individual_class_names[predicted_class_idx]
            else:
                predicted_class = self.class_names[predicted_class_idx]
            
            return {
                'architecture': arch_name,
                'predicted_class': predicted_class,
                'predicted_class_idx': predicted_class_idx,
                'confidence': confidence,
                'all_probabilities': predictions[0],
                'prediction_time': prediction_time,
                'model_size': self.get_model_size(model),
                'param_count': model.count_params()
            }
            
        except Exception as e:
            st.error(f"Error en predicción {arch_name}: {str(e)}")
            return None
    
    def get_model_size(self, model):
        """Calcula el tamaño del modelo en MB"""
        try:
            param_count = model.count_params()
            size_mb = (param_count * 4) / (1024 * 1024)
            return size_mb
        except:
            return 0
    
    # ========== NUEVAS FUNCIONES ESTADÍSTICAS ==========
    
    def calculate_matthews_correlation(self, y_true, y_pred):
        """Calcula el Coeficiente de Correlación de Matthews"""
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
            return mcc
        except Exception as e:
            st.error(f"Error calculando MCC: {str(e)}")
            return 0.0
    
    def mcnemar_test(self, y_true, y_pred1, y_pred2):
        """Realiza la prueba de McNemar entre dos modelos"""
        try:
            # Crear tabla de contingencia 2x2
            # Casos donde modelo1 correcto, modelo2 incorrecto
            correct_1_incorrect_2 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
            # Casos donde modelo1 incorrecto, modelo2 correcto  
            incorrect_1_correct_2 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
            
            # Tabla de contingencia
            contingency_table = np.array([
                [correct_1_incorrect_2, incorrect_1_correct_2],
                [incorrect_1_correct_2, correct_1_incorrect_2]
            ])
            
            # Calcular estadístico de McNemar con corrección de continuidad
            n = correct_1_incorrect_2 + incorrect_1_correct_2
            
            if n == 0:
                return {
                    'statistic': 0.0,
                    'p_value': 1.0,
                    'significant': False,
                    'contingency_table': contingency_table,
                    'interpretation': 'No hay diferencias entre modelos'
                }
            
            # McNemar con corrección de continuidad de Yates
            mcnemar_stat = (abs(correct_1_incorrect_2 - incorrect_1_correct_2) - 1)**2 / n
            p_value = 1 - chi2.cdf(mcnemar_stat, df=1)
            
            # Interpretación
            significant = p_value < 0.05
            
            if significant:
                if correct_1_incorrect_2 > incorrect_1_correct_2:
                    interpretation = "Modelo 1 significativamente mejor que Modelo 2"
                else:
                    interpretation = "Modelo 2 significativamente mejor que Modelo 1"
            else:
                interpretation = "No hay diferencia significativa entre modelos"
            
            return {
                'statistic': mcnemar_stat,
                'p_value': p_value,
                'significant': significant,
                'contingency_table': contingency_table,
                'interpretation': interpretation,
                'n_disagreements': n
            }
            
        except Exception as e:
            st.error(f"Error en prueba McNemar: {str(e)}")
            return None
    
    def calculate_confidence_interval_mcc(self, y_true, y_pred, confidence=0.95):
        """Calcula intervalo de confianza bootstrap para MCC"""
        try:
            n_bootstrap = 1000
            bootstrap_mccs = []
            
            n_samples = len(y_true)
            
            for _ in range(n_bootstrap):
                # Bootstrap resampling
                indices = np.random.choice(n_samples, n_samples, replace=True)
                y_true_bootstrap = y_true[indices]
                y_pred_bootstrap = y_pred[indices]
                
                try:
                    mcc_bootstrap = matthews_corrcoef(y_true_bootstrap, y_pred_bootstrap)
                    bootstrap_mccs.append(mcc_bootstrap)
                except:
                    continue
            
            if len(bootstrap_mccs) == 0:
                return None, None
            
            alpha = 1 - confidence
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100
            
            ci_lower = np.percentile(bootstrap_mccs, lower_percentile)
            ci_upper = np.percentile(bootstrap_mccs, upper_percentile)
            
            return ci_lower, ci_upper
            
        except Exception as e:
            st.error(f"Error calculando IC para MCC: {str(e)}")
            return None, None
    
    def scan_dataset_folder(self, dataset_path):
        """Escanea carpeta de dataset y crea lista de imágenes con etiquetas"""
        try:
            dataset_path = Path(dataset_path)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            
            images_data = []
            class_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
            
            if not class_folders:
                st.error("No se encontraron carpetas de clases en el dataset")
                return None
            
            # Mapear nombres de carpetas a índices
            class_name_to_idx = {}
            
            st.info(f"📁 Clases encontradas: {len(class_folders)}")
            
            for class_idx, class_folder in enumerate(sorted(class_folders)):
                class_name = class_folder.name
                class_name_to_idx[class_name] = class_idx
                
                st.write(f"• **Clase {class_idx}**: {class_name}")
                
                # Buscar imágenes en la carpeta
                images_in_class = []
                for ext in image_extensions:
                    images_in_class.extend(class_folder.glob(f'*{ext}'))
                    images_in_class.extend(class_folder.glob(f'*{ext.upper()}'))
                
                for img_path in images_in_class:
                    images_data.append({
                        'image_path': str(img_path),
                        'true_label': class_idx,
                        'class_name': class_name
                    })
            
            st.success(f"✅ Total de imágenes encontradas: {len(images_data)}")
            
            return images_data, class_name_to_idx
            
        except Exception as e:
            st.error(f"Error escaneando dataset: {str(e)}")
            return None, None

    def evaluate_models_on_dataset(self, dataset_input):
        """Evalúa todos los modelos en un dataset (carpeta o CSV)"""
        try:
            # Determinar si es carpeta o archivo CSV
            if isinstance(dataset_input, str) and dataset_input.endswith('.csv'):
                # Leer CSV
                df_test = pd.read_csv(dataset_input)
                
                if 'image_path' not in df_test.columns or 'true_label' not in df_test.columns:
                    st.error("El archivo CSV debe contener columnas 'image_path' y 'true_label'")
                    return None
                
                images_data = df_test.to_dict('records')
                
            else:
                # Escanear carpeta de dataset
                images_data, class_mapping = self.scan_dataset_folder(dataset_input)
                if images_data is None:
                    return None
            
            results = {
                'true_labels': [],
                'predictions': {arch: [] for arch in self.models.keys()},
                'confidences': {arch: [] for arch in self.models.keys()},
                'prediction_times': {arch: [] for arch in self.models.keys()}
            }
            
            total_images = len(images_data)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Procesar imágenes en lotes para eficiencia
            for idx, img_data in enumerate(images_data):
                status_text.text(f"Evaluando imagen {idx+1}/{total_images}: {Path(img_data['image_path']).name}")
                
                # Preprocesar imagen
                img_array = self.preprocess_image_from_path(img_data['image_path'])
                if img_array is None:
                    continue
                
                # Etiqueta verdadera
                true_label = img_data['true_label']
                results['true_labels'].append(true_label)
                
                # Predecir con cada modelo
                for arch_name, model in self.models.items():
                    pred_result = self.predict_with_timing(model, img_array, arch_name)
                    
                    if pred_result:
                        results['predictions'][arch_name].append(pred_result['predicted_class_idx'])
                        results['confidences'][arch_name].append(pred_result['confidence'])
                        results['prediction_times'][arch_name].append(pred_result['prediction_time'])
                    else:
                        results['predictions'][arch_name].append(-1)  # Error
                        results['confidences'][arch_name].append(0.0)
                        results['prediction_times'][arch_name].append(0.0)
                
                # Actualizar progreso
                progress_bar.progress((idx + 1) / total_images)
                
                # Mostrar progreso cada 50 imágenes
                if (idx + 1) % 50 == 0:
                    st.write(f"✅ Procesadas {idx + 1}/{total_images} imágenes")
            
            progress_bar.empty()
            status_text.empty()
            
            # Convertir a arrays numpy
            results['true_labels'] = np.array(results['true_labels'])
            for arch in self.models.keys():
                results['predictions'][arch] = np.array(results['predictions'][arch])
                results['confidences'][arch] = np.array(results['confidences'][arch])
                results['prediction_times'][arch] = np.array(results['prediction_times'][arch])
            
            return results
            
        except Exception as e:
            st.error(f"Error evaluando modelos: {str(e)}")
            return None
    
    def perform_statistical_analysis(self, evaluation_results):
        """Realiza análisis estadístico completo"""
        try:
            y_true = evaluation_results['true_labels']
            architectures = list(self.models.keys())
            
            statistical_results = {
                'mcc_scores': {},
                'mcc_confidence_intervals': {},
                'accuracy_scores': {},
                'mcnemar_results': {},
                'confusion_matrices': {},
                'classification_reports': {}
            }
            
            # Calcular MCC y accuracy para cada modelo
            for arch in architectures:
                y_pred = evaluation_results['predictions'][arch]
                
                # MCC
                mcc = self.calculate_matthews_correlation(y_true, y_pred)
                statistical_results['mcc_scores'][arch] = mcc
                
                # Intervalo de confianza para MCC
                ci_lower, ci_upper = self.calculate_confidence_interval_mcc(y_true, y_pred)
                statistical_results['mcc_confidence_intervals'][arch] = (ci_lower, ci_upper)
                
                # Accuracy
                accuracy = np.mean(y_true == y_pred)
                statistical_results['accuracy_scores'][arch] = accuracy
                
                # Matriz de confusión
                cm = confusion_matrix(y_true, y_pred)
                statistical_results['confusion_matrices'][arch] = cm
                
                # Reporte de clasificación
                try:
                    class_report = classification_report(y_true, y_pred, output_dict=True)
                    statistical_results['classification_reports'][arch] = class_report
                except:
                    statistical_results['classification_reports'][arch] = {}
            
            # Pruebas de McNemar entre pares de modelos
            for arch1, arch2 in itertools.combinations(architectures, 2):
                y_pred1 = evaluation_results['predictions'][arch1]
                y_pred2 = evaluation_results['predictions'][arch2]
                
                mcnemar_result = self.mcnemar_test(y_true, y_pred1, y_pred2)
                statistical_results['mcnemar_results'][f"{arch1}_vs_{arch2}"] = mcnemar_result
            
            return statistical_results
            
        except Exception as e:
            st.error(f"Error en análisis estadístico: {str(e)}")
            return None
    
    def display_statistical_analysis_section(self):
        """Sección completa de análisis estadístico"""
        st.markdown("---")
        st.header("📊 ANÁLISIS ESTADÍSTICO INFERENCIAL")
        st.markdown("""
        **Evaluación rigurosa con pruebas estadísticas:**
        - 🎯 **Coeficiente de Matthews (MCC)**: Métrica balanceada que considera todos los casos de la matriz de confusión
        - 🔬 **Prueba de McNemar**: Comparación estadística entre pares de modelos
        - 📈 **Intervalos de Confianza**: Bootstrap CI para robustez estadística
        """)
        
        # Dataset de evaluación
        st.subheader("📂 Dataset de Evaluación")
        
        # Input de ruta de carpeta
        dataset_folder = st.text_input(
            "🗂️ Ruta de la carpeta de pruebas:",
            value="Pruebas",  # Valor por defecto
            help="Ejemplo: Pruebas, ./Pruebas, /path/to/Pruebas"
        )
        
        # Mostrar estructura esperada
        with st.expander("📋 Estructura de carpetas esperada"):
            st.code("""
    📂 Pruebas/
    ├── 📁 Central_Serous_Chorioretinopathy/
    │   ├── 🖼️ test001.jpg
    │   ├── 🖼️ test002.jpg
    │   └── ...
    ├── 📁 Diabetic_Retinopathy/
    │   ├── 🖼️ test003.jpg
    │   ├── 🖼️ test004.jpg
    │   └── ...
    ├── 📁 Glaucoma/
    │   ├── 🖼️ test005.jpg
    │   └── ...
    └── 📁 Healthy/
        ├── 🖼️ test006.jpg
        └── ...

    ✅ Cada carpeta = una clase
    ✅ Nombres de carpetas = nombres de clases
    ✅ Formatos soportados: .jpg, .jpeg, .png, .bmp, .tiff
            """)
        
        # Verificar si la carpeta existe
        if dataset_folder:
            dataset_path = Path(dataset_folder)
            if dataset_path.exists() and dataset_path.is_dir():
                st.success(f"✅ Carpeta encontrada: {dataset_path.absolute()}")
                
                # Vista previa del dataset
                if st.button("👀 Vista Previa del Dataset", key="preview_dataset"):
                    with st.spinner("🔍 Escaneando dataset..."):
                        preview_data, class_mapping = self.scan_dataset_folder(dataset_path)
                        
                        if preview_data:
                            st.markdown("#### 📊 Resumen del Dataset:")
                            
                            # Crear DataFrame para mostrar distribución
                            df_preview = pd.DataFrame(preview_data)
                            class_counts = df_preview['class_name'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📈 Distribución por Clase:**")
                                for class_name, count in class_counts.items():
                                    st.markdown(f"• **{class_name}**: {count} imágenes")
                            
                            with col2:
                                # Gráfico de distribución
                                fig_dist = go.Figure(data=[
                                    go.Bar(x=class_counts.index, y=class_counts.values)
                                ])
                                fig_dist.update_layout(
                                    title="📊 Distribución de Imágenes por Clase",
                                    xaxis_title="Clases",
                                    yaxis_title="Número de Imágenes",
                                    height=400
                                )
                                st.plotly_chart(fig_dist, use_container_width=True)
                            
                            st.dataframe(df_preview.head(10), use_container_width=True)
                
                # Botón de evaluación
                if st.button("🚀 INICIAR EVALUACIÓN ESTADÍSTICA", type="primary", use_container_width=True, key="eval_folder"):
                    st.info("🔄 Evaluando modelos en dataset completo... Esto puede tomar varios minutos.")
                    
                    # Evaluar modelos
                    evaluation_results = self.evaluate_models_on_dataset(str(dataset_path))
                    
                    if evaluation_results is not None:
                        st.success("✅ Evaluación completada! Realizando análisis estadístico...")
                        
                        # Análisis estadístico
                        statistical_results = self.perform_statistical_analysis(evaluation_results)
                        
                        if statistical_results is not None:
                            # Guardar en session state
                            st.session_state.statistical_results = statistical_results
                            st.session_state.evaluation_results = evaluation_results
                            
                            # Mostrar resultados
                            self.display_statistical_results(statistical_results, evaluation_results)
            
            else:
                st.error(f"❌ No se encontró la carpeta: {dataset_folder}")
                st.markdown("**💡 Sugerencias:**")
                st.markdown("• Verifica que la ruta sea correcta")
                st.markdown("• Usa rutas relativas como `Pruebas` o `./Pruebas`")
                st.markdown("• O rutas absolutas como `/ruta/completa/Pruebas`")
        
        # Mostrar resultados si ya están calculados
        if hasattr(st.session_state, 'statistical_results') and st.session_state.statistical_results:
            st.markdown("---")
            st.info("📊 Mostrando resultados de análisis estadístico previo")
            self.display_statistical_results(
                st.session_state.statistical_results, 
                st.session_state.evaluation_results
            )
    
    def display_statistical_results(self, statistical_results, evaluation_results):
        """Muestra resultados del análisis estadístico"""
        
        # === SECCIÓN 1: COEFICIENTE DE MATTHEWS ===
        st.subheader("🎯 Coeficiente de Correlación de Matthews (MCC)")
        
        st.markdown("""
        **MCC** es una métrica balanceada que funciona bien incluso con clases desbalanceadas.
        - **Rango**: -1 (completamente incorrecto) a +1 (predicción perfecta)
        - **0**: Predicción aleatoria
        - **>0.5**: Excelente rendimiento
        """)
        
        # Tabla de MCC con intervalos de confianza
        mcc_data = []
        for arch in self.models.keys():
            mcc_score = statistical_results['mcc_scores'][arch]
            ci_lower, ci_upper = statistical_results['mcc_confidence_intervals'][arch]
            accuracy = statistical_results['accuracy_scores'][arch]
            
            mcc_data.append({
                'Arquitectura': arch.replace('_', ' '),
                'MCC': f"{mcc_score:.4f}",
                'IC 95% Inferior': f"{ci_lower:.4f}" if ci_lower else "N/A",
                'IC 95% Superior': f"{ci_upper:.4f}" if ci_upper else "N/A",
                'Accuracy': f"{accuracy:.4f}",
                'Interpretación': self.interpret_mcc(mcc_score)
            })
        
        df_mcc = pd.DataFrame(mcc_data)
        st.dataframe(df_mcc, use_container_width=True)
        
        # Gráfico de MCC con intervalos de confianza
        fig_mcc = go.Figure()
        
        architectures = list(self.models.keys())
        mcc_scores = [statistical_results['mcc_scores'][arch] for arch in architectures]
        ci_lowers = [statistical_results['mcc_confidence_intervals'][arch][0] for arch in architectures]
        ci_uppers = [statistical_results['mcc_confidence_intervals'][arch][1] for arch in architectures]
        
        # Barras con intervalos de confianza
        fig_mcc.add_trace(go.Bar(
            x=[arch.replace('_', ' ') for arch in architectures],
            y=mcc_scores,
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ci_upper - mcc for ci_upper, mcc in zip(ci_uppers, mcc_scores)],
                arrayminus=[mcc - ci_lower for ci_lower, mcc in zip(ci_lowers, mcc_scores)]
            ),
            marker_color=[self.architecture_info[arch]['color'] for arch in architectures],
            text=[f"{mcc:.3f}" for mcc in mcc_scores],
            textposition='auto'
        ))
        
        fig_mcc.update_layout(
            title="🎯 Coeficiente de Matthews con Intervalos de Confianza (95%)",
            yaxis_title="MCC Score",
            yaxis=dict(range=[-1, 1]),
            height=500
        )
        
        # Líneas de referencia
        fig_mcc.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Predicción Aleatoria")
        fig_mcc.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="Excelente Rendimiento")
        
        st.plotly_chart(fig_mcc, use_container_width=True)
        
        # === SECCIÓN 2: PRUEBAS DE MCNEMAR ===
        st.subheader("🔬 Pruebas de McNemar - Comparación entre Modelos")
        
        st.markdown("""
        **Prueba de McNemar** compara estadísticamente el rendimiento entre pares de modelos:
        - **H₀**: No hay diferencia entre los modelos
        - **H₁**: Hay diferencia significativa
        - **α = 0.05**: Nivel de significancia
        """)
        
        mcnemar_data = []
        for comparison, result in statistical_results['mcnemar_results'].items():
            if result:
                arch1, arch2 = comparison.split('_vs_')
                
                mcnemar_data.append({
                    'Comparación': f"{arch1.replace('_', ' ')} vs {arch2.replace('_', ' ')}",
                    'Estadístico McNemar': f"{result['statistic']:.4f}",
                    'p-valor': f"{result['p_value']:.6f}",
                    'Significativo (α=0.05)': "✅ Sí" if result['significant'] else "❌ No",
                    'Interpretación': result['interpretation'],
                    'N° Desacuerdos': result['n_disagreements']
                })
        
        df_mcnemar = pd.DataFrame(mcnemar_data)
        st.dataframe(df_mcnemar, use_container_width=True)
        
        # Heatmap de p-valores
        self.plot_mcnemar_heatmap(statistical_results['mcnemar_results'])
        
        # === SECCIÓN 3: MATRICES DE CONFUSIÓN ===
        st.subheader("🎭 Matrices de Confusión por Arquitectura")
        
        cols = st.columns(len(self.models))
        
        for i, (arch, cm) in enumerate(statistical_results['confusion_matrices'].items()):
            with cols[i]:
                fig_cm = self.plot_confusion_matrix(cm, arch)
                st.plotly_chart(fig_cm, use_container_width=True)
        
        # === SECCIÓN 4: ANÁLISIS DE SIGNIFICANCIA ===
        st.subheader("📈 Análisis de Significancia Estadística")
        
        # Resumen de significancia
        significant_comparisons = [
            comp for comp, result in statistical_results['mcnemar_results'].items() 
            if result and result['significant']
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="🔬 Comparaciones Significativas",
                value=len(significant_comparisons),
                delta=f"de {len(statistical_results['mcnemar_results'])} totales"
            )
        
        with col2:
            best_mcc_arch = max(statistical_results['mcc_scores'], key=statistical_results['mcc_scores'].get)
            best_mcc_score = statistical_results['mcc_scores'][best_mcc_arch]
            
            st.metric(
                label="🏆 Mejor MCC",
                value=f"{best_mcc_score:.4f}",
                delta=f"{best_mcc_arch.replace('_', ' ')}"
            )
        
        # Recomendaciones estadísticas
        st.subheader("💡 Recomendaciones Estadísticas")
        
        if len(significant_comparisons) == 0:
            st.warning("""
            ⚠️ **No se encontraron diferencias estadísticamente significativas** entre los modelos.
            
            **Implicaciones:**
            - Los modelos tienen rendimiento similar estadísticamente
            - Otros criterios (velocidad, tamaño) pueden ser decisivos
            - Se recomienda aumentar el tamaño del dataset de prueba
            """)
        else:
            st.success(f"""
            ✅ **Se encontraron {len(significant_comparisons)} diferencias significativas**
            
            **Modelos con diferencias estadísticamente probadas:**
            """)
            
            for comp in significant_comparisons:
                result = statistical_results['mcnemar_results'][comp]
                st.markdown(f"• **{comp.replace('_', ' ')}**: {result['interpretation']}")
        
        # === SECCIÓN 5: EXPORTAR RESULTADOS ESTADÍSTICOS ===
        st.subheader("📤 Exportar Resultados Estadísticos")
        
        if st.button("📊 Generar Reporte Estadístico Completo", use_container_width=True):
            self.generate_statistical_report(statistical_results, evaluation_results)
    
    def interpret_mcc(self, mcc_score):
        """Interpreta el score MCC"""
        if mcc_score >= 0.8:
            return "🟢 Excelente"
        elif mcc_score >= 0.6:
            return "🔵 Muy bueno"
        elif mcc_score >= 0.4:
            return "🟡 Bueno"
        elif mcc_score >= 0.2:
            return "🟠 Regular"
        elif mcc_score >= 0:
            return "🔴 Bajo"
        else:
            return "🔴 Muy bajo"
    
    def plot_mcnemar_heatmap(self, mcnemar_results):
        """Crea heatmap de p-valores de McNemar"""
        try:
            architectures = list(self.models.keys())
            n_archs = len(architectures)
            
            # Matriz de p-valores
            p_value_matrix = np.ones((n_archs, n_archs))
            significance_matrix = np.zeros((n_archs, n_archs))
            
            for i, arch1 in enumerate(architectures):
                for j, arch2 in enumerate(architectures):
                    if i != j:
                        comp_key = f"{arch1}_vs_{arch2}"
                        if comp_key in mcnemar_results:
                            result = mcnemar_results[comp_key]
                            p_value_matrix[i, j] = result['p_value']
                            significance_matrix[i, j] = 1 if result['significant'] else 0
                        else:
                            # Buscar comparación inversa
                            comp_key_inv = f"{arch2}_vs_{arch1}"
                            if comp_key_inv in mcnemar_results:
                                result = mcnemar_results[comp_key_inv]
                                p_value_matrix[i, j] = result['p_value']
                                significance_matrix[i, j] = 1 if result['significant'] else 0
            
            # Crear heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=p_value_matrix,
                x=[arch.replace('_', ' ') for arch in architectures],
                y=[arch.replace('_', ' ') for arch in architectures],
                colorscale='RdYlBu_r',
                text=[[f"p={p_value_matrix[i,j]:.4f}" for j in range(n_archs)] for i in range(n_archs)],
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            # Añadir línea de significancia
            fig_heatmap.add_shape(
                type="line",
                x0=-0.5, y0=-0.5, x1=n_archs-0.5, y1=n_archs-0.5,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_heatmap.update_layout(
                title="🔬 Heatmap de p-valores (Pruebas de McNemar)<br>Valores < 0.05 indican diferencia significativa",
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creando heatmap: {str(e)}")
    
    def plot_confusion_matrix(self, cm, architecture_name):
        """Crea matriz de confusión interactiva"""
        try:
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig_cm.update_layout(
                title=f"📊 {architecture_name.replace('_', ' ')}",
                xaxis_title="Predicción",
                yaxis_title="Verdadero",
                height=300
            )
            
            return fig_cm
            
        except Exception as e:
            st.error(f"Error creando matriz de confusión: {str(e)}")
            return go.Figure()
    
    def generate_statistical_report(self, statistical_results, evaluation_results):
        """Genera reporte estadístico completo"""
        try:
            # Crear reporte en formato texto
            report_content = self.create_statistical_report_content(statistical_results, evaluation_results)
            
            # Crear archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"reporte_estadistico_{timestamp}.txt"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            # Crear también JSON con datos estructurados
            json_filename = f"datos_estadisticos_{timestamp}.json"
            
            # Convertir numpy arrays para JSON
            json_data = {}
            for key, value in statistical_results.items():
                if isinstance(value, dict):
                    json_data[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            json_data[key][subkey] = subvalue.tolist()
                        elif isinstance(subvalue, np.integer):
                            json_data[key][subkey] = int(subvalue)
                        elif isinstance(subvalue, np.floating):
                            json_data[key][subkey] = float(subvalue)
                        else:
                            json_data[key][subkey] = subvalue
                else:
                    json_data[key] = value
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # Botones de descarga
            col1, col2 = st.columns(2)
            
            with col1:
                with open(report_filename, 'r', encoding='utf-8') as f:
                    report_data = f.read()
                
                st.download_button(
                    label="📄 Descargar Reporte TXT",
                    data=report_data,
                    file_name=report_filename,
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                with open(json_filename, 'r', encoding='utf-8') as f:
                    json_data = f.read()
                
                st.download_button(
                    label="📊 Descargar Datos JSON",
                    data=json_data,
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )
            
            st.success("✅ Reportes estadísticos generados correctamente!")
            
            # Limpiar archivos temporales
            try:
                os.remove(report_filename)
                os.remove(json_filename)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error generando reporte estadístico: {str(e)}")
    
    def create_statistical_report_content(self, statistical_results, evaluation_results):
        """Crea contenido del reporte estadístico"""
        report = f"""
REPORTE DE ANÁLISIS ESTADÍSTICO INFERENCIAL
===========================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Sistema: Comparación de Arquitecturas CNN para Diagnóstico Ocular
Número de modelos evaluados: {len(self.models)}
Tamaño del dataset de prueba: {len(evaluation_results['true_labels'])}

1. COEFICIENTE DE CORRELACIÓN DE MATTHEWS (MCC)
===============================================

El MCC es una métrica balanceada que considera todas las categorías de la matriz de confusión.
Rango: -1 (completamente incorrecto) a +1 (predicción perfecta)

"""
        
        # Resultados MCC
        for arch in self.models.keys():
            mcc_score = statistical_results['mcc_scores'][arch]
            ci_lower, ci_upper = statistical_results['mcc_confidence_intervals'][arch]
            accuracy = statistical_results['accuracy_scores'][arch]
            
            report += f"""
{arch.replace('_', ' ')}:
  - MCC Score: {mcc_score:.6f}
  - IC 95%: [{ci_lower:.6f}, {ci_upper:.6f}]
  - Accuracy: {accuracy:.6f}
  - Interpretación: {self.interpret_mcc(mcc_score)}
"""
        
        report += f"""

2. PRUEBAS DE MCNEMAR
====================

La prueba de McNemar compara estadísticamente el rendimiento entre pares de modelos.
H₀: No hay diferencia entre los modelos
H₁: Hay diferencia significativa (α = 0.05)

"""
        
        # Resultados McNemar
        for comparison, result in statistical_results['mcnemar_results'].items():
            if result:
                report += f"""
{comparison.replace('_', ' ')}:
  - Estadístico McNemar: {result['statistic']:.6f}
  - p-valor: {result['p_value']:.6f}
  - Significativo: {'Sí' if result['significant'] else 'No'}
  - N° desacuerdos: {result['n_disagreements']}
  - Interpretación: {result['interpretation']}
"""
        
        # Resumen y recomendaciones
        significant_comparisons = [
            comp for comp, result in statistical_results['mcnemar_results'].items() 
            if result and result['significant']
        ]
        
        best_mcc_arch = max(statistical_results['mcc_scores'], key=statistical_results['mcc_scores'].get)
        best_mcc_score = statistical_results['mcc_scores'][best_mcc_arch]
        
        report += f"""

3. RESUMEN Y RECOMENDACIONES
============================

Mejor modelo por MCC: {best_mcc_arch.replace('_', ' ')} (MCC = {best_mcc_score:.6f})
Comparaciones significativas: {len(significant_comparisons)} de {len(statistical_results['mcnemar_results'])}

"""
        
        if len(significant_comparisons) == 0:
            report += """
CONCLUSIÓN:
No se encontraron diferencias estadísticamente significativas entre los modelos.
Esto sugiere que todos los modelos tienen un rendimiento similar estadísticamente.
La selección del modelo puede basarse en otros criterios como velocidad o eficiencia.

"""
        else:
            report += f"""
CONCLUSIÓN:
Se encontraron {len(significant_comparisons)} diferencias estadísticamente significativas:

"""
            for comp in significant_comparisons:
                result = statistical_results['mcnemar_results'][comp]
                report += f"- {comp.replace('_', ' ')}: {result['interpretation']}\n"
        
        report += f"""

4. DETALLES TÉCNICOS
====================

Dataset de evaluación: {len(evaluation_results['true_labels'])} imágenes
Métodos estadísticos utilizados:
- Coeficiente de Correlación de Matthews
- Prueba de McNemar con corrección de continuidad de Yates
- Intervalos de confianza bootstrap (95%)
- Matrices de confusión

Arquitecturas evaluadas:
"""
        
        for arch in self.models.keys():
            info = self.architecture_info[arch]
            report += f"- {info['nombre_completo']}: {info['descripcion']}\n"
        
        return report
    
    # ========== FUNCIONES ORIGINALES (MANTENER TODAS) ==========
    
    def find_best_architecture(self, predictions):
        """Encuentra la mejor arquitectura por diferentes métricas"""
        if not predictions or len(predictions) < 2:
            return {}
        
        # Mejor por confianza
        best_confidence = max(predictions, key=lambda x: x['confidence'])
        
        # Más rápido
        fastest = min(predictions, key=lambda x: x['prediction_time'])
        
        # Más eficiente (mayor confianza / tiempo)
        for pred in predictions:
            pred['efficiency'] = pred['confidence'] / pred['prediction_time']
        most_efficient = max(predictions, key=lambda x: x['efficiency'])
        
        # Más ligero
        lightest = min(predictions, key=lambda x: x['model_size'])
        
        return {
            'highest_confidence': best_confidence,
            'fastest': fastest,
            'most_efficient': most_efficient,
            'lightest': lightest
        }
    
    def display_header(self):
        """Header de la aplicación"""
        st.title("🏆 DETECCION DE ENFERMEDADES OCULARES 👁️")
        st.subheader("MobileNetV2 vs EfficientNet-B0 vs ResNet-50 V2 + Análisis Estadístico")
        st.markdown("---")
    
    def display_architecture_showcase(self):
        """Muestra las características de cada arquitectura"""
        st.header("🏗️ LAS 3 ARQUITECTURAS EN COMPETENCIA")
        
        cols = st.columns(3)
        
        for i, (arch_name, info) in enumerate(self.architecture_info.items()):
            with cols[i]:
                # Header de la arquitectura
                st.subheader(f"{info['icon']} {info['nombre_completo']}")
                
                # Descripción
                st.info(f"**{info['descripcion']}**")
                
                # Características técnicas
                st.markdown("**📊 Características:**")
                st.markdown(f"• **Tipo:** {info['caracteristicas']['Tipo']}")
                st.markdown(f"• **Parámetros:** {info['caracteristicas']['Parámetros']}")
                st.markdown(f"• **Ventaja:** {info['caracteristicas']['Ventaja principal']}")
                st.markdown(f"• **Año:** {info['caracteristicas']['Año']}")
                
                # Ventajas
                st.markdown("**✅ Ventajas:**")
                for ventaja in info['ventajas']:
                    st.markdown(f"• {ventaja}")
                
                st.markdown("---")
    
    def display_prediction_results(self, predictions):
        """Muestra resultados de las 3 arquitecturas lado a lado"""
        st.header("🎯 RESULTADOS DE PREDICCIÓN")
        
        cols = st.columns(3)
        
        for i, pred in enumerate(predictions):
            arch_name = pred['architecture']
            info = self.architecture_info[arch_name]
            
            with cols[i]:
                # Nombre de la arquitectura
                st.subheader(f"{info['icon']} {arch_name.replace('_', ' ')}")
                
                # Diagnóstico
                predicted_class = pred['predicted_class']
                class_info = self.class_info.get(predicted_class, {})
                nombre_es = class_info.get('nombre', predicted_class)
                
                st.success(f"**Diagnóstico:** {nombre_es}")
                
                # Confianza (métrica principal)
                st.metric(
                    label="🎯 Confianza",
                    value=f"{pred['confidence']:.1%}",
                    delta=None
                )
                
                # Métricas técnicas
                st.markdown("**📊 Métricas Técnicas:**")
                st.markdown(f"⏱️ **Tiempo:** {pred['prediction_time']:.3f}s")
                st.markdown(f"💾 **Tamaño:** {pred['model_size']:.1f}MB")
                st.markdown(f"🔢 **Parámetros:** {pred['param_count']:,}")
                
                st.markdown("---")
    
    def display_performance_comparison(self, predictions):
        """Gráficos comparativos de rendimiento"""
        st.markdown("## 📊 ANÁLISIS COMPARATIVO DE RENDIMIENTO")
        
        # Crear DataFrame para gráficos
        df = pd.DataFrame([
            {
                'Arquitectura': pred['architecture'].replace('_', ' '),
                'Confianza': pred['confidence'],
                'Tiempo (s)': pred['prediction_time'],
                'Tamaño (MB)': pred['model_size'],
                'Parámetros (M)': pred['param_count'] / 1_000_000,
                'Eficiencia (Conf/Tiempo)': pred['confidence'] / pred['prediction_time']
            }
            for pred in predictions
        ])
        
        # Colores para gráficos
        colors = [self.architecture_info[pred['architecture']]['color'] for pred in predictions]
        
        # 4 gráficos en 2x2
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de confianza
            fig_conf = go.Figure(data=[
                go.Bar(
                    x=df['Arquitectura'],
                    y=df['Confianza'],
                    text=[f"{conf:.1%}" for conf in df['Confianza']],
                    textposition='auto',
                    marker_color=colors,
                    name='Confianza'
                )
            ])
            fig_conf.update_layout(
                title='🎯 Confianza de Predicción',
                yaxis=dict(tickformat='.0%'),
                height=400
            )
            st.plotly_chart(fig_conf, use_container_width=True)
            
            # Gráfico de tamaño
            fig_size = go.Figure(data=[
                go.Bar(
                    x=df['Arquitectura'],
                    y=df['Tamaño (MB)'],
                    text=[f"{size:.1f}MB" for size in df['Tamaño (MB)']],
                    textposition='auto',
                    marker_color=colors,
                    name='Tamaño'
                )
            ])
            fig_size.update_layout(
                title='💾 Tamaño del Modelo',
                yaxis_title='Tamaño (MB)',
                height=400
            )
            st.plotly_chart(fig_size, use_container_width=True)
        
        with col2:
            # Gráfico de tiempo
            fig_time = go.Figure(data=[
                go.Bar(
                    x=df['Arquitectura'],
                    y=df['Tiempo (s)'],
                    text=[f"{time:.3f}s" for time in df['Tiempo (s)']],
                    textposition='auto',
                    marker_color=colors,
                    name='Tiempo'
                )
            ])
            fig_time.update_layout(
                title='⏱️ Tiempo de Predicción',
                yaxis_title='Tiempo (segundos)',
                height=400
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Gráfico de eficiencia
            fig_eff = go.Figure(data=[
                go.Bar(
                    x=df['Arquitectura'],
                    y=df['Eficiencia (Conf/Tiempo)'],
                    text=[f"{eff:.1f}" for eff in df['Eficiencia (Conf/Tiempo)']],
                    textposition='auto',
                    marker_color=colors,
                    name='Eficiencia'
                )
            ])
            fig_eff.update_layout(
                title='⚡ Eficiencia (Confianza/Tiempo)',
                yaxis_title='Eficiencia Score',
                height=400
            )
            st.plotly_chart(fig_eff, use_container_width=True)
    
    def display_radar_comparison(self, predictions):
        """Gráfico radar comparando todas las métricas"""
        st.markdown("### 🕸️ Comparación Multidimensional")
        
        # Normalizar métricas para el radar (0-1)
        max_conf = max(pred['confidence'] for pred in predictions)
        min_time = min(pred['prediction_time'] for pred in predictions)
        max_time = max(pred['prediction_time'] for pred in predictions)
        min_size = min(pred['model_size'] for pred in predictions)
        max_size = max(pred['model_size'] for pred in predictions)
        
        fig = go.Figure()
        
        categories = ['Confianza', 'Velocidad', 'Eficiencia Memoria', 'Score General']
        
        for pred in predictions:
            arch_name = pred['architecture']
            info = self.architecture_info[arch_name]
            
            # Normalizar valores (más alto = mejor)
            norm_conf = pred['confidence'] / max_conf if max_conf > 0 else 0
            norm_speed = (max_time - pred['prediction_time']) / (max_time - min_time) if max_time > min_time else 1
            norm_memory = (max_size - pred['model_size']) / (max_size - min_size) if max_size > min_size else 1
            norm_general = (norm_conf + norm_speed + norm_memory) / 3
            
            values = [norm_conf, norm_speed, norm_memory, norm_general]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=arch_name.replace('_', ' '),
                line_color=info['color']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.0%'
                )),
            title="🕸️ Perfil Multidimensional de Arquitecturas",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_winners_podium(self, best_models):
        """Muestra el podio de ganadores por categoría"""
        st.header("🏆 PODIO DE GANADORES")
        
        categories = [
            ('highest_confidence', '🎯 Mayor Confianza', 'El más preciso'),
            ('fastest', '⚡ Más Rápido', 'El velocista'),
            ('lightest', '🪶 Más Ligero', 'El eficiente'),
            ('most_efficient', '⚖️ Más Eficiente', 'El balanceado')
        ]
        
        cols = st.columns(2)
        
        for i, (key, title, subtitle) in enumerate(categories):
            col = cols[i % 2]
            
            with col:
                if key in best_models:
                    winner = best_models[key]
                    arch_name = winner['architecture']
                    info = self.architecture_info[arch_name]
                    
                    if key == 'highest_confidence':
                        metric_value = f"{winner['confidence']:.1%}"
                    elif key == 'fastest':
                        metric_value = f"{winner['prediction_time']:.3f}s"
                    elif key == 'lightest':
                        metric_value = f"{winner['model_size']:.1f}MB"
                    else:  # most_efficient
                        metric_value = f"{winner['efficiency']:.1f}"
                    
                    # Usar diferentes tipos de alertas para cada categoría
                    if key == 'highest_confidence':
                        st.success(f"**{title}**\n\n{info['icon']} **{arch_name.replace('_', ' ')}**\n\n{metric_value}\n\n*{subtitle}*")
                    elif key == 'fastest':
                        st.info(f"**{title}**\n\n{info['icon']} **{arch_name.replace('_', ' ')}**\n\n{metric_value}\n\n*{subtitle}*")
                    elif key == 'lightest':
                        st.warning(f"**{title}**\n\n{info['icon']} **{arch_name.replace('_', ' ')}**\n\n{metric_value}\n\n*{subtitle}*")
                    else:  # most_efficient
                        st.error(f"**{title}**\n\n{info['icon']} **{arch_name.replace('_', ' ')}**\n\n{metric_value}\n\n*{subtitle}*")
    
    def display_detailed_analysis(self, predictions, best_models):
        """Análisis detallado y recomendaciones"""
        st.markdown("## 🔬 ANÁLISIS DETALLADO")
        
        # Encontrar el mejor general (combinación de métricas)
        for pred in predictions:
            # Score combinado: 50% confianza + 25% velocidad + 25% eficiencia memoria
            max_conf = max(p['confidence'] for p in predictions)
            min_time = min(p['prediction_time'] for p in predictions)
            min_size = min(p['model_size'] for p in predictions)
            
            conf_score = pred['confidence'] / max_conf
            speed_score = min_time / pred['prediction_time']
            memory_score = min_size / pred['model_size']
            
            pred['overall_score'] = 0.5 * conf_score + 0.25 * speed_score + 0.25 * memory_score
        
        best_overall = max(predictions, key=lambda x: x['overall_score'])
        
        # Mostrar ganador general
        arch_name = best_overall['architecture']
        info = self.architecture_info[arch_name]
        
        st.balloons()  # Celebración!
        st.success(f"## 👑 GANADOR GENERAL: {info['icon']} {arch_name.replace('_', ' ')}")
        st.metric(
            label="🏆 Score General",
            value=f"{best_overall['overall_score']:.3f}",
            delta="¡El mejor balance de todas las métricas!"
        )
        
        # Análisis por arquitectura
        st.markdown("### 📋 Fortalezas y Debilidades")
        
        for pred in predictions:
            arch_name = pred['architecture']
            info = self.architecture_info[arch_name]
            
            with st.expander(f"{info['icon']} {arch_name.replace('_', ' ')} - Análisis Detallado"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🟢 Fortalezas:**")
                    strengths = []
                    
                    if pred == best_models.get('highest_confidence'):
                        strengths.append("✅ Mayor confianza de predicción")
                    if pred == best_models.get('fastest'):
                        strengths.append("✅ Tiempo de respuesta más rápido")
                    if pred == best_models.get('lightest'):
                        strengths.append("✅ Menor uso de memoria")
                    if pred == best_models.get('most_efficient'):
                        strengths.append("✅ Mejor relación confianza/tiempo")
                    
                    # Agregar fortalezas generales
                    for ventaja in info['ventajas']:
                        strengths.append(f"✅ {ventaja}")
                    
                    for strength in strengths:
                        st.markdown(strength)
                
                with col2:
                    st.markdown("**🔴 Áreas de mejora:**")
                    weaknesses = []
                    
                    if pred != best_models.get('highest_confidence'):
                        weaknesses.append(f"🔸 Confianza: {pred['confidence']:.1%} vs {best_models['highest_confidence']['confidence']:.1%}")
                    if pred != best_models.get('fastest'):
                        weaknesses.append(f"🔸 Velocidad: {pred['prediction_time']:.3f}s vs {best_models['fastest']['prediction_time']:.3f}s")
                    if pred != best_models.get('lightest'):
                        weaknesses.append(f"🔸 Tamaño: {pred['model_size']:.1f}MB vs {best_models['lightest']['model_size']:.1f}MB")
                    
                    for weakness in weaknesses:
                        st.markdown(weakness)
                
                # Métricas técnicas
                st.markdown("**📊 Métricas Técnicas:**")
                st.markdown(f"""
                - **Parámetros**: {pred['param_count']:,}
                - **Tiempo de predicción**: {pred['prediction_time']:.3f}s
                - **Tamaño del modelo**: {pred['model_size']:.1f}MB
                - **Eficiencia**: {pred['efficiency']:.1f} (confianza/tiempo)
                - **Score general**: {pred['overall_score']:.3f}
                """)
        
        # Recomendaciones de uso
        st.markdown("### 💡 RECOMENDACIONES DE USO")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.markdown("""
            **🏥 Aplicaciones Clínicas:**
            - Usa el modelo con **mayor confianza**
            - Prioriza precisión sobre velocidad
            - Ideal para diagnósticos complejos
            """)
        
        with rec_col2:
            st.markdown("""
            **📱 Aplicaciones Móviles:**
            - Usa el modelo **más rápido y ligero**
            - Balance entre precisión y recursos
            - Ideal para apps en tiempo real
            """)
        
        with rec_col3:
            st.markdown("""
            **🔄 Sistemas de Producción:**
            - Usa el modelo **más eficiente**
            - Considera el volumen de procesamiento
            - Ideal para escalabilidad
            """)

    
    def generate_comprehensive_pdf_report(self, predictions, image, analysis_timestamp):
        """Genera reporte PDF profesional completo"""
        try:
            # Crear PDF
            pdf = FPDF()
            pdf.add_page()
            
            # --- PORTADA ---
            pdf.set_font('Arial', 'B', 20)
            pdf.cell(0, 15, 'REPORTE DE DIAGNÓSTICO OCULAR AVANZADO', 0, 1, 'C')
            pdf.ln(5)
            
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Sistema Multi-Arquitectura CNN', 0, 1, 'C')
            pdf.ln(10)
            
            # Información general
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f'Fecha del análisis: {analysis_timestamp}', 0, 1)
            pdf.cell(0, 8, f'Arquitecturas analizadas: {len(predictions)}', 0, 1)
            pdf.cell(0, 8, f'Enfermedades detectables: 10 patologías especializadas', 0, 1)
            pdf.ln(10)
            
            # --- RESUMEN EJECUTIVO ---
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'RESUMEN EJECUTIVO', 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            
            # Encontrar ganador general
            best_overall = max(predictions, key=lambda x: x.get('overall_score', 0))
            ganador_info = self.architecture_info[best_overall['architecture']]
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, f'ARQUITECTURA RECOMENDADA: {ganador_info["nombre_completo"]}', 0, 1)
            
            pdf.set_font('Arial', '', 11)
            predicted_class = best_overall['predicted_class']
            class_info = self.class_info.get(predicted_class, {})
            
            pdf.cell(0, 6, f'Diagnóstico principal: {class_info.get("nombre", predicted_class)}', 0, 1)
            pdf.cell(0, 6, f'Nivel de confianza: {best_overall["confidence"]:.1%}', 0, 1)
            pdf.cell(0, 6, f'Gravedad: {class_info.get("gravedad", "No especificada")}', 0, 1)
            pdf.ln(8)
            
            # Agregar imagen de manera segura
            try:
                if image is not None:
                    # Crear nombre único para imagen temporal
                    temp_img_name = f"temp_img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    
                    # Convertir y guardar imagen
                    if hasattr(image, 'save'):
                        # Es una imagen PIL
                        image_rgb = image.convert('RGB')
                        image_rgb.save(temp_img_name, 'JPEG', quality=85)
                    else:
                        # Crear imagen placeholder si hay problemas
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                        ax.text(0.5, 0.5, 'Imagen Analizada', ha='center', va='center', fontsize=14)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
                        plt.savefig(temp_img_name, dpi=150, bbox_inches='tight')
                        plt.close()
                    
                    # Agregar imagen al PDF
                    pdf.cell(0, 10, 'IMAGEN ANALIZADA:', 0, 1)
                    pdf.image(temp_img_name, w=80)
                    pdf.ln(5)
                    
            except Exception as img_error:
                # Si hay error con la imagen, continuar sin ella
                pdf.set_font('Arial', 'I', 10)
                pdf.cell(0, 6, f'[Imagen no disponible: {str(img_error)[:50]}...]', 0, 1)
                pdf.ln(5)
            
            # --- NUEVA PÁGINA: COMPARACIÓN DE ARQUITECTURAS ---
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, 'COMPARACIÓN DE ARQUITECTURAS CNN', 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)
            
            # Tabla comparativa
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(50, 8, 'Arquitectura', 1, 0, 'C')
            pdf.cell(35, 8, 'Confianza', 1, 0, 'C')
            pdf.cell(30, 8, 'Tiempo (ms)', 1, 0, 'C')
            pdf.cell(25, 8, 'Tamaño (MB)', 1, 0, 'C')
            pdf.cell(30, 8, 'Eficiencia', 1, 1, 'C')
            
            pdf.set_font('Arial', '', 9)
            for pred in predictions:
                arch_name = pred['architecture'].replace('_', ' ')
                pdf.cell(50, 6, arch_name, 1, 0)
                pdf.cell(35, 6, f"{pred['confidence']:.1%}", 1, 0, 'C')
                pdf.cell(30, 6, f"{pred['prediction_time']*1000:.1f}", 1, 0, 'C')
                pdf.cell(25, 6, f"{pred['model_size']:.1f}", 1, 0, 'C')
                pdf.cell(30, 6, f"{pred.get('efficiency', 0):.1f}", 1, 1, 'C')
            
            pdf.ln(10)
            
            # --- ANÁLISIS CLÍNICO DETALLADO ---
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'ANÁLISIS CLÍNICO DETALLADO', 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            
            # Para cada predicción única
            unique_diagnoses = list(set(pred['predicted_class'] for pred in predictions))
            
            for diagnosis in unique_diagnoses:
                class_info = self.class_info.get(diagnosis, {})
                
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 8, f'{class_info.get("nombre", diagnosis)}', 0, 1)
                
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 5, f'Descripción: {class_info.get("descripcion", "No disponible")}', 0, 1)
                pdf.cell(0, 5, f'Gravedad: {class_info.get("gravedad", "No especificada")}', 0, 1)
                pdf.cell(0, 5, f'Tratamiento: {class_info.get("tratamiento", "Consultar especialista")}', 0, 1)
                pdf.cell(0, 5, f'Pronóstico: {class_info.get("pronostico", "Variable")}', 0, 1)
                pdf.ln(5)
            
            # --- RECOMENDACIONES TÉCNICAS ---
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, 'RECOMENDACIONES TÉCNICAS', 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)
            
            # Ganadores por categoría
            best_models = self.find_best_architecture(predictions)
            
            categories = [
                ('highest_confidence', 'Mayor Confianza', 'Uso clínico de alta precisión'),
                ('fastest', 'Más Rápido', 'Aplicaciones tiempo real/móviles'),
                ('lightest', 'Más Ligero', 'Dispositivos recursos limitados'),
                ('most_efficient', 'Más Eficiente', 'Sistemas de producción escalables')
            ]
            
            for key, title, context in categories:
                if key in best_models:
                    winner = best_models[key]
                    arch_info = self.architecture_info[winner['architecture']]
                    
                    pdf.set_font('Arial', 'B', 11)
                    pdf.cell(0, 7, f'{title}: {arch_info["nombre_completo"]}', 0, 1)
                    pdf.set_font('Arial', '', 10)
                    pdf.cell(0, 5, f'Contexto: {context}', 0, 1)
                    
                    if key == 'highest_confidence':
                        pdf.cell(0, 5, f'Confianza: {winner["confidence"]:.1%}', 0, 1)
                    elif key == 'fastest':
                        pdf.cell(0, 5, f'Tiempo: {winner["prediction_time"]:.3f}s', 0, 1)
                    elif key == 'lightest':
                        pdf.cell(0, 5, f'Tamaño: {winner["model_size"]:.1f}MB', 0, 1)
                    else:
                        pdf.cell(0, 5, f'Eficiencia: {winner.get("efficiency", 0):.1f}', 0, 1)
                    
                    pdf.ln(3)
            
            # --- DISCLAIMER MÉDICO ---
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'DISCLAIMER MÉDICO', 0, 1)
            pdf.set_font('Arial', '', 9)
            pdf.multi_cell(0, 4, 
                'Este reporte es generado por un sistema de inteligencia artificial y debe ser '
                'utilizado únicamente como herramienta de apoyo diagnóstico. No reemplaza el '
                'criterio clínico profesional. Se recomienda confirmación por oftalmólogo '
                'certificado antes de tomar decisiones terapéuticas.')
            
            # Generar archivo PDF
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"reporte_diagnostico_ocular_{timestamp}.pdf"
            pdf.output(pdf_filename)
            
            # Limpiar archivo temporal de imagen si existe
            try:
                if 'temp_img_name' in locals() and os.path.exists(temp_img_name):
                    os.remove(temp_img_name)
            except:
                pass
            
            return pdf_filename
            
        except Exception as e:
            st.error(f"Error generando PDF: {str(e)}")
            return None
    
    def export_technical_data(self, predictions, analysis_timestamp):
        """Exporta datos técnicos completos en JSON"""
        try:
            # Crear estructura de datos completa
            technical_data = {
                'metadata': {
                    'timestamp': analysis_timestamp,
                    'system_version': '2.1 Multi-Architecture + Statistical Analysis',
                    'total_architectures': len(predictions),
                    'diseases_detectable': len(self.class_info),
                    'analysis_type': 'Comparative Multi-CNN with Statistical Inference'
                },
                'architecture_comparison': [],
                'performance_metrics': {},
                'clinical_analysis': {},
                'recommendations': {}
            }
            
            # Datos por arquitectura
            for pred in predictions:
                arch_data = {
                    'architecture_name': pred['architecture'],
                    'model_info': self.architecture_info[pred['architecture']],
                    'prediction_results': {
                        'predicted_class': pred['predicted_class'],
                        'confidence': float(pred['confidence']),
                        'all_probabilities': [float(p) for p in pred['all_probabilities']],
                        'prediction_time_seconds': float(pred['prediction_time']),
                        'model_size_mb': float(pred['model_size']),
                        'parameter_count': int(pred['param_count']),
                        'efficiency_score': float(pred.get('efficiency', 0)),
                        'overall_score': float(pred.get('overall_score', 0))
                    },
                    'clinical_info': self.class_info.get(pred['predicted_class'], {})
                }
                technical_data['architecture_comparison'].append(arch_data)
            
            # Métricas de rendimiento
            confidences = [pred['confidence'] for pred in predictions]
            times = [pred['prediction_time'] for pred in predictions]
            sizes = [pred['model_size'] for pred in predictions]
            
            technical_data['performance_metrics'] = {
                'confidence_stats': {
                    'mean': float(np.mean(confidences)),
                    'std': float(np.std(confidences)),
                    'min': float(np.min(confidences)),
                    'max': float(np.max(confidences))
                },
                'timing_stats': {
                    'mean_ms': float(np.mean(times) * 1000),
                    'std_ms': float(np.std(times) * 1000),
                    'fastest_ms': float(np.min(times) * 1000),
                    'slowest_ms': float(np.max(times) * 1000)
                },
                'size_stats': {
                    'mean_mb': float(np.mean(sizes)),
                    'std_mb': float(np.std(sizes)),
                    'lightest_mb': float(np.min(sizes)),
                    'heaviest_mb': float(np.max(sizes))
                }
            }
            
            # Análisis clínico
            diagnoses = [pred['predicted_class'] for pred in predictions]
            unique_diagnoses = list(set(diagnoses))
            
            technical_data['clinical_analysis'] = {
                'unique_diagnoses': len(unique_diagnoses),
                'consensus_diagnosis': max(set(diagnoses), key=diagnoses.count) if diagnoses else None,
                'diagnostic_agreement': (diagnoses.count(max(set(diagnoses), key=diagnoses.count)) / len(diagnoses)) if diagnoses else 0,
                'severity_distribution': {
                    diagnosis: self.class_info.get(diagnosis, {}).get('gravedad', 'Unknown')
                    for diagnosis in unique_diagnoses
                }
            }
            
            # Recomendaciones
            best_models = self.find_best_architecture(predictions)
            technical_data['recommendations'] = {
                category: {
                    'architecture': model_data['architecture'],
                    'reason': f'Best {category.replace("_", " ")}',
                    'metric_value': model_data.get('confidence' if 'confidence' in category else 
                                                  'prediction_time' if 'fast' in category else
                                                  'model_size' if 'light' in category else 'efficiency', 0)
                }
                for category, model_data in best_models.items()
            }
            
            # Guardar archivo JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"technical_analysis_{timestamp}.json"
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(technical_data, f, indent=2, ensure_ascii=False)
            
            return json_filename
            
        except Exception as e:
            st.error(f"Error exportando datos técnicos: {str(e)}")
            return None
    
    def display_advanced_reporting_section(self, predictions, image, analysis_timestamp):
        """Sección avanzada de reportes y exportación"""
        st.markdown("---")
        st.header("📋 SISTEMA AVANZADO DE REPORTES")
        
        # Métricas de cobertura del sistema
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🏥 Enfermedades Detectables",
                value="10",
                delta="6 más que sistemas básicos",
                help="Nuestro sistema detecta 10 vs 4 de sistemas convencionales"
            )
        
        with col2:
            st.metric(
                label="🧠 Arquitecturas CNN",
                value=len(predictions),
                delta="Análisis multi-arquitectura",
                help="Comparación simultánea de múltiples modelos"
            )
        
        with col3:
            unique_diagnoses = len(set(pred['predicted_class'] for pred in predictions))
            st.metric(
                label="🎯 Diagnósticos Únicos",
                value=unique_diagnoses,
                delta="En este análisis",
                help="Número de diagnósticos diferentes detectados"
            )
        
        with col4:
            avg_confidence = np.mean([pred['confidence'] for pred in predictions])
            st.metric(
                label="📊 Confianza Promedio",
                value=f"{avg_confidence:.1%}",
                delta=f"±{np.std([pred['confidence'] for pred in predictions]):.1%}",
                help="Confianza promedio entre todas las arquitecturas"
            )
        
        # Sección de exportación
        st.markdown("### 📤 Exportar Análisis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generar Reporte PDF Completo", type="primary", use_container_width=True, key="pdf_btn"):
                try:
                    pdf_status = st.empty()
                    pdf_status.info("🔄 Generando reporte PDF profesional...")
                    pdf_file = self.generate_comprehensive_pdf_report(predictions, image, analysis_timestamp)
                    
                    if pdf_file and os.path.exists(pdf_file):
                        pdf_status.success("✅ PDF generado exitosamente!")
                        
                        with open(pdf_file, "rb") as f:
                            pdf_bytes = f.read()
                        
                        st.download_button(
                            label="⬇️ DESCARGAR REPORTE PDF",
                            data=pdf_bytes,
                            file_name=f"reporte_diagnostico_ocular_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="download_pdf"
                        )
                        
                        st.balloons()
                        
                        try:
                            os.remove(pdf_file)
                        except:
                            pass
                    else:
                        st.error("❌ Error generando el reporte PDF")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("📊 Exportar Datos Técnicos (JSON)", use_container_width=True, key="json_btn"):
                try:
                    json_status = st.empty()
                    json_status.info("🔄 Exportando datos técnicos...")
                    
                    json_file = self.export_technical_data(predictions, analysis_timestamp)
                    
                    if json_file and os.path.exists(json_file):
                        json_status.success("✅ Datos técnicos exportados!")
                        
                        with open(json_file, "r", encoding='utf-8') as f:
                            json_data = f.read()
                        
                        st.download_button(
                            label="⬇️ DESCARGAR DATOS JSON",
                            data=json_data,
                            file_name=f"technical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True,
                            key="download_json"
                        )
                        
                        try:
                            os.remove(json_file)
                        except:
                            pass
                    else:
                        st.error("❌ Error exportando datos técnicos")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col3:
            if st.button("📈 Exportar CSV Comparativo", use_container_width=True, key="csv_btn"):
                try:
                    csv_status = st.empty()
                    csv_status.info("🔄 Preparando CSV...")
                    
                    df_export = pd.DataFrame([
                        {
                            'Timestamp': analysis_timestamp,
                            'Arquitectura': pred['architecture'].replace('_', ' '),
                            'Diagnóstico': pred['predicted_class'],
                            'Diagnóstico_ES': self.class_info.get(pred['predicted_class'], {}).get('nombre', pred['predicted_class']),
                            'Confianza': pred['confidence'],
                            'Tiempo_ms': pred['prediction_time'] * 1000,
                            'Tamaño_MB': pred['model_size'],
                            'Parámetros': pred['param_count'],
                            'Eficiencia': pred.get('efficiency', 0),
                            'Score_General': pred.get('overall_score', 0),
                            'Gravedad': self.class_info.get(pred['predicted_class'], {}).get('gravedad', 'No especificada')
                        }
                        for pred in predictions
                    ])
                    
                    csv_data = df_export.to_csv(index=False, encoding='utf-8')
                    csv_status.success("✅ CSV listo!")
                    
                    st.download_button(
                        label="⬇️ DESCARGAR CSV",
                        data=csv_data,
                        file_name=f"analisis_comparativo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Información adicional sobre las descargas
        st.markdown("---")
        st.info("""
        💡 **Información sobre las descargas:**
        - **PDF**: Reporte completo con análisis clínico y recomendaciones técnicas
        - **JSON**: Datos técnicos estructurados para análisis posterior 
        - **CSV**: Tabla comparativa simple para Excel/análisis estadístico
        
        📁 Los archivos se descargan automáticamente a tu carpeta de Descargas
        """)
    
    # ========== FUNCIÓN RUN PRINCIPAL (MODIFICADA) ==========
    def run(self):
        """Ejecuta la aplicación principal CON ANÁLISIS ESTADÍSTICO"""
        # Inicializar session state
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'analysis_image' not in st.session_state:
            st.session_state.analysis_image = None
        if 'analysis_timestamp' not in st.session_state:
            st.session_state.analysis_timestamp = None
        if 'statistical_results' not in st.session_state:
            st.session_state.statistical_results = None
        
        # Header
        self.display_header()
        
        # Sidebar
        st.sidebar.markdown("## 🎛️ Panel de Control")
        st.sidebar.markdown("---")
        
        # Cargar modelos
        if self.models is None:
            with st.spinner("🔄 Cargando las 3 arquitecturas..."):
                self.models, self.class_names, self.individual_class_names = self.load_models()
        
        if len(self.models) < 2:
            st.error("❌ Se necesitan al menos 2 modelos para comparar")
            st.stop()
        
        # Info en sidebar
        st.sidebar.success(f"✅ {len(self.models)} arquitecturas cargadas")
        
        # Pestañas principales
        tab1, tab2 = st.tabs(["🔬 Análisis Individual", "📊 Evaluación Estadística"])
        
        with tab1:
            # Botón para limpiar análisis
            if st.sidebar.button("🔄 Nuevo Análisis", help="Limpia el análisis actual"):
                st.session_state.analysis_complete = False
                st.session_state.predictions = None
                st.session_state.analysis_image = None
                st.session_state.analysis_timestamp = None
                st.rerun()
            
            # Mostrar características
            self.display_architecture_showcase()
            
            st.markdown("---")
            
            # Si ya hay un análisis completo, mostrar resultados
            if st.session_state.analysis_complete and st.session_state.predictions:
                st.success("🎉 **Análisis ya completado!** Puedes descargar los reportes o hacer un nuevo análisis.")
                
                # Mostrar imagen analizada
                if st.session_state.analysis_image:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(st.session_state.analysis_image, caption="Imagen analizada", use_container_width=True)
                
                # Mostrar todos los resultados usando el estado guardado
                predictions = st.session_state.predictions
                
                self.display_prediction_results(predictions)
                st.markdown("---")
                self.display_performance_comparison(predictions)
                st.markdown("---")
                self.display_radar_comparison(predictions)
                
                best_models = self.find_best_architecture(predictions)
                st.markdown("---")
                self.display_winners_podium(best_models)
                st.markdown("---")
                self.display_detailed_analysis(predictions, best_models)
                
                # SECCIÓN DE REPORTES AVANZADOS
                self.display_advanced_reporting_section(predictions, st.session_state.analysis_image, st.session_state.analysis_timestamp)
                
                # Tabla resumen
                with st.expander("📊 Tabla Resumen de Métricas"):
                    df_summary = pd.DataFrame([
                        {
                            'Arquitectura': pred['architecture'].replace('_', ' '),
                            'Diagnóstico': pred['predicted_class'],
                            'Diagnóstico_ES': self.class_info.get(pred['predicted_class'], {}).get('nombre', pred['predicted_class']),
                            'Confianza': f"{pred['confidence']:.1%}",
                            'Tiempo': f"{pred['prediction_time']:.3f}s",
                            'Tamaño': f"{pred['model_size']:.1f}MB",
                            'Parámetros': f"{pred['param_count']:,}",
                            'Eficiencia': f"{pred.get('efficiency', 0):.1f}",
                            'Score General': f"{pred.get('overall_score', 0):.3f}",
                            'Gravedad': self.class_info.get(pred['predicted_class'], {}).get('gravedad', 'No especificada')
                        }
                        for pred in predictions
                    ])
                    
                    st.dataframe(df_summary, use_container_width=True)
                
                # Timestamp
                st.markdown("---")
                st.markdown(f"📅 Análisis realizado: {st.session_state.analysis_timestamp}")
                
            else:
                # Interfaz para nuevo análisis
                st.markdown("## 📸 Subir Imagen para Comparar Arquitecturas")
                uploaded_file = st.file_uploader(
                    "Selecciona una imagen de retina para la batalla de arquitecturas",
                    type=['png', 'jpg', 'jpeg'],
                    help="La imagen será analizada por las 3 arquitecturas simultáneamente"
                )
                
                if uploaded_file is not None:
                    # Mostrar imagen
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Imagen para la batalla", use_container_width=True)
                    
                    # Botón de análisis
                    if st.button("🚀 INICIAR BATALLA DE ARQUITECTURAS", type="primary", use_container_width=True):
                        
                        analysis_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Preprocesamiento y predicciones
                        with st.spinner("🔄 Procesando imagen para todas las arquitecturas..."):
                            img_array = self.preprocess_image(image)
                        
                        if img_array is not None:
                            predictions = []
                            
                            with st.spinner("🏗️ Analizando con las 3 arquitecturas..."):
                                progress_bar = st.progress(0)
                                
                                for i, (arch_name, model) in enumerate(self.models.items()):
                                    pred = self.predict_with_timing(model, img_array, arch_name)
                                    if pred:
                                        predictions.append(pred)
                                    progress_bar.progress((i + 1) / len(self.models))
                            
                            if len(predictions) >= 2:
                                st.success("✅ ¡Batalla completada! Analizando resultados...")
                                
                                # Calcular scores adicionales
                                for pred in predictions:
                                    max_conf = max(p['confidence'] for p in predictions)
                                    min_time = min(p['prediction_time'] for p in predictions)
                                    min_size = min(p['model_size'] for p in predictions)
                                    
                                    conf_score = pred['confidence'] / max_conf
                                    speed_score = min_time / pred['prediction_time']
                                    memory_score = min_size / pred['model_size']
                                    
                                    pred['overall_score'] = 0.5 * conf_score + 0.25 * speed_score + 0.25 * memory_score
                                    pred['efficiency'] = pred['confidence'] / pred['prediction_time']
                                
                                # GUARDAR EN SESSION STATE
                                st.session_state.predictions = predictions
                                st.session_state.analysis_image = image
                                st.session_state.analysis_timestamp = analysis_timestamp
                                st.session_state.analysis_complete = True
                                
                                # Forzar rerun para mostrar resultados
                                st.rerun()
                            
                            else:
                                st.error("❌ Error en las predicciones")
        
        with tab2:
            # NUEVA PESTAÑA: ANÁLISIS ESTADÍSTICO
            self.display_statistical_analysis_section()
        
        # Footer técnico (expandido)
        st.markdown("---")
        st.markdown("""
        ### ⚙️ Sobre Este Sistema Avanzado con Análisis Estadístico
        
        **🚀 Sistema de Diagnóstico Ocular de Nueva Generación**
        
        **🔬 Nuevas Funcionalidades Estadísticas:**
        - **📊 Coeficiente de Matthews (MCC)**: Métrica balanceada para clases desbalanceadas
        - **🔬 Prueba de McNemar**: Comparación estadística rigurosa entre modelos
        - **📈 Intervalos de Confianza Bootstrap**: Robustez estadística (95% CI)
        - **🎭 Matrices de Confusión**: Análisis detallado por clase
        - **📋 Reportes Estadísticos**: Exportación completa de resultados
        
        **🔬 Ventajas Competitivas:**
        - **10 enfermedades especializadas** vs 4 básicas de sistemas convencionales
        - **Análisis multi-arquitectura** con comparación simultánea de CNNs
        - **Evaluación estadística rigurosa** con pruebas de significancia
        - **Reportes profesionales PDF** con análisis clínico y estadístico
        - **Exportación técnica completa** (JSON, CSV, TXT) para investigación
        - **Recomendaciones contextuales** basadas en evidencia estadística
        
        **🏗️ Arquitecturas Implementadas:**
        - **🧠 CNN Híbrida (MobileNetV2)**: Transfer Learning especializado
        - **⚡ EfficientNet-B0**: Compound Scaling balanceado
        - **🔗 ResNet-50 V2**: Conexiones residuales profundas
        
        **📊 Métricas Evaluadas:**
        - 🎯 **Precisión**: Confianza, MCC y consenso diagnóstico
        - ⚡ **Velocidad**: Tiempo de inferencia optimizado
        - 💾 **Eficiencia**: Uso de memoria y escalabilidad
        - 🏆 **Balance**: Score general multi-criterio
        - 📈 **Significancia**: Pruebas estadísticas inferenciales
        
        **🎯 Aplicaciones:**
        - 🏥 **Clínicas**: Diagnóstico de alta precisión con validación estadística
        - 📱 **Móviles**: Apps de telemedicina con métricas robustas
        - 🔄 **Producción**: Sistemas hospitalarios escalables con evidencia estadística
        - 🔬 **Investigación**: Datos completos para publicaciones científicas
        
        **💡 Innovación**: Primer sistema que combina múltiples arquitecturas CNN con 
        análisis estadístico inferencial completo para diagnóstico ocular especializado.
        
        **📚 Métodos Estadísticos:**
        - **MCC**: Coeficiente de Correlación de Matthews para métricas balanceadas
        - **McNemar**: Prueba chi-cuadrado para comparación de clasificadores
        - **Bootstrap**: Intervalos de confianza no paramétricos
        - **Corrección de Yates**: Para muestras pequeñas en McNemar
        """)

# Ejecutar aplicación
if __name__ == "__main__":
    app = ThreeArchitecturesApp()
    app.run()