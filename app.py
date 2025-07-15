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
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy.stats import chi2
from scipy import stats
import itertools
from pathlib import Path
from translations import get_text, get_available_languages
import download_models

warnings.filterwarnings('ignore')
#download_models.descargar_modelos()

# Configuración de página
def configurar_pagina(lang='es'):
    st.set_page_config(
        page_title=get_text('page_title', lang),
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Agregar selector de idioma en el sidebar - ACTUALIZADO PARA 3 IDIOMAS
def mostrar_selector_idioma():
    if 'language' not in st.session_state:
        st.session_state.language = 'es'
    
    languages = get_available_languages()
    
    # Encontrar el índice actual
    current_index = list(languages.keys()).index(st.session_state.language) if st.session_state.language in languages else 0
    
    selected_lang = st.sidebar.selectbox(
        get_text('select_language', st.session_state.language),
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=current_index,
        key="language_selector"
    )
    
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()
    
    return st.session_state.language

class AplicacionTresArquitecturas:    
    def __init__(self):
        # Inicializar idioma
        if 'language' not in st.session_state:
            st.session_state.language = 'es'
        self.lang = st.session_state.language
        
        self.informacion_clases = {
            'Central Serous Chorioretinopathy [Color Fundus]': {
                'nombre': get_text('CentralSerous_nombre', self.lang),
                'descripcion': get_text('CentralSerous_descripcion', self.lang),
                'gravedad': get_text('CentralSerous_gravedad', self.lang),
                'color': '#FFA07A',
                'tratamiento': get_text('CentralSerous_tratamiento', self.lang),
                'pronostico': get_text('CentralSerous_pronostico', self.lang)
            },
            'Diabetic Retinopathy': {
                'nombre': get_text('Diabetic_nombre', self.lang),
                'descripcion': get_text('Diabetic_descripcion', self.lang),
                'gravedad': get_text('Diabetic_gravedad', self.lang),
                'color': '#FF6B6B',
                'tratamiento': get_text('Diabetic_tratamiento', self.lang),
                'pronostico': get_text('Diabetic_pronostico', self.lang)
            },
            'Disc Edema': {
                'nombre': get_text('DiscEdema_nombre', self.lang),
                'descripcion': get_text('DiscEdema_descripcion', self.lang),
                'gravedad': get_text('DiscEdema_gravedad', self.lang),
                'color': '#FF4444',
                'tratamiento': get_text('DiscEdema_tratamiento', self.lang),
                'pronostico': get_text('DiscEdema_pronostico', self.lang)
            },
            'Glaucoma': {
                'nombre': get_text('Glaucoma_nombre', self.lang),
                'descripcion': get_text('Glaucoma_descripcion', self.lang),
                'gravedad': get_text('Glaucoma_gravedad', self.lang),
                'color': '#DC143C',
                'tratamiento': get_text('Glaucoma_tratamiento', self.lang),
                'pronostico': get_text('Glaucoma_pronostico', self.lang)
            },
            'Healthy': {
                'nombre': get_text('Healthy_nombre', self.lang),
                'descripcion': get_text('Healthy_descripcion', self.lang),
                'gravedad': get_text('Healthy_gravedad', self.lang),
                'color': '#32CD32',
                'tratamiento': get_text('Healthy_tratamiento', self.lang),
                'pronostico': get_text('Healthy_pronostico', self.lang)
            },
            'Macular Scar': {
                'nombre': get_text('MacularScar_nombre', self.lang),
                'descripcion': get_text('MacularScar_descripcion', self.lang),
                'gravedad': get_text('MacularScar_gravedad', self.lang),
                'color': '#DAA520',
                'tratamiento': get_text('MacularScar_tratamiento', self.lang),
                'pronostico': get_text('MacularScar_pronostico', self.lang)
            },
            'Myopia': {
                'nombre': get_text('Myopia_nombre', self.lang),
                'descripcion': get_text('Myopia_descripcion', self.lang),
                'gravedad': get_text('Myopia_gravedad', self.lang),
                'color': '#87CEEB',
                'tratamiento': get_text('Myopia_tratamiento', self.lang),
                'pronostico': get_text('Myopia_pronostico', self.lang)
            },
            'Pterygium': {
                'nombre': get_text('Pterygium_nombre', self.lang),
                'descripcion': get_text('Pterygium_descripcion', self.lang),
                'gravedad': get_text('Pterygium_gravedad', self.lang),
                'color': '#DDA0DD',
                'tratamiento': get_text('Pterygium_tratamiento', self.lang),
                'pronostico': get_text('Pterygium_pronostico', self.lang)
            },
            'Retinal Detachment': {
                'nombre': get_text('RetinalDetachment_nombre', self.lang),
                'descripcion': get_text('RetinalDetachment_descripcion', self.lang),
                'gravedad': get_text('RetinalDetachment_gravedad', self.lang),
                'color': '#B22222',
                'tratamiento': get_text('RetinalDetachment_tratamiento', self.lang),
                'pronostico': get_text('RetinalDetachment_pronostico', self.lang)
            },
            'Retinitis Pigmentosa': {
                'nombre': get_text('Retinitis_nombre', self.lang),
                'descripcion': get_text('Retinitis_descripcion', self.lang),
                'gravedad': get_text('Retinitis_gravedad', self.lang),
                'color': '#8B0000',
                'tratamiento': get_text('Retinitis_tratamiento', self.lang),
                'pronostico': get_text('Retinitis_pronostico', self.lang)
            }
        }

        self.informacion_arquitecturas = {
            'CNN_Original': {
                'nombre_completo': get_text('CNN_original_nombre', self.lang),
                'descripcion': get_text('CNN_original_descripcion', self.lang),
                'color': '#E91E63',
                'icon': '🧠',
                'ventajas': [
                    get_text('CNN_original_ventaja1', self.lang),
                    get_text('CNN_original_ventaja2', self.lang),
                    get_text('CNN_original_ventaja3', self.lang)
                ],
                'caracteristicas': {
                    'Tipo': get_text('CNN_original_tipo', self.lang),
                    'Parámetros': '~3.5M',
                    'Ventaja principal': get_text('CNN_original_ventaja_principal', self.lang),
                    'Año': '2018'
                }
            },
            'EfficientNetB0': {
                'nombre_completo': get_text('EfficientNet_nombre', self.lang),
                'descripcion': get_text('EfficientNet_descripcion', self.lang),
                'color': '#2196F3',
                'icon': '⚡',
                'ventajas': [
                    get_text('EfficientNet_ventaja1', self.lang),
                    get_text('EfficientNet_ventaja2', self.lang),
                    get_text('EfficientNet_ventaja3', self.lang)
                ],
                'caracteristicas': {
                    'Tipo': get_text('EfficientNet_tipo', self.lang),
                    'Parámetros': '~5.3M',
                    'Ventaja principal': get_text('EfficientNet_ventaja_principal', self.lang),
                    'Año': '2019'
                }
            },
            'ResNet50V2': {
                'nombre_completo': get_text('ResNet_nombre', self.lang),
                'descripcion': get_text('ResNet_descripcion', self.lang),
                'color': '#FF9800',
                'icon': '🔗',
                'ventajas': [
                    get_text('ResNet_ventaja1', self.lang),
                    get_text('ResNet_ventaja2', self.lang),
                    get_text('ResNet_ventaja3', self.lang)
                ],
                'caracteristicas': {
                    'Tipo': get_text('ResNet_tipo', self.lang),
                    'Parámetros': '~25.6M',
                    'Ventaja principal': get_text('ResNet_ventaja_principal', self.lang),
                    'Año': '2016'
                }
            }
        }
        
        self.modelos = None
        self.nombres_clases = None
        self.nombres_clases_individuales = None
        self.analisis_actual = None
        self.resultados_estadisticos = None  # Para almacenar resultados estadísticos
    
    
    def limpiar_texto_pdf(self, texto):
        """Limpia texto para compatibilidad con FPDF"""
        if not isinstance(texto, str):
            texto = str(texto)
        
        # Reemplazos de caracteres especiales
        reemplazos = {
            '\u2022': '- ',     # Viñeta
            '•': '- ',          # Viñeta
            '✅': '[OK] ',
            '❌': '[X] ',
            '⚠️': '[!] ',
            '🔬': '[TEST] ',
            '📊': '[CHART] ',
            '🎯': '[TARGET] ',
            '🟢': '[EXCELENTE] ',
            '🔵': '[BUENO] ',
            '🟡': '[REGULAR] ',
            '🟠': '[BAJO] ',
            '🔴': '[MUY BAJO] ',
            '→': '-> ',
            '←': '<- ',
            '↑': '^',
            '↓': 'v',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'δ': 'delta',
            'ε': 'epsilon',
            'θ': 'theta',
            'μ': 'mu',
            'σ': 'sigma',
            'π': 'pi',
            '₀': '0',
            '₁': '1',
            '₂': '2',
            '°': ' grados',
            '±': '+/-',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '≈': '~=',
            '∞': 'infinito'
        }
        
        # Aplicar reemplazos
        for unicode_char, ascii_char in reemplazos.items():
            texto = texto.replace(unicode_char, ascii_char)
        
        # Remover cualquier carácter que no sea ASCII
        texto = texto.encode('ascii', errors='ignore').decode('ascii')
        
        return texto
    
    
    @st.cache_resource
    def cargar_modelos(_self):
        """Carga las 3 arquitecturas para comparar"""
        try:
            modelos = {}
            
            # Mapeo de archivos a arquitecturas
            archivos_modelos = {
                'CNN_Original': 'eye_disease_model.h5',
                'EfficientNetB0': 'ensemble_efficientnet_model.h5', 
                'ResNet50V2': 'ensemble_resnet_model.h5'
            }
            
            for nombre_arq, nombre_archivo in archivos_modelos.items():
                if os.path.exists(nombre_archivo):
                    modelos[nombre_arq] = tf.keras.models.load_model(nombre_archivo)
                    st.success(get_text('model_loaded', _self.lang, name=nombre_arq))
                else:
                    st.warning(get_text('model_not_found', _self.lang, filename=nombre_archivo))
            
            # Cargar nombres de clases
            nombres_clases_conjunto = {}
            if os.path.exists('ensemble_class_indices.npy'):
                indices_clases = np.load('ensemble_class_indices.npy', allow_pickle=True).item()
                nombres_clases_conjunto = {v: k for k, v in indices_clases.items()}
            
            nombres_clases_individuales = {}
            if os.path.exists('class_indices.npy'):
                indices_clases = np.load('class_indices.npy', allow_pickle=True).item()
                nombres_clases_individuales = {v: k for k, v in indices_clases.items()}
            
            # Nombres por defecto si no hay archivos
            if not nombres_clases_conjunto:
                nombres_clases_conjunto = {i: f"Clase_{i}" for i in range(10)}
            if not nombres_clases_individuales:
                nombres_clases_individuales = {i: f"Clase_{i}" for i in range(10)}
            
            return modelos, nombres_clases_conjunto, nombres_clases_individuales
            
        except Exception as e:
            st.error(get_text('loading_error', _self.lang, error=str(e)))
            return {}, {}, {}
    
    def preprocesar_imagen(self, imagen):
        """Preprocesa imagen para predicción"""
        try:
            if imagen.mode != 'RGB':
                imagen = imagen.convert('RGB')
            
            imagen = imagen.resize((224, 224))
            array_img = np.array(imagen)
            array_img = array_img.astype('float32') / 255.0
            array_img = np.expand_dims(array_img, axis=0)
            
            return array_img
            
        except Exception as e:
            st.error(f"Error procesando imagen: {str(e)}")
            return None

    def preprocesar_imagen_desde_ruta(self, ruta_imagen):
        """Preprocesa imagen desde ruta para evaluación estadística"""
        try:
            imagen = Image.open(ruta_imagen)
            return self.preprocesar_imagen(imagen)
        except Exception as e:
            st.error(f"Error procesando imagen {ruta_imagen}: {str(e)}")
            return None
    
    def predecir_con_cronometraje(self, modelo, array_img, nombre_arq):
        """Realiza predicción midiendo tiempo y métricas"""
        try:
            # Medir tiempo de predicción
            tiempo_inicio = time.time()
            predicciones = modelo.predict(array_img, verbose=0)
            tiempo_fin = time.time()
            
            tiempo_prediccion = tiempo_fin - tiempo_inicio
            
            indice_clase_predicha = np.argmax(predicciones[0])
            confianza = float(predicciones[0][indice_clase_predicha])
            
            # Usar nombres de clases correctos
            if nombre_arq == 'CNN_Original':
                clase_predicha = self.nombres_clases_individuales[indice_clase_predicha]
            else:
                clase_predicha = self.nombres_clases[indice_clase_predicha]
            
            return {
                'arquitectura': nombre_arq,
                'clase_predicha': clase_predicha,
                'indice_clase_predicha': indice_clase_predicha,
                'confianza': confianza,
                'todas_probabilidades': predicciones[0],
                'tiempo_prediccion': tiempo_prediccion,
                'tamaño_modelo': self.obtener_tamaño_modelo(modelo),
                'conteo_parametros': modelo.count_params()
            }
            
        except Exception as e:
            st.error(f"Error en predicción {nombre_arq}: {str(e)}")
            return None
    
    def obtener_tamaño_modelo(self, modelo):
        """Calcula el tamaño del modelo en MB"""
        try:
            conteo_parametros = modelo.count_params()
            tamaño_mb = (conteo_parametros * 4) / (1024 * 1024)
            return tamaño_mb
        except:
            return 0
    
    # ========== NUEVAS FUNCIONES ESTADÍSTICAS ==========
    
    def calcular_correlacion_matthews(self, y_verdadero, y_predicho):
        """Calcula el Coeficiente de Correlación de Matthews"""
        try:
            mcc = matthews_corrcoef(y_verdadero, y_predicho)
            return mcc
        except Exception as e:
            st.error(f"Error calculando MCC: {str(e)}")
            return 0.0
    
    def prueba_mcnemar(self, y_verdadero, y_pred1, y_pred2):
        """Realiza la prueba de McNemar entre dos modelos"""
        try:
            # Crear tabla de contingencia 2x2
            # Casos donde modelo1 correcto, modelo2 incorrecto
            correcto_1_incorrecto_2 = np.sum((y_pred1 == y_verdadero) & (y_pred2 != y_verdadero))
            # Casos donde modelo1 incorrecto, modelo2 correcto  
            incorrecto_1_correcto_2 = np.sum((y_pred1 != y_verdadero) & (y_pred2 == y_verdadero))
            
            # Tabla de contingencia
            tabla_contingencia = np.array([
                [correcto_1_incorrecto_2, incorrecto_1_correcto_2],
                [incorrecto_1_correcto_2, correcto_1_incorrecto_2]
            ])
            
            # Calcular estadístico de McNemar con corrección de continuidad
            n = correcto_1_incorrecto_2 + incorrecto_1_correcto_2
            
            if n == 0:
                return {
                    'estadistico': 0.0,
                    'valor_p': 1.0,
                    'significativo': False,
                    'tabla_contingencia': tabla_contingencia,
                    'interpretacion': 'No hay diferencias entre modelos'
                }
            
            # McNemar con corrección de continuidad de Yates
            estadistico_mcnemar = (abs(correcto_1_incorrecto_2 - incorrecto_1_correcto_2) - 1)**2 / n
            valor_p = 1 - chi2.cdf(estadistico_mcnemar, df=1)
            
            # Interpretación
            significativo = valor_p < 0.05
            
            if significativo:
                if correcto_1_incorrecto_2 > incorrecto_1_correcto_2:
                    interpretacion = "Modelo 1 significativamente mejor que Modelo 2"
                else:
                    interpretacion = "Modelo 2 significativamente mejor que Modelo 1"
            else:
                interpretacion = "No hay diferencia significativa entre modelos"
            
            return {
                'estadistico': estadistico_mcnemar,
                'valor_p': valor_p,
                'significativo': significativo,
                'tabla_contingencia': tabla_contingencia,
                'interpretacion': interpretacion,
                'n_desacuerdos': n
            }
            
        except Exception as e:
            st.error(f"Error en prueba McNemar: {str(e)}")
            return None
    
    def calcular_intervalo_confianza_mcc(self, y_verdadero, y_predicho, confianza=0.95):
        """Calcula intervalo de confianza bootstrap para MCC"""
        try:
            n_bootstrap = 1000
            mccs_bootstrap = []
            
            n_muestras = len(y_verdadero)
            
            for _ in range(n_bootstrap):
                # Bootstrap resampling
                indices = np.random.choice(n_muestras, n_muestras, replace=True)
                y_verdadero_bootstrap = y_verdadero[indices]
                y_predicho_bootstrap = y_predicho[indices]
                
                try:
                    mcc_bootstrap = matthews_corrcoef(y_verdadero_bootstrap, y_predicho_bootstrap)
                    mccs_bootstrap.append(mcc_bootstrap)
                except:
                    continue
            
            if len(mccs_bootstrap) == 0:
                return None, None
            
            alpha = 1 - confianza
            percentil_inferior = (alpha/2) * 100
            percentil_superior = (1 - alpha/2) * 100
            
            ci_inferior = np.percentile(mccs_bootstrap, percentil_inferior)
            ci_superior = np.percentile(mccs_bootstrap, percentil_superior)
            
            return ci_inferior, ci_superior
            
        except Exception as e:
            st.error(f"Error calculando IC para MCC: {str(e)}")
            return None, None
    
    def escanear_carpeta_dataset(self, ruta_dataset):
        """Escanea carpeta de dataset y crea lista de imágenes con etiquetas"""
        try:
            ruta_dataset = Path(ruta_dataset)
            extensiones_imagen = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            
            datos_imagenes = []
            carpetas_clases = [d for d in ruta_dataset.iterdir() if d.is_dir()]
            
            if not carpetas_clases:
                st.error("No se encontraron carpetas de clases en el dataset")
                return None
            
            # Mapear nombres de carpetas a índices
            nombre_clase_a_indice = {}
            
            st.info(f"📁 Clases encontradas: {len(carpetas_clases)}")
            
            for indice_clase, carpeta_clase in enumerate(sorted(carpetas_clases)):
                nombre_clase = carpeta_clase.name
                nombre_clase_a_indice[nombre_clase] = indice_clase
                
                st.write(f"• **Clase {indice_clase}**: {nombre_clase}")
                
                # Buscar imágenes en la carpeta
                imagenes_en_clase = []
                for ext in extensiones_imagen:
                    imagenes_en_clase.extend(carpeta_clase.glob(f'*{ext}'))
                    imagenes_en_clase.extend(carpeta_clase.glob(f'*{ext.upper()}'))
                
                for ruta_img in imagenes_en_clase:
                    datos_imagenes.append({
                        'ruta_imagen': str(ruta_img),
                        'etiqueta_verdadera': indice_clase,
                        'nombre_clase': nombre_clase
                    })
            
            st.success(f"✅ Total de imágenes encontradas: {len(datos_imagenes)}")
            
            return datos_imagenes, nombre_clase_a_indice
            
        except Exception as e:
            st.error(f"Error escaneando dataset: {str(e)}")
            return None, None

    def evaluar_modelos_en_dataset(self, entrada_dataset):
        """Evalúa todos los modelos en un dataset (carpeta o CSV)"""
        try:
            # Determinar si es carpeta o archivo CSV
            if isinstance(entrada_dataset, str) and entrada_dataset.endswith('.csv'):
                # Leer CSV
                df_prueba = pd.read_csv(entrada_dataset)
                
                if 'ruta_imagen' not in df_prueba.columns or 'etiqueta_verdadera' not in df_prueba.columns:
                    st.error("El archivo CSV debe contener columnas 'ruta_imagen' y 'etiqueta_verdadera'")
                    return None
                
                datos_imagenes = df_prueba.to_dict('records')
                
            else:
                # Escanear carpeta de dataset
                datos_imagenes, mapeo_clases = self.escanear_carpeta_dataset(entrada_dataset)
                if datos_imagenes is None:
                    return None
            
            resultados = {
                'etiquetas_verdaderas': [],
                'predicciones': {arq: [] for arq in self.modelos.keys()},
                'confianzas': {arq: [] for arq in self.modelos.keys()},
                'tiempos_prediccion': {arq: [] for arq in self.modelos.keys()}
            }
            
            total_imagenes = len(datos_imagenes)
            barra_progreso = st.progress(0)
            texto_estado = st.empty()
            
            # Procesar imágenes en lotes para eficiencia
            for idx, datos_img in enumerate(datos_imagenes):
                texto_estado.text(f"Evaluando imagen {idx+1}/{total_imagenes}: {Path(datos_img['ruta_imagen']).name}")
                
                # Preprocesar imagen
                array_img = self.preprocesar_imagen_desde_ruta(datos_img['ruta_imagen'])
                if array_img is None:
                    continue
                
                # Etiqueta verdadera
                etiqueta_verdadera = datos_img['etiqueta_verdadera']
                resultados['etiquetas_verdaderas'].append(etiqueta_verdadera)
                
                # Predecir con cada modelo
                for nombre_arq, modelo in self.modelos.items():
                    resultado_pred = self.predecir_con_cronometraje(modelo, array_img, nombre_arq)
                    
                    if resultado_pred:
                        resultados['predicciones'][nombre_arq].append(resultado_pred['indice_clase_predicha'])
                        resultados['confianzas'][nombre_arq].append(resultado_pred['confianza'])
                        resultados['tiempos_prediccion'][nombre_arq].append(resultado_pred['tiempo_prediccion'])
                    else:
                        resultados['predicciones'][nombre_arq].append(-1)  # Error
                        resultados['confianzas'][nombre_arq].append(0.0)
                        resultados['tiempos_prediccion'][nombre_arq].append(0.0)
                
                # Actualizar progreso
                barra_progreso.progress((idx + 1) / total_imagenes)
                
                # Mostrar progreso cada 50 imágenes
                if (idx + 1) % 50 == 0:
                    st.write(f"✅ Procesadas {idx + 1}/{total_imagenes} imágenes")
            
            barra_progreso.empty()
            texto_estado.empty()
            
            # Convertir a arrays numpy
            resultados['etiquetas_verdaderas'] = np.array(resultados['etiquetas_verdaderas'])
            for arq in self.modelos.keys():
                resultados['predicciones'][arq] = np.array(resultados['predicciones'][arq])
                resultados['confianzas'][arq] = np.array(resultados['confianzas'][arq])
                resultados['tiempos_prediccion'][arq] = np.array(resultados['tiempos_prediccion'][arq])
            
            return resultados
            
        except Exception as e:
            st.error(f"Error evaluando modelos: {str(e)}")
            return None
    
    def realizar_analisis_estadistico(self, resultados_evaluacion):
        """Realiza análisis estadístico completo"""
        try:
            y_verdadero = resultados_evaluacion['etiquetas_verdaderas']
            arquitecturas = list(self.modelos.keys())
            
            resultados_estadisticos = {
                'puntuaciones_mcc': {},
                'intervalos_confianza_mcc': {},
                'puntuaciones_accuracy': {},
                'resultados_mcnemar': {},
                'matrices_confusion': {},
                'reportes_clasificacion': {}
            }
            
            # Calcular MCC y accuracy para cada modelo
            for arq in arquitecturas:
                y_pred = resultados_evaluacion['predicciones'][arq]
                
                # MCC
                mcc = self.calcular_correlacion_matthews(y_verdadero, y_pred)
                resultados_estadisticos['puntuaciones_mcc'][arq] = mcc
                
                # Intervalo de confianza para MCC
                ci_inferior, ci_superior = self.calcular_intervalo_confianza_mcc(y_verdadero, y_pred)
                resultados_estadisticos['intervalos_confianza_mcc'][arq] = (ci_inferior, ci_superior)
                
                # Accuracy
                accuracy = np.mean(y_verdadero == y_pred)
                resultados_estadisticos['puntuaciones_accuracy'][arq] = accuracy
                
                # Matriz de confusión
                cm = confusion_matrix(y_verdadero, y_pred)
                resultados_estadisticos['matrices_confusion'][arq] = cm
                
                # Reporte de clasificación
                try:
                    reporte_clases = classification_report(y_verdadero, y_pred, output_dict=True)
                    resultados_estadisticos['reportes_clasificacion'][arq] = reporte_clases
                except:
                    resultados_estadisticos['reportes_clasificacion'][arq] = {}
            
            # Pruebas de McNemar entre pares de modelos
            for arq1, arq2 in itertools.combinations(arquitecturas, 2):
                y_pred1 = resultados_evaluacion['predicciones'][arq1]
                y_pred2 = resultados_evaluacion['predicciones'][arq2]
                
                resultado_mcnemar = self.prueba_mcnemar(y_verdadero, y_pred1, y_pred2)
                resultados_estadisticos['resultados_mcnemar'][f"{arq1}_vs_{arq2}"] = resultado_mcnemar
            
            return resultados_estadisticos
            
        except Exception as e:
            st.error(f"Error en análisis estadístico: {str(e)}")
            return None
    
    def mostrar_seccion_analisis_estadistico(self):
        """Sección completa de análisis estadístico - TRADUCIDA"""
        st.markdown("---")
        st.header(get_text('statistical_analysis_title', self.lang))
        st.markdown(get_text('statistical_description', self.lang))
        
        # Dataset de evaluación
        st.subheader(get_text('dataset_evaluation', self.lang))
        
        # Input de ruta de carpeta
        carpeta_dataset = st.text_input(
            get_text('dataset_path', self.lang),
            value="Pruebas",  # Valor por defecto
            help=get_text('dataset_path_help', self.lang)
        )
        
        # Mostrar estructura esperada
        with st.expander(get_text('expected_structure', self.lang)):
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
        if carpeta_dataset:
            ruta_dataset = Path(carpeta_dataset)
            if ruta_dataset.exists() and ruta_dataset.is_dir():
                st.success(get_text('folder_found', self.lang, path=str(ruta_dataset.absolute())))
                
                # Vista previa del dataset
                if st.button(get_text('dataset_preview', self.lang), key="vista_previa_dataset"):
                    with st.spinner("🔍 Escaneando dataset..."):
                        datos_vista_previa, mapeo_clases = self.escanear_carpeta_dataset(ruta_dataset)
                        
                        if datos_vista_previa:
                            st.markdown("#### 📊 Resumen del Dataset:")
                            
                            # Crear DataFrame para mostrar distribución
                            df_vista_previa = pd.DataFrame(datos_vista_previa)
                            conteos_clases = df_vista_previa['nombre_clase'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📈 Distribución por Clase:**")
                                for nombre_clase, conteo in conteos_clases.items():
                                    st.markdown(f"• **{nombre_clase}**: {conteo} imágenes")
                            
                            with col2:
                                # Gráfico de distribución
                                fig_dist = go.Figure(data=[
                                    go.Bar(x=conteos_clases.index, y=conteos_clases.values)
                                ])
                                fig_dist.update_layout(
                                    title="📊 Distribución de Imágenes por Clase",
                                    xaxis_title="Clases",
                                    yaxis_title="Número de Imágenes",
                                    height=400
                                )
                                st.plotly_chart(fig_dist, use_container_width=True)
                            
                            st.dataframe(df_vista_previa.head(10), use_container_width=True)
                
                # Botón de evaluación
                if st.button(get_text('start_evaluation', self.lang), type="primary", use_container_width=True, key="eval_carpeta"):
                    st.info("🔄 Evaluando modelos en dataset completo... Esto puede tomar varios minutos.")
                    
                    # Evaluar modelos
                    resultados_evaluacion = self.evaluar_modelos_en_dataset(str(ruta_dataset))
                    
                    if resultados_evaluacion is not None:
                        st.success("✅ Evaluación completada! Realizando análisis estadístico...")
                        
                        # Análisis estadístico
                        resultados_estadisticos = self.realizar_analisis_estadistico(resultados_evaluacion)
                        
                        if resultados_estadisticos is not None:
                            # Guardar en session state
                            st.session_state.resultados_estadisticos = resultados_estadisticos
                            st.session_state.resultados_evaluacion = resultados_evaluacion
                            
                            # Mostrar resultados
                            self.mostrar_resultados_estadisticos(resultados_estadisticos, resultados_evaluacion, key_suffix="_actual")
            
            else:
                st.error(get_text('folder_not_found', self.lang, path=carpeta_dataset))
                st.markdown("**💡 Sugerencias:**")
                st.markdown("• Verifica que la ruta sea correcta")
                st.markdown("• Usa rutas relativas como `Pruebas` o `./Pruebas`")
                st.markdown("• O rutas absolutas como `/ruta/completa/Pruebas`")
        
        # Mostrar resultados si ya están calculados
        if hasattr(st.session_state, 'resultados_estadisticos') and st.session_state.resultados_estadisticos:
            st.markdown("---")
            st.info("📊 Mostrando resultados de análisis estadístico previo")
            self.mostrar_resultados_estadisticos(
                st.session_state.resultados_estadisticos, 
                st.session_state.resultados_evaluacion,
                key_suffix="_previo"
            )
    
    def mostrar_resultados_estadisticos(self, resultados_estadisticos, resultados_evaluacion,key_suffix=""):
        """Muestra resultados del análisis estadístico"""
        
        # Crear timestamp único para evitar keys duplicados
        import time
        timestamp = str(int(time.time() * 1000))  # timestamp en milisegundos
        
        # === SECCIÓN 1: COEFICIENTE DE MATTHEWS ===
        st.subheader("🎯 Coeficiente de Correlación de Matthews (MCC)")
        
        st.markdown("""
        **MCC** es una métrica balanceada que funciona bien incluso con clases desbalanceadas.
        - **Rango**: -1 (completamente incorrecto) a +1 (predicción perfecta)
        - **0**: Predicción aleatoria
        - **>0.5**: Excelente rendimiento
        """)
        
        # Tabla de MCC con intervalos de confianza
        datos_mcc = []
        for arq in self.modelos.keys():
            puntuacion_mcc = resultados_estadisticos['puntuaciones_mcc'][arq]
            ci_inferior, ci_superior = resultados_estadisticos['intervalos_confianza_mcc'][arq]
            accuracy = resultados_estadisticos['puntuaciones_accuracy'][arq]
            
            datos_mcc.append({
                get_text('architecture', self.lang): arq.replace('_', ' '),
                'MCC': f"{puntuacion_mcc:.4f}",
                'IC 95% Inferior': f"{ci_inferior:.4f}" if ci_inferior else "N/A",
                'IC 95% Superior': f"{ci_superior:.4f}" if ci_superior else "N/A",
                'Accuracy': f"{accuracy:.4f}",
                'Interpretación': self.interpretar_mcc(puntuacion_mcc)
            })
        
        df_mcc = pd.DataFrame(datos_mcc)
        st.dataframe(df_mcc, use_container_width=True)
        
        # Gráfico de MCC con intervalos de confianza
        fig_mcc = go.Figure()
        
        arquitecturas = list(self.modelos.keys())
        puntuaciones_mcc = [resultados_estadisticos['puntuaciones_mcc'][arq] for arq in arquitecturas]
        ci_inferiores = [resultados_estadisticos['intervalos_confianza_mcc'][arq][0] for arq in arquitecturas]
        ci_superiores = [resultados_estadisticos['intervalos_confianza_mcc'][arq][1] for arq in arquitecturas]
        
        # Barras con intervalos de confianza
        fig_mcc.add_trace(go.Bar(
            x=[arq.replace('_', ' ') for arq in arquitecturas],
            y=puntuaciones_mcc,
            error_y=dict(
                type='data',
                symmetric=False,
                array=[ci_superior - mcc for ci_superior, mcc in zip(ci_superiores, puntuaciones_mcc)],
                arrayminus=[mcc - ci_inferior for ci_inferior, mcc in zip(ci_inferiores, puntuaciones_mcc)]
            ),
            marker_color=[self.informacion_arquitecturas[arq]['color'] for arq in arquitecturas],
            text=[f"{mcc:.3f}" for mcc in puntuaciones_mcc],
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
        
        st.plotly_chart(fig_mcc, use_container_width=True, key=f"grafico_mcc_principal_{timestamp}")
        
        # === SECCIÓN 2: PRUEBAS DE MCNEMAR ===
        st.subheader("🔬 Pruebas de McNemar - Comparación entre Modelos")
        
        st.markdown("""
        **Prueba de McNemar** compara estadísticamente el rendimiento entre pares de modelos:
        - **H₀**: No hay diferencia entre los modelos
        - **H₁**: Hay diferencia significativa
        - **α = 0.05**: Nivel de significancia
        """)
        
        datos_mcnemar = []
        for comparacion, resultado in resultados_estadisticos['resultados_mcnemar'].items():
            if resultado:
                arq1, arq2 = comparacion.split('_vs_')
                
                datos_mcnemar.append({
                    'Comparación': f"{arq1.replace('_', ' ')} vs {arq2.replace('_', ' ')}",
                    'Estadístico McNemar': f"{resultado['estadistico']:.4f}",
                    'p-valor': f"{resultado['valor_p']:.6f}",
                    'Significativo (α=0.05)': "✅ Sí" if resultado['significativo'] else "❌ No",
                    'Interpretación': resultado['interpretacion'],
                    'N° Desacuerdos': resultado['n_desacuerdos']
                })
        
        df_mcnemar = pd.DataFrame(datos_mcnemar)
        st.dataframe(df_mcnemar, use_container_width=True)
        
        # Heatmap de p-valores
        self.graficar_mapa_calor_mcnemar(resultados_estadisticos['resultados_mcnemar'], timestamp)
        
        # === SECCIÓN 3: MATRICES DE CONFUSIÓN ===
        st.subheader("🎭 Matrices de Confusión por Arquitectura")
        
        cols = st.columns(len(self.modelos))
        
        for i, (arq, cm) in enumerate(resultados_estadisticos['matrices_confusion'].items()):
            with cols[i]:
                fig_cm = self.graficar_matriz_confusion(cm, arq)
                st.plotly_chart(fig_cm, use_container_width=True, key=f"matriz_confusion_{arq}_{timestamp}")
        
        # === SECCIÓN 4: ANÁLISIS DE SIGNIFICANCIA ===
        st.subheader("📈 Análisis de Significancia Estadística")
        
        # Resumen de significancia
        comparaciones_significativas = [
            comp for comp, resultado in resultados_estadisticos['resultados_mcnemar'].items() 
            if resultado and resultado['significativo']
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="🔬 Comparaciones Significativas",
                value=len(comparaciones_significativas),
                delta=f"de {len(resultados_estadisticos['resultados_mcnemar'])} totales"
            )
        
        with col2:
            mejor_mcc_arq = max(resultados_estadisticos['puntuaciones_mcc'], key=resultados_estadisticos['puntuaciones_mcc'].get)
            mejor_puntuacion_mcc = resultados_estadisticos['puntuaciones_mcc'][mejor_mcc_arq]
            
            st.metric(
                label="🏆 Mejor MCC",
                value=f"{mejor_puntuacion_mcc:.4f}",
                delta=f"{mejor_mcc_arq.replace('_', ' ')}"
            )
        
        # Recomendaciones estadísticas
        st.subheader("💡 Recomendaciones Estadísticas")
        
        if len(comparaciones_significativas) == 0:
            st.warning("""
            ⚠️ **No se encontraron diferencias estadísticamente significativas** entre los modelos.
            
            **Implicaciones:**
            - Los modelos tienen rendimiento similar estadísticamente
            - Otros criterios (velocidad, tamaño) pueden ser decisivos
            - Se recomienda aumentar el tamaño del dataset de prueba
            """)
        else:
            st.success(f"""
            ✅ **Se encontraron {len(comparaciones_significativas)} diferencias significativas**
            
            **Modelos con diferencias estadísticamente probadas:**
            """)
            
            for comp in comparaciones_significativas:
                resultado = resultados_estadisticos['resultados_mcnemar'][comp]
                st.markdown(f"• **{comp.replace('_', ' ')}**: {resultado['interpretacion']}")
                
        st.subheader("📈 Curvas ROC - Análisis de Rendimiento")
        
        with st.spinner("🔄 Calculando curvas ROC..."):
            curvas_roc = self.calcular_curvas_roc(resultados_evaluacion)
            
            if curvas_roc:
                st.markdown("""
                **Curvas ROC (Receiver Operating Characteristic)**:
                - **AUC > 0.9**: Excelente capacidad discriminatoria
                - **AUC 0.8-0.9**: Buena capacidad discriminatoria  
                - **AUC 0.7-0.8**: Capacidad discriminatoria aceptable
                - **AUC = 0.5**: Sin capacidad discriminatoria (aleatorio)
                """)
                
                self.graficar_curvas_roc(curvas_roc, timestamp)
                
                # Tabla de AUC scores
                st.markdown("#### 📊 Puntuaciones AUC por Arquitectura")
                datos_auc = []
                for arq, datos_roc in curvas_roc.items():
                    if datos_roc['n_clases'] == 2:
                        auc_score = datos_roc['roc_auc'][1]
                    else:
                        auc_score = datos_roc['roc_auc']['micro']
                    
                    datos_auc.append({
                        get_text('architecture', self.lang): arq.replace('_', ' '),
                        'AUC Score': f"{auc_score:.4f}",
                        'Interpretación': self.interpretar_auc(auc_score)
                    })
                
                df_auc = pd.DataFrame(datos_auc)
                st.dataframe(df_auc, use_container_width=True)
            else:
                st.warning("⚠️ No se pudieron calcular las curvas ROC")
        
        # === SECCIÓN 5: EXPORTAR RESULTADOS ESTADÍSTICOS ===
        st.subheader("📤 Exportar Resultados Estadísticos")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Generar Reporte Estadístico TXT", use_container_width=True, key=f"btn_reporte_txt_{timestamp}"):
                self.generar_reporte_estadistico(resultados_estadisticos, resultados_evaluacion)

        with col2:
            if st.button("📄 Generar Reporte PDF Estadístico", type="primary", use_container_width=True, key=f"btn_reporte_pdf_estadistico{key_suffix}"):
                try:
                    estado_pdf = st.empty()
                    estado_pdf.info("🔄 Generando reporte PDF estadístico profesional...")
                    archivo_pdf = self.generar_reporte_pdf_estadistico(resultados_estadisticos, resultados_evaluacion)
                    
                    if archivo_pdf and os.path.exists(archivo_pdf):
                        estado_pdf.success("✅ PDF estadístico generado exitosamente!")
                        
                        with open(archivo_pdf, "rb") as f:
                            bytes_pdf = f.read()
                        
                        st.download_button(
                            label="⬇️ DESCARGAR REPORTE PDF ESTADÍSTICO",
                            data=bytes_pdf,
                            file_name=f"reporte_estadistico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key=f"descargar_pdf_estadistico{key_suffix}"
                        )
                        
                        st.balloons()
                        
                        try:
                            os.remove(archivo_pdf)
                        except:
                            pass
                    else:
                        st.error("❌ Error generando el reporte PDF estadístico")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    def interpretar_mcc(self, puntuacion_mcc):
        """Interpreta el score MCC"""
        if puntuacion_mcc >= 0.8:
            return "🟢 Excelente"
        elif puntuacion_mcc >= 0.6:
            return "🔵 Muy bueno"
        elif puntuacion_mcc >= 0.4:
            return "🟡 Bueno"
        elif puntuacion_mcc >= 0.2:
            return "🟠 Regular"
        elif puntuacion_mcc >= 0:
            return "🔴 Bajo"
        else:
            return "🔴 Muy bajo"
    
    
    def interpretar_auc(self, auc_score):
        """Interpreta el score AUC"""
        if auc_score >= 0.9:
            return "🟢 Excelente"
        elif auc_score >= 0.8:
            return "🔵 Bueno"
        elif auc_score >= 0.7:
            return "🟡 Aceptable"
        elif auc_score >= 0.6:
            return "🟠 Bajo"
        else:
            return "🔴 Muy bajo"
    
    
    def graficar_mapa_calor_mcnemar(self, resultados_mcnemar, timestamp=None):
        """Crea heatmap de p-valores de McNemar"""
        try:
            if timestamp is None:
                import time
                timestamp = str(int(time.time() * 1000))
                
            arquitecturas = list(self.modelos.keys())
            n_arqs = len(arquitecturas)
            
            # Matriz de p-valores
            matriz_valores_p = np.ones((n_arqs, n_arqs))
            matriz_significancia = np.zeros((n_arqs, n_arqs))
            
            for i, arq1 in enumerate(arquitecturas):
                for j, arq2 in enumerate(arquitecturas):
                    if i != j:
                        clave_comp = f"{arq1}_vs_{arq2}"
                        if clave_comp in resultados_mcnemar:
                            resultado = resultados_mcnemar[clave_comp]
                            matriz_valores_p[i, j] = resultado['valor_p']
                            matriz_significancia[i, j] = 1 if resultado['significativo'] else 0
                        else:
                            # Buscar comparación inversa
                            clave_comp_inv = f"{arq2}_vs_{arq1}"
                            if clave_comp_inv in resultados_mcnemar:
                                resultado = resultados_mcnemar[clave_comp_inv]
                                matriz_valores_p[i, j] = resultado['valor_p']
                                matriz_significancia[i, j] = 1 if resultado['significativo'] else 0
            
            # Crear heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=matriz_valores_p,
                x=[arq.replace('_', ' ') for arq in arquitecturas],
                y=[arq.replace('_', ' ') for arq in arquitecturas],
                colorscale='RdYlBu_r',
                text=[[f"p={matriz_valores_p[i,j]:.4f}" for j in range(n_arqs)] for i in range(n_arqs)],
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            # Añadir línea de significancia
            fig_heatmap.add_shape(
                type="line",
                x0=-0.5, y0=-0.5, x1=n_arqs-0.5, y1=n_arqs-0.5,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig_heatmap.update_layout(
                title="🔬 Heatmap de p-valores (Pruebas de McNemar)<br>Valores < 0.05 indican diferencia significativa",
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True, key=f"heatmap_mcnemar_{timestamp}")
            
        except Exception as e:
            st.error(f"Error creando heatmap: {str(e)}")
    
    def graficar_matriz_confusion(self, cm, nombre_arquitectura):
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
                title=f"📊 {nombre_arquitectura.replace('_', ' ')}",
                xaxis_title="Predicción",
                yaxis_title="Verdadero",
                height=300
            )
            
            return fig_cm
            
        except Exception as e:
            st.error(f"Error creando matriz de confusión: {str(e)}")
            return go.Figure()
    
    def generar_reporte_estadistico(self, resultados_estadisticos, resultados_evaluacion):
        """Genera reporte estadístico completo"""
        try:
            # Crear reporte en formato texto
            contenido_reporte = self.crear_contenido_reporte_estadistico(resultados_estadisticos, resultados_evaluacion)
            
            # Crear archivo
            marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo_reporte = f"reporte_estadistico_{marca_tiempo}.txt"
            
            with open(nombre_archivo_reporte, 'w', encoding='utf-8') as f:
                f.write(contenido_reporte)
            
            # Crear también JSON con datos estructurados
            nombre_archivo_json = f"datos_estadisticos_{marca_tiempo}.json"
            
            # Convertir numpy arrays para JSON
            datos_json = {}
            for clave, valor in resultados_estadisticos.items():
                if isinstance(valor, dict):
                    datos_json[clave] = {}
                    for subclave, subvalor in valor.items():
                        if isinstance(subvalor, np.ndarray):
                            datos_json[clave][subclave] = subvalor.tolist()
                        elif isinstance(subvalor, np.integer):
                            datos_json[clave][subclave] = int(subvalor)
                        elif isinstance(subvalor, np.floating):
                            datos_json[clave][subclave] = float(subvalor)
                        else:
                            datos_json[clave][subclave] = subvalor
                else:
                    datos_json[clave] = valor
            
            with open(nombre_archivo_json, 'w', encoding='utf-8') as f:
                json.dump(datos_json, f, indent=2, ensure_ascii=False)
            
            # Botones de descarga
            col1, col2 = st.columns(2)
            
            with col1:
                with open(nombre_archivo_reporte, 'r', encoding='utf-8') as f:
                    datos_reporte = f.read()
                
                st.download_button(
                    label="📄 Descargar Reporte TXT",
                    data=datos_reporte,
                    file_name=nombre_archivo_reporte,
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                with open(nombre_archivo_json, 'r', encoding='utf-8') as f:
                    datos_json = f.read()
                
                st.download_button(
                    label="📊 Descargar Datos JSON",
                    data=datos_json,
                    file_name=nombre_archivo_json,
                    mime="application/json",
                    use_container_width=True
                )
            
            st.success("✅ Reportes estadísticos generados correctamente!")
            
            # Limpiar archivos temporales
            try:
                os.remove(nombre_archivo_reporte)
                os.remove(nombre_archivo_json)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error generando reporte estadístico: {str(e)}")
            
    def generar_reporte_pdf_estadistico(self, resultados_estadisticos, resultados_evaluacion):
        """Genera reporte PDF específico para análisis estadístico"""
        try:
            # Crear PDF
            pdf = FPDF()
            pdf.add_page()
            
            # --- PORTADA ---
            pdf.set_font('Arial', 'B', 24)
            pdf.cell(0, 20, self.limpiar_texto_pdf('REPORTE ESTADÍSTICO INFERENCIAL'), 0, 1, 'C')
            pdf.ln(5)
            
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, self.limpiar_texto_pdf('Análisis Comparativo de Arquitecturas CNN'), 0, 1, 'C')
            pdf.ln(5)

            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, self.limpiar_texto_pdf(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'), 0, 1, 'C')
            pdf.cell(0, 8, self.limpiar_texto_pdf(f'Dataset de prueba: {len(resultados_evaluacion["etiquetas_verdaderas"])} imágenes'), 0, 1, 'C')
            pdf.cell(0, 8, self.limpiar_texto_pdf(f'Arquitecturas evaluadas: {len(self.modelos)}'), 0, 1, 'C')
            pdf.ln(10)

            # --- RESUMEN EJECUTIVO ESTADÍSTICO ---
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, self.limpiar_texto_pdf('RESUMEN EJECUTIVO ESTADÍSTICO'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)

            # Mejor modelo por MCC
            mejor_mcc_arq = max(resultados_estadisticos['puntuaciones_mcc'], 
                            key=resultados_estadisticos['puntuaciones_mcc'].get)
            mejor_mcc = resultados_estadisticos['puntuaciones_mcc'][mejor_mcc_arq]

            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 8, self.limpiar_texto_pdf('MODELO CON MEJOR RENDIMIENTO ESTADÍSTICO:'), 0, 1)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 6, self.limpiar_texto_pdf(f'• Arquitectura: {self.informacion_arquitecturas[mejor_mcc_arq]["nombre_completo"]}'), 0, 1)
            pdf.cell(0, 6, self.limpiar_texto_pdf(f'• MCC Score: {mejor_mcc:.6f}'), 0, 1)
            pdf.cell(0, 6, self.limpiar_texto_pdf(f'• Interpretación: {self.interpretar_mcc(mejor_mcc)}'), 0, 1)

            # Significancia estadística
            comparaciones_sig = [comp for comp, res in resultados_estadisticos['resultados_mcnemar'].items() 
                                if res and res['significativo']]

            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 6, self.limpiar_texto_pdf(f'DIFERENCIAS ESTADÍSTICAMENTE SIGNIFICATIVAS: {len(comparaciones_sig)}'), 0, 1)
            pdf.ln(5)

            # --- TABLA DE RESULTADOS MCC ---
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, self.limpiar_texto_pdf('COEFICIENTE DE CORRELACIÓN DE MATTHEWS'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, self.limpiar_texto_pdf('El MCC es una métrica balanceada que considera todos los elementos de la matriz de confusión. '
                                        'Rango: -1 (predicción completamente incorrecta) a +1 (predicción perfecta). '
                                        'Un valor de 0 indica rendimiento aleatorio.'))
            pdf.ln(5)

            # Tabla MCC
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(60, 8, self.limpiar_texto_pdf('Arquitectura'), 1, 0, 'C')
            pdf.cell(25, 8, self.limpiar_texto_pdf('MCC'), 1, 0, 'C')
            pdf.cell(25, 8, self.limpiar_texto_pdf('IC 95% Inf'), 1, 0, 'C')
            pdf.cell(25, 8, self.limpiar_texto_pdf('IC 95% Sup'), 1, 0, 'C')
            pdf.cell(25, 8, self.limpiar_texto_pdf('Accuracy'), 1, 0, 'C')
            pdf.cell(30, 8, self.limpiar_texto_pdf('Interpretación'), 1, 1, 'C')

            pdf.set_font('Arial', '', 8)
            for arq in self.modelos.keys():
                mcc = resultados_estadisticos['puntuaciones_mcc'][arq]
                ci_inf, ci_sup = resultados_estadisticos['intervalos_confianza_mcc'][arq]
                acc = resultados_estadisticos['puntuaciones_accuracy'][arq]
                
                pdf.cell(60, 6, self.limpiar_texto_pdf(self.informacion_arquitecturas[arq]['nombre_completo'][:25]), 1, 0)
                pdf.cell(25, 6, self.limpiar_texto_pdf(f'{mcc:.4f}'), 1, 0, 'C')
                pdf.cell(25, 6, self.limpiar_texto_pdf(f'{ci_inf:.4f}' if ci_inf else 'N/A'), 1, 0, 'C')
                pdf.cell(25, 6, self.limpiar_texto_pdf(f'{ci_sup:.4f}' if ci_sup else 'N/A'), 1, 0, 'C')
                pdf.cell(25, 6, self.limpiar_texto_pdf(f'{acc:.4f}'), 1, 0, 'C')
                pdf.cell(30, 6, self.limpiar_texto_pdf(self.interpretar_mcc(mcc)[:15]), 1, 1, 'C')

            # --- PRUEBAS DE MCNEMAR ---
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self.limpiar_texto_pdf('PRUEBAS DE MCNEMAR - COMPARACIONES PAREADAS'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 4, self.limpiar_texto_pdf('La prueba de McNemar evalúa si existe diferencia estadísticamente significativa '
                                        'entre el rendimiento de dos modelos (α = 0.05).'))
            pdf.ln(3)

            # Tabla McNemar
            pdf.set_font('Arial', 'B', 9)
            pdf.cell(50, 6, self.limpiar_texto_pdf('Comparación'), 1, 0, 'C')
            pdf.cell(25, 6, self.limpiar_texto_pdf('Estadístico'), 1, 0, 'C')
            pdf.cell(20, 6, self.limpiar_texto_pdf('p-valor'), 1, 0, 'C')
            pdf.cell(20, 6, self.limpiar_texto_pdf('Significativo'), 1, 0, 'C')
            pdf.cell(75, 6, self.limpiar_texto_pdf('Interpretación'), 1, 1, 'C')

            pdf.set_font('Arial', '', 8)
            for comp, resultado in resultados_estadisticos['resultados_mcnemar'].items():
                if resultado:
                    comp_text = comp.replace('_vs_', ' vs ').replace('_', ' ')
                    pdf.cell(50, 5, self.limpiar_texto_pdf(comp_text[:20]), 1, 0)
                    pdf.cell(25, 5, self.limpiar_texto_pdf(f'{resultado["estadistico"]:.4f}'), 1, 0, 'C')
                    pdf.cell(20, 5, self.limpiar_texto_pdf(f'{resultado["valor_p"]:.4f}'), 1, 0, 'C')
                    pdf.cell(20, 5, self.limpiar_texto_pdf('Sí' if resultado['significativo'] else 'No'), 1, 0, 'C')
                    pdf.cell(75, 5, self.limpiar_texto_pdf(resultado['interpretacion'][:35]), 1, 1)

            # --- CONCLUSIONES Y RECOMENDACIONES ---
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, self.limpiar_texto_pdf('CONCLUSIONES Y RECOMENDACIONES'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)

            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, self.limpiar_texto_pdf('1. RANKING DE MODELOS POR RENDIMIENTO ESTADÍSTICO:'), 0, 1)

            # Ranking por MCC
            ranking_mcc = sorted(resultados_estadisticos['puntuaciones_mcc'].items(), 
                            key=lambda x: x[1], reverse=True)

            pdf.set_font('Arial', '', 10)
            for i, (arq, mcc) in enumerate(ranking_mcc, 1):
                nombre_arq = self.informacion_arquitecturas[arq]['nombre_completo']
                pdf.cell(0, 6, self.limpiar_texto_pdf(f'{i}. {nombre_arq}: MCC = {mcc:.4f} ({self.interpretar_mcc(mcc)})'), 0, 1)

            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, self.limpiar_texto_pdf('2. SIGNIFICANCIA ESTADÍSTICA:'), 0, 1)

            pdf.set_font('Arial', '', 10)
            if len(comparaciones_sig) == 0:
                pdf.multi_cell(0, 5, self.limpiar_texto_pdf('No se encontraron diferencias estadísticamente significativas entre los modelos. '
                                            'Esto sugiere que todos tienen rendimiento similar desde el punto de vista estadístico.'))
            else:
                pdf.cell(0, 5, self.limpiar_texto_pdf(f'Se encontraron {len(comparaciones_sig)} diferencias significativas:'), 0, 1)
                for comp in comparaciones_sig[:3]:  # Mostrar máximo 3
                    resultado = resultados_estadisticos['resultados_mcnemar'][comp]
                    pdf.cell(0, 4, self.limpiar_texto_pdf(f'• {comp.replace("_", " ")}: {resultado["interpretacion"][:50]}'), 0, 1)

            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, self.limpiar_texto_pdf('3. RECOMENDACIÓN FINAL:'), 0, 1)

            pdf.set_font('Arial', '', 10)
            if len(comparaciones_sig) > 0:
                pdf.multi_cell(0, 5, self.limpiar_texto_pdf(f'Se recomienda el uso de {self.informacion_arquitecturas[mejor_mcc_arq]["nombre_completo"]} '
                                            f'por su superior rendimiento estadístico (MCC = {mejor_mcc:.4f}) y evidencia '
                                            'de diferencias significativas con otros modelos.'))
            else:
                pdf.multi_cell(0, 5, self.limpiar_texto_pdf('Dado que no hay diferencias estadísticamente significativas, la selección '
                                            'puede basarse en otros criterios como velocidad, tamaño o eficiencia energética.'))

            # --- INFORMACIÓN TÉCNICA ---
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, self.limpiar_texto_pdf('INFORMACIÓN TÉCNICA'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)

            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, self.limpiar_texto_pdf(f'''
    METODOLOGÍA ESTADÍSTICA:

    - Coeficiente de Correlación de Matthews (MCC):
    - Métrica balanceada que considera verdaderos/falsos positivos y negativos
    - Especialmente útil para datasets con clases desbalanceadas
    - Más informativo que la accuracy tradicional

    - Prueba de McNemar:
    - Test estadístico pareado para comparar clasificadores
    - H₀: No hay diferencia entre modelos
    - H₁: Existe diferencia significativa (α = 0.05)
    - Usa corrección de continuidad de Yates

    - Intervalos de Confianza Bootstrap:
    - Método no paramétrico para estimar incertidumbre
    - 1000 muestras bootstrap por modelo
    - Nivel de confianza: 95%

    DATASET DE EVALUACIÓN:
    - Total de imágenes: {len(resultados_evaluacion["etiquetas_verdaderas"])}
    - Arquitecturas evaluadas: {len(self.modelos)}
    - Clases únicas: {len(np.unique(resultados_evaluacion["etiquetas_verdaderas"]))}

    LIMITACIONES:
    - Los resultados son válidos para este dataset específico
    - Se recomienda validación cruzada para mayor robustez
    - El rendimiento puede variar con diferentes poblaciones
            '''))
            
            # Generar archivo
            marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo_pdf = f"reporte_estadistico_completo_{marca_tiempo}.pdf"  # ← CAMBIAR ESTA LÍNEA
            pdf.output(nombre_archivo_pdf)
        
            return nombre_archivo_pdf
            
        except Exception as e:
            st.error(f"Error generando reporte PDF estadístico: {str(e)}")
            return None
    
    def crear_contenido_reporte_estadistico(self, resultados_estadisticos, resultados_evaluacion):
        """Crea contenido del reporte estadístico"""
        reporte = f"""
        REPORTE DE ANÁLISIS ESTADÍSTICO INFERENCIAL
        ===========================================

        Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Sistema: Comparación de Arquitecturas CNN para Diagnóstico Ocular
        Número de modelos evaluados: {len(self.modelos)}
        Tamaño del dataset de prueba: {len(resultados_evaluacion['etiquetas_verdaderas'])}

        1. COEFICIENTE DE CORRELACIÓN DE MATTHEWS (MCC)
        ===============================================

        El MCC es una métrica balanceada que considera todas las categorías de la matriz de confusión.
        Rango: -1 (completamente incorrecto) a +1 (predicción perfecta)

        """
        
        # Resultados MCC
        for arq in self.modelos.keys():
            puntuacion_mcc = resultados_estadisticos['puntuaciones_mcc'][arq]
            ci_inferior, ci_superior = resultados_estadisticos['intervalos_confianza_mcc'][arq]
            accuracy = resultados_estadisticos['puntuaciones_accuracy'][arq]
            
            reporte += f"""
{arq.replace('_', ' ')}:
  - MCC Score: {puntuacion_mcc:.6f}
  - IC 95%: [{ci_inferior:.6f}, {ci_superior:.6f}]
  - Accuracy: {accuracy:.6f}
  - Interpretación: {self.interpretar_mcc(puntuacion_mcc)}
"""
        
        reporte += f"""

2. PRUEBAS DE MCNEMAR
====================

La prueba de McNemar compara estadísticamente el rendimiento entre pares de modelos.
H₀: No hay diferencia entre los modelos
H₁: Hay diferencia significativa (α = 0.05)

"""
        
        # Resultados McNemar
        for comparacion, resultado in resultados_estadisticos['resultados_mcnemar'].items():
            if resultado:
                reporte += f"""
{comparacion.replace('_', ' ')}:
  - Estadístico McNemar: {resultado['estadistico']:.6f}
  - p-valor: {resultado['valor_p']:.6f}
  - Significativo: {'Sí' if resultado['significativo'] else 'No'}
  - N° desacuerdos: {resultado['n_desacuerdos']}
  - Interpretación: {resultado['interpretacion']}
"""
        
        # Resumen y recomendaciones
        comparaciones_significativas = [
            comp for comp, resultado in resultados_estadisticos['resultados_mcnemar'].items() 
            if resultado and resultado['significativo']
        ]
        
        mejor_mcc_arq = max(resultados_estadisticos['puntuaciones_mcc'], key=resultados_estadisticos['puntuaciones_mcc'].get)
        mejor_puntuacion_mcc = resultados_estadisticos['puntuaciones_mcc'][mejor_mcc_arq]
        
        reporte += f"""

3. RESUMEN Y RECOMENDACIONES
============================

Mejor modelo por MCC: {mejor_mcc_arq.replace('_', ' ')} (MCC = {mejor_puntuacion_mcc:.6f})
Comparaciones significativas: {len(comparaciones_significativas)} de {len(resultados_estadisticos['resultados_mcnemar'])}

"""
        
        if len(comparaciones_significativas) == 0:
            reporte += """
CONCLUSIÓN:
No se encontraron diferencias estadísticamente significativas entre los modelos.
Esto sugiere que todos los modelos tienen un rendimiento similar estadísticamente.
La selección del modelo puede basarse en otros criterios como velocidad o eficiencia.

"""
        else:
            reporte += f"""
CONCLUSIÓN:
Se encontraron {len(comparaciones_significativas)} diferencias estadísticamente significativas:

"""
            for comp in comparaciones_significativas:
                resultado = resultados_estadisticos['resultados_mcnemar'][comp]
                reporte += f"- {comp.replace('_', ' ')}: {resultado['interpretacion']}\n"
        
        reporte += f"""

4. DETALLES TÉCNICOS
====================

Dataset de evaluación: {len(resultados_evaluacion['etiquetas_verdaderas'])} imágenes
Métodos estadísticos utilizados:
- Coeficiente de Correlación de Matthews
- Prueba de McNemar con corrección de continuidad de Yates
- Intervalos de confianza bootstrap (95%)
- Matrices de confusión

Arquitecturas evaluadas:
"""
        
        for arq in self.modelos.keys():
            info = self.informacion_arquitecturas[arq]
            reporte += f"- {info['nombre_completo']}: {info['descripcion']}\n"
        
        return reporte
    
    # ========== FUNCIONES ORIGINALES TRADUCIDAS ==========
    
    def calcular_curvas_roc(self, resultados_evaluacion):
        """Calcula curvas ROC para cada modelo"""
        try:
            y_verdadero = resultados_evaluacion['etiquetas_verdaderas']
            n_clases = len(np.unique(y_verdadero))
            
            # Binarizar las etiquetas para ROC multiclase
            y_verdadero_bin = label_binarize(y_verdadero, classes=range(n_clases))
            
            curvas_roc = {}
            
            for arq in self.modelos.keys():
                # Obtener probabilidades de predicción
                probabilidades = []
                for i, pred in enumerate(resultados_evaluacion['predicciones'][arq]):
                    # Crear vector de probabilidades one-hot simulado
                    prob_vec = np.zeros(n_clases)
                    prob_vec[pred] = resultados_evaluacion['confianzas'][arq][i]
                    # Distribuir probabilidad restante
                    prob_restante = (1 - resultados_evaluacion['confianzas'][arq][i]) / (n_clases - 1)
                    for j in range(n_clases):
                        if j != pred:
                            prob_vec[j] = prob_restante
                    probabilidades.append(prob_vec)
                
                probabilidades = np.array(probabilidades)
                
                # Calcular ROC para cada clase
                fpr = {}
                tpr = {}
                roc_auc = {}
                
                for i in range(n_clases):
                    if n_clases == 2:
                        fpr[i], tpr[i], _ = roc_curve(y_verdadero_bin, probabilidades[:, 1])
                    else:
                        fpr[i], tpr[i], _ = roc_curve(y_verdadero_bin[:, i], probabilidades[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # ROC micro-average
                if n_clases > 2:
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_verdadero_bin.ravel(), probabilidades.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                curvas_roc[arq] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'roc_auc': roc_auc,
                    'n_clases': n_clases
                }
            
            return curvas_roc
        
        except Exception as e:
            st.error(f"Error calculando curvas ROC: {str(e)}")
            return None
    
    def graficar_curvas_roc(self, curvas_roc, timestamp=None):
        """Grafica curvas ROC comparativas"""
        try:
            if timestamp is None:
                import time
                timestamp = str(int(time.time() * 1000))
            
            fig_roc = go.Figure()
            
            # Colores para cada arquitectura
            colores = ['#E91E63', '#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
            
            for i, (arq, datos_roc) in enumerate(curvas_roc.items()):
                color = colores[i % len(colores)]
                info_arq = self.informacion_arquitecturas[arq]
                
                if datos_roc['n_clases'] == 2:
                    # ROC binario
                    fpr = datos_roc['fpr'][1]
                    tpr = datos_roc['tpr'][1]
                    auc_score = datos_roc['roc_auc'][1]
                    
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f"{arq.replace('_', ' ')} (AUC = {auc_score:.3f})",
                        line=dict(color=color, width=3)
                    ))
                else:
                    # ROC multiclase - mostrar micro-average
                    fpr = datos_roc['fpr']["micro"]
                    tpr = datos_roc['tpr']["micro"]
                    auc_score = datos_roc['roc_auc']["micro"]
                    
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f"{arq.replace('_', ' ')} (AUC = {auc_score:.3f})",
                        line=dict(color=color, width=3)
                    ))
            
            # Línea diagonal (clasificador aleatorio)
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Clasificador Aleatorio (AUC = 0.500)',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig_roc.update_layout(
                title='📈 Curvas ROC - Comparación de Arquitecturas',
                xaxis_title='Tasa de Falsos Positivos (1 - Especificidad)',
                yaxis_title='Tasa de Verdaderos Positivos (Sensibilidad)',
                width=800,
                height=600,
                legend=dict(x=0.6, y=0.1)
            )
            
            st.plotly_chart(fig_roc, use_container_width=True, key=f"curvas_roc_{timestamp}")
            
        except Exception as e:
            st.error(f"Error graficando curvas ROC: {str(e)}")
    
    
    def encontrar_mejor_arquitectura(self, predicciones):
        """Encuentra la mejor arquitectura por diferentes métricas"""
        if not predicciones or len(predicciones) < 2:
            return {}
        
        # Mejor por confianza
        mejor_confianza = max(predicciones, key=lambda x: x['confianza'])
        
        # Más rápido
        mas_rapido = min(predicciones, key=lambda x: x['tiempo_prediccion'])
        
        # Más eficiente (mayor confianza / tiempo)
        for pred in predicciones:
            pred['eficiencia'] = pred['confianza'] / pred['tiempo_prediccion']
        mas_eficiente = max(predicciones, key=lambda x: x['eficiencia'])
        
        # Más ligero
        mas_ligero = min(predicciones, key=lambda x: x['tamaño_modelo'])
        
        return {
            'mayor_confianza': mejor_confianza,
            'mas_rapido': mas_rapido,
            'mas_eficiente': mas_eficiente,
            'mas_ligero': mas_ligero
        }
    
    def mostrar_encabezado(self):
        """Header de la aplicación - TRADUCIDO"""
        st.title(get_text('main_title', self.lang))
        st.subheader(get_text('page_subtitle', self.lang))
        st.markdown("---")
        
    def mostrar_vitrina_arquitecturas(self):
        """Muestra las características de cada arquitectura - TRADUCIDA"""
        st.header(get_text('architectures_title', self.lang))
        
        cols = st.columns(3)
        
        for i, (nombre_arq, info) in enumerate(self.informacion_arquitecturas.items()):
            with cols[i]:
                # Header de la arquitectura
                st.subheader(f"{info['icon']} {info['nombre_completo']}")
                
                # Descripción
                st.info(f"**{info['descripcion']}**")
                
                # Características técnicas
                st.markdown(f"**📊 {get_text('characteristics', self.lang)}:**")
                st.markdown(f"• **{get_text('type', self.lang)}:** {info['caracteristicas']['Tipo']}")
                st.markdown(f"• **{get_text('parameters_count', self.lang)}:** {info['caracteristicas']['Parámetros']}")
                st.markdown(f"• **{get_text('main_advantage', self.lang)}:** {info['caracteristicas']['Ventaja principal']}")
                st.markdown(f"• **{get_text('year', self.lang)}:** {info['caracteristicas']['Año']}")
                
                # Ventajas
                st.markdown(f"**✅ {get_text('advantages', self.lang)}:**")
                for ventaja in info['ventajas']:
                    st.markdown(f"• {ventaja}")
                
                st.markdown("---")
    
    def mostrar_resultados_prediccion(self, predicciones):
        """Muestra resultados de las 3 arquitecturas lado a lado - TRADUCIDA"""
        st.header(get_text('results_title', self.lang))
        
        cols = st.columns(3)
        
        for i, pred in enumerate(predicciones):
            nombre_arq = pred['arquitectura']
            info = self.informacion_arquitecturas[nombre_arq]
            
            with cols[i]:
                # Nombre de la arquitectura
                st.subheader(f"{info['icon']} {nombre_arq.replace('_', ' ')}")
                
                # Diagnóstico
                clase_predicha = pred['clase_predicha']
                info_clase = self.informacion_clases.get(clase_predicha, {})
                nombre_es = info_clase.get('nombre', clase_predicha)
                
                st.success(get_text('diagnosis', self.lang, diagnosis=nombre_es))
                
                # Confianza (métrica principal)
                st.metric(
                    label=get_text('confidence', self.lang),
                    value=f"{pred['confianza']:.1%}",
                    delta=None
                )
                
                # Métricas técnicas
                st.markdown(get_text('technical_metrics', self.lang))
                st.markdown(get_text('time', self.lang, time=pred['tiempo_prediccion']))
                st.markdown(get_text('size', self.lang, size=pred['tamaño_modelo']))
                st.markdown(get_text('parameters', self.lang, params=pred['conteo_parametros']))
                
                st.markdown("---")
    
    def mostrar_comparacion_rendimiento(self, predicciones):
        """Gráficos comparativos de rendimiento - TRADUCIDA"""
        st.markdown(f"## {get_text('comparison_title', self.lang)}")
        
        # Crear timestamp único
        import time
        timestamp = str(int(time.time() * 1000))
        
        # Crear DataFrame para gráficos
        df = pd.DataFrame([
            {
                get_text('architecture', self.lang): pred['arquitectura'].replace('_', ' '),
                get_text('confidence', self.lang): pred['confianza'],
                f"{get_text('time_table', self.lang)} (s)": pred['tiempo_prediccion'],
                f"{get_text('size_table', self.lang)} (MB)": pred['tamaño_modelo'],
                f"{get_text('parameters_table', self.lang)} (M)": pred['conteo_parametros'] / 1_000_000,
                f"{get_text('efficiency_table', self.lang)} (Conf/Tiempo)": pred['confianza'] / pred['tiempo_prediccion']
            }
            for pred in predicciones
        ])
        
        # Colores para gráficos
        colores = [self.informacion_arquitecturas[pred['arquitectura']]['color'] for pred in predicciones]
        
        # 4 gráficos en 2x2
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de confianza
            fig_conf = go.Figure(data=[
                go.Bar(
                    x=df[get_text('architecture', self.lang)],
                    y=df[get_text('confidence', self.lang)],
                    text=[f"{conf:.1%}" for conf in df[get_text('confidence', self.lang)]],
                    textposition='auto',
                    marker_color=colores,
                    name=get_text('confidence', self.lang)
                )
            ])
            fig_conf.update_layout(
                title=get_text('confidence_chart', self.lang),
                yaxis=dict(tickformat='.0%'),
                height=400
            )
            st.plotly_chart(fig_conf, use_container_width=True, key=f"grafico_confianza_{timestamp}")
            
            # Gráfico de tamaño
            fig_tamaño = go.Figure(data=[
                go.Bar(
                    x=df[get_text('architecture', self.lang)],
                    y=df[f"{get_text('size_table', self.lang)} (MB)"],
                    text=[f"{tamaño:.1f}MB" for tamaño in df[f"{get_text('size_table', self.lang)} (MB)"]],
                    textposition='auto',
                    marker_color=colores,
                    name=get_text('size_table', self.lang)
                )
            ])
            fig_tamaño.update_layout(
                title=get_text('size_chart', self.lang),
                yaxis_title=f"{get_text('size_table', self.lang)} (MB)",
                height=400
            )
            st.plotly_chart(fig_tamaño, use_container_width=True, key=f"grafico_tamaño_{timestamp}")
        
        with col2:
            # Gráfico de tiempo
            fig_tiempo = go.Figure(data=[
                go.Bar(
                    x=df[get_text('architecture', self.lang)],
                    y=df[f"{get_text('time_table', self.lang)} (s)"],
                    text=[f"{tiempo:.3f}s" for tiempo in df[f"{get_text('time_table', self.lang)} (s)"]],
                    textposition='auto',
                    marker_color=colores,
                    name=get_text('time_table', self.lang)
                )
            ])
            fig_tiempo.update_layout(
                title=get_text('time_chart', self.lang),
                yaxis_title=f"{get_text('time_table', self.lang)} (segundos)",
                height=400
            )
            st.plotly_chart(fig_tiempo, use_container_width=True, key=f"grafico_tiempo_{timestamp}")
            
            # Gráfico de eficiencia
            fig_eff = go.Figure(data=[
                go.Bar(
                    x=df[get_text('architecture', self.lang)],
                    y=df[f"{get_text('efficiency_table', self.lang)} (Conf/Tiempo)"],
                    text=[f"{eff:.1f}" for eff in df[f"{get_text('efficiency_table', self.lang)} (Conf/Tiempo)"]],
                    textposition='auto',
                    marker_color=colores,
                    name=get_text('efficiency_table', self.lang)
                )
            ])
            fig_eff.update_layout(
                title=get_text('efficiency_chart', self.lang),
                yaxis_title=f"{get_text('efficiency_table', self.lang)} Score",
                height=400
            )
            st.plotly_chart(fig_eff, use_container_width=True, key=f"grafico_eficiencia_{timestamp}")
    
    def mostrar_comparacion_radar(self, predicciones):
        """Gráfico radar comparando todas las métricas - TRADUCIDA"""
        st.markdown(f"### {get_text('radar_title', self.lang)}")
        
        # Crear timestamp único
        import time
        timestamp = str(int(time.time() * 1000))
        
        # Normalizar métricas para el radar (0-1)
        max_conf = max(pred['confianza'] for pred in predicciones)
        min_tiempo = min(pred['tiempo_prediccion'] for pred in predicciones)
        max_tiempo = max(pred['tiempo_prediccion'] for pred in predicciones)
        min_tamaño = min(pred['tamaño_modelo'] for pred in predicciones)
        max_tamaño = max(pred['tamaño_modelo'] for pred in predicciones)
        
        fig = go.Figure()
        
        categorias = [
            get_text('confidence', self.lang), 
            'Velocidad', 
            'Eficiencia Memoria', 
            'Score General'
        ]
        
        for pred in predicciones:
            nombre_arq = pred['arquitectura']
            info = self.informacion_arquitecturas[nombre_arq]
            
            # Normalizar valores (más alto = mejor)
            norm_conf = pred['confianza'] / max_conf if max_conf > 0 else 0
            norm_velocidad = (max_tiempo - pred['tiempo_prediccion']) / (max_tiempo - min_tiempo) if max_tiempo > min_tiempo else 1
            norm_memoria = (max_tamaño - pred['tamaño_modelo']) / (max_tamaño - min_tamaño) if max_tamaño > min_tamaño else 1
            norm_general = (norm_conf + norm_velocidad + norm_memoria) / 3
            
            valores = [norm_conf, norm_velocidad, norm_memoria, norm_general]
            
            fig.add_trace(go.Scatterpolar(
                r=valores,
                theta=categorias,
                fill='toself',
                name=nombre_arq.replace('_', ' '),
                line_color=info['color']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.0%'
                )),
            title=get_text('radar_chart', self.lang),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"grafico_radar_{timestamp}")
    
    def mostrar_podio_ganadores(self, mejores_modelos):
        """Muestra el podio de ganadores por categoría - TRADUCIDA"""
        st.header(get_text('winners_title', self.lang))
        
        categorias = [
            ('mayor_confianza', get_text('highest_confidence', self.lang), get_text('most_accurate', self.lang)),
            ('mas_rapido', get_text('fastest', self.lang), get_text('speedster', self.lang)),
            ('mas_ligero', get_text('lightest', self.lang), get_text('efficient', self.lang)),
            ('mas_eficiente', get_text('most_efficient', self.lang), get_text('balanced', self.lang))
        ]
        
        cols = st.columns(2)
        
        for i, (clave, titulo, subtitulo) in enumerate(categorias):
            col = cols[i % 2]
            
            with col:
                if clave in mejores_modelos:
                    ganador = mejores_modelos[clave]
                    nombre_arq = ganador['arquitectura']
                    info = self.informacion_arquitecturas[nombre_arq]
                    
                    if clave == 'mayor_confianza':
                        valor_metrica = f"{ganador['confianza']:.1%}"
                    elif clave == 'mas_rapido':
                        valor_metrica = f"{ganador['tiempo_prediccion']:.3f}s"
                    elif clave == 'mas_ligero':
                        valor_metrica = f"{ganador['tamaño_modelo']:.1f}MB"
                    else:  # mas_eficiente
                        valor_metrica = f"{ganador['eficiencia']:.1f}"
                    
                    # Usar diferentes tipos de alertas para cada categoría
                    if clave == 'mayor_confianza':
                        st.success(f"**{titulo}**\n\n{info['icon']} **{nombre_arq.replace('_', ' ')}**\n\n{valor_metrica}\n\n*{subtitulo}*")
                    elif clave == 'mas_rapido':
                        st.info(f"**{titulo}**\n\n{info['icon']} **{nombre_arq.replace('_', ' ')}**\n\n{valor_metrica}\n\n*{subtitulo}*")
                    elif clave == 'mas_ligero':
                        st.warning(f"**{titulo}**\n\n{info['icon']} **{nombre_arq.replace('_', ' ')}**\n\n{valor_metrica}\n\n*{subtitulo}*")
                    else:  # mas_eficiente
                        st.error(f"**{titulo}**\n\n{info['icon']} **{nombre_arq.replace('_', ' ')}**\n\n{valor_metrica}\n\n*{subtitulo}*")
    
    def mostrar_analisis_detallado(self, predicciones, mejores_modelos):
        """Análisis detallado y recomendaciones - TRADUCIDA"""
        st.markdown(f"## {get_text('detailed_analysis_title', self.lang)}")
        
        # Encontrar el mejor general (combinación de métricas)
        for pred in predicciones:
            # Score combinado: 50% confianza + 25% velocidad + 25% eficiencia memoria
            max_conf = max(p['confianza'] for p in predicciones)
            min_tiempo = min(p['tiempo_prediccion'] for p in predicciones)
            min_tamaño = min(p['tamaño_modelo'] for p in predicciones)
            
            score_conf = pred['confianza'] / max_conf
            score_velocidad = min_tiempo / pred['tiempo_prediccion']
            score_memoria = min_tamaño / pred['tamaño_modelo']
            
            pred['score_general'] = 0.5 * score_conf + 0.25 * score_velocidad + 0.25 * score_memoria
        
        mejor_general = max(predicciones, key=lambda x: x['score_general'])
        
        # Mostrar ganador general
        nombre_arq = mejor_general['arquitectura']
        info = self.informacion_arquitecturas[nombre_arq]
        
        st.balloons()  # Celebración!
        st.success(get_text('general_winner', self.lang, name=f"{info['icon']} {nombre_arq.replace('_', ' ')}"))
        st.metric(
            label=get_text('general_score', self.lang),
            value=f"{mejor_general['score_general']:.3f}",
            delta=get_text('best_balance', self.lang)
        )
        
        # Análisis por arquitectura
        st.markdown(f"### {get_text('strengths_weaknesses', self.lang)}")
        
        for pred in predicciones:
            nombre_arq = pred['arquitectura']
            info = self.informacion_arquitecturas[nombre_arq]
            
            with st.expander(f"{info['icon']} {nombre_arq.replace('_', ' ')} - Análisis Detallado"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(get_text('strengths', self.lang))
                    fortalezas = []
                    
                    if pred == mejores_modelos.get('mayor_confianza'):
                        fortalezas.append("✅ Mayor confianza de predicción")
                    if pred == mejores_modelos.get('mas_rapido'):
                        fortalezas.append("✅ Tiempo de respuesta más rápido")
                    if pred == mejores_modelos.get('mas_ligero'):
                        fortalezas.append("✅ Menor uso de memoria")
                    if pred == mejores_modelos.get('mas_eficiente'):
                        fortalezas.append("✅ Mejor relación confianza/tiempo")
                    
                    # Agregar fortalezas generales
                    for ventaja in info['ventajas']:
                        fortalezas.append(f"✅ {ventaja}")
                    
                    for fortaleza in fortalezas:
                        st.markdown(fortaleza)
                
                with col2:
                    st.markdown(get_text('weaknesses', self.lang))
                    debilidades = []
                    
                    if pred != mejores_modelos.get('mayor_confianza'):
                        debilidades.append(f"🔸 Confianza: {pred['confianza']:.1%} vs {mejores_modelos['mayor_confianza']['confianza']:.1%}")
                    if pred != mejores_modelos.get('mas_rapido'):
                        debilidades.append(f"🔸 Velocidad: {pred['tiempo_prediccion']:.3f}s vs {mejores_modelos['mas_rapido']['tiempo_prediccion']:.3f}s")
                    if pred != mejores_modelos.get('mas_ligero'):
                        debilidades.append(f"🔸 Tamaño: {pred['tamaño_modelo']:.1f}MB vs {mejores_modelos['mas_ligero']['tamaño_modelo']:.1f}MB")
                    
                    for debilidad in debilidades:
                        st.markdown(debilidad)
                
                # Métricas técnicas
                st.markdown(get_text('technical_details', self.lang))
                st.markdown(f"""
                - **{get_text('parameters_count', self.lang)}**: {pred['conteo_parametros']:,}
                - **Tiempo de predicción**: {pred['tiempo_prediccion']:.3f}s
                - **Tamaño del modelo**: {pred['tamaño_modelo']:.1f}MB
                - **Eficiencia**: {pred['eficiencia']:.1f} (confianza/tiempo)
                - **Score general**: {pred['score_general']:.3f}
                """)
        
        # Recomendaciones de uso
        st.markdown(f"### {get_text('usage_recommendations', self.lang)}")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.markdown(get_text('clinical_apps', self.lang))
            st.markdown(get_text('clinical_desc', self.lang))
        
        with rec_col2:
            st.markdown(get_text('mobile_apps', self.lang))
            st.markdown(get_text('mobile_desc', self.lang))
        
        with rec_col3:
            st.markdown(get_text('production_apps', self.lang))
            st.markdown(get_text('production_desc', self.lang))

    def generar_reporte_pdf_completo(self, predicciones, imagen, marca_tiempo_analisis):
        """Genera reporte PDF profesional completo con gráficos"""
        try:
            # Crear PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Lista para archivos temporales
            archivos_temporales = []
            
            # --- PORTADA ---
            pdf.set_font('Arial', 'B', 20)
            pdf.cell(0, 15, self.limpiar_texto_pdf('REPORTE DE DIAGNÓSTICO OCULAR AVANZADO'), 0, 1, 'C')
            pdf.ln(5)
            
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self.limpiar_texto_pdf('Sistema Multi-Arquitectura CNN'), 0, 1, 'C')
            pdf.ln(10)
            
            # Información general
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, self.limpiar_texto_pdf(f'Fecha del análisis: {marca_tiempo_analisis}'), 0, 1)
            pdf.cell(0, 8, self.limpiar_texto_pdf(f'Arquitecturas analizadas: {len(predicciones)}'), 0, 1)
            pdf.cell(0, 8, self.limpiar_texto_pdf('Enfermedades detectables: 10 patologías especializadas'), 0, 1)
            pdf.ln(10)
            
            # --- RESUMEN EJECUTIVO ---
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self.limpiar_texto_pdf('RESUMEN EJECUTIVO'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            
            # Encontrar ganador general
            mejor_general = max(predicciones, key=lambda x: x.get('score_general', 0))
            info_ganador = self.informacion_arquitecturas[mejor_general['arquitectura']]
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, self.limpiar_texto_pdf(f'ARQUITECTURA RECOMENDADA: {info_ganador["nombre_completo"]}'), 0, 1)
            
            pdf.set_font('Arial', '', 11)
            clase_predicha = mejor_general['clase_predicha']
            info_clase = self.informacion_clases.get(clase_predicha, {})
            
            pdf.cell(0, 6, self.limpiar_texto_pdf(f'Diagnóstico principal: {info_clase.get("nombre", clase_predicha)}'), 0, 1)
            pdf.cell(0, 6, self.limpiar_texto_pdf(f'Nivel de confianza: {mejor_general["confianza"]:.1%}'), 0, 1)
            pdf.cell(0, 6, self.limpiar_texto_pdf(f'Gravedad: {info_clase.get("gravedad", "No especificada")}'), 0, 1)
            pdf.ln(8)
            
            # Agregar imagen analizada
            try:
                if imagen is not None:
                    nombre_img_temp = f"temp_img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    archivos_temporales.append(nombre_img_temp)
                    
                    if hasattr(imagen, 'save'):
                        imagen_rgb = imagen.convert('RGB')
                        imagen_rgb.save(nombre_img_temp, 'JPEG', quality=85)
                    else:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                        ax.text(0.5, 0.5, 'Imagen Analizada', ha='center', va='center', fontsize=14)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
                        plt.savefig(nombre_img_temp, dpi=150, bbox_inches='tight')
                        plt.close()
                    
                    pdf.cell(0, 10, self.limpiar_texto_pdf('IMAGEN ANALIZADA:'), 0, 1)
                    pdf.image(nombre_img_temp, w=80)
                    pdf.ln(5)
                    
            except Exception as error_img:
                pdf.set_font('Arial', 'I', 10)
                pdf.cell(0, 6, self.limpiar_texto_pdf(f'[Imagen no disponible: {str(error_img)[:50]}...]'), 0, 1)
                pdf.ln(5)
            
            # --- NUEVA PÁGINA: COMPARACIÓN DE ARQUITECTURAS ---
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, self.limpiar_texto_pdf('COMPARACIÓN DE ARQUITECTURAS CNN'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)
            
            # Tabla comparativa
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(50, 8, self.limpiar_texto_pdf('Arquitectura'), 1, 0, 'C')
            pdf.cell(35, 8, self.limpiar_texto_pdf('Confianza'), 1, 0, 'C')
            pdf.cell(30, 8, self.limpiar_texto_pdf('Tiempo (ms)'), 1, 0, 'C')
            pdf.cell(25, 8, self.limpiar_texto_pdf('Tamaño (MB)'), 1, 0, 'C')
            pdf.cell(30, 8, self.limpiar_texto_pdf('Eficiencia'), 1, 1, 'C')
            
            pdf.set_font('Arial', '', 9)
            for pred in predicciones:
                nombre_arq = pred['arquitectura'].replace('_', ' ')
                pdf.cell(50, 6, self.limpiar_texto_pdf(nombre_arq), 1, 0)
                pdf.cell(35, 6, self.limpiar_texto_pdf(f"{pred['confianza']:.1%}"), 1, 0, 'C')
                pdf.cell(30, 6, self.limpiar_texto_pdf(f"{pred['tiempo_prediccion']*1000:.1f}"), 1, 0, 'C')
                pdf.cell(25, 6, self.limpiar_texto_pdf(f"{pred['tamaño_modelo']:.1f}"), 1, 0, 'C')
                pdf.cell(30, 6, self.limpiar_texto_pdf(f"{pred.get('eficiencia', 0):.1f}"), 1, 1, 'C')
            
            pdf.ln(10)
            
            # --- GRÁFICOS DE RENDIMIENTO ---
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, self.limpiar_texto_pdf('ANÁLISIS GRÁFICO DE RENDIMIENTO'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)
            
            # Generar gráficos de comparación
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Crear DataFrame para gráficos
                df_datos = pd.DataFrame([
                    {
                        'Arquitectura': pred['arquitectura'].replace('_', ' '),
                        'Confianza': pred['confianza'],
                        'Tiempo': pred['tiempo_prediccion'],
                        'Tamaño': pred['tamaño_modelo'],
                        'Eficiencia': pred.get('eficiencia', 0)
                    }
                    for pred in predicciones
                ])
                
                # Colores para gráficos
                colores = [self.informacion_arquitecturas[pred['arquitectura']]['color'] for pred in predicciones]
                
                # 1. Gráfico de Confianza
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(df_datos['Arquitectura'], df_datos['Confianza'], color=colores)
                ax.set_title('Comparación de Confianza por Arquitectura', fontsize=16, fontweight='bold')
                ax.set_ylabel('Confianza (%)')
                ax.set_ylim(0, 1)
                
                # Agregar valores en las barras
                for bar, conf in zip(bars, df_datos['Confianza']):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{conf:.1%}', ha='center', va='bottom', fontweight='bold')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                nombre_grafico_conf = f"grafico_confianza_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(nombre_grafico_conf, dpi=300, bbox_inches='tight')
                archivos_temporales.append(nombre_grafico_conf)
                plt.close()
                
                pdf.cell(0, 8, self.limpiar_texto_pdf('1. ANÁLISIS DE CONFIANZA'), 0, 1)
                pdf.image(nombre_grafico_conf, w=180)
                pdf.ln(5)
                
                # 2. Gráfico de Tiempo
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(df_datos['Arquitectura'], df_datos['Tiempo'] * 1000, color=colores)
                ax.set_title('Comparación de Tiempo de Respuesta por Arquitectura', fontsize=16, fontweight='bold')
                ax.set_ylabel('Tiempo (ms)')
                
                for bar, tiempo in zip(bars, df_datos['Tiempo']):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{tiempo*1000:.1f}ms', ha='center', va='bottom', fontweight='bold')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                nombre_grafico_tiempo = f"grafico_tiempo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(nombre_grafico_tiempo, dpi=300, bbox_inches='tight')
                archivos_temporales.append(nombre_grafico_tiempo)
                plt.close()
                
                pdf.add_page()
                pdf.cell(0, 8, self.limpiar_texto_pdf('2. ANÁLISIS DE VELOCIDAD'), 0, 1)
                pdf.image(nombre_grafico_tiempo, w=180)
                pdf.ln(5)
                
                # 3. Gráfico de Tamaño
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(df_datos['Arquitectura'], df_datos['Tamaño'], color=colores)
                ax.set_title('Comparación de Tamaño de Modelo por Arquitectura', fontsize=16, fontweight='bold')
                ax.set_ylabel('Tamaño (MB)')
                
                for bar, tamaño in zip(bars, df_datos['Tamaño']):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{tamaño:.1f}MB', ha='center', va='bottom', fontweight='bold')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                nombre_grafico_tamaño = f"grafico_tamaño_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(nombre_grafico_tamaño, dpi=300, bbox_inches='tight')
                archivos_temporales.append(nombre_grafico_tamaño)
                plt.close()
                
                pdf.add_page()
                pdf.cell(0, 8, self.limpiar_texto_pdf('3. ANÁLISIS DE EFICIENCIA DE MEMORIA'), 0, 1)
                pdf.image(nombre_grafico_tamaño, w=180)
                pdf.ln(5)
                
                # 4. Gráfico de Eficiencia
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(df_datos['Arquitectura'], df_datos['Eficiencia'], color=colores)
                ax.set_title('Comparación de Eficiencia por Arquitectura', fontsize=16, fontweight='bold')
                ax.set_ylabel('Eficiencia (Confianza/Tiempo)')
                
                for bar, eficiencia in zip(bars, df_datos['Eficiencia']):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{eficiencia:.1f}', ha='center', va='bottom', fontweight='bold')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                nombre_grafico_eficiencia = f"grafico_eficiencia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(nombre_grafico_eficiencia, dpi=300, bbox_inches='tight')
                archivos_temporales.append(nombre_grafico_eficiencia)
                plt.close()
                
                pdf.add_page()
                pdf.cell(0, 8, self.limpiar_texto_pdf('4. ANÁLISIS DE EFICIENCIA GENERAL'), 0, 1)
                pdf.image(nombre_grafico_eficiencia, w=180)
                pdf.ln(5)
                
                # 5. Gráfico Radar
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                
                # Normalizar métricas para el radar
                max_conf = max(pred['confianza'] for pred in predicciones)
                min_tiempo = min(pred['tiempo_prediccion'] for pred in predicciones)
                max_tiempo = max(pred['tiempo_prediccion'] for pred in predicciones)
                min_tamaño = min(pred['tamaño_modelo'] for pred in predicciones)
                max_tamaño = max(pred['tamaño_modelo'] for pred in predicciones)
                
                categorias = ['Confianza', 'Velocidad', 'Eficiencia Memoria', 'Score General']
                N = len(categorias)
                
                # Ángulos para cada categoría
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Cerrar el círculo
                
                for i, pred in enumerate(predicciones):
                    # Normalizar valores
                    norm_conf = pred['confianza'] / max_conf if max_conf > 0 else 0
                    norm_velocidad = (max_tiempo - pred['tiempo_prediccion']) / (max_tiempo - min_tiempo) if max_tiempo > min_tiempo else 1
                    norm_memoria = (max_tamaño - pred['tamaño_modelo']) / (max_tamaño - min_tamaño) if max_tamaño > min_tamaño else 1
                    norm_general = (norm_conf + norm_velocidad + norm_memoria) / 3
                    
                    valores = [norm_conf, norm_velocidad, norm_memoria, norm_general]
                    valores += valores[:1]  # Cerrar el círculo
                    
                    ax.plot(angles, valores, 'o-', linewidth=2, 
                        label=pred['arquitectura'].replace('_', ' '), 
                        color=self.informacion_arquitecturas[pred['arquitectura']]['color'])
                    ax.fill(angles, valores, alpha=0.25, 
                        color=self.informacion_arquitecturas[pred['arquitectura']]['color'])
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categorias)
                ax.set_ylim(0, 1)
                ax.set_title('Análisis Radar Comparativo', size=16, fontweight='bold', pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                ax.grid(True)
                
                plt.tight_layout()
                
                nombre_grafico_radar = f"grafico_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(nombre_grafico_radar, dpi=300, bbox_inches='tight')
                archivos_temporales.append(nombre_grafico_radar)
                plt.close()
                
                pdf.add_page()
                pdf.cell(0, 8, self.limpiar_texto_pdf('5. ANÁLISIS RADAR COMPARATIVO'), 0, 1)
                pdf.image(nombre_grafico_radar, w=180)
                pdf.ln(5)
                
                # 6. Gráfico de Barras Comparativo General
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Confianza
                ax1.bar(df_datos['Arquitectura'], df_datos['Confianza'], color=colores)
                ax1.set_title('Confianza', fontweight='bold')
                ax1.set_ylabel('Confianza (%)')
                ax1.tick_params(axis='x', rotation=45)
                
                # Tiempo
                ax2.bar(df_datos['Arquitectura'], df_datos['Tiempo'] * 1000, color=colores)
                ax2.set_title('Tiempo de Respuesta', fontweight='bold')
                ax2.set_ylabel('Tiempo (ms)')
                ax2.tick_params(axis='x', rotation=45)
                
                # Tamaño
                ax3.bar(df_datos['Arquitectura'], df_datos['Tamaño'], color=colores)
                ax3.set_title('Tamaño del Modelo', fontweight='bold')
                ax3.set_ylabel('Tamaño (MB)')
                ax3.tick_params(axis='x', rotation=45)
                
                # Eficiencia
                ax4.bar(df_datos['Arquitectura'], df_datos['Eficiencia'], color=colores)
                ax4.set_title('Eficiencia', fontweight='bold')
                ax4.set_ylabel('Eficiencia')
                ax4.tick_params(axis='x', rotation=45)
                
                plt.suptitle('Comparación Completa de Métricas', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                nombre_grafico_completo = f"grafico_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(nombre_grafico_completo, dpi=300, bbox_inches='tight')
                archivos_temporales.append(nombre_grafico_completo)
                plt.close()
                
                pdf.add_page()
                pdf.cell(0, 8, self.limpiar_texto_pdf('6. COMPARACIÓN COMPLETA DE MÉTRICAS'), 0, 1)
                pdf.image(nombre_grafico_completo, w=180)
                pdf.ln(5)
                
            except Exception as e:
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 6, self.limpiar_texto_pdf(f'Error generando gráficos: {str(e)}'), 0, 1)
            
            # --- ANÁLISIS CLÍNICO DETALLADO ---
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self.limpiar_texto_pdf('ANÁLISIS CLÍNICO DETALLADO'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            
            # Para cada predicción única
            diagnosticos_unicos = list(set(pred['clase_predicha'] for pred in predicciones))
            
            for diagnostico in diagnosticos_unicos:
                info_clase = self.informacion_clases.get(diagnostico, {})
                
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 8, self.limpiar_texto_pdf(f'{info_clase.get("nombre", diagnostico)}'), 0, 1)
                
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 5, self.limpiar_texto_pdf(f'Descripción: {info_clase.get("descripcion", "No disponible")}'), 0, 1)
                pdf.cell(0, 5, self.limpiar_texto_pdf(f'Gravedad: {info_clase.get("gravedad", "No especificada")}'), 0, 1)
                pdf.cell(0, 5, self.limpiar_texto_pdf(f'Tratamiento: {info_clase.get("tratamiento", "Consultar especialista")}'), 0, 1)
                pdf.cell(0, 5, self.limpiar_texto_pdf(f'Pronóstico: {info_clase.get("pronostico", "Variable")}'), 0, 1)
                pdf.ln(5)
            
            # --- RECOMENDACIONES TÉCNICAS ---
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, self.limpiar_texto_pdf('RECOMENDACIONES TÉCNICAS'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)
            
            # Ganadores por categoría
            mejores_modelos = self.encontrar_mejor_arquitectura(predicciones)
            
            categorias = [
                ('mayor_confianza', 'Mayor Confianza', 'Uso clínico de alta precisión'),
                ('mas_rapido', 'Más Rápido', 'Aplicaciones tiempo real/móviles'),
                ('mas_ligero', 'Más Ligero', 'Dispositivos recursos limitados'),
                ('mas_eficiente', 'Más Eficiente', 'Sistemas de producción escalables')
            ]
            
            for clave, titulo, contexto in categorias:
                if clave in mejores_modelos:
                    ganador = mejores_modelos[clave]
                    info_arq = self.informacion_arquitecturas[ganador['arquitectura']]
                    
                    pdf.set_font('Arial', 'B', 11)
                    pdf.cell(0, 7, self.limpiar_texto_pdf(f'{titulo}: {info_arq["nombre_completo"]}'), 0, 1)
                    pdf.set_font('Arial', '', 10)
                    pdf.cell(0, 5, self.limpiar_texto_pdf(f'Contexto: {contexto}'), 0, 1)
                    
                    if clave == 'mayor_confianza':
                        pdf.cell(0, 5, self.limpiar_texto_pdf(f'Confianza: {ganador["confianza"]:.1%}'), 0, 1)
                    elif clave == 'mas_rapido':
                        pdf.cell(0, 5, self.limpiar_texto_pdf(f'Tiempo: {ganador["tiempo_prediccion"]:.3f}s'), 0, 1)
                    elif clave == 'mas_ligero':
                        pdf.cell(0, 5, self.limpiar_texto_pdf(f'Tamaño: {ganador["tamaño_modelo"]:.1f}MB'), 0, 1)
                    else:
                        pdf.cell(0, 5, self.limpiar_texto_pdf(f'Eficiencia: {ganador.get("eficiencia", 0):.1f}'), 0, 1)
                    
                    pdf.ln(3)
            
            # --- DISCLAIMER MÉDICO ---
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, self.limpiar_texto_pdf('DISCLAIMER MÉDICO'), 0, 1)
            pdf.set_font('Arial', '', 9)
            pdf.multi_cell(0, 4, self.limpiar_texto_pdf(
                'Este reporte es generado por un sistema de inteligencia artificial y debe ser '
                'utilizado únicamente como herramienta de apoyo diagnóstico. No reemplaza el '
                'criterio clínico profesional. Se recomienda confirmación por oftalmólogo '
                'certificado antes de tomar decisiones terapéuticas.'))
            
            # Generar archivo PDF
            marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo_pdf = f"reporte_diagnostico_completo_{marca_tiempo}.pdf"
            pdf.output(nombre_archivo_pdf)
            
            # Limpiar archivos temporales
            for archivo in archivos_temporales:
                try:
                    if os.path.exists(archivo):
                        os.remove(archivo)
                except:
                    pass
            
            return nombre_archivo_pdf
            
        except Exception as e:
            st.error(f"Error generando PDF completo: {str(e)}")
            # Limpiar archivos temporales en caso de error
            for archivo in archivos_temporales:
                try:
                    if os.path.exists(archivo):
                        os.remove(archivo)
                except:
                    pass
            return None
        """Genera reporte PDF profesional completo"""
        try:
            # Crear PDF
            pdf = FPDF()
            pdf.add_page()
            
            # --- PORTADA ---
            pdf.set_font('Arial', 'B', 20)
            pdf.cell(0, 15, self.limpiar_texto_pdf('REPORTE DE DIAGNÓSTICO OCULAR AVANZADO'), 0, 1, 'C')
            pdf.ln(5)
            
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self.limpiar_texto_pdf('Sistema Multi-Arquitectura CNN'), 0, 1, 'C')
            pdf.ln(10)
            
            # Información general
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, self.limpiar_texto_pdf(f'Fecha del análisis: {marca_tiempo_analisis}'), 0, 1)
            pdf.cell(0, 8, self.limpiar_texto_pdf(f'Arquitecturas analizadas: {len(predicciones)}'), 0, 1)
            pdf.cell(0, 8, self.limpiar_texto_pdf('Enfermedades detectables: 10 patologías especializadas'), 0, 1)
            pdf.ln(10)
            
            # --- RESUMEN EJECUTIVO ---
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self.limpiar_texto_pdf('RESUMEN EJECUTIVO'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            
            # Encontrar ganador general
            mejor_general = max(predicciones, key=lambda x: x.get('score_general', 0))
            info_ganador = self.informacion_arquitecturas[mejor_general['arquitectura']]
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, self.limpiar_texto_pdf(f'ARQUITECTURA RECOMENDADA: {info_ganador["nombre_completo"]}'), 0, 1)
            
            pdf.set_font('Arial', '', 11)
            clase_predicha = mejor_general['clase_predicha']
            info_clase = self.informacion_clases.get(clase_predicha, {})
            
            pdf.cell(0, 6, self.limpiar_texto_pdf(f'Diagnóstico principal: {info_clase.get("nombre", clase_predicha)}'), 0, 1)
            pdf.cell(0, 6, self.limpiar_texto_pdf(f'Nivel de confianza: {mejor_general["confianza"]:.1%}'), 0, 1)
            pdf.cell(0, 6, self.limpiar_texto_pdf(f'Gravedad: {info_clase.get("gravedad", "No especificada")}'), 0, 1)
            pdf.ln(8)
            
            # Agregar imagen de manera segura
            try:
                if imagen is not None:
                    # Crear nombre único para imagen temporal
                    nombre_img_temp = f"temp_img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    
                    # Convertir y guardar imagen
                    if hasattr(imagen, 'save'):
                        # Es una imagen PIL
                        imagen_rgb = imagen.convert('RGB')
                        imagen_rgb.save(nombre_img_temp, 'JPEG', quality=85)
                    else:
                        # Crear imagen placeholder si hay problemas
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                        ax.text(0.5, 0.5, 'Imagen Analizada', ha='center', va='center', fontsize=14)
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
                        plt.savefig(nombre_img_temp, dpi=150, bbox_inches='tight')
                        plt.close()
                    
                    # Agregar imagen al PDF
                    pdf.cell(0, 10, self.limpiar_texto_pdf('IMAGEN ANALIZADA:'), 0, 1)
                    pdf.image(nombre_img_temp, w=80)
                    pdf.ln(5)
                    
            except Exception as error_img:
                # Si hay error con la imagen, continuar sin ella
                pdf.set_font('Arial', 'I', 10)
                pdf.cell(0, 6, self.limpiar_texto_pdf(f'[Imagen no disponible: {str(error_img)[:50]}...]'), 0, 1)
                pdf.ln(5)
            
            # --- NUEVA PÁGINA: COMPARACIÓN DE ARQUITECTURAS ---
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, self.limpiar_texto_pdf('COMPARACIÓN DE ARQUITECTURAS CNN'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)
            
            # Tabla comparativa
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(50, 8, self.limpiar_texto_pdf('Arquitectura'), 1, 0, 'C')
            pdf.cell(35, 8, self.limpiar_texto_pdf('Confianza'), 1, 0, 'C')
            pdf.cell(30, 8, self.limpiar_texto_pdf('Tiempo (ms)'), 1, 0, 'C')
            pdf.cell(25, 8, self.limpiar_texto_pdf('Tamaño (MB)'), 1, 0, 'C')
            pdf.cell(30, 8, self.limpiar_texto_pdf('Eficiencia'), 1, 1, 'C')
            
            pdf.set_font('Arial', '', 9)
            for pred in predicciones:
                nombre_arq = pred['arquitectura'].replace('_', ' ')
                pdf.cell(50, 6, self.limpiar_texto_pdf(nombre_arq), 1, 0)
                pdf.cell(35, 6, self.limpiar_texto_pdf(f"{pred['confianza']:.1%}"), 1, 0, 'C')
                pdf.cell(30, 6, self.limpiar_texto_pdf(f"{pred['tiempo_prediccion']*1000:.1f}"), 1, 0, 'C')
                pdf.cell(25, 6, self.limpiar_texto_pdf(f"{pred['tamaño_modelo']:.1f}"), 1, 0, 'C')
                pdf.cell(30, 6, self.limpiar_texto_pdf(f"{pred.get('eficiencia', 0):.1f}"), 1, 1, 'C')
            
            pdf.ln(10)
            
            # --- ANÁLISIS CLÍNICO DETALLADO ---
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, self.limpiar_texto_pdf('ANÁLISIS CLÍNICO DETALLADO'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            
            # Para cada predicción única
            diagnosticos_unicos = list(set(pred['clase_predicha'] for pred in predicciones))
            
            for diagnostico in diagnosticos_unicos:
                info_clase = self.informacion_clases.get(diagnostico, {})
                
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 8, self.limpiar_texto_pdf(f'{info_clase.get("nombre", diagnostico)}'), 0, 1)
                
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 5, self.limpiar_texto_pdf(f'Descripción: {info_clase.get("descripcion", "No disponible")}'), 0, 1)
                pdf.cell(0, 5, self.limpiar_texto_pdf(f'Gravedad: {info_clase.get("gravedad", "No especificada")}'), 0, 1)
                pdf.cell(0, 5, self.limpiar_texto_pdf(f'Tratamiento: {info_clase.get("tratamiento", "Consultar especialista")}'), 0, 1)
                pdf.cell(0, 5, self.limpiar_texto_pdf(f'Pronóstico: {info_clase.get("pronostico", "Variable")}'), 0, 1)
                pdf.ln(5)
            
            # --- RECOMENDACIONES TÉCNICAS ---
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 12, self.limpiar_texto_pdf('RECOMENDACIONES TÉCNICAS'), 0, 1)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(8)
            
            # Ganadores por categoría
            mejores_modelos = self.encontrar_mejor_arquitectura(predicciones)
            
            categorias = [
                ('mayor_confianza', 'Mayor Confianza', 'Uso clínico de alta precisión'),
                ('mas_rapido', 'Más Rápido', 'Aplicaciones tiempo real/móviles'),
                ('mas_ligero', 'Más Ligero', 'Dispositivos recursos limitados'),
                ('mas_eficiente', 'Más Eficiente', 'Sistemas de producción escalables')
            ]
            
            for clave, titulo, contexto in categorias:
                if clave in mejores_modelos:
                    ganador = mejores_modelos[clave]
                    info_arq = self.informacion_arquitecturas[ganador['arquitectura']]
                    
                    pdf.set_font('Arial', 'B', 11)
                    pdf.cell(0, 7, self.limpiar_texto_pdf(f'{titulo}: {info_arq["nombre_completo"]}'), 0, 1)
                    pdf.set_font('Arial', '', 10)
                    pdf.cell(0, 5, self.limpiar_texto_pdf(f'Contexto: {contexto}'), 0, 1)
                    
                    if clave == 'mayor_confianza':
                        pdf.cell(0, 5, self.limpiar_texto_pdf(f'Confianza: {ganador["confianza"]:.1%}'), 0, 1)
                    elif clave == 'mas_rapido':
                        pdf.cell(0, 5, self.limpiar_texto_pdf(f'Tiempo: {ganador["tiempo_prediccion"]:.3f}s'), 0, 1)
                    elif clave == 'mas_ligero':
                        pdf.cell(0, 5, self.limpiar_texto_pdf(f'Tamaño: {ganador["tamaño_modelo"]:.1f}MB'), 0, 1)
                    else:
                        pdf.cell(0, 5, self.limpiar_texto_pdf(f'Eficiencia: {ganador.get("eficiencia", 0):.1f}'), 0, 1)
                    
                    pdf.ln(3)
            
            # --- DISCLAIMER MÉDICO ---
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, self.limpiar_texto_pdf('DISCLAIMER MÉDICO'), 0, 1)
            pdf.set_font('Arial', '', 9)
            pdf.multi_cell(0, 4, self.limpiar_texto_pdf(
                'Este reporte es generado por un sistema de inteligencia artificial y debe ser '
                'utilizado únicamente como herramienta de apoyo diagnóstico. No reemplaza el '
                'criterio clínico profesional. Se recomienda confirmación por oftalmólogo '
                'certificado antes de tomar decisiones terapéuticas.'))
            
            # Generar archivo PDF
            marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo_pdf = f"reporte_diagnostico_ocular_{marca_tiempo}.pdf"
            pdf.output(nombre_archivo_pdf)
            
            # Limpiar archivo temporal de imagen si existe
            try:
                if 'nombre_img_temp' in locals() and os.path.exists(nombre_img_temp):
                    os.remove(nombre_img_temp)
            except:
                pass
            
            return nombre_archivo_pdf
            
        except Exception as e:
            st.error(f"Error generando PDF: {str(e)}")
            return None
    
    def exportar_datos_tecnicos(self, predicciones, marca_tiempo_analisis):
        """Exporta datos técnicos completos en JSON"""
        try:
            # Crear estructura de datos completa
            datos_tecnicos = {
                'metadatos': {
                    'marca_tiempo': marca_tiempo_analisis,
                    'version_sistema': '2.1 Multi-Architecture + Statistical Analysis',
                    'total_arquitecturas': len(predicciones),
                    'enfermedades_detectables': len(self.informacion_clases),
                    'tipo_analisis': 'Comparative Multi-CNN with Statistical Inference'
                },
                'comparacion_arquitecturas': [],
                'metricas_rendimiento': {},
                'analisis_clinico': {},
                'recomendaciones': {}
            }
            
            # Datos por arquitectura
            for pred in predicciones:
                datos_arq = {
                    'nombre_arquitectura': pred['arquitectura'],
                    'info_modelo': self.informacion_arquitecturas[pred['arquitectura']],
                    'resultados_prediccion': {
                        'clase_predicha': pred['clase_predicha'],
                        'confianza': float(pred['confianza']),
                        'todas_probabilidades': [float(p) for p in pred['todas_probabilidades']],
                        'tiempo_prediccion_segundos': float(pred['tiempo_prediccion']),
                        'tamaño_modelo_mb': float(pred['tamaño_modelo']),
                        'conteo_parametros': int(pred['conteo_parametros']),
                        'score_eficiencia': float(pred.get('eficiencia', 0)),
                        'score_general': float(pred.get('score_general', 0))
                    },
                    'info_clinica': self.informacion_clases.get(pred['clase_predicha'], {})
                }
                datos_tecnicos['comparacion_arquitecturas'].append(datos_arq)
            
            # Métricas de rendimiento
            confianzas = [pred['confianza'] for pred in predicciones]
            tiempos = [pred['tiempo_prediccion'] for pred in predicciones]
            tamaños = [pred['tamaño_modelo'] for pred in predicciones]
            
            datos_tecnicos['metricas_rendimiento'] = {
                'estadisticas_confianza': {
                    'media': float(np.mean(confianzas)),
                    'desviacion_estandar': float(np.std(confianzas)),
                    'minimo': float(np.min(confianzas)),
                    'maximo': float(np.max(confianzas))
                },
                'estadisticas_tiempo': {
                    'media_ms': float(np.mean(tiempos) * 1000),
                    'desviacion_estandar_ms': float(np.std(tiempos) * 1000),
                    'mas_rapido_ms': float(np.min(tiempos) * 1000),
                    'mas_lento_ms': float(np.max(tiempos) * 1000)
                },
                'estadisticas_tamaño': {
                    'media_mb': float(np.mean(tamaños)),
                    'desviacion_estandar_mb': float(np.std(tamaños)),
                    'mas_ligero_mb': float(np.min(tamaños)),
                    'mas_pesado_mb': float(np.max(tamaños))
                }
            }
            
            # Análisis clínico
            diagnosticos = [pred['clase_predicha'] for pred in predicciones]
            diagnosticos_unicos = list(set(diagnosticos))
            
            datos_tecnicos['analisis_clinico'] = {
                'diagnosticos_unicos': len(diagnosticos_unicos),
                'diagnostico_consenso': max(set(diagnosticos), key=diagnosticos.count) if diagnosticos else None,
                'acuerdo_diagnostico': (diagnosticos.count(max(set(diagnosticos), key=diagnosticos.count)) / len(diagnosticos)) if diagnosticos else 0,
                'distribucion_gravedad': {
                    diagnostico: self.informacion_clases.get(diagnostico, {}).get('gravedad', 'Desconocida')
                    for diagnostico in diagnosticos_unicos
                }
            }
            
            # Recomendaciones
            mejores_modelos = self.encontrar_mejor_arquitectura(predicciones)
            datos_tecnicos['recomendaciones'] = {
                categoria: {
                    'arquitectura': datos_modelo['arquitectura'],
                    'razon': f'Mejor {categoria.replace("_", " ")}',
                    'valor_metrica': datos_modelo.get('confianza' if 'confianza' in categoria else 
                                                  'tiempo_prediccion' if 'rapido' in categoria else
                                                  'tamaño_modelo' if 'ligero' in categoria else 'eficiencia', 0)
                }
                for categoria, datos_modelo in mejores_modelos.items()
            }
            
            # Guardar archivo JSON
            marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo_json = f"analisis_tecnico_{marca_tiempo}.json"
            
            with open(nombre_archivo_json, 'w', encoding='utf-8') as f:
                json.dump(datos_tecnicos, f, indent=2, ensure_ascii=False)
            
            return nombre_archivo_json
            
        except Exception as e:
            st.error(f"Error exportando datos técnicos: {str(e)}")
            return None
    
    def mostrar_seccion_reportes_avanzados(self, predicciones, imagen, marca_tiempo_analisis):
        """Sección avanzada de reportes y exportación - TRADUCIDA"""
        st.markdown("---")
        st.header(get_text('advanced_reports_title', self.lang))
        
        # Métricas de cobertura del sistema
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=get_text('detectable_diseases', self.lang),
                value="10",
                delta="6 más que sistemas básicos",
                help="Nuestro sistema detecta 10 vs 4 de sistemas convencionales"
            )
        
        with col2:
            st.metric(
                label=get_text('cnn_architectures', self.lang),
                value=len(predicciones),
                delta="Análisis multi-arquitectura",
                help="Comparación simultánea de múltiples modelos"
            )
        
        with col3:
            diagnosticos_unicos = len(set(pred['clase_predicha'] for pred in predicciones))
            st.metric(
                label=get_text('unique_diagnoses', self.lang),
                value=diagnosticos_unicos,
                delta="En este análisis",
                help="Número de diagnósticos diferentes detectados"
            )
        
        with col4:
            confianza_promedio = np.mean([pred['confianza'] for pred in predicciones])
            st.metric(
                label=get_text('average_confidence', self.lang),
                value=f"{confianza_promedio:.1%}",
                delta=f"±{np.std([pred['confianza'] for pred in predicciones]):.1%}",
                help="Confianza promedio entre todas las arquitecturas"
            )
        
        # Sección de exportación
        st.markdown(f"### {get_text('export_analysis', self.lang)}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(get_text('generate_pdf', self.lang), type="primary", use_container_width=True, key="boton_pdf"):
                try:
                    estado_pdf = st.empty()
                    estado_pdf.info(get_text('generating_pdf', self.lang))
                    archivo_pdf = self.generar_reporte_pdf_completo(predicciones, imagen, marca_tiempo_analisis)
                    
                    if archivo_pdf and os.path.exists(archivo_pdf):
                        estado_pdf.success(get_text('pdf_generated', self.lang))
                        
                        with open(archivo_pdf, "rb") as f:
                            bytes_pdf = f.read()
                        
                        st.download_button(
                            label=get_text('download_pdf', self.lang),
                            data=bytes_pdf,
                            file_name=f"reporte_diagnostico_ocular_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="descargar_pdf"
                        )
                        
                        st.balloons()
                        
                        try:
                            os.remove(archivo_pdf)
                        except:
                            pass
                    else:
                        st.error(get_text('pdf_error', self.lang))
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button(get_text('export_json', self.lang), use_container_width=True, key="boton_json"):
                try:
                    estado_json = st.empty()
                    estado_json.info(get_text('exporting_data', self.lang))
                    
                    archivo_json = self.exportar_datos_tecnicos(predicciones, marca_tiempo_analisis)
                    
                    if archivo_json and os.path.exists(archivo_json):
                        estado_json.success(get_text('data_exported', self.lang))
                        
                        with open(archivo_json, "r", encoding='utf-8') as f:
                            datos_json = f.read()
                        
                        st.download_button(
                            label=get_text('download_json', self.lang),
                            data=datos_json,
                            file_name=f"analisis_tecnico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True,
                            key="descargar_json"
                        )
                        
                        try:
                            os.remove(archivo_json)
                        except:
                            pass
                    else:
                        st.error(get_text('data_error', self.lang))
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col3:
            if st.button(get_text('export_csv', self.lang), use_container_width=True, key="boton_csv"):
                try:
                    estado_csv = st.empty()
                    estado_csv.info(get_text('preparing_csv', self.lang))
                    
                    df_exportar = pd.DataFrame([
                        {
                            'Marca_Tiempo': marca_tiempo_analisis,
                            get_text('architecture', self.lang): pred['arquitectura'].replace('_', ' '),
                            get_text('diagnosis_en', self.lang): pred['clase_predicha'],
                            get_text('diagnosis_es', self.lang): self.informacion_clases.get(pred['clase_predicha'], {}).get('nombre', pred['clase_predicha']),
                            get_text('confidence_table', self.lang): pred['confianza'],
                            f"{get_text('time_table', self.lang)}_ms": pred['tiempo_prediccion'] * 1000,
                            f"{get_text('size_table', self.lang)}_MB": pred['tamaño_modelo'],
                            get_text('parameters_table', self.lang): pred['conteo_parametros'],
                            get_text('efficiency_table', self.lang): pred.get('eficiencia', 0),
                            get_text('general_score_table', self.lang): pred.get('score_general', 0),
                            get_text('severity', self.lang): self.informacion_clases.get(pred['clase_predicha'], {}).get('gravedad', 'No especificada')
                        }
                        for pred in predicciones
                    ])
                    
                    datos_csv = df_exportar.to_csv(index=False, encoding='utf-8')
                    estado_csv.success(get_text('csv_ready', self.lang))
                    
                    st.download_button(
                        label=get_text('download_csv', self.lang),
                        data=datos_csv,
                        file_name=f"analisis_comparativo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="descargar_csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Información adicional sobre las descargas
        st.markdown("---")
        st.info(get_text('download_info', self.lang))
    
    def encontrar_mejor_arquitectura(self, predicciones):
        """Encuentra la mejor arquitectura por diferentes métricas"""
        if not predicciones or len(predicciones) < 2:
            return {}
        
        # Mejor por confianza
        mejor_confianza = max(predicciones, key=lambda x: x['confianza'])
        
        # Más rápido
        mas_rapido = min(predicciones, key=lambda x: x['tiempo_prediccion'])
        
        # Más eficiente (mayor confianza / tiempo)
        for pred in predicciones:
            pred['eficiencia'] = pred['confianza'] / pred['tiempo_prediccion']
        mas_eficiente = max(predicciones, key=lambda x: x['eficiencia'])
        
        # Más ligero
        mas_ligero = min(predicciones, key=lambda x: x['tamaño_modelo'])
        
        return {
            'mayor_confianza': mejor_confianza,
            'mas_rapido': mas_rapido,
            'mas_eficiente': mas_eficiente,
            'mas_ligero': mas_ligero
        }

    def mostrar_encabezado(self):
        """Header de la aplicación - TRADUCIDO"""
        st.title(get_text('main_title', self.lang))
        st.subheader(get_text('page_subtitle', self.lang))
        st.markdown("---")

    # ========== FUNCIÓN EJECUTAR PRINCIPAL (MODIFICADA PARA MULTILENGUAJE) ==========
    def ejecutar(self):
        """Ejecuta la aplicación principal CON MULTILENGUAJE COMPLETO"""
        # Reinicializar datos con el nuevo idioma
        self.__init__()
        
        # Inicializar session state
        if 'analisis_completado' not in st.session_state:
            st.session_state.analisis_completado = False
        if 'predicciones' not in st.session_state:
            st.session_state.predicciones = None
        if 'imagen_analisis' not in st.session_state:
            st.session_state.imagen_analisis = None
        if 'marca_tiempo_analisis' not in st.session_state:
            st.session_state.marca_tiempo_analisis = None
        if 'resultados_estadisticos' not in st.session_state:
            st.session_state.resultados_estadisticos = None
        
        # Header
        self.mostrar_encabezado()
        
        # Sidebar
        st.sidebar.markdown(f"## {get_text('sidebar_title', self.lang)}")
        st.sidebar.markdown("---")
        
        # Cargar modelos
        if self.modelos is None:
            with st.spinner(get_text('loading_models', self.lang)):
                self.modelos, self.nombres_clases, self.nombres_clases_individuales = self.cargar_modelos()
        
        if len(self.modelos) < 2:
            st.error(get_text('models_error', self.lang))
            st.stop()
        
        # Info en sidebar
        st.sidebar.success(get_text('models_loaded', self.lang, count=len(self.modelos)))
        
        # Pestañas principales
        tab1, tab2 = st.tabs([
            get_text('tab_individual', self.lang), 
            get_text('tab_statistical', self.lang)
        ])
        
        with tab1:
            # Botón para limpiar análisis
            if st.sidebar.button(get_text('new_analysis', self.lang), help=get_text('new_analysis_help', self.lang)):
                st.session_state.analisis_completado = False
                st.session_state.predicciones = None
                st.session_state.imagen_analisis = None
                st.session_state.marca_tiempo_analisis = None
                st.rerun()
            
            # Mostrar características
            self.mostrar_vitrina_arquitecturas()
            
            st.markdown("---")
            
            # Si ya hay un análisis completo, mostrar resultados
            if st.session_state.analisis_completado and st.session_state.predicciones:
                st.success(get_text('analysis_completed', self.lang))
                
                # Mostrar imagen analizada
                if st.session_state.imagen_analisis:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(st.session_state.imagen_analisis, caption=get_text('analyzed_image', self.lang), use_container_width=True)
                
                # Mostrar todos los resultados usando el estado guardado
                predicciones = st.session_state.predicciones
                
                self.mostrar_resultados_prediccion(predicciones)
                st.markdown("---")
                self.mostrar_comparacion_rendimiento(predicciones)
                st.markdown("---")
                self.mostrar_comparacion_radar(predicciones)
                
                mejores_modelos = self.encontrar_mejor_arquitectura(predicciones)
                st.markdown("---")
                self.mostrar_podio_ganadores(mejores_modelos)
                st.markdown("---")
                self.mostrar_analisis_detallado(predicciones, mejores_modelos)
                
                # SECCIÓN DE REPORTES AVANZADOS
                self.mostrar_seccion_reportes_avanzados(predicciones, st.session_state.imagen_analisis, st.session_state.marca_tiempo_analisis)
                
                # Tabla resumen
                with st.expander(get_text('summary_table', self.lang)):
                    df_resumen = pd.DataFrame([
                        {
                            get_text('architecture', self.lang): pred['arquitectura'].replace('_', ' '),
                            get_text('diagnosis_en', self.lang): pred['clase_predicha'],
                            get_text('diagnosis_es', self.lang): self.informacion_clases.get(pred['clase_predicha'], {}).get('nombre', pred['clase_predicha']),
                            get_text('confidence_table', self.lang): f"{pred['confianza']:.1%}",
                            get_text('time_table', self.lang): f"{pred['tiempo_prediccion']:.3f}s",
                            get_text('size_table', self.lang): f"{pred['tamaño_modelo']:.1f}MB",
                            get_text('parameters_table', self.lang): f"{pred['conteo_parametros']:,}",
                            get_text('efficiency_table', self.lang): f"{pred.get('eficiencia', 0):.1f}",
                            get_text('general_score_table', self.lang): f"{pred.get('score_general', 0):.3f}",
                            get_text('severity', self.lang): self.informacion_clases.get(pred['clase_predicha'], {}).get('gravedad', get_text('unspecified', self.lang))
                        }
                        for pred in predicciones
                    ])
                    
                    st.dataframe(df_resumen, use_container_width=True)
                
                # Timestamp
                st.markdown("---")
                st.markdown(get_text('analysis_timestamp', self.lang, timestamp=st.session_state.marca_tiempo_analisis))
                
            else:
                # Interfaz para nuevo análisis
                st.markdown(f"## {get_text('upload_title', self.lang)}")
                archivo_subido = st.file_uploader(
                    get_text('upload_help', self.lang),
                    type=['png', 'jpg', 'jpeg'],
                    help=get_text('upload_description', self.lang)
                )
                
                if archivo_subido is not None:
                    # Mostrar imagen
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        imagen = Image.open(archivo_subido)
                        st.image(imagen, caption=get_text('image_caption', self.lang), use_container_width=True)
                    
                    # Botón de análisis
                    if st.button(get_text('battle_button', self.lang), type="primary", use_container_width=True):
                        
                        marca_tiempo_analisis = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Preprocesamiento y predicciones
                        with st.spinner(get_text('processing_image', self.lang)):
                            array_img = self.preprocesar_imagen(imagen)
                        
                        if array_img is not None:
                            predicciones = []
                            
                            with st.spinner(get_text('analyzing_architectures', self.lang)):
                                barra_progreso = st.progress(0)
                                
                                for i, (nombre_arq, modelo) in enumerate(self.modelos.items()):
                                    pred = self.predecir_con_cronometraje(modelo, array_img, nombre_arq)
                                    if pred:
                                        predicciones.append(pred)
                                    barra_progreso.progress((i + 1) / len(self.modelos))
                            
                            if len(predicciones) >= 2:
                                st.success(get_text('battle_completed', self.lang))
                                
                                # Calcular scores adicionales
                                for pred in predicciones:
                                    max_conf = max(p['confianza'] for p in predicciones)
                                    min_tiempo = min(p['tiempo_prediccion'] for p in predicciones)
                                    min_tamaño = min(p['tamaño_modelo'] for p in predicciones)
                                    
                                    score_conf = pred['confianza'] / max_conf
                                    score_velocidad = min_tiempo / pred['tiempo_prediccion']
                                    score_memoria = min_tamaño / pred['tamaño_modelo']
                                    
                                    pred['score_general'] = 0.5 * score_conf + 0.25 * score_velocidad + 0.25 * score_memoria
                                    pred['eficiencia'] = pred['confianza'] / pred['tiempo_prediccion']
                                
                                # GUARDAR EN SESSION STATE
                                st.session_state.predicciones = predicciones
                                st.session_state.imagen_analisis = imagen
                                st.session_state.marca_tiempo_analisis = marca_tiempo_analisis
                                st.session_state.analisis_completado = True
                                
                                # Forzar rerun para mostrar resultados
                                st.rerun()
                            
                            else:
                                st.error(get_text('prediction_error', self.lang))
        
        with tab2:
            # NUEVA PESTAÑA: ANÁLISIS ESTADÍSTICO
            self.mostrar_seccion_analisis_estadistico()
        
        # Footer técnico (expandido) - TRADUCIDO
        st.markdown("---")
        st.markdown(f"""
        ### {get_text('system_title', self.lang)}
        
        **{get_text('system_subtitle', self.lang)}**
        
        **{get_text('statistical_features_title', self.lang)}**
        - {get_text('mcc_description', self.lang)}
        - {get_text('mcnemar_description', self.lang)}
        - {get_text('bootstrap_description', self.lang)}
        - {get_text('confusion_matrices', self.lang)}
        - {get_text('statistical_reports', self.lang)}
        
        **{get_text('competitive_advantages_title', self.lang)}**
        - {get_text('specialized_diseases', self.lang)}
        - {get_text('multi_architecture', self.lang)}
        - {get_text('statistical_evaluation', self.lang)}
        - {get_text('professional_reports', self.lang)}
        - {get_text('complete_export', self.lang)}
        - {get_text('contextual_recommendations', self.lang)}
        
        **{get_text('implemented_architectures_title', self.lang)}**
        - {get_text('hybrid_cnn', self.lang)}
        - {get_text('efficientnet_desc', self.lang)}
        - {get_text('resnet_desc', self.lang)}
        
        **{get_text('evaluated_metrics_title', self.lang)}**
        - {get_text('precision_metric', self.lang)}
        - {get_text('speed_metric', self.lang)}
        - {get_text('efficiency_metric', self.lang)}
        - {get_text('balance_metric', self.lang)}
        - {get_text('significance_metric', self.lang)}
        
        **{get_text('applications_title', self.lang)}**
        - {get_text('clinical_application', self.lang)}
        - {get_text('mobile_application', self.lang)}
        - {get_text('production_application', self.lang)}
        - {get_text('research_application', self.lang)}
        
        {get_text('innovation_text', self.lang)}
        
        **{get_text('statistical_methods_title', self.lang)}**
        - {get_text('mcc_method', self.lang)}
        - {get_text('mcnemar_method', self.lang)}
        - {get_text('bootstrap_method', self.lang)}
        - {get_text('yates_correction', self.lang)}
        """)

if __name__ == "__main__":
    configurar_pagina("es")              # Evita usar `st` antes de esto
    lang = mostrar_selector_idioma()     # Selector de idioma una sola vez

    aplicacion = AplicacionTresArquitecturas()
    aplicacion.lang = lang               # Asigna el idioma seleccionado
    aplicacion.ejecutar()