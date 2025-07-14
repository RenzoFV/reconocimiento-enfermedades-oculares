# translations.py
translations = {
    'es': {
        # Configuración de página
        'page_title': "🏥 Comparación de 3 Arquitecturas CNN + Estadísticas",
        'page_subtitle': "MobileNetV2 vs EfficientNet-B0 vs ResNet-50 V2 + Análisis Estadístico",
        
        # Menú y navegación
        'sidebar_title': "🎛️ Panel de Control",
        'select_language': "🌍 Seleccionar Idioma",
        'new_analysis': "🔄 Nuevo Análisis",
        'new_analysis_help': "Limpia el análisis actual",
        
        # Pestañas principales
        'tab_individual': "🔬 Análisis Individual",
        'tab_statistical': "📊 Evaluación Estadística",
        
        # Encabezados principales
        'main_title': "🏆 DETECCIÓN DE ENFERMEDADES OCULARES 👁️",
        'architectures_title': "🏗️ LAS 3 ARQUITECTURAS EN COMPETENCIA",
        'results_title': "🎯 RESULTADOS DE PREDICCIÓN",
        'comparison_title': "📊 ANÁLISIS COMPARATIVO DE RENDIMIENTO",
        'radar_title': "🕸️ Comparación Multidimensional",
        'winners_title': "🏆 PODIO DE GANADORES",
        'detailed_analysis_title': "🔬 ANÁLISIS DETALLADO",
        'statistical_analysis_title': "📊 ANÁLISIS ESTADÍSTICO INFERENCIAL",
        'advanced_reports_title': "📋 SISTEMA AVANZADO DE REPORTES",
        
        # Carga de modelos
        'loading_models': "🔄 Cargando las 3 arquitecturas...",
        'models_loaded': "✅ {count} arquitecturas cargadas",
        'model_loaded': "✅ {name} cargado correctamente",
        'model_not_found': "⚠️ No se encontró {filename}",
        'models_error': "❌ Se necesitan al menos 2 modelos para comparar",
        'loading_error': "Error cargando modelos: {error}",
        
        # Subida de imágenes
        'upload_title': "📸 Subir Imagen para Comparar Arquitecturas",
        'upload_help': "Selecciona una imagen de retina para la batalla de arquitecturas",
        'upload_description': "La imagen será analizada por las 3 arquitecturas simultáneamente",
        'battle_button': "🚀 INICIAR BATALLA DE ARQUITECTURAS",
        'image_caption': "Imagen para la batalla",
        
        # Procesamiento
        'processing_image': "🔄 Procesando imagen para todas las arquitecturas...",
        'analyzing_architectures': "🏗️ Analizando con las 3 arquitecturas...",
        'battle_completed': "✅ ¡Batalla completada! Analizando resultados...",
        'prediction_error': "❌ Error en las predicciones",
        'analysis_completed': "🎉 **Análisis ya completado!** Puedes descargar los reportes o hacer un nuevo análisis.",
        'analyzed_image': "Imagen analizada",
        'analysis_timestamp': "📅 Análisis realizado: {timestamp}",
        
        # Métricas y resultados
        'diagnosis': "**Diagnóstico:** {diagnosis}",
        'confidence': "🎯 Confianza",
        'technical_metrics': "**📊 Métricas Técnicas:**",
        'time': "⏱️ **Tiempo:** {time:.3f}s",
        'size': "💾 **Tamaño:** {size:.1f}MB",
        'parameters': "🔢 **Parámetros:** {params:,}",
        
        # Gráficos
        'confidence_chart': "🎯 Confianza de Predicción",
        'time_chart': "⏱️ Tiempo de Predicción",
        'size_chart': "💾 Tamaño del Modelo",
        'efficiency_chart': "⚡ Eficiencia (Confianza/Tiempo)",
        'radar_chart': "🕸️ Perfil Multidimensional de Arquitecturas",
        
        # Podio de ganadores
        'highest_confidence': "🎯 Mayor Confianza",
        'fastest': "⚡ Más Rápido",
        'lightest': "🪶 Más Ligero",
        'most_efficient': "⚖️ Más Eficiente",
        'most_accurate': "El más preciso",
        'speedster': "El velocista",
        'efficient': "El eficiente",
        'balanced': "El balanceado",
        
        # Análisis detallado
        'general_winner': "👑 GANADOR GENERAL: {name}",
        'general_score': "🏆 Score General",
        'best_balance': "¡El mejor balance de todas las métricas!",
        'strengths_weaknesses': "📋 Fortalezas y Debilidades",
        'strengths': "**🟢 Fortalezas:**",
        'weaknesses': "**🔴 Áreas de mejora:**",
        'technical_details': "**📊 Métricas Técnicas:**",
        
        # Recomendaciones
        'usage_recommendations': "💡 RECOMENDACIONES DE USO",
        'clinical_apps': "**🏥 Aplicaciones Clínicas:**",
        'clinical_desc': "- Usa el modelo con **mayor confianza**\n- Prioriza precisión sobre velocidad\n- Ideal para diagnósticos complejos",
        'mobile_apps': "**📱 Aplicaciones Móviles:**",
        'mobile_desc': "- Usa el modelo **más rápido y ligero**\n- Balance entre precisión y recursos\n- Ideal para apps en tiempo real",
        'production_apps': "**🔄 Sistemas de Producción:**",
        'production_desc': "- Usa el modelo **más eficiente**\n- Considera el volumen de procesamiento\n- Ideal para escalabilidad",
        
        # Análisis estadístico
        'statistical_description': """**Evaluación rigurosa con pruebas estadísticas:**
        - 🎯 **Coeficiente de Matthews (MCC)**: Métrica balanceada que considera todos los casos de la matriz de confusión
        - 🔬 **Prueba de McNemar**: Comparación estadística entre pares de modelos
        - 📈 **Intervalos de Confianza**: Bootstrap CI para robustez estadística""",
        
        'dataset_evaluation': "📂 Dataset de Evaluación",
        'dataset_path': "🗂️ Ruta de la carpeta de pruebas:",
        'dataset_path_help': "Ejemplo: Pruebas, ./Pruebas, /path/to/Pruebas",
        'expected_structure': "📋 Estructura de carpetas esperada",
        'folder_found': "✅ Carpeta encontrada: {path}",
        'folder_not_found': "❌ No se encontró la carpeta: {path}",
        'dataset_preview': "👀 Vista Previa del Dataset",
        'start_evaluation': "🚀 INICIAR EVALUACIÓN ESTADÍSTICA",
        
        # Reportes
        'detectable_diseases': "🏥 Enfermedades Detectables",
        'cnn_architectures': "🧠 Arquitecturas CNN",
        'unique_diagnoses': "🎯 Diagnósticos Únicos",
        'average_confidence': "📊 Confianza Promedio",
        'export_analysis': "📤 Exportar Análisis",
        'generate_pdf': "📄 Generar Reporte PDF Completo",
        'export_json': "📊 Exportar Datos Técnicos (JSON)",
        'export_csv': "📈 Exportar CSV Comparativo",
        'download_pdf': "⬇️ DESCARGAR REPORTE PDF",
        'download_json': "⬇️ DESCARGAR DATOS JSON",
        'download_csv': "⬇️ DESCARGAR CSV",
        
        # Mensajes de estado
        'generating_pdf': "🔄 Generando reporte PDF profesional...",
        'pdf_generated': "✅ PDF generado exitosamente!",
        'pdf_error': "❌ Error generando el reporte PDF",
        'exporting_data': "🔄 Exportando datos técnicos...",
        'data_exported': "✅ Datos técnicos exportados!",
        'data_error': "❌ Error exportando datos técnicos",
        'preparing_csv': "🔄 Preparando CSV...",
        'csv_ready': "✅ CSV listo!",
        
        # Información sobre descargas
        'download_info': """💡 **Información sobre las descargas:**
        - **PDF**: Reporte completo con análisis clínico y recomendaciones técnicas
        - **JSON**: Datos técnicos estructurados para análisis posterior 
        - **CSV**: Tabla comparativa simple para Excel/análisis estadístico
        
        📁 Los archivos se descargan automáticamente a tu carpeta de Descargas""",
        
        # Tabla resumen
        'summary_table': "📊 Tabla Resumen de Métricas",
        'architecture': "Arquitectura",
        'diagnosis_en': "Diagnóstico",
        'diagnosis_es': "Diagnóstico_ES",
        'confidence_table': "Confianza",
        'time_table': "Tiempo",
        'size_table': "Tamaño",
        'parameters_table': "Parámetros",
        'efficiency_table': "Eficiencia",
        'general_score_table': "Score General",
        'severity': "Gravedad",
        
        # Sistema técnico
        'system_title': "⚙️ Sobre Este Sistema Avanzado con Análisis Estadístico",
        'system_subtitle': "🚀 Sistema de Diagnóstico Ocular de Nueva Generación",
        
        # Información de clases (enfermedades)
        'normal': 'Normal',
        'moderate': 'Moderada',
        'high': 'Alta',
        'critical': 'Crítica',
        'mild': 'Leve',
        'unspecified': 'No especificada',
        
        # Botones y acciones
        'cancel': 'Cancelar',
        'accept': 'Aceptar',
        'close': 'Cerrar',
        'save': 'Guardar',
        'load': 'Cargar',
        'error': 'Error',
        'success': 'Éxito',
        'warning': 'Advertencia',
        'info': 'Información'
    },
    
    'en': {
        # Page configuration
        'page_title': "🏥 3 CNN Architectures Comparison + Statistics",
        'page_subtitle': "MobileNetV2 vs EfficientNet-B0 vs ResNet-50 V2 + Statistical Analysis",
        
        # Menu and navigation
        'sidebar_title': "🎛️ Control Panel",
        'select_language': "🌍 Select Language",
        'new_analysis': "🔄 New Analysis",
        'new_analysis_help': "Clear current analysis",
        
        # Main tabs
        'tab_individual': "🔬 Individual Analysis",
        'tab_statistical': "📊 Statistical Evaluation",
        
        # Main headers
        'main_title': "🏆 EYE DISEASE DETECTION 👁️",
        'architectures_title': "🏗️ THE 3 COMPETING ARCHITECTURES",
        'results_title': "🎯 PREDICTION RESULTS",
        'comparison_title': "📊 COMPARATIVE PERFORMANCE ANALYSIS",
        'radar_title': "🕸️ Multidimensional Comparison",
        'winners_title': "🏆 WINNERS PODIUM",
        'detailed_analysis_title': "🔬 DETAILED ANALYSIS",
        'statistical_analysis_title': "📊 INFERENTIAL STATISTICAL ANALYSIS",
        'advanced_reports_title': "📋 ADVANCED REPORTING SYSTEM",
        
        # Model loading
        'loading_models': "🔄 Loading the 3 architectures...",
        'models_loaded': "✅ {count} architectures loaded",
        'model_loaded': "✅ {name} loaded successfully",
        'model_not_found': "⚠️ {filename} not found",
        'models_error': "❌ At least 2 models are needed for comparison",
        'loading_error': "Error loading models: {error}",
        
        # Image upload
        'upload_title': "📸 Upload Image to Compare Architectures",
        'upload_help': "Select a retina image for the architecture battle",
        'upload_description': "The image will be analyzed by all 3 architectures simultaneously",
        'battle_button': "🚀 START ARCHITECTURE BATTLE",
        'image_caption': "Image for battle",
        
        # Processing
        'processing_image': "🔄 Processing image for all architectures...",
        'analyzing_architectures': "🏗️ Analyzing with the 3 architectures...",
        'battle_completed': "✅ Battle completed! Analyzing results...",
        'prediction_error': "❌ Error in predictions",
        'analysis_completed': "🎉 **Analysis completed!** You can download reports or start a new analysis.",
        'analyzed_image': "Analyzed image",
        'analysis_timestamp': "📅 Analysis performed: {timestamp}",
        
        # Metrics and results
        'diagnosis': "**Diagnosis:** {diagnosis}",
        'confidence': "🎯 Confidence",
        'technical_metrics': "**📊 Technical Metrics:**",
        'time': "⏱️ **Time:** {time:.3f}s",
        'size': "💾 **Size:** {size:.1f}MB",
        'parameters': "🔢 **Parameters:** {params:,}",
        
        # Charts
        'confidence_chart': "🎯 Prediction Confidence",
        'time_chart': "⏱️ Prediction Time",
        'size_chart': "💾 Model Size",
        'efficiency_chart': "⚡ Efficiency (Confidence/Time)",
        'radar_chart': "🕸️ Multidimensional Architecture Profile",
        
        # Winners podium
        'highest_confidence': "🎯 Highest Confidence",
        'fastest': "⚡ Fastest",
        'lightest': "🪶 Lightest",
        'most_efficient': "⚖️ Most Efficient",
        'most_accurate': "The most accurate",
        'speedster': "The speedster",
        'efficient': "The efficient",
        'balanced': "The balanced",
        
        # Detailed analysis
        'general_winner': "👑 OVERALL WINNER: {name}",
        'general_score': "🏆 Overall Score",
        'best_balance': "The best balance of all metrics!",
        'strengths_weaknesses': "📋 Strengths and Weaknesses",
        'strengths': "**🟢 Strengths:**",
        'weaknesses': "**🔴 Areas for improvement:**",
        'technical_details': "**📊 Technical Metrics:**",
        
        # Recommendations
        'usage_recommendations': "💡 USAGE RECOMMENDATIONS",
        'clinical_apps': "**🏥 Clinical Applications:**",
        'clinical_desc': "- Use the model with **highest confidence**\n- Prioritize accuracy over speed\n- Ideal for complex diagnoses",
        'mobile_apps': "**📱 Mobile Applications:**",
        'mobile_desc': "- Use the **fastest and lightest** model\n- Balance between accuracy and resources\n- Ideal for real-time apps",
        'production_apps': "**🔄 Production Systems:**",
        'production_desc': "- Use the **most efficient** model\n- Consider processing volume\n- Ideal for scalability",
        
        # Statistical analysis
        'statistical_description': """**Rigorous evaluation with statistical tests:**
        - 🎯 **Matthews Coefficient (MCC)**: Balanced metric considering all confusion matrix cases
        - 🔬 **McNemar Test**: Statistical comparison between model pairs
        - 📈 **Confidence Intervals**: Bootstrap CI for statistical robustness""",
        
        'dataset_evaluation': "📂 Evaluation Dataset",
        'dataset_path': "🗂️ Test folder path:",
        'dataset_path_help': "Example: Tests, ./Tests, /path/to/Tests",
        'expected_structure': "📋 Expected folder structure",
        'folder_found': "✅ Folder found: {path}",
        'folder_not_found': "❌ Folder not found: {path}",
        'dataset_preview': "👀 Dataset Preview",
        'start_evaluation': "🚀 START STATISTICAL EVALUATION",
        
        # Reports
        'detectable_diseases': "🏥 Detectable Diseases",
        'cnn_architectures': "🧠 CNN Architectures",
        'unique_diagnoses': "🎯 Unique Diagnoses",
        'average_confidence': "📊 Average Confidence",
        'export_analysis': "📤 Export Analysis",
        'generate_pdf': "📄 Generate Complete PDF Report",
        'export_json': "📊 Export Technical Data (JSON)",
        'export_csv': "📈 Export Comparative CSV",
        'download_pdf': "⬇️ DOWNLOAD PDF REPORT",
        'download_json': "⬇️ DOWNLOAD JSON DATA",
        'download_csv': "⬇️ DOWNLOAD CSV",
        
        # Status messages
        'generating_pdf': "🔄 Generating professional PDF report...",
        'pdf_generated': "✅ PDF generated successfully!",
        'pdf_error': "❌ Error generating PDF report",
        'exporting_data': "🔄 Exporting technical data...",
        'data_exported': "✅ Technical data exported!",
        'data_error': "❌ Error exporting technical data",
        'preparing_csv': "🔄 Preparing CSV...",
        'csv_ready': "✅ CSV ready!",
        
        # Download information
        'download_info': """💡 **Download information:**
        - **PDF**: Complete report with clinical analysis and technical recommendations
        - **JSON**: Structured technical data for further analysis 
        - **CSV**: Simple comparative table for Excel/statistical analysis
        
        📁 Files are automatically downloaded to your Downloads folder""",
        
        # Summary table
        'summary_table': "📊 Metrics Summary Table",
        'architecture': "Architecture",
        'diagnosis_en': "Diagnosis",
        'diagnosis_es': "Diagnosis_ES",
        'confidence_table': "Confidence",
        'time_table': "Time",
        'size_table': "Size",
        'parameters_table': "Parameters",
        'efficiency_table': "Efficiency",
        'general_score_table': "Overall Score",
        'severity': "Severity",
        
        # Technical system
        'system_title': "⚙️ About This Advanced System with Statistical Analysis",
        'system_subtitle': "🚀 Next-Generation Eye Diagnosis System",
        
        # Class information (diseases)
        'normal': 'Normal',
        'moderate': 'Moderate',
        'high': 'High',
        'critical': 'Critical',
        'mild': 'Mild',
        'unspecified': 'Unspecified',
        
        # Buttons and actions
        'cancel': 'Cancel',
        'accept': 'Accept',
        'close': 'Close',
        'save': 'Save',
        'load': 'Load',
        'error': 'Error',
        'success': 'Success',
        'warning': 'Warning',
        'info': 'Information'
    }
}

def get_text(key, lang='es', **kwargs):
    """
    Obtiene texto traducido para una clave específica
    
    Args:
        key: Clave del texto a traducir
        lang: Idioma ('es' o 'en')
        **kwargs: Parámetros para formatear el texto
    
    Returns:
        Texto traducido y formateado
    """
    try:
        text = translations.get(lang, translations['es']).get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text
    except Exception as e:
        # Fallback al español si hay error
        text = translations['es'].get(key, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except:
                return text
        return text

def get_available_languages():
    """Retorna lista de idiomas disponibles"""
    return {
        'es': '🇪🇸 Español',
        'en': '🇺🇸 English'
    }