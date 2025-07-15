def get_available_languages():
    """Retorna lista de idiomas disponibles"""
    return {
        'es': '🇪🇸 Español',
        'en': '🇺🇸 English',
        'fr': '🇫🇷 Français',
        'pt': '🇧🇷 Português'
    }

def get_text(key, lang='es', **kwargs):
    """
    Obtiene texto traducido por clave y idioma
    
    Args:
        key (str): Clave del texto a obtener
        lang (str): Código del idioma ('es', 'en', 'fr')
        **kwargs: Parámetros para formatear el texto (ej: {name}, {count})
    
    Returns:
        str: Texto traducido y formateado
    """
    try:
        text = translations[lang].get(key, translations['es'].get(key, f"[{key}]"))
        if kwargs:
            return text.format(**kwargs)
        return text
    except (KeyError, ValueError):
        # Si hay error, devolver la clave entre corchetes
        return f"[{key}]"


# translations.py - VERSIÓN COMPLETA CON TODAS LAS TRADUCCIONES
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
        
        # === ENFERMEDADES OCULARES (FALTABAN) ===
        'CentralSerous_nombre': 'Corioretinopatía Serosa Central',
        'CentralSerous_descripcion': 'Acumulación de líquido bajo la retina',
        'CentralSerous_gravedad': 'Moderada',
        'CentralSerous_tratamiento': 'Observación, láser focal en casos persistentes',
        'CentralSerous_pronostico': 'Bueno, resolución espontánea en 80% casos',
        
        'Diabetic_nombre': 'Retinopatía Diabética',
        'Diabetic_descripcion': 'Daño vascular por diabetes',
        'Diabetic_gravedad': 'Alta',
        'Diabetic_tratamiento': 'Control glucémico, inyecciones intravítreas, láser',
        'Diabetic_pronostico': 'Manejo temprano previene ceguera',
        
        'DiscEdema_nombre': 'Edema del Disco Óptico',
        'DiscEdema_descripcion': 'Hinchazón por presión intracraneal',
        'DiscEdema_gravedad': 'Alta',
        'DiscEdema_tratamiento': 'Urgente: reducir presión intracraneal',
        'DiscEdema_pronostico': 'Depende de causa subyacente',
        
        'Glaucoma_nombre': 'Glaucoma',
        'Glaucoma_descripcion': 'Daño del nervio óptico',
        'Glaucoma_gravedad': 'Alta',
        'Glaucoma_tratamiento': 'Gotas hipotensoras, láser, cirugía',
        'Glaucoma_pronostico': 'Progresión lenta con tratamiento',
        
        'Healthy_nombre': 'Ojo Sano',
        'Healthy_descripcion': 'Sin patologías detectadas',
        'Healthy_gravedad': 'Normal',
        'Healthy_tratamiento': 'Exámenes preventivos anuales',
        'Healthy_pronostico': 'Excelente',
        
        'MacularScar_nombre': 'Cicatriz Macular',
        'MacularScar_descripcion': 'Tejido cicatricial en mácula',
        'MacularScar_gravedad': 'Moderada',
        'MacularScar_tratamiento': 'Rehabilitación visual, ayudas ópticas',
        'MacularScar_pronostico': 'Estable, visión central afectada',
        
        'Myopia_nombre': 'Miopía',
        'Myopia_descripcion': 'Error refractivo',
        'Myopia_gravedad': 'Leve',
        'Myopia_tratamiento': 'Lentes correctivos, cirugía refractiva',
        'Myopia_pronostico': 'Excelente con corrección',
        
        'Pterygium_nombre': 'Pterigión',
        'Pterygium_descripcion': 'Crecimiento anormal en córnea',
        'Pterygium_gravedad': 'Leve',
        'Pterygium_tratamiento': 'Observación, cirugía si afecta visión',
        'Pterygium_pronostico': 'Bueno, puede recurrir post-cirugía',
        
        'RetinalDetachment_nombre': 'Desprendimiento de Retina',
        'RetinalDetachment_descripcion': 'Emergencia: separación retinal',
        'RetinalDetachment_gravedad': 'Crítica',
        'RetinalDetachment_tratamiento': 'URGENTE: cirugía inmediata',
        'RetinalDetachment_pronostico': 'Bueno si se trata en <24-48h',
        
        'Retinitis_nombre': 'Retinitis Pigmentosa',
        'Retinitis_descripcion': 'Degeneración progresiva',
        'Retinitis_gravedad': 'Alta',
        'Retinitis_tratamiento': 'Suplementos, implantes retinales',
        'Retinitis_pronostico': 'Progresivo, investigación activa',
        
        # === ARQUITECTURAS CNN (FALTABAN) ===
        'CNN_original_nombre': 'CNN MobileNetV2 Original',
        'CNN_original_descripcion': 'Tu modelo inicial entrenado (70.44% accuracy)',
        'CNN_original_ventaja1': 'Tu modelo base',
        'CNN_original_ventaja2': 'Conocido',
        'CNN_original_ventaja3': 'Optimizado móvil',
        'CNN_original_tipo': 'Depthwise Separable Convolutions',
        'CNN_original_ventaja_principal': 'Eficiencia computacional',
        
        'EfficientNet_nombre': 'EfficientNet-B0',
        'EfficientNet_descripcion': 'Arquitectura con compound scaling balanceado',
        'EfficientNet_ventaja1': 'Compound scaling',
        'EfficientNet_ventaja2': 'Balance accuracy/params',
        'EfficientNet_ventaja3': 'Estado del arte',
        'EfficientNet_tipo': 'Compound Scaling CNN',
        'EfficientNet_ventaja_principal': 'Balance óptimo accuracy/eficiencia',
        
        'ResNet_nombre': 'ResNet-50 V2',
        'ResNet_descripcion': 'Red residual profunda con conexiones skip',
        'ResNet_ventaja1': 'Conexiones residuales',
        'ResNet_ventaja2': 'Red profunda',
        'ResNet_ventaja3': 'Estable',
        'ResNet_tipo': 'Residual Network',
        'ResNet_ventaja_principal': 'Capacidad de representación profunda',
        
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
        'info': 'Información',
        
        # Textos adicionales que aparecen en el código
        'characteristics': 'Características',
        'advantages': 'Ventajas',
        'type': 'Tipo',
        'parameters_count': 'Parámetros',
        'main_advantage': 'Ventaja principal',
        'year': 'Año',
        
        # Footer técnico - Funcionalidades estadísticas
        'statistical_features_title': '🔬 Nuevas Funcionalidades Estadísticas:',
        'mcc_description': '📊 **Coeficiente de Matthews (MCC)**: Métrica balanceada para clases desbalanceadas',
        'mcnemar_description': '🔬 **Prueba de McNemar**: Comparación estadística rigurosa entre modelos',
        'bootstrap_description': '📈 **Intervalos de Confianza Bootstrap**: Robustez estadística (95% CI)',
        'confusion_matrices': '🎭 **Matrices de Confusión**: Análisis detallado por clase',
        'statistical_reports': '📋 **Reportes Estadísticos**: Exportación completa de resultados',
        
        # Ventajas competitivas
        'competitive_advantages_title': '🔬 Ventajas Competitivas:',
        'specialized_diseases': '**10 enfermedades especializadas** vs 4 básicas de sistemas convencionales',
        'multi_architecture': '**Análisis multi-arquitectura** con comparación simultánea de CNNs',
        'statistical_evaluation': '**Evaluación estadística rigurosa** con pruebas de significancia',
        'professional_reports': '**Reportes profesionales PDF** con análisis clínico y estadístico',
        'complete_export': '**Exportación técnica completa** (JSON, CSV, TXT) para investigación',
        'contextual_recommendations': '**Recomendaciones contextuales** basadas en evidencia estadística',
        
        # Arquitecturas implementadas
        'implemented_architectures_title': '🏗️ Arquitecturas Implementadas:',
        'hybrid_cnn': '**🧠 CNN Híbrida (MobileNetV2)**: Transfer Learning especializado',
        'efficientnet_desc': '**⚡ EfficientNet-B0**: Compound Scaling balanceado',
        'resnet_desc': '**🔗 ResNet-50 V2**: Conexiones residuales profundas',
        
        # Métricas evaluadas
        'evaluated_metrics_title': '📊 Métricas Evaluadas:',
        'precision_metric': '🎯 **Precisión**: Confianza, MCC y consenso diagnóstico',
        'speed_metric': '⚡ **Velocidad**: Tiempo de inferencia optimizado',
        'efficiency_metric': '💾 **Eficiencia**: Uso de memoria y escalabilidad',
        'balance_metric': '🏆 **Balance**: Score general multi-criterio',
        'significance_metric': '📈 **Significancia**: Pruebas estadísticas inferenciales',
        
        # Aplicaciones
        'applications_title': '🎯 Aplicaciones:',
        'clinical_application': '🏥 **Clínicas**: Diagnóstico de alta precisión con validación estadística',
        'mobile_application': '📱 **Móviles**: Apps de telemedicina con métricas robustas',
        'production_application': '🔄 **Producción**: Sistemas hospitalarios escalables con evidencia estadística',
        'research_application': '🔬 **Investigación**: Datos completos para publicaciones científicas',
        
        # Innovación
        'innovation_text': '**💡 Innovación**: Primer sistema que combina múltiples arquitecturas CNN con análisis estadístico inferencial completo para diagnóstico ocular especializado.',
        
        # Métodos estadísticos
        'statistical_methods_title': '📚 Métodos Estadísticos:',
        'mcc_method': '**MCC**: Coeficiente de Correlación de Matthews para métricas balanceadas',
        'mcnemar_method': '**McNemar**: Prueba chi-cuadrado para comparación de clasificadores',
        'bootstrap_method': '**Bootstrap**: Intervalos de confianza no paramétricos',
        'yates_correction': '**Corrección de Yates**: Para muestras pequeñas en McNemar',
    },
    'pt': {
        # Configuração da página
        'page_title': "🏥 Comparação de 3 Arquiteturas CNN + Estatísticas",
        'page_subtitle': "MobileNetV2 vs EfficientNet-B0 vs ResNet-50 V2 + Análise Estatística",
        
        # Menu e navegação
        'sidebar_title': "🎛️ Painel de Controle",
        'select_language': "🌍 Selecionar Idioma",
        'new_analysis': "🔄 Nova Análise",
        'new_analysis_help': "Limpar análise atual",
        
        # Abas principais
        'tab_individual': "🔬 Análise Individual",
        'tab_statistical': "📊 Avaliação Estatística",
        
        # Cabeçalhos principais
        'main_title': "🏆 DETECÇÃO DE DOENÇAS OCULARES 👁️",
        'architectures_title': "🏗️ AS 3 ARQUITETURAS EM COMPETIÇÃO",
        'results_title': "🎯 RESULTADOS DE PREDIÇÃO",
        'comparison_title': "📊 ANÁLISE COMPARATIVA DE DESEMPENHO",
        'radar_title': "🕸️ Comparação Multidimensional",
        'winners_title': "🏆 PÓDIO DOS VENCEDORES",
        'detailed_analysis_title': "🔬 ANÁLISE DETALHADA",
        'statistical_analysis_title': "📊 ANÁLISE ESTATÍSTICA INFERENCIAL",
        'advanced_reports_title': "📋 SISTEMA AVANÇADO DE RELATÓRIOS",
        
        # Carregamento de modelos
        'loading_models': "🔄 Carregando as 3 arquiteturas...",
        'models_loaded': "✅ {count} arquiteturas carregadas",
        'model_loaded': "✅ {name} carregado corretamente",
        'model_not_found': "⚠️ Não foi encontrado {filename}",
        'models_error': "❌ São necessários pelo menos 2 modelos para comparar",
        'loading_error': "Erro carregando modelos: {error}",
        
        # Upload de imagens
        'upload_title': "📸 Carregar Imagem para Comparar Arquiteturas",
        'upload_help': "Selecione uma imagem de retina para a batalha de arquiteturas",
        'upload_description': "A imagem será analisada pelas 3 arquiteturas simultaneamente",
        'battle_button': "🚀 INICIAR BATALHA DE ARQUITETURAS",
        'image_caption': "Imagem para a batalha",
        
        # Processamento
        'processing_image': "🔄 Processando imagem para todas as arquiteturas...",
        'analyzing_architectures': "🏗️ Analisando com as 3 arquiteturas...",
        'battle_completed': "✅ Batalha concluída! Analisando resultados...",
        'prediction_error': "❌ Erro nas predições",
        'analysis_completed': "🎉 **Análise já concluída!** Você pode baixar os relatórios ou fazer uma nova análise.",
        'analyzed_image': "Imagem analisada",
        'analysis_timestamp': "📅 Análise realizada: {timestamp}",
        
        # Métricas e resultados
        'diagnosis': "**Diagnóstico:** {diagnosis}",
        'confidence': "🎯 Confiança",
        'technical_metrics': "**📊 Métricas Técnicas:**",
        'time': "⏱️ **Tempo:** {time:.3f}s",
        'size': "💾 **Tamanho:** {size:.1f}MB",
        'parameters': "🔢 **Parâmetros:** {params:,}",
        
        # Gráficos
        'confidence_chart': "🎯 Confiança de Predição",
        'time_chart': "⏱️ Tempo de Predição",
        'size_chart': "💾 Tamanho do Modelo",
        'efficiency_chart': "⚡ Eficiência (Confiança/Tempo)",
        'radar_chart': "🕸️ Perfil Multidimensional de Arquiteturas",
        
        # Pódio dos vencedores
        'highest_confidence': "🎯 Maior Confiança",
        'fastest': "⚡ Mais Rápido",
        'lightest': "🪶 Mais Leve",
        'most_efficient': "⚖️ Mais Eficiente",
        'most_accurate': "O mais preciso",
        'speedster': "O velocista",
        'efficient': "O eficiente",
        'balanced': "O equilibrado",
        
        # Análise detalhada
        'general_winner': "👑 VENCEDOR GERAL: {name}",
        'general_score': "🏆 Score Geral",
        'best_balance': "O melhor equilíbrio de todas as métricas!",
        'strengths_weaknesses': "📋 Forças e Fraquezas",
        'strengths': "**🟢 Forças:**",
        'weaknesses': "**🔴 Áreas de melhoria:**",
        'technical_details': "**📊 Métricas Técnicas:**",
        
        # Recomendações
        'usage_recommendations': "💡 RECOMENDAÇÕES DE USO",
        'clinical_apps': "**🏥 Aplicações Clínicas:**",
        'clinical_desc': "- Use o modelo com **maior confiança**\n- Priorize precisão sobre velocidade\n- Ideal para diagnósticos complexos",
        'mobile_apps': "**📱 Aplicações Móveis:**",
        'mobile_desc': "- Use o modelo **mais rápido e leve**\n- Equilíbrio entre precisão e recursos\n- Ideal para apps em tempo real",
        'production_apps': "**🔄 Sistemas de Produção:**",
        'production_desc': "- Use o modelo **mais eficiente**\n- Considere o volume de processamento\n- Ideal para escalabilidade",
        
        # Análise estatística
        'statistical_description': """**Avaliação rigorosa com testes estatísticos:**
        - 🎯 **Coeficiente de Matthews (MCC)**: Métrica balanceada que considera todos os casos da matriz de confusão
        - 🔬 **Teste de McNemar**: Comparação estatística entre pares de modelos
        - 📈 **Intervalos de Confiança**: Bootstrap CI para robustez estatística""",
        
        'dataset_evaluation': "📂 Dataset de Avaliação",
        'dataset_path': "🗂️ Caminho da pasta de testes:",
        'dataset_path_help': "Exemplo: Testes, ./Testes, /caminho/para/Testes",
        'expected_structure': "📋 Estrutura de pastas esperada",
        'folder_found': "✅ Pasta encontrada: {path}",
        'folder_not_found': "❌ Pasta não encontrada: {path}",
        'dataset_preview': "👀 Prévia do Dataset",
        'start_evaluation': "🚀 INICIAR AVALIAÇÃO ESTATÍSTICA",
        
        # Relatórios
        'detectable_diseases': "🏥 Doenças Detectáveis",
        'cnn_architectures': "🧠 Arquiteturas CNN",
        'unique_diagnoses': "🎯 Diagnósticos Únicos",
        'average_confidence': "📊 Confiança Média",
        'export_analysis': "📤 Exportar Análise",
        'generate_pdf': "📄 Gerar Relatório PDF Completo",
        'export_json': "📊 Exportar Dados Técnicos (JSON)",
        'export_csv': "📈 Exportar CSV Comparativo",
        'download_pdf': "⬇️ BAIXAR RELATÓRIO PDF",
        'download_json': "⬇️ BAIXAR DADOS JSON",
        'download_csv': "⬇️ BAIXAR CSV",
        
        # Mensagens de status
        'generating_pdf': "🔄 Gerando relatório PDF profissional...",
        'pdf_generated': "✅ PDF gerado com sucesso!",
        'pdf_error': "❌ Erro gerando o relatório PDF",
        'exporting_data': "🔄 Exportando dados técnicos...",
        'data_exported': "✅ Dados técnicos exportados!",
        'data_error': "❌ Erro exportando dados técnicos",
        'preparing_csv': "🔄 Preparando CSV...",
        'csv_ready': "✅ CSV pronto!",
        
        # Informações sobre downloads
        'download_info': """💡 **Informações sobre os downloads:**
        - **PDF**: Relatório completo com análise clínica e recomendações técnicas
        - **JSON**: Dados técnicos estruturados para análise posterior 
        - **CSV**: Tabela comparativa simples para Excel/análise estatística
        
        📁 Os arquivos são baixados automaticamente para sua pasta de Downloads""",
        
        # Tabela resumo
        'summary_table': "📊 Tabela Resumo de Métricas",
        'architecture': "Arquitetura",
        'diagnosis_en': "Diagnóstico",
        'diagnosis_es': "Diagnóstico_PT",
        'confidence_table': "Confiança",
        'time_table': "Tempo",
        'size_table': "Tamanho",
        'parameters_table': "Parâmetros",
        'efficiency_table': "Eficiência",
        'general_score_table': "Score Geral",
        'severity': "Gravidade",
        
        # Sistema técnico
        'system_title': "⚙️ Sobre Este Sistema Avançado com Análise Estatística",
        'system_subtitle': "🚀 Sistema de Diagnóstico Ocular de Nova Geração",
        
        # === DOENÇAS OCULARES ===
        'CentralSerous_nombre': 'Corioretinopatia Serosa Central',
        'CentralSerous_descripcion': 'Acúmulo de líquido sob a retina',
        'CentralSerous_gravedad': 'Moderada',
        'CentralSerous_tratamiento': 'Observação, laser focal em casos persistentes',
        'CentralSerous_pronostico': 'Bom, resolução espontânea em 80% dos casos',
        
        'Diabetic_nombre': 'Retinopatia Diabética',
        'Diabetic_descripcion': 'Dano vascular por diabetes',
        'Diabetic_gravedad': 'Alta',
        'Diabetic_tratamiento': 'Controle glicêmico, injeções intravítreas, laser',
        'Diabetic_pronostico': 'Manejo precoce previne cegueira',
        
        'DiscEdema_nombre': 'Edema do Disco Óptico',
        'DiscEdema_descripcion': 'Inchaço por pressão intracraniana',
        'DiscEdema_gravedad': 'Alta',
        'DiscEdema_tratamiento': 'Urgente: reduzir pressão intracraniana',
        'DiscEdema_pronostico': 'Depende da causa subjacente',
        
        'Glaucoma_nombre': 'Glaucoma',
        'Glaucoma_descripcion': 'Dano do nervo óptico',
        'Glaucoma_gravedad': 'Alta',
        'Glaucoma_tratamiento': 'Colírios hipotensores, laser, cirurgia',
        'Glaucoma_pronostico': 'Progressão lenta com tratamento',
        
        'Healthy_nombre': 'Olho Saudável',
        'Healthy_descripcion': 'Sem patologias detectadas',
        'Healthy_gravedad': 'Normal',
        'Healthy_tratamiento': 'Exames preventivos anuais',
        'Healthy_pronostico': 'Excelente',
        
        'MacularScar_nombre': 'Cicatriz Macular',
        'MacularScar_descripcion': 'Tecido cicatricial na mácula',
        'MacularScar_gravedad': 'Moderada',
        'MacularScar_tratamiento': 'Reabilitação visual, auxílios ópticos',
        'MacularScar_pronostico': 'Estável, visão central afetada',
        
        'Myopia_nombre': 'Miopia',
        'Myopia_descripcion': 'Erro refrativo',
        'Myopia_gravedad': 'Leve',
        'Myopia_tratamiento': 'Lentes corretivas, cirurgia refrativa',
        'Myopia_pronostico': 'Excelente com correção',
        
        'Pterygium_nombre': 'Pterígio',
        'Pterygium_descripcion': 'Crescimento anormal na córnea',
        'Pterygium_gravedad': 'Leve',
        'Pterygium_tratamiento': 'Observação, cirurgia se afetar a visão',
        'Pterygium_pronostico': 'Bom, pode recorrer pós-cirurgia',
        
        'RetinalDetachment_nombre': 'Descolamento de Retina',
        'RetinalDetachment_descripcion': 'Emergência: separação retiniana',
        'RetinalDetachment_gravedad': 'Crítica',
        'RetinalDetachment_tratamiento': 'URGENTE: cirurgia imediata',
        'RetinalDetachment_pronostico': 'Bom se tratado em <24-48h',
        
        'Retinitis_nombre': 'Retinite Pigmentosa',
        'Retinitis_descripcion': 'Degeneração progressiva',
        'Retinitis_gravedad': 'Alta',
        'Retinitis_tratamiento': 'Suplementos, implantes retinianos',
        'Retinitis_pronostico': 'Progressivo, pesquisa ativa',
        
        # === ARQUITETURAS CNN ===
        'CNN_original_nombre': 'CNN MobileNetV2 Original',
        'CNN_original_descripcion': 'Seu modelo inicial treinado (70.44% precisão)',
        'CNN_original_ventaja1': 'Seu modelo base',
        'CNN_original_ventaja2': 'Conhecido',
        'CNN_original_ventaja3': 'Otimizado para móvel',
        'CNN_original_tipo': 'Convoluções Separáveis em Profundidade',
        'CNN_original_ventaja_principal': 'Eficiência computacional',
        
        'EfficientNet_nombre': 'EfficientNet-B0',
        'EfficientNet_descripcion': 'Arquitetura com compound scaling balanceado',
        'EfficientNet_ventaja1': 'Compound scaling',
        'EfficientNet_ventaja2': 'Equilíbrio precisão/parâmetros',
        'EfficientNet_ventaja3': 'Estado da arte',
        'EfficientNet_tipo': 'CNN de Escalonamento Composto',
        'EfficientNet_ventaja_principal': 'Equilíbrio ótimo precisão/eficiência',
        
        'ResNet_nombre': 'ResNet-50 V2',
        'ResNet_descripcion': 'Rede residual profunda com conexões skip',
        'ResNet_ventaja1': 'Conexões residuais',
        'ResNet_ventaja2': 'Rede profunda',
        'ResNet_ventaja3': 'Estável',
        'ResNet_tipo': 'Rede Residual',
        'ResNet_ventaja_principal': 'Capacidade de representação profunda',
        
        # Informações de classes (doenças)
        'normal': 'Normal',
        'moderate': 'Moderada',
        'high': 'Alta',
        'critical': 'Crítica',
        'mild': 'Leve',
        'unspecified': 'Não especificada',
        
        # Botões e ações
        'cancel': 'Cancelar',
        'accept': 'Aceitar',
        'close': 'Fechar',
        'save': 'Salvar',
        'load': 'Carregar',
        'error': 'Erro',
        'success': 'Sucesso',
        'warning': 'Aviso',
        'info': 'Informação',
        
        # Textos adicionais que aparecem no código
        'characteristics': 'Características',
        'advantages': 'Vantagens',
        'type': 'Tipo',
        'parameters_count': 'Parâmetros',
        'main_advantage': 'Vantagem principal',
        'year': 'Ano',
        
        # Footer técnico - Funcionalidades estatísticas
        'statistical_features_title': '🔬 Novas Funcionalidades Estatísticas:',
        'mcc_description': '📊 **Coeficiente de Matthews (MCC)**: Métrica balanceada para classes desbalanceadas',
        'mcnemar_description': '🔬 **Teste de McNemar**: Comparação estatística rigorosa entre modelos',
        'bootstrap_description': '📈 **Intervalos de Confiança Bootstrap**: Robustez estatística (95% CI)',
        'confusion_matrices': '🎭 **Matrizes de Confusão**: Análise detalhada por classe',
        'statistical_reports': '📋 **Relatórios Estatísticos**: Exportação completa de resultados',
        
        # Vantagens competitivas
        'competitive_advantages_title': '🔬 Vantagens Competitivas:',
        'specialized_diseases': '**10 doenças especializadas** vs 4 básicas de sistemas convencionais',
        'multi_architecture': '**Análise multi-arquitetura** com comparação simultânea de CNNs',
        'statistical_evaluation': '**Avaliação estatística rigorosa** com testes de significância',
        'professional_reports': '**Relatórios profissionais PDF** com análise clínica e estatística',
        'complete_export': '**Exportação técnica completa** (JSON, CSV, TXT) para pesquisa',
        'contextual_recommendations': '**Recomendações contextuais** baseadas em evidência estatística',
        
        # Arquiteturas implementadas
        'implemented_architectures_title': '🏗️ Arquiteturas Implementadas:',
        'hybrid_cnn': '**🧠 CNN Híbrida (MobileNetV2)**: Transfer Learning especializado',
        'efficientnet_desc': '**⚡ EfficientNet-B0**: Compound Scaling balanceado',
        'resnet_desc': '**🔗 ResNet-50 V2**: Conexões residuais profundas',
        
        # Métricas avaliadas
        'evaluated_metrics_title': '📊 Métricas Avaliadas:',
        'precision_metric': '🎯 **Precisão**: Confiança, MCC e consenso diagnóstico',
        'speed_metric': '⚡ **Velocidade**: Tempo de inferência otimizado',
        'efficiency_metric': '💾 **Eficiência**: Uso de memória e escalabilidade',
        'balance_metric': '🏆 **Equilíbrio**: Score geral multi-critério',
        'significance_metric': '📈 **Significância**: Testes estatísticos inferenciais',
        
        # Aplicações
        'applications_title': '🎯 Aplicações:',
        'clinical_application': '🏥 **Clínicas**: Diagnóstico de alta precisão com validação estatística',
        'mobile_application': '📱 **Móveis**: Apps de telemedicina com métricas robustas',
        'production_application': '🔄 **Produção**: Sistemas hospitalares escaláveis com evidência estatística',
        'research_application': '🔬 **Pesquisa**: Dados completos para publicações científicas',
        
        # Inovação
        'innovation_text': '**💡 Inovação**: Primeiro sistema que combina múltiplas arquiteturas CNN com análise estatística inferencial completa para diagnóstico ocular especializado.',
        
        # Métodos estatísticos
        'statistical_methods_title': '📚 Métodos Estatísticos:',
        'mcc_method': '**MCC**: Coeficiente de Correlação de Matthews para métricas balanceadas',
        'mcnemar_method': '**McNemar**: Teste qui-quadrado para comparação de classificadores',
        'bootstrap_method': '**Bootstrap**: Intervalos de confiança não paramétricos',
        'yates_correction': '**Correção de Yates**: Para amostras pequenas em McNemar',
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
        
        # === EYE DISEASES (MISSING) ===
        'CentralSerous_nombre': 'Central Serous Chorioretinopathy',
        'CentralSerous_descripcion': 'Fluid accumulation under the retina',
        'CentralSerous_gravedad': 'Moderate',
        'CentralSerous_tratamiento': 'Observation, focal laser in persistent cases',
        'CentralSerous_pronostico': 'Good, spontaneous resolution in 80% of cases',
        
        'Diabetic_nombre': 'Diabetic Retinopathy',
        'Diabetic_descripcion': 'Vascular damage from diabetes',
        'Diabetic_gravedad': 'High',
        'Diabetic_tratamiento': 'Glycemic control, intravitreal injections, laser',
        'Diabetic_pronostico': 'Early management prevents blindness',
        
        'DiscEdema_nombre': 'Optic Disc Edema',
        'DiscEdema_descripcion': 'Swelling from intracranial pressure',
        'DiscEdema_gravedad': 'High',
        'DiscEdema_tratamiento': 'Urgent: reduce intracranial pressure',
        'DiscEdema_pronostico': 'Depends on underlying cause',
        
        'Glaucoma_nombre': 'Glaucoma',
        'Glaucoma_descripcion': 'Optic nerve damage',
        'Glaucoma_gravedad': 'High',
        'Glaucoma_tratamiento': 'Hypotensive drops, laser, surgery',
        'Glaucoma_pronostico': 'Slow progression with treatment',
        
        'Healthy_nombre': 'Healthy Eye',
        'Healthy_descripcion': 'No pathologies detected',
        'Healthy_gravedad': 'Normal',
        'Healthy_tratamiento': 'Annual preventive exams',
        'Healthy_pronostico': 'Excellent',
        
        'MacularScar_nombre': 'Macular Scar',
        'MacularScar_descripcion': 'Scar tissue in macula',
        'MacularScar_gravedad': 'Moderate',
        'MacularScar_tratamiento': 'Visual rehabilitation, optical aids',
        'MacularScar_pronostico': 'Stable, central vision affected',
        
        'Myopia_nombre': 'Myopia',
        'Myopia_descripcion': 'Refractive error',
        'Myopia_gravedad': 'Mild',
        'Myopia_tratamiento': 'Corrective lenses, refractive surgery',
        'Myopia_pronostico': 'Excellent with correction',
        
        'Pterygium_nombre': 'Pterygium',
        'Pterygium_descripcion': 'Abnormal growth on cornea',
        'Pterygium_gravedad': 'Mild',
        'Pterygium_tratamiento': 'Observation, surgery if affecting vision',
        'Pterygium_pronostico': 'Good, may recur post-surgery',
        
        'RetinalDetachment_nombre': 'Retinal Detachment',
        'RetinalDetachment_descripcion': 'Emergency: retinal separation',
        'RetinalDetachment_gravedad': 'Critical',
        'RetinalDetachment_tratamiento': 'URGENT: immediate surgery',
        'RetinalDetachment_pronostico': 'Good if treated within <24-48h',
        
        'Retinitis_nombre': 'Retinitis Pigmentosa',
        'Retinitis_descripcion': 'Progressive degeneration',
        'Retinitis_gravedad': 'High',
        'Retinitis_tratamiento': 'Supplements, retinal implants',
        'Retinitis_pronostico': 'Progressive, active research',
        
        # === CNN ARCHITECTURES (MISSING) ===
        'CNN_original_nombre': 'Original CNN MobileNetV2',
        'CNN_original_descripcion': 'Your initial trained model (70.44% accuracy)',
        'CNN_original_ventaja1': 'Your base model',
        'CNN_original_ventaja2': 'Known',
        'CNN_original_ventaja3': 'Mobile optimized',
        'CNN_original_tipo': 'Depthwise Separable Convolutions',
        'CNN_original_ventaja_principal': 'Computational efficiency',
        
        'EfficientNet_nombre': 'EfficientNet-B0',
        'EfficientNet_descripcion': 'Architecture with balanced compound scaling',
        'EfficientNet_ventaja1': 'Compound scaling',
        'EfficientNet_ventaja2': 'Balance accuracy/params',
        'EfficientNet_ventaja3': 'State of the art',
        'EfficientNet_tipo': 'Compound Scaling CNN',
        'EfficientNet_ventaja_principal': 'Optimal balance accuracy/efficiency',
        
        'ResNet_nombre': 'ResNet-50 V2',
        'ResNet_descripcion': 'Deep residual network with skip connections',
        'ResNet_ventaja1': 'Residual connections',
        'ResNet_ventaja2': 'Deep network',
        'ResNet_ventaja3': 'Stable',
        'ResNet_tipo': 'Residual Network',
        'ResNet_ventaja_principal': 'Deep representation capacity',
        
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
        'info': 'Information',
        
        # Additional texts that appear in code
        'characteristics': 'Characteristics',
        'advantages': 'Advantages',
        'type': 'Type',
        'parameters_count': 'Parameters',
        'main_advantage': 'Main advantage',
        'year': 'Year',
        
        # Footer técnico - Funcionalidades estadísticas
        'statistical_features_title': '🔬 New Statistical Features:',
        'mcc_description': '📊 **Matthews Coefficient (MCC)**: Balanced metric for unbalanced classes',
        'mcnemar_description': '🔬 **McNemar Test**: Rigorous statistical comparison between models',
        'bootstrap_description': '📈 **Bootstrap Confidence Intervals**: Statistical robustness (95% CI)',
        'confusion_matrices': '🎭 **Confusion Matrices**: Detailed analysis by class',
        'statistical_reports': '📋 **Statistical Reports**: Complete results export',
        
        # Ventajas competitivas
        'competitive_advantages_title': '🔬 Competitive Advantages:',
        'specialized_diseases': '**10 specialized diseases** vs 4 basic ones from conventional systems',
        'multi_architecture': '**Multi-architecture analysis** with simultaneous CNN comparison',
        'statistical_evaluation': '**Rigorous statistical evaluation** with significance tests',
        'professional_reports': '**Professional PDF reports** with clinical and statistical analysis',
        'complete_export': '**Complete technical export** (JSON, CSV, TXT) for research',
        'contextual_recommendations': '**Contextual recommendations** based on statistical evidence',
        
        # Arquitecturas implementadas
        'implemented_architectures_title': '🏗️ Implemented Architectures:',
        'hybrid_cnn': '**🧠 Hybrid CNN (MobileNetV2)**: Specialized Transfer Learning',
        'efficientnet_desc': '**⚡ EfficientNet-B0**: Balanced Compound Scaling',
        'resnet_desc': '**🔗 ResNet-50 V2**: Deep residual connections',
        
        # Métricas evaluadas
        'evaluated_metrics_title': '📊 Evaluated Metrics:',
        'precision_metric': '🎯 **Precision**: Confidence, MCC and diagnostic consensus',
        'speed_metric': '⚡ **Speed**: Optimized inference time',
        'efficiency_metric': '💾 **Efficiency**: Memory usage and scalability',
        'balance_metric': '🏆 **Balance**: Multi-criteria overall score',
        'significance_metric': '📈 **Significance**: Inferential statistical tests',
        
        # Aplicaciones
        'applications_title': '🎯 Applications:',
        'clinical_application': '🏥 **Clinical**: High-precision diagnosis with statistical validation',
        'mobile_application': '📱 **Mobile**: Telemedicine apps with robust metrics',
        'production_application': '🔄 **Production**: Scalable hospital systems with statistical evidence',
        'research_application': '🔬 **Research**: Complete data for scientific publications',
        
        # Innovación
        'innovation_text': '**💡 Innovation**: First system combining multiple CNN architectures with complete inferential statistical analysis for specialized eye diagnosis.',
        
        # Métodos estadísticos
        'statistical_methods_title': '📚 Statistical Methods:',
        'mcc_method': '**MCC**: Matthews Correlation Coefficient for balanced metrics',
        'mcnemar_method': '**McNemar**: Chi-square test for classifier comparison',
        'bootstrap_method': '**Bootstrap**: Non-parametric confidence intervals',
        'yates_correction': '**Yates Correction**: For small samples in McNemar',
    },
    
    'fr': {
        # Configuration de page
        'page_title': "🏥 Comparaison de 3 Architectures CNN + Statistiques",
        'page_subtitle': "MobileNetV2 vs EfficientNet-B0 vs ResNet-50 V2 + Analyse Statistique",
        
        # Menu et navigation
        'sidebar_title': "🎛️ Panneau de Contrôle",
        'select_language': "🌍 Sélectionner la Langue",
        'new_analysis': "🔄 Nouvelle Analyse",
        'new_analysis_help': "Effacer l'analyse actuelle",
        
        # Onglets principaux
        'tab_individual': "🔬 Analyse Individuelle",
        'tab_statistical': "📊 Évaluation Statistique",
        
        # En-têtes principaux
        'main_title': "🏆 DÉTECTION DE MALADIES OCULAIRES 👁️",
        'architectures_title': "🏗️ LES 3 ARCHITECTURES EN COMPÉTITION",
        'results_title': "🎯 RÉSULTATS DE PRÉDICTION",
        'comparison_title': "📊 ANALYSE COMPARATIVE DE PERFORMANCE",
        'radar_title': "🕸️ Comparaison Multidimensionnelle",
        'winners_title': "🏆 PODIUM DES GAGNANTS",
        'detailed_analysis_title': "🔬 ANALYSE DÉTAILLÉE",
        'statistical_analysis_title': "📊 ANALYSE STATISTIQUE INFÉRENTIELLE",
        'advanced_reports_title': "📋 SYSTÈME DE RAPPORTS AVANCÉS",
        
        # Chargement des modèles
        'loading_models': "🔄 Chargement des 3 architectures...",
        'models_loaded': "✅ {count} architectures chargées",
        'model_loaded': "✅ {name} chargé avec succès",
        'model_not_found': "⚠️ {filename} non trouvé",
        'models_error': "❌ Au moins 2 modèles sont nécessaires pour la comparaison",
        'loading_error': "Erreur lors du chargement des modèles: {error}",
        
        # Téléchargement d'images
        'upload_title': "📸 Télécharger une Image pour Comparer les Architectures",
        'upload_help': "Sélectionnez une image de rétine pour la bataille d'architectures",
        'upload_description': "L'image sera analysée par les 3 architectures simultanément",
        'battle_button': "🚀 COMMENCER LA BATAILLE D'ARCHITECTURES",
        'image_caption': "Image pour la bataille",
        
        # Traitement
        'processing_image': "🔄 Traitement de l'image pour toutes les architectures...",
        'analyzing_architectures': "🏗️ Analyse avec les 3 architectures...",
        'battle_completed': "✅ Bataille terminée! Analyse des résultats...",
        'prediction_error': "❌ Erreur dans les prédictions",
        'analysis_completed': "🎉 **Analyse terminée!** Vous pouvez télécharger les rapports ou commencer une nouvelle analyse.",
        'analyzed_image': "Image analysée",
        'analysis_timestamp': "📅 Analyse effectuée: {timestamp}",
        
        # Métriques et résultats
        'diagnosis': "**Diagnostic:** {diagnosis}",
        'confidence': "🎯 Confiance",
        'technical_metrics': "**📊 Métriques Techniques:**",
        'time': "⏱️ **Temps:** {time:.3f}s",
        'size': "💾 **Taille:** {size:.1f}MB",
        'parameters': "🔢 **Paramètres:** {params:,}",
        
        # Graphiques
        'confidence_chart': "🎯 Confiance de Prédiction",
        'time_chart': "⏱️ Temps de Prédiction",
        'size_chart': "💾 Taille du Modèle",
        'efficiency_chart': "⚡ Efficacité (Confiance/Temps)",
        'radar_chart': "🕸️ Profil Multidimensionnel des Architectures",
        
        # Podium des gagnants
        'highest_confidence': "🎯 Plus Haute Confiance",
        'fastest': "⚡ Le Plus Rapide",
        'lightest': "🪶 Le Plus Léger",
        'most_efficient': "⚖️ Le Plus Efficace",
        'most_accurate': "Le plus précis",
        'speedster': "Le sprinteur",
        'efficient': "L'efficace",
        'balanced': "L'équilibré",
        
        # Analyse détaillée
        'general_winner': "👑 GAGNANT GÉNÉRAL: {name}",
        'general_score': "🏆 Score Général",
        'best_balance': "Le meilleur équilibre de toutes les métriques!",
        'strengths_weaknesses': "📋 Forces et Faiblesses",
        'strengths': "**🟢 Forces:**",
        'weaknesses': "**🔴 Domaines d'amélioration:**",
        'technical_details': "**📊 Métriques Techniques:**",
        
        # Recommandations
        'usage_recommendations': "💡 RECOMMANDATIONS D'USAGE",
        'clinical_apps': "**🏥 Applications Cliniques:**",
        'clinical_desc': "- Utilisez le modèle avec la **plus haute confiance**\n- Priorisez la précision sur la vitesse\n- Idéal pour les diagnostics complexes",
        'mobile_apps': "**📱 Applications Mobiles:**",
        'mobile_desc': "- Utilisez le modèle **le plus rapide et léger**\n- Équilibre entre précision et ressources\n- Idéal pour les apps temps réel",
        'production_apps': "**🔄 Systèmes de Production:**",
        'production_desc': "- Utilisez le modèle **le plus efficace**\n- Considérez le volume de traitement\n- Idéal pour la scalabilité",
        
        # Analyse statistique
        'statistical_description': """**Évaluation rigoureuse avec tests statistiques:**
        - 🎯 **Coefficient de Matthews (MCC)**: Métrique équilibrée considérant tous les cas de la matrice de confusion
        - 🔬 **Test de McNemar**: Comparaison statistique entre paires de modèles
        - 📈 **Intervalles de Confiance**: Bootstrap CI pour la robustesse statistique""",
        
        'dataset_evaluation': "📂 Dataset d'Évaluation",
        'dataset_path': "🗂️ Chemin du dossier de tests:",
        'dataset_path_help': "Exemple: Tests, ./Tests, /chemin/vers/Tests",
        'expected_structure': "📋 Structure de dossiers attendue",
        'folder_found': "✅ Dossier trouvé: {path}",
        'folder_not_found': "❌ Dossier non trouvé: {path}",
        'dataset_preview': "👀 Aperçu du Dataset",
        'start_evaluation': "🚀 COMMENCER L'ÉVALUATION STATISTIQUE",
        
        # Rapports
        'detectable_diseases': "🏥 Maladies Détectables",
        'cnn_architectures': "🧠 Architectures CNN",
        'unique_diagnoses': "🎯 Diagnostics Uniques",
        'average_confidence': "📊 Confiance Moyenne",
        'export_analysis': "📤 Exporter l'Analyse",
        'generate_pdf': "📄 Générer un Rapport PDF Complet",
        'export_json': "📊 Exporter les Données Techniques (JSON)",
        'export_csv': "📈 Exporter un CSV Comparatif",
        'download_pdf': "⬇️ TÉLÉCHARGER LE RAPPORT PDF",
        'download_json': "⬇️ TÉLÉCHARGER LES DONNÉES JSON",
        'download_csv': "⬇️ TÉLÉCHARGER LE CSV",
        
        # Messages de statut
        'generating_pdf': "🔄 Génération du rapport PDF professionnel...",
        'pdf_generated': "✅ PDF généré avec succès!",
        'pdf_error': "❌ Erreur lors de la génération du rapport PDF",
        'exporting_data': "🔄 Exportation des données techniques...",
        'data_exported': "✅ Données techniques exportées!",
        'data_error': "❌ Erreur lors de l'exportation des données techniques",
        'preparing_csv': "🔄 Préparation du CSV...",
        'csv_ready': "✅ CSV prêt!",
        
        # Informations sur les téléchargements
        'download_info': """💡 **Informations sur les téléchargements:**
        - **PDF**: Rapport complet avec analyse clinique et recommandations techniques
        - **JSON**: Données techniques structurées pour analyse ultérieure 
        - **CSV**: Tableau comparatif simple pour Excel/analyse statistique
        
        📁 Les fichiers sont automatiquement téléchargés dans votre dossier Téléchargements""",
        
        # Tableau résumé
        'summary_table': "📊 Tableau Résumé des Métriques",
        'architecture': "Architecture",
        'diagnosis_en': "Diagnostic",
        'diagnosis_es': "Diagnostic_ES",
        'confidence_table': "Confiance",
        'time_table': "Temps",
        'size_table': "Taille",
        'parameters_table': "Paramètres",
        'efficiency_table': "Efficacité",
        'general_score_table': "Score Général",
        'severity': "Gravité",
        
        # Système technique
        'system_title': "⚙️ À Propos de ce Système Avancé avec Analyse Statistique",
        'system_subtitle': "🚀 Système de Diagnostic Oculaire de Nouvelle Génération",
        
        # === MALADIES OCULAIRES ===
        'CentralSerous_nombre': 'Choriorétinopathie Séreuse Centrale',
        'CentralSerous_descripcion': 'Accumulation de liquide sous la rétine',
        'CentralSerous_gravedad': 'Modérée',
        'CentralSerous_tratamiento': 'Observation, laser focal dans les cas persistants',
        'CentralSerous_pronostico': 'Bon, résolution spontanée dans 80% des cas',
        
        'Diabetic_nombre': 'Rétinopathie Diabétique',
        'Diabetic_descripcion': 'Dommages vasculaires dus au diabète',
        'Diabetic_gravedad': 'Élevée',
        'Diabetic_tratamiento': 'Contrôle glycémique, injections intravitréennes, laser',
        'Diabetic_pronostico': 'La gestion précoce prévient la cécité',
        
        'DiscEdema_nombre': 'Œdème du Disque Optique',
        'DiscEdema_descripcion': 'Gonflement dû à la pression intracrânienne',
        'DiscEdema_gravedad': 'Élevée',
        'DiscEdema_tratamiento': 'Urgent: réduire la pression intracrânienne',
        'DiscEdema_pronostico': 'Dépend de la cause sous-jacente',
        
        'Glaucoma_nombre': 'Glaucome',
        'Glaucoma_descripcion': 'Dommages du nerf optique',
        'Glaucoma_gravedad': 'Élevée',
        'Glaucoma_tratamiento': 'Gouttes hypotensives, laser, chirurgie',
        'Glaucoma_pronostico': 'Progression lente avec traitement',
        
        'Healthy_nombre': 'Œil Sain',
        'Healthy_descripcion': 'Aucune pathologie détectée',
        'Healthy_gravedad': 'Normal',
        'Healthy_tratamiento': 'Examens préventifs annuels',
        'Healthy_pronostico': 'Excellent',
        
        'MacularScar_nombre': 'Cicatrice Maculaire',
        'MacularScar_descripcion': 'Tissu cicatriciel dans la macula',
        'MacularScar_gravedad': 'Modérée',
        'MacularScar_tratamiento': 'Rééducation visuelle, aides optiques',
        'MacularScar_pronostico': 'Stable, vision centrale affectée',
        
        'Myopia_nombre': 'Myopie',
        'Myopia_descripcion': 'Erreur de réfraction',
        'Myopia_gravedad': 'Légère',
        'Myopia_tratamiento': 'Lentilles correctives, chirurgie réfractive',
        'Myopia_pronostico': 'Excellent avec correction',
        
        'Pterygium_nombre': 'Ptérygion',
        'Pterygium_descripcion': 'Croissance anormale sur la cornée',
        'Pterygium_gravedad': 'Légère',
        'Pterygium_tratamiento': 'Observation, chirurgie si affecte la vision',
        'Pterygium_pronostico': 'Bon, peut récidiver post-chirurgie',
        
        'RetinalDetachment_nombre': 'Décollement de Rétine',
        'RetinalDetachment_descripcion': 'Urgence: séparation rétinienne',
        'RetinalDetachment_gravedad': 'Critique',
        'RetinalDetachment_tratamiento': 'URGENT: chirurgie immédiate',
        'RetinalDetachment_pronostico': 'Bon si traité dans <24-48h',
        
        'Retinitis_nombre': 'Rétinite Pigmentaire',
        'Retinitis_descripcion': 'Dégénérescence progressive',
        'Retinitis_gravedad': 'Élevée',
        'Retinitis_tratamiento': 'Suppléments, implants rétiniens',
        'Retinitis_pronostico': 'Progressif, recherche active',
        
        # === ARCHITECTURES CNN ===
        'CNN_original_nombre': 'CNN MobileNetV2 Original',
        'CNN_original_descripcion': 'Votre modèle initial entraîné (70.44% précision)',
        'CNN_original_ventaja1': 'Votre modèle de base',
        'CNN_original_ventaja2': 'Connu',
        'CNN_original_ventaja3': 'Optimisé mobile',
        'CNN_original_tipo': 'Convolutions Séparables en Profondeur',
        'CNN_original_ventaja_principal': 'Efficacité computationnelle',
        
        'EfficientNet_nombre': 'EfficientNet-B0',
        'EfficientNet_descripcion': 'Architecture avec mise à l\'échelle composée équilibrée',
        'EfficientNet_ventaja1': 'Mise à l\'échelle composée',
        'EfficientNet_ventaja2': 'Équilibre précision/paramètres',
        'EfficientNet_ventaja3': 'État de l\'art',
        'EfficientNet_tipo': 'CNN à Mise à l\'Échelle Composée',
        'EfficientNet_ventaja_principal': 'Équilibre optimal précision/efficacité',
        
        'ResNet_nombre': 'ResNet-50 V2',
        'ResNet_descripcion': 'Réseau résiduel profond avec connexions de saut',
        'ResNet_ventaja1': 'Connexions résiduelles',
        'ResNet_ventaja2': 'Réseau profond',
        'ResNet_ventaja3': 'Stable',
        'ResNet_tipo': 'Réseau Résiduel',
        'ResNet_ventaja_principal': 'Capacité de représentation profonde',
        
        # Informations sur les classes (maladies)
        'normal': 'Normal',
        'moderate': 'Modérée',
        'high': 'Élevée',
        'critical': 'Critique',
        'mild': 'Légère',
        'unspecified': 'Non spécifiée',
        
        # Boutons et actions
        'cancel': 'Annuler',
        'accept': 'Accepter',
        'close': 'Fermer',
        'save': 'Sauvegarder',
        'load': 'Charger',
        'error': 'Erreur',
        'success': 'Succès',
        'warning': 'Avertissement',
        'info': 'Information',
        
        # Textes additionnels qui apparaissent dans le code
        'characteristics': 'Caractéristiques',
        'advantages': 'Avantages',
        'type': 'Type',
        'parameters_count': 'Paramètres',
        'main_advantage': 'Avantage principal',
        'year': 'Année',
        
        # Footer técnico - Funcionalidades estadísticas
        'statistical_features_title': '🔬 Nouvelles Fonctionnalités Statistiques:',
        'mcc_description': '📊 **Coefficient de Matthews (MCC)**: Métrique équilibrée pour classes déséquilibrées',
        'mcnemar_description': '🔬 **Test de McNemar**: Comparaison statistique rigoureuse entre modèles',
        'bootstrap_description': '📈 **Intervalles de Confiance Bootstrap**: Robustesse statistique (95% CI)',
        'confusion_matrices': '🎭 **Matrices de Confusion**: Analyse détaillée par classe',
        'statistical_reports': '📋 **Rapports Statistiques**: Exportation complète des résultats',
        
        # Ventajas competitivas
        'competitive_advantages_title': '🔬 Avantages Concurrentiels:',
        'specialized_diseases': '**10 maladies spécialisées** vs 4 de base des systèmes conventionnels',
        'multi_architecture': '**Analyse multi-architecture** avec comparaison simultanée de CNNs',
        'statistical_evaluation': '**Évaluation statistique rigoureuse** avec tests de signification',
        'professional_reports': '**Rapports PDF professionnels** avec analyse clinique et statistique',
        'complete_export': '**Exportation technique complète** (JSON, CSV, TXT) pour la recherche',
        'contextual_recommendations': '**Recommandations contextuelles** basées sur des preuves statistiques',
        
        # Arquitecturas implementadas
        'implemented_architectures_title': '🏗️ Architectures Implémentées:',
        'hybrid_cnn': '**🧠 CNN Hybride (MobileNetV2)**: Transfer Learning spécialisé',
        'efficientnet_desc': '**⚡ EfficientNet-B0**: Mise à l\'échelle composée équilibrée',
        'resnet_desc': '**🔗 ResNet-50 V2**: Connexions résiduelles profondes',
        
        # Métricas evaluadas
        'evaluated_metrics_title': '📊 Métriques Évaluées:',
        'precision_metric': '🎯 **Précision**: Confiance, MCC et consensus diagnostique',
        'speed_metric': '⚡ **Vitesse**: Temps d\'inférence optimisé',
        'efficiency_metric': '💾 **Efficacité**: Utilisation mémoire et évolutivité',
        'balance_metric': '🏆 **Équilibre**: Score général multi-critères',
        'significance_metric': '📈 **Signification**: Tests statistiques inférentiels',
        
        # Aplicaciones
        'applications_title': '🎯 Applications:',
        'clinical_application': '🏥 **Cliniques**: Diagnostic haute précision avec validation statistique',
        'mobile_application': '📱 **Mobiles**: Apps de télémédecine avec métriques robustes',
        'production_application': '🔄 **Production**: Systèmes hospitaliers évolutifs avec preuves statistiques',
        'research_application': '🔬 **Recherche**: Données complètes pour publications scientifiques',
        
        # Innovación
        'innovation_text': '**💡 Innovation**: Premier système combinant multiples architectures CNN avec analyse statistique inférentielle complète pour diagnostic oculaire spécialisé.',
        
        # Métodos estadísticos
        'statistical_methods_title': '📚 Méthodes Statistiques:',
        'mcc_method': '**MCC**: Coefficient de Corrélation de Matthews pour métriques équilibrées',
        'mcnemar_method': '**McNemar**: Test chi-carré pour comparaison de classificateurs',
        'bootstrap_method': '**Bootstrap**: Intervalles de confiance non paramétriques',
        'yates_correction': '**Correction de Yates**: Pour petits échantillons dans McNemar',
    },
}