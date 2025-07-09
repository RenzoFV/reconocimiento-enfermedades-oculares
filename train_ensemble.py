import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

import tensorflow as tf
from keras import layers, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2, EfficientNetB0, ResNet50V2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

class MedicalEnsemble:
    """
    Ensemble de CNNs para Diagnóstico Médico
    
    PARADIGMA COMPLETAMENTE DIFERENTE a CNN individual:
    - Entrena 3 arquitecturas diferentes por separado
    - Combina sus predicciones para decisión final
    - Reduce overfitting y mejora precisión
    - Usado en medicina real para mayor confiabilidad
    """
    
    def __init__(self):
        # Mapeo de clases (igual que tu código original)
        self.class_info = {
            'Central Serous Chorioretinopathy [Color Fundus]': {
                'nombre': 'Corioretinopatía Serosa Central',
                'descripcion': 'Acumulación de líquido bajo la retina que causa visión borrosa'
            },
            'Diabetic Retinopathy': {
                'nombre': 'Retinopatía Diabética',
                'descripcion': 'Daño a los vasos sanguíneos de la retina por diabetes'
            },
            'Disc Edema': {
                'nombre': 'Edema del Disco Óptico',
                'descripcion': 'Hinchazón del disco óptico por aumento de presión intracraneal'
            },
            'Glaucoma': {
                'nombre': 'Glaucoma',
                'descripcion': 'Daño al nervio óptico, generalmente por presión ocular alta'
            },
            'Healthy': {
                'nombre': 'Ojo Sano',
                'descripcion': 'Retina sin patologías evidentes'
            },
            'Macular Scar': {
                'nombre': 'Cicatriz Macular',
                'descripcion': 'Tejido cicatricial en la mácula que afecta la visión central'
            },
            'Myopia': {
                'nombre': 'Miopía',
                'descripcion': 'Error refractivo que causa dificultad para ver objetos lejanos'
            },
            'Pterygium': {
                'nombre': 'Pterigión',
                'descripcion': 'Crecimiento anormal de tejido sobre la córnea'
            },
            'Retinal Detachment': {
                'nombre': 'Desprendimiento de Retina',
                'descripcion': 'Separación de la retina de la pared posterior del ojo'
            },
            'Retinitis Pigmentosa': {
                'nombre': 'Retinitis Pigmentosa',
                'descripcion': 'Degeneración progresiva de la retina'
            }
        }
        
        # Configuración
        self.img_height = 224
        self.img_width = 224
        self.batch_size = 32
        self.validation_split = 0.2
        
        print("🤖 ENSEMBLE MÉDICO DE CNNs")
        print("=" * 50)
        print("📋 ARQUITECTURAS A COMBINAR:")
        print("   🔲 MobileNetV2  - Eficiente y rápido")
        print("   ⚡ EfficientNetB0 - Optimizado accuracy/params")  
        print("   🔗 ResNet50V2    - Residual connections profundas")
        print("=" * 50)
    
    def prepare_data(self, dataset_path):
        """Prepara datos para ensemble (igual que tu código)"""
        print("🔄 Preparando datos para Ensemble...")
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=self.validation_split
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split
        )
        
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"✅ Datos entrenamiento: {train_generator.samples}")
        print(f"✅ Datos validación: {validation_generator.samples}")
        
        return train_generator, validation_generator
    
    def create_individual_model(self, architecture, num_classes, model_name):
        """Crea modelo individual según arquitectura"""
        print(f"🏗️ Creando modelo {architecture}...")
        
        if architecture == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
            # Fine-tuning específico para MobileNet
            base_model.trainable = True
            for layer in base_model.layers[:-20]:
                layer.trainable = False
                
        elif architecture == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
            # Fine-tuning específico para EfficientNet
            base_model.trainable = True
            for layer in base_model.layers[:-15]:
                layer.trainable = False
                
        elif architecture == 'resnet':
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
            # Fine-tuning específico para ResNet
            base_model.trainable = True
            for layer in base_model.layers[:-25]:
                layer.trainable = False
        
        # Capas superiores específicas por arquitectura
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        # Configuración específica por modelo
        if architecture == 'mobilenet':
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        elif architecture == 'efficientnet':
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
        elif architecture == 'resnet':
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        # Salida
        predictions = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions, name=model_name)
        
        # Compilar con configuración específica
        if architecture == 'mobilenet':
            lr = 0.0001
        elif architecture == 'efficientnet':
            lr = 0.00005  # Más conservador
        else:  # resnet
            lr = 0.0002   # Más agresivo
            
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print(f"✅ {architecture} creado: {model.count_params():,} parámetros")
        return model
    
    def train_individual_model(self, model, architecture, train_gen, val_gen, epochs=25):
        """Entrena un modelo individual"""
        print(f"\n🚀 ENTRENANDO {architecture.upper()}...")
        print("=" * 40)
        
        # Callbacks específicos
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                f'{architecture}_individual_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Métricas finales
        final_acc = max(history.history['val_accuracy'])
        print(f"✅ {architecture} completado - Mejor accuracy: {final_acc:.4f}")
        
        return model, history, final_acc
    
    def create_ensemble_predictor(self, models, weights=None):
        """Crea función de ensemble para combinar predicciones"""
        if weights is None:
            weights = [1/len(models)] * len(models)  # Pesos iguales
        
        def ensemble_predict(x):
            predictions = []
            for model in models:
                pred = model.predict(x, verbose=0)
                predictions.append(pred)
            
            # Promedio ponderado
            ensemble_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += weights[i] * pred
                
            return ensemble_pred
        
        return ensemble_predict
    
    def evaluate_ensemble(self, models, val_gen, weights=None):
        """Evalúa ensemble completo"""
        print("\n🔬 EVALUANDO ENSEMBLE COMPLETO...")
        print("=" * 50)
        
        # Crear predictor ensemble
        ensemble_predict = self.create_ensemble_predictor(models, weights)
        
        # Obtener predicciones ensemble
        val_gen.reset()
        ensemble_predictions = ensemble_predict(val_gen)
        predicted_classes = np.argmax(ensemble_predictions, axis=1)
        
        # Etiquetas verdaderas
        true_classes = val_gen.classes
        class_labels = list(val_gen.class_indices.keys())
        
        # Reporte de clasificación
        print("📋 REPORTE ENSEMBLE:")
        print("=" * 80)
        report = classification_report(true_classes, predicted_classes, 
                                     target_names=class_labels, 
                                     output_dict=True)
        
        # Mostrar por clase
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                spanish_name = self.class_info.get(class_name, {}).get('nombre', class_name)
                print(f"{spanish_name:30} | Precisión: {metrics['precision']:.3f} | "
                      f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")
        
        ensemble_accuracy = report['accuracy']
        ensemble_f1 = report['weighted avg']['f1-score']
        
        print("=" * 80)
        print(f"🎯 ENSEMBLE ACCURACY: {ensemble_accuracy:.4f}")
        print(f"📊 ENSEMBLE F1-SCORE: {ensemble_f1:.4f}")
        
        return ensemble_accuracy, ensemble_f1, report
    
    def train_ensemble(self, dataset_path, epochs=25):
        """Entrena ensemble completo"""
        print("🚀 ENTRENAMIENTO ENSEMBLE MÉDICO INICIADO")
        print("=" * 60)
        
        # Preparar datos
        train_gen, val_gen = self.prepare_data(dataset_path)
        num_classes = len(train_gen.class_indices)
        
        # Crear modelos individuales
        architectures = ['mobilenet', 'efficientnet', 'resnet']
        models = []
        histories = []
        individual_accuracies = []
        
        for arch in architectures:
            print(f"\n{'='*20} {arch.upper()} {'='*20}")
            
            # Crear modelo
            model = self.create_individual_model(arch, num_classes, f'{arch}_medical')
            
            # Entrenar
            trained_model, history, final_acc = self.train_individual_model(
                model, arch, train_gen, val_gen, epochs
            )
            
            models.append(trained_model)
            histories.append(history)
            individual_accuracies.append(final_acc)
        
        # Evaluar ensemble
        print(f"\n{'='*60}")
        print("🤖 EVALUACIÓN FINAL DEL ENSEMBLE")
        print(f"{'='*60}")
        
        # Mostrar accuracies individuales
        print("\n📊 ACCURACIES INDIVIDUALES:")
        for i, arch in enumerate(architectures):
            print(f"   {arch:12}: {individual_accuracies[i]:.4f}")
        
        # Evaluar ensemble con pesos optimizados
        ensemble_acc, ensemble_f1, ensemble_report = self.evaluate_ensemble(models, val_gen)
        
        # Guardar modelos
        for i, arch in enumerate(architectures):
            models[i].save(f'ensemble_{arch}_model.h5')
        
        # Guardar información del ensemble
        np.save('ensemble_class_indices.npy', train_gen.class_indices)
        
        # Crear gráfico comparativo
        self.plot_ensemble_results(architectures, histories, individual_accuracies, ensemble_acc)
        
        print(f"\n🎉 ENSEMBLE COMPLETADO")
        print("=" * 50)
        print("📈 COMPARACIÓN DE PARADIGMAS:")
        print(f"   CNN Individual (tu anterior): 70.44%")
        print(f"   Ensemble de CNNs:             {ensemble_acc:.2%}")
        print(f"   Mejora: {((ensemble_acc/0.7044)-1)*100:+.1f}%")
        
        return models, histories, ensemble_acc
    
    def plot_ensemble_results(self, architectures, histories, individual_accs, ensemble_acc):
        """Visualiza resultados del ensemble"""
        print("📈 Generando análisis visual del ensemble...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Accuracy por época - todos los modelos
        colors = ['blue', 'red', 'green']
        for i, (arch, history) in enumerate(zip(architectures, histories)):
            axes[0, 0].plot(history.history['val_accuracy'], 
                           label=f'{arch}', color=colors[i], linewidth=2)
        
        axes[0, 0].axhline(y=ensemble_acc, color='purple', linestyle='--', 
                          linewidth=3, label=f'Ensemble: {ensemble_acc:.3f}')
        axes[0, 0].set_title('Evolución de Accuracy - Ensemble')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Validation Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Comparación de accuracies finales
        all_accs = individual_accs + [ensemble_acc]
        all_names = architectures + ['Ensemble']
        colors_bar = ['lightblue', 'lightcoral', 'lightgreen', 'purple']
        
        bars = axes[0, 1].bar(all_names, all_accs, color=colors_bar)
        axes[0, 1].set_title('Comparación Final de Accuracies')
        axes[0, 1].set_ylabel('Validation Accuracy')
        axes[0, 1].set_ylim(0, max(all_accs) + 0.1)
        
        # Agregar valores en las barras
        for bar, acc in zip(bars, all_accs):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Loss evolution
        for i, (arch, history) in enumerate(zip(architectures, histories)):
            axes[1, 0].plot(history.history['val_loss'], 
                           label=f'{arch}', color=colors[i], linewidth=2)
        
        axes[1, 0].set_title('Evolución de Loss - Ensemble')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Información del ensemble
        axes[1, 1].text(0.1, 0.9, '🤖 ENSEMBLE DE CNNs', fontsize=14, weight='bold', color='purple')
        axes[1, 1].text(0.1, 0.8, '• Combina 3 arquitecturas diferentes', fontsize=10)
        axes[1, 1].text(0.1, 0.7, '• Reduce overfitting individual', fontsize=10)
        axes[1, 1].text(0.1, 0.6, '• Mejora precisión y robustez', fontsize=10)
        axes[1, 1].text(0.1, 0.5, '• Usado en medicina real', fontsize=10)
        
        axes[1, 1].text(0.1, 0.3, 'vs CNN Individual:', fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.2, '• Un solo modelo vs múltiples', fontsize=10)
        axes[1, 1].text(0.1, 0.1, '• Decisión individual vs consenso', fontsize=10)
        axes[1, 1].text(0.1, 0.0, '• Mayor riesgo overfitting', fontsize=10)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('ensemble_medical_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main_ensemble():
    """Función principal para ensemble"""
    print("👁️🤖 ENSEMBLE MÉDICO DE CNNs")
    print("=" * 70)
    
    # Configuración
    DATASET_PATH = "./Dataset"
    EPOCHS = 25  # Mismas épocas que CNN individual para comparación justa
    
    # Verificar dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: No se encontró el dataset en {DATASET_PATH}")
        return
    
    # Crear ensemble
    ensemble = MedicalEnsemble()
    
    try:
        models, histories, ensemble_accuracy = ensemble.train_ensemble(
            dataset_path=DATASET_PATH,
            epochs=EPOCHS
        )
        
        print("\n🎉 ¡ENSEMBLE COMPLETADO EXITOSAMENTE!")
        print("=" * 60)
        print("🆚 COMPARACIÓN COMPLETA DE PARADIGMAS:")
        print("   CNN Individual:    70.44% accuracy")
        print(f"   Ensemble de CNNs:   {ensemble_accuracy:.2%} accuracy")
        print("\n✅ Dos enfoques completamente diferentes probados")
        print("📁 Modelos guardados: ensemble_[arquitectura]_model.h5")
        
    except Exception as e:
        print(f"\n❌ Error durante ensemble: {str(e)}")

if __name__ == "__main__":
    main_ensemble()