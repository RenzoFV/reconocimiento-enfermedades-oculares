import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Importaciones modernas de TensorFlow
import tensorflow as tf
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

class EyeDiseaseTrainer:
    def __init__(self):
        # Mapeo de clases en español
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
        
        # Configuración del modelo
        self.img_height = 224
        self.img_width = 224
        self.batch_size = 64
        self.validation_split = 0.15
        
    def analyze_dataset(self, dataset_path):
        """Analiza el dataset antes del entrenamiento"""
        print("📊 Analizando dataset...")
        print("=" * 50)
        
        class_counts = {}
        total_images = 0
        
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                count = len(image_files)
                spanish_name = self.class_info.get(class_name, {}).get('nombre', class_name)
                class_counts[spanish_name] = count
                total_images += count
                print(f"✅ {spanish_name}: {count} imágenes")
        
        print("=" * 50)
        print(f"📈 Total de imágenes: {total_images}")
        print(f"📁 Número de clases: {len(class_counts)}")
        print(f"📊 Promedio por clase: {total_images/len(class_counts):.0f}")
        
        # Verificar balance
        counts = list(class_counts.values())
        min_count, max_count = min(counts), max(counts)
        balance_ratio = min_count / max_count
        
        if balance_ratio < 0.5:
            print("⚠️  Dataset desbalanceado - considera técnicas de balanceo")
        else:
            print("✅ Dataset relativamente balanceado")
        
        print("=" * 50)
        return class_counts, total_images
    
    def prepare_data(self, dataset_path):
        """Prepara los generadores de datos"""
        print("🔄 Preparando generadores de datos...")
        
        # Data augmentation para entrenamiento
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
        
        # Solo rescaling para validación
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split
        )
        
        # Generador de entrenamiento
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Generador de validación
        validation_generator = val_datagen.flow_from_directory(
            dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"✅ Datos de entrenamiento: {train_generator.samples} imágenes")
        print(f"✅ Datos de validación: {validation_generator.samples} imágenes")
        
        return train_generator, validation_generator
    
    def create_model(self, num_classes):
        """Crea el modelo con transfer learning"""
        print("🧠 Creando modelo con transfer learning...")
        
        # Modelo base preentrenado
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        # Fine-tuning: descongelar las últimas capas
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Construir modelo completo
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        # Mostrar resumen
        print("📋 Resumen del modelo:")
        model.summary()
        
        return model
    
    def train_model(self, dataset_path, epochs=20, save_path='eye_disease_model.h5'):
        """Entrena el modelo completo"""
        print("🚀 Iniciando entrenamiento...")
        print("=" * 50)
        
        # Analizar dataset
        self.analyze_dataset(dataset_path)
        
        # Preparar datos
        train_gen, val_gen = self.prepare_data(dataset_path)
        num_classes = len(train_gen.class_indices)
        
        # Crear modelo
        model = self.create_model(num_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=4,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar
        print(f"🎯 Entrenando por {epochs} épocas...")
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Guardar clases para la aplicación principal
        class_indices = train_gen.class_indices
        np.save('class_indices.npy', class_indices)
        
        print("=" * 50)
        print(f"✅ Modelo guardado como: {save_path}")
        print(f"✅ Índices de clases guardados como: class_indices.npy")
        
        # Mostrar métricas finales
        self.plot_training_results(history)
        self.evaluate_model(model, val_gen)
        
        return model, history
    
    def plot_training_results(self, history):
        """Visualiza los resultados del entrenamiento"""
        print("📈 Generando gráficos de entrenamiento...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Entrenamiento', color='blue')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validación', color='red')
        axes[0, 0].set_title('Precisión del Modelo')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Precisión')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Entrenamiento', color='blue')
        axes[0, 1].plot(history.history['val_loss'], label='Validación', color='red')
        axes[0, 1].set_title('Pérdida del Modelo')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Pérdida')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-k accuracy
        axes[1, 0].plot(history.history['top_k_categorical_accuracy'], label='Top-K Entrenamiento', color='green')
        axes[1, 0].plot(history.history['val_top_k_categorical_accuracy'], label='Top-K Validación', color='orange')
        axes[1, 0].set_title('Top-K Categorical Accuracy')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Top-K Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (si está disponible)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], color='purple')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nno disponible', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Mostrar métricas finales
        final_acc = history.history['val_accuracy'][-1]
        final_loss = history.history['val_loss'][-1]
        print(f"📊 Precisión final en validación: {final_acc:.4f}")
        print(f"📊 Pérdida final en validación: {final_loss:.4f}")
    
    def evaluate_model(self, model, val_gen):
        """Evalúa el modelo en el conjunto de validación"""
        print("🔬 Evaluando modelo en conjunto de validación...")
        
        # Resetear generador
        val_gen.reset()
        
        # Predicciones
        predictions = model.predict(val_gen, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Etiquetas verdaderas
        true_classes = val_gen.classes
        class_labels = list(val_gen.class_indices.keys())
        
        # Reporte de clasificación
        print("\n📋 Reporte de Clasificación:")
        print("=" * 80)
        report = classification_report(true_classes, predicted_classes, 
                                     target_names=class_labels, 
                                     output_dict=True)
        
        # Mostrar métricas por clase
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                spanish_name = self.class_info.get(class_name, {}).get('nombre', class_name)
                print(f"{spanish_name:30} | Precisión: {metrics['precision']:.3f} | "
                      f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")
        
        print("=" * 80)
        print(f"Precisión general: {report['accuracy']:.4f}")
        print(f"F1-score promedio: {report['weighted avg']['f1-score']:.4f}")

def main():
    """Función principal para entrenar el modelo"""
    print("👁️ ENTRENADOR DE CLASIFICADOR DE ENFERMEDADES OCULARES")
    print("=" * 60)
    
    # Configuración
    DATASET_PATH = "./Dataset"  # Cambia esta ruta si es necesario
    EPOCHS = 25
    MODEL_NAME = "eye_disease_model.h5"
    
    # Verificar que existe el dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: No se encontró el dataset en {DATASET_PATH}")
        print("📁 Asegúrate de que la carpeta Dataset esté en el directorio actual")
        return
    
    # Crear entrenador
    trainer = EyeDiseaseTrainer()
    
    # Entrenar modelo
    try:
        model, history = trainer.train_model(
            dataset_path=DATASET_PATH,
            epochs=EPOCHS,
            save_path=MODEL_NAME
        )
        
        print("\n🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print("=" * 60)
        print(f"✅ Modelo guardado: {MODEL_NAME}")
        print(f"✅ Índices de clases: class_indices.npy")
        print(f"✅ Gráficos: training_results.png")
        print("\n🚀 Ahora puedes ejecutar 'streamlit run app.py' para usar la aplicación")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {str(e)}")
        print("🔧 Verifica que todas las dependencias están instaladas correctamente")

if __name__ == "__main__":
    main()