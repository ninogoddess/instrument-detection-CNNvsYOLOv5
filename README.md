## Preprocesamiento de Imágenes

El preprocesamiento de las imágenes es un paso crítico en cualquier proyecto de visión artificial, ya que condiciona la calidad de las características extraídas y, en consecuencia, el desempeño del modelo. En este proyecto se utilizaron técnicas de preprocesamiento específicas para cada enfoque: CNN personalizada y YOLOv5. A continuación, se describen las técnicas empleadas en cada caso, justificando su elección y comparando su impacto.

---

### Para el modelo CNN

El preprocesamiento de imágenes en la red neuronal convolucional (CNN) se implementó dentro del módulo `data_loader.py`. Las transformaciones aplicadas son las siguientes:

- **Redimensionamiento (`Resize`)**:  
  Todas las imágenes se escalaron a 224x224 píxeles. Este tamaño se eligió por ser un estándar compatible con muchas arquitecturas CNN y porque permite una buena relación entre precisión y velocidad de entrenamiento.

- **Conversión a tensor (`ToTensor`)**:  
  Convierte las imágenes de PIL a tensores PyTorch con formato `[C, H, W]`, necesario para el entrenamiento.

- **Normalización (`Normalize`)**:  
  Se utilizó una normalización por canal RGB con media `[0.5, 0.5, 0.5]` y desviación estándar `[0.5, 0.5, 0.5]`, lo que reescala los valores de píxeles al rango `[-1, 1]`. Esto facilita la convergencia del modelo durante el entrenamiento.

**Ventajas**:
- Simplicidad y velocidad en el preprocesamiento.
- Reducción del sobreajuste mediante normalización adecuada.

**Desventajas**:
- No se incluyeron técnicas de data augmentation, lo que limita la capacidad del modelo para generalizar ante imágenes con variaciones de posición, escala o iluminación.

---

### Para el modelo YOLOv5

En el caso de YOLOv5, el preprocesamiento se define tanto en el código como en el archivo `hyp.scratch-med.yaml`, el cual fue utilizado en el entrenamiento. Las técnicas aplicadas son más variadas y avanzadas:

- **Resize**:  
  Las imágenes se escalaron a 640x640 píxeles, tamaño ideal para mantener un equilibrio entre precisión y rendimiento en detección.

- **Normalización interna**:  
  YOLOv5 realiza una normalización interna al procesar las imágenes (valores en `[0, 1]`), optimizada para su backbone preentrenado.

- **Data Augmentation avanzada**:
  - `hsv_h`, `hsv_s`, `hsv_v`: Ajustes aleatorios de matiz, saturación y valor para simular diferentes condiciones de iluminación.
  - `translate`, `scale`: Transformaciones de posición y escala para simular distintos encuadres y tamaños.
  - `fliplr`: Volteo horizontal con 50% de probabilidad.
  - `mosaic`: Técnica que combina 4 imágenes en una sola para mejorar la diversidad del contexto y objetos.
  - `mixup`: Superposición parcial de imágenes, útil para mejorar la generalización en clases balanceadas.

**Ventajas**:
- Aumenta significativamente la robustez y generalización del modelo.
- Mejora la detección en condiciones reales variadas (iluminación, ángulos, posiciones).

**Desventajas**:
- Entrenamiento más lento debido a la complejidad de las transformaciones.
- Mayor dependencia de una correcta anotación y preprocesamiento para evitar errores de detección.

---

### Comparación y conclusiones

El uso de un preprocesamiento básico en la CNN limita su capacidad para adaptarse a variaciones comunes en imágenes reales, como se evidenció en las pruebas realizadas. En contraste, YOLOv5 mostró una notable superioridad, especialmente en escenarios difíciles o con múltiples instrumentos, gracias a sus técnicas de data augmentation más agresivas y diversas.

Esto explica en parte por qué YOLO logró mejores resultados en las pruebas comparativas:  
- Mayor precisión en la detección de instrumentos presentes.  
- Menor tasa de falsos positivos.  
- Mayor robustez ante condiciones adversas.

Se concluye que el preprocesamiento debe adaptarse no solo al modelo, sino también al entorno y al tipo de imágenes utilizadas, siendo un factor determinante en el rendimiento final del sistema.
