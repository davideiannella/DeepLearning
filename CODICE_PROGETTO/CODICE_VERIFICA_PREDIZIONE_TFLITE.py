import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# CIFAR-10 labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']

# Preprocessing per CIFAR10 compatibile con il tuo modello
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 🔹 Carica CIFAR-10 test
cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
image, label = cifar10[7]  # Cambia indice per testare altre immagini

# 🔹 Prepara input per TFLite
input_data = image.unsqueeze(0).numpy().astype(np.float32)

# 🔹 Carica modello da Drive
model_path = "/content/drive/My Drive/vit_tiny_student.tflite"  # <-- cambia il path se necessario
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 🔹 Ottieni input/output details
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# 🔹 Esegui inferenza
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_index)

# 🔹 Risultato
pred = np.argmax(output[0])
true = label
print(f"✅ Predizione: {labels[pred]} ({output[0][pred]:.2f})")
print(f"🏷️ Classe Reale: {labels[true]}")

# Mostra immagine
unnorm = image.permute(1, 2, 0).numpy() * 0.5 + 0.5
plt.imshow(unnorm)
plt.axis("off")
plt.title(f"Pred: {labels[pred]} | True: {labels[true]}")
plt.show()
