!pip install timm torchsummary


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
import timm
import time
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📦 Device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Subset CIFAR-10
np.random.seed(42)
train_idx = np.random.choice(len(trainset_full), 4000, replace=False)
test_idx = np.random.choice(len(testset_full), 1000, replace=False)

trainset = Subset(trainset_full, train_idx)
testset = Subset(testset_full, test_idx)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

print(f"🧠 Trainset: {len(trainset)}\n🧪 Testset: {len(testset)}")




teacher = torchvision.models.resnet18(weights='IMAGENET1K_V1')
teacher.fc = nn.Linear(teacher.fc.in_features, 10)
teacher = teacher.to(device)
teacher.eval()

student = timm.create_model('vit_tiny_patch16_224', pretrained=True)
student.head = nn.Linear(student.head.in_features, 10)
student = student.to(device)

summary(student, input_size=(3, 224, 224), device=str(device))



kd_loss = nn.KLDivLoss(reduction='batchmean')
ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(student.parameters(), lr=3e-4)

def train_distill(epochs=3):
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        start = time.time()

        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                t_logits = teacher(images)
            s_logits = student(images)

            loss = kd_loss(torch.log_softmax(s_logits, dim=1),
                           torch.softmax(t_logits, dim=1)) + 0.5 * ce_loss(s_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Accuracy calcolo
            _, predicted = torch.max(s_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(trainloader)
        accuracy = 100 * correct / total
        print(f"✅ Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%, Time = {time.time() - start:.2f}s")
        
        
        
        
 for name, module in student.named_modules():
    if hasattr(module, 'attn_drop'):
        module.attn_drop.p = 0.0
        
        
        
        
        
 def test_model(model, testloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"🎯 Test Accuracy: {accuracy:.2f}%")



# Dopo training e pruning
student.eval()  # Importante!
test_model(student, testloader)



import matplotlib.pyplot as plt

def show_prediction_with_metrics(index, model, dataset, class_names, device='cpu'):
    model.eval()
    image, label = dataset[index]
    image_tensor = image.unsqueeze(0).to(device)
    label_tensor = torch.tensor([label]).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        prob = torch.softmax(output, dim=1)[0, predicted.item()].item()
        ce = ce_loss(output, label_tensor).item()
        acc = float(predicted.item() == label)

        # Cattura i nomi classe
        predicted_class = class_names[predicted.item()]
        true_class = class_names[label]

    # Plot
    plt.figure(figsize=(10, 4))

    # Immagine originale
    plt.subplot(1, 2, 1)
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5 + 0.5).clip(0, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'🎯 Pred: {predicted_class}\n✅ True: {true_class}')

    # Grafico metrica
    plt.subplot(1, 2, 2)
    plt.bar(['Accuracy', 'Loss'], [acc * 100, ce])
    plt.ylim(0, 100)
    plt.title(f'📊 Confidence: {prob:.2f}')
    plt.ylabel('Percentuale / Valore')

    plt.tight_layout()
    plt.show()

    # ✅ Stampa risultati in console
    print("\n📢 RISULTATI DETTAGLIATI")
    print("-" * 30)
    print(f"✅ Classe reale      : {true_class}")
    print(f"🎯 Classe predetta   : {predicted_class}")
    print(f"📈 Accuratezza       : {acc * 100:.2f}%")
    print(f"📉 CrossEntropy Loss : {ce:.4f}")
    print(f"🔢 Confidenza Softmax: {prob:.2f}")
    
    
    
user_index = 42  # Inserisci un numero da 1 a 1000
show_prediction_with_metrics(user_index - 1, student, testset, trainset_full.classes, device=device)



!pip install -q ai-edge-torch

import torch
import timm
import ai_edge_torch

# Ricrea modello student
student = timm.create_model('vit_tiny_patch16_224', pretrained=False)
student.head = torch.nn.Linear(student.head.in_features, 10)

# Carica i pesi distillati
student.load_state_dict(torch.load("/content/drive/My Drive/vit_tiny_student_distilled.pth", map_location='cpu'))
student.eval()

# ❗️ ATTENZIONE: il dummy input dev'essere su CPU e NON tracciato
sample_input = (torch.randn(1, 3, 224, 224),)

# 🔁 Disabilita compilatori dinamici
with torch.no_grad():
    edge_model = ai_edge_torch.convert(student, sample_input)

# Salva in Drive
edge_model.export("/content/drive/My Drive/vit_tiny_student.tflite")
print("✅ Modello TFLite salvato correttamente.")



import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 🔹 Percorso modello TFLite salvato
tflite_model_path = "/content/drive/My Drive/vit_tiny_student.tflite"

# 🔹 Carica modello TFLite
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 🔹 Info input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 🔹 CIFAR-10 preprocessing compatibile con ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 🔹 Carica il dataset CIFAR-10 (test set)
cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
class_names = cifar10.classes

# 🔹 Scegli un'immagine da testare (es: 42)
idx = 42
image, label = cifar10[idx]
np_image = image.unsqueeze(0).numpy().astype(np.float32)

# 🔹 Imposta input
interpreter.set_tensor(input_details[0]['index'], np_image)

# 🔍 Inference
interpreter.invoke()

# 🔹 Ottieni output
output = interpreter.get_tensor(output_details[0]['index'])
pred_class = np.argmax(output)
pred_label = class_names[pred_class]
true_label = class_names[label]

# 🔸 Visualizza immagine e risultato
plt.imshow(np.transpose(image.numpy(), (1, 2, 0)) * 0.5 + 0.5)
plt.axis('off')
plt.title(f"🎯 Predetto: {pred_label} | ✅ Reale: {true_label}")
plt.show()

# 📢 Stampa dettagli
print(f"✅ Predizione TFLite: {pred_label}")
print(f"🎯 Classe reale     : {true_label}")
print(f"📈 Probabilità max  : {np.max(output):.2f}")