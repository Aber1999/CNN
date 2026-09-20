import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image


# ==========================================
# 1. تحميل وتحضير بيانات MNIST
# ==========================================

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)


# ==========================================
# 2. بناء نموذج CNN
# ==========================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc = nn.Linear(
            16 * 13 * 13,
            10
        )

    def forward(self, x):

        x = self.conv(x)

        x = torch.relu(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


# إنشاء النموذج
model = CNN()


# ==========================================
# 3. Loss و Optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 4. التدريب
# ==========================================

epochs = 10

print("Training started...")

for epoch in range(epochs):

    total_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {average_loss:.4f}"
    )


print("Training finished!")


# ==========================================
# 5. إدخال الصورة
# ==========================================

print()
print("====================================")
print("Enter the path of your image")
print("Example:")
print(r"E:\projcte\7.jpg.")
print("====================================")

image_path = input("Image path: ").strip()


# ==========================================
# 6. تجهيز الصورة
# ==========================================

try:

    image = Image.open(image_path)

    # تحويل الصورة إلى Grayscale
    image = image.convert("L")

    # تحويلها إلى 28x28 مثل MNIST
    image = image.resize((28, 28))

    # تحويل الصورة إلى Tensor
    image_tensor = transforms.ToTensor()(image)

    # إضافة Batch Dimension
    image_tensor = image_tensor.unsqueeze(0)


    # ==========================================
    # 7. التنبؤ
    # ==========================================

    model.eval()

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    # ==========================================
    # 8. عرض النتيجة
    # ==========================================

    print()
    print("====================================")
    print("Prediction:", prediction.item())
    print(
        "Confidence:",
        f"{confidence.item() * 100:.2f}%"
    )
    print("====================================")


except FileNotFoundError:

    print()
    print("ERROR: Image file was not found.")
    print("Check the image path.")


except Exception as e:

    print()
    print("ERROR:", e)