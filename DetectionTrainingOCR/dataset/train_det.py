#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Add Rotated Images (Data Augmentation) ~~~~~~~~~~~~~~~~~~

import cv2, os, random, json
from glob import glob
from pathlib import Path

img_dir = "datasets/train"
for img_path in glob(f"{img_dir}/*.jpg"):
    img = cv2.imread(img_path)
    angle = random.choice([15, 30, 45, 90, 135, 180])
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite(img_path.replace(".jpg", f"_rot{angle}.jpg"), rotated)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training Configuration ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import torch
from doctr.datasets import DetectionDataset
from doctr.models import detection
from doctr.models.detection import db_resnet50
from doctr.transforms import Resize
from doctr.utils.metrics import LocalizationConfusion
from doctr import transforms as T

# 1. Define data pipeline
train_set = DetectionDataset(
    img_folder="datasets/train",
    label_path="datasets/train_labels.json",
    img_transforms=T.Compose([
        T.RandomRotate(10),  # augment rotation
        T.Resize((1024, 1024))
    ])
)

val_set = DetectionDataset(
    img_folder="datasets/val",
    label_path="datasets/val_labels.json",
    img_transforms=T.Resize((1024, 1024))
)

# 2. Model: DBNet (resnet50 backbone)
model = db_resnet50(pretrained=False)

# 3. Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = detection.DBPostProcessor()
metric = LocalizationConfusion(iou_thresh=0.5)

# 4. Training loop (simplified)
for epoch in range(10):  # number of epochs
    model.train()
    for imgs, targets in train_set:
        out = model(imgs)
        loss = criterion(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")

# 5. Save weights
torch.save(model.state_dict(), "dbnet_custom.pth")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from doctr.models.detection import db_resnet50
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page
import matplotlib.pyplot as plt

model = db_resnet50(pretrained=False)
model.load_state_dict(torch.load("dbnet_custom.pth"))
model.eval()

doc = DocumentFile.from_images(["test_image.jpg"])
preds = model(doc)

fig, ax = plt.subplots()
ax = visualize_page(preds.pages[0], image=doc[0], ax=ax)
plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~ 💾 STEP 6: Save / Reuse Trained Model ~~~~~~~~~~~~~~~~

model = db_resnet50(pretrained=False)
model.load_state_dict(torch.load("dbnet_custom.pth"))
model.eval()
