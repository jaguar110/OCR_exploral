from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import matplotlib.pyplot as plt
from doctr.utils.visualization import visualize_page
import json

# -----------------------------
# 1️⃣ Load your document (image or PDF)
# -----------------------------
doc = DocumentFile.from_images("receipt.jpeg")
# For PDFs, uncomment the next line:
# doc = DocumentFile.from_pdf("21F1004013.pdf")
print(f"Loaded {len(doc)} pages")

# -----------------------------
# 2️⃣ Load pretrained OCR model
# -----------------------------
model = ocr_predictor(pretrained=True)

# -----------------------------
# 3️⃣ Run OCR
# -----------------------------
result = model(doc)

# -----------------------------
# 4️⃣ Print structure summary
# -----------------------------
for i, page in enumerate(result.pages, start=1):
    num_blocks = len(page.blocks)
    num_lines = sum(len(block.lines) for block in page.blocks)
    num_words = sum(len(line.words) for block in page.blocks for line in block.lines)
    print(f"Page {i}: {num_blocks} blocks, {num_lines} lines, {num_words} words")

# -----------------------------
# 5️⃣ Print recognized text
# -----------------------------
print("\nRecognized Text:\n" + "-" * 30)
for page in result.pages:
    for block in page.blocks:
        for line in block.lines:
            print(" ".join([word.value for word in line.words]))

# -----------------------------
# 6️⃣ Save results manually to JSON
# -----------------------------
# In v1.0, there's no built-in serializer, so we build a dict manually
ocr_dict = {
    "pages": []
}

for page_idx, page in enumerate(result.pages, start=1):
    page_data = {"page": page_idx, "blocks": []}
    for block in page.blocks:
        block_data = {"lines": []}
        for line in block.lines:
            words = [{"value": w.value, "geometry": w.geometry} for w in line.words]
            block_data["lines"].append({"words": words})
        page_data["blocks"].append(block_data)
    ocr_dict["pages"].append(page_data)

# Save to JSON
with open("ocr_output.json", "w", encoding="utf-8") as f:
    json.dump(ocr_dict, f, indent=2, ensure_ascii=False)

print("\n✅ OCR results saved to: ocr_output.json")

# -----------------------------
# 7️⃣ Visualize bounding boxes (UNIVERSAL FIX)
# -----------------------------

print("\n🖼️ Generating visualization...")

# 1️⃣ Get the first OCR page
page = result.pages[0]

# 2️⃣ Get the original image (works for both image and PDF)
if isinstance(doc, list):
    image = doc[0]
elif hasattr(doc, "as_images"):
    image = doc.as_images()[0]
else:
    raise TypeError("Unsupported document type for visualization")

# 3️⃣ Convert the Page object to a dict
page_dict = page.export()

# 4️⃣ Create a figure (no ax passed to visualize_page)
fig = plt.figure(figsize=(10, 10))

# 5️⃣ Draw bounding boxes
visualize_page(page_dict, image=image)

plt.axis("off")
plt.tight_layout()
plt.savefig("ocr_visualization.png", bbox_inches="tight")
print("✅ Visualization saved as ocr_visualization.png")

# 6️⃣ Optional display (skip safely if headless)
try:
    plt.show()
except Exception as e:
    print(f"(Non-interactive backend, skipping display: {e})")
