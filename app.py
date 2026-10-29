import torch
import torch.nn as nn
from torchvision import models, transforms
import gradio as gr

# ZeroGPU-სთვის საჭიროა spaces.GPU დეკორატორი. სხვაგან უბრალოდ არაფერს აკეთებს.
try:
    import spaces
    gpu = spaces.GPU
except Exception:
    def gpu(f):
        return f

CLASSES = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound",
    "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair",
    "Chihuahua", "Egyptian Mau", "English Cocker Spaniel", "English Setter",
    "German Shorthaired", "Great Pyrenees", "Havanese", "Japanese Chin",
    "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard",
    "Samoyed", "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx",
    "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier",
]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("finetune_resnet18.pth", map_location="cpu"))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@gpu
def predict(img):
    if img is None:
        return {}
    x = preprocess(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, 3)
    return {CLASSES[i]: float(p) for p, i in zip(top.values, top.indices)}


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🐾 Pet Breed Classifier\n"
        "ატვირთეთ კატის ან ძაღლის ფოტო და მოდელი 37 ჯიშიდან ამოიცნობს მას.  \n"
        "**მოდელი:** ResNet18 (fine-tuned) · **მონაცემები:** Oxford-IIIT Pet · **Test accuracy:** 85.7%"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="ატვირთეთ ცხოველის ფოტო")
            btn = gr.Button("ამოიცანი ჯიში", variant="primary")
        with gr.Column():
            out = gr.Label(num_top_classes=3, label="ყველაზე სავარაუდო ჯიშები")
    btn.click(predict, inputs=inp, outputs=out)

demo.launch()
