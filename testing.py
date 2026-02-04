import torch
import clip
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
MODEL_PATH = "image_edit_generator.pth"

# -------- LOAD CLIP --------
clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# -------- GENERATOR--------
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.txt = nn.Linear(512, 256)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, img, text_emb):
        x = self.enc(img)
        t = self.txt(text_emb).unsqueeze(-1).unsqueeze(-1)
        t = t.expand(-1, -1, x.shape[2], x.shape[3])
        return self.dec(torch.cat([x, t], dim=1))

# -------- LOAD MODEL --------
G = Generator().to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()

# -------- IMAGE TRANSFORM --------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# -------- LOAD IMAGE --------
image_path = "C:\\Users\\Aditya\\Desktop\\Guitar_AI_Project\\synthetic_dataset\\input\\1185_0.jpg"
image = Image.open(image_path).convert("RGB")
inp = transform(image).unsqueeze(0).to(DEVICE)

# -------- TEXT PROMPT --------
prompt = "edge-detected sketch"

with torch.no_grad():
    tokens = clip.tokenize([prompt]).to(DEVICE)
    text_emb = clip_model.encode_text(tokens).float()
    out = G(inp, text_emb)

# -------- POSTPROCESS --------
out = (out.squeeze(0).cpu() + 1) / 2
out = out.permute(1, 2, 0).numpy()

# -------- SHOW --------
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Input")
plt.imshow(image)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Edited")
plt.imshow(out)
plt.axis("off")

plt.show()