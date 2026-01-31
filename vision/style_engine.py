import torch
import cv2
import numpy as np
from torchvision import transforms

class StyleEngine:

    def __init__(self, model_path):
        self.device = "cpu"
        self.model = torch.load(model_path,
                                map_location=self.device)
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

    def stylize(self, frame):

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (256,256))

        t = self.tf(img).unsqueeze(0)

        with torch.no_grad():
            out = self.model(t)

        out = out.squeeze().clamp(0,255).numpy()
        out = out.transpose(1,2,0).astype(np.uint8)

        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        return out

