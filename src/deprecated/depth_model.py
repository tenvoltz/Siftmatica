import torch
import cv2


class DepthEstimator:
    def __init__(self):
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.model.eval()

        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def predict(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        return depth
