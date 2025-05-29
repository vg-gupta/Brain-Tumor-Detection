import torch
import torch.nn as nn
import timm
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

class BrainTumorClassifier:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = None

    def create_model(self, num_classes):
        """Create a DeiT model with the specified number of classes"""
        model = timm.create_model("deit_small_patch16_224", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    def load_model(self):
        """Load the model and class names"""
        if self.model is None or self.class_names is None:
            # Load saved model data
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.class_names = checkpoint["class_names"]

            # Create a new model with the correct number of classes
            self.model = self.create_model(len(self.class_names))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"Model loaded from {self.model_path}")
            print(f"Classes: {self.class_names}")

        return self.model, self.class_names

    def preprocess_image(self, image):
        """Preprocess an image for inference"""
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        # Preprocess for model
        image_tensor = transform(image)

        # Original image for display
        display_image = np.array(image.resize((224, 224))) / 255.0

        return image_tensor, display_image

    def predict(self, image_tensor):
        """Run inference on a preprocessed image tensor"""
        model, class_names = self.load_model()

        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Get top prediction
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = class_names[predicted_class]
        confidence = probabilities[predicted_class].item()

        # Get all class probabilities
        all_probs = {class_names[i]: prob.item() for i, prob in enumerate(probabilities)}

        return predicted_label, confidence, all_probs

    def get_visualization(self, image, predicted_label, confidence, all_probs):
        """Generate a visualization of the prediction and return as base64 string"""
        plt.figure(figsize=(12, 6))

        # Plot image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2%}")
        plt.axis("off")

        # Plot probability bar chart
        plt.subplot(1, 2, 2)
        classes = list(all_probs.keys())
        probs = list(all_probs.values())

        # Sort by probability in descending order
        sorted_indices = np.argsort(probs)[::-1]
        classes = [classes[i] for i in sorted_indices]
        probs = [probs[i] for i in sorted_indices]

        bar_colors = ["#6366f1" if p == max(probs) else "#e2e8f0" for p in probs]

        plt.barh(classes, probs, color=bar_colors)
        plt.xlabel("Probability")
        plt.title("Class Probabilities")
        plt.tight_layout()

        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)

        # Convert to base64
        plot_base64 = base64.b64encode(buf.getvalue()).decode()
        return plot_base64 