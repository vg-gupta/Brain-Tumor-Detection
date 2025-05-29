# model_training.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import timm
from PIL import Image
import json
import random
from tqdm import tqdm

# Configuration
class Config:
    DATA_DIR = 'NeuroVision-master/Data/Training/'
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 10
    K_FOLDS = 5
    NUM_CLASSES = 4  # For 4 tumor classes
    MODEL_NAME = 'deit_small_patch16_224'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    SAVE_DIR = 'saved_models'
    
    # Augmentation parameters
    ROTATION_RANGE = 15
    BRIGHTNESS_RANGE = (0.9, 1.1)
    CONTRAST_RANGE = (0.9, 1.1)
    NOISE_STD = 0.01

# Set random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)

# Named functions for transforms (replacing lambdas)
def apply_brightness_contrast(img):
    brightness_factor = random.uniform(*Config.BRIGHTNESS_RANGE)
    contrast_factor = random.uniform(*Config.CONTRAST_RANGE)
    img = transforms.functional.adjust_brightness(img, brightness_factor)
    img = transforms.functional.adjust_contrast(img, contrast_factor)
    return img

def apply_gaussian_noise(img):
    if isinstance(img, Image.Image):
        img = transforms.functional.to_tensor(img)
    noise = torch.randn_like(img) * Config.NOISE_STD
    noisy_img = img + noise
    return transforms.functional.to_pil_image(noisy_img.clamp(0, 1))

# Dataset class
class BrainTumorDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('L')
        image = transforms.functional.to_tensor(image)
        image = image.repeat(3, 1, 1)
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Transformations
def get_train_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(Config.ROTATION_RANGE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.9, 1.1)),
        transforms.Lambda(apply_brightness_contrast),
        transforms.Lambda(apply_gaussian_noise),
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Model class
class BrainTumorClassifier:
    def __init__(self, num_classes):
        self.model = timm.create_model(Config.MODEL_NAME, pretrained=True, num_classes=num_classes)
        self.model.to(Config.DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2, factor=0.5)
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Disable multiprocessing for Windows compatibility
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_acc
    
    def evaluate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average='weighted')
        epoch_recall = recall_score(all_labels, all_preds, average='weighted')
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)
        
        self.scheduler.step(epoch_acc)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'precision': epoch_precision,
            'recall': epoch_recall,
            'f1': epoch_f1,
            'confusion_matrix': cm.tolist()
        }
    
    def save_model(self, fold, metrics, class_names, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        model_path = os.path.join(save_dir, f'fold_{fold}_model.pth')
        torch.save(self.model.state_dict(), model_path)
        
        metadata = {
            'model_name': Config.MODEL_NAME,
            'fold': fold,
            'class_names': class_names,
            'training_params': {
                'batch_size': Config.BATCH_SIZE,
                'epochs': Config.EPOCHS,
                'optimizer': 'AdamW',
                'learning_rate': 1e-4,
                'scheduler': 'ReduceLROnPlateau',
                'criterion': 'CrossEntropyLoss'
            },
            'augmentation_settings': {
                'rotation_range': Config.ROTATION_RANGE,
                'brightness_range': Config.BRIGHTNESS_RANGE,
                'contrast_range': Config.CONTRAST_RANGE,
                'noise_std': Config.NOISE_STD,
                'horizontal_flip': True,
                'vertical_flip': True,
                'random_translation': True,
                'scaling_variations': True
            },
            'metrics': metrics
        }
        
        metadata_path = os.path.join(save_dir, f'fold_{fold}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

# Main training function
def main():
    print("Loading dataset...")
    all_file_paths = []
    all_labels = []
    class_names = ['glioma', 'meningioma', 'notumor' ,'pituitary']
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(Config.DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            raise ValueError(f"Class directory not found: {class_dir}")
        
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_file_paths.append(os.path.join(class_dir, filename))
                all_labels.append(class_idx)
    
    if len(all_file_paths) == 0:
        raise ValueError("No images found in the dataset directory.")
    
    print(f"Found {len(all_file_paths)} images across {len(class_names)} classes.")
    
    print("\nStarting K-Fold Cross Validation...")
    results = []
    kf = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=Config.SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_file_paths)):
        print(f'\n{"="*50}')
        print(f'Fold {fold + 1}/{Config.K_FOLDS}')
        print(f'{"="*50}')
        
        train_files = [all_file_paths[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_files = [all_file_paths[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        train_dataset = BrainTumorDataset(train_files, train_labels, transform=get_train_transform())
        val_dataset = BrainTumorDataset(val_files, val_labels, transform=get_val_transform())
        
        # Set num_workers=0 for Windows compatibility
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
        
        model = BrainTumorClassifier(Config.NUM_CLASSES)
        best_val_acc = 0.0
        fold_metrics = []
        
        for epoch in range(Config.EPOCHS):
            print(f'\nEpoch {epoch + 1}/{Config.EPOCHS}')
            train_loss, train_acc = model.train_epoch(train_loader)
            val_results = model.evaluate(val_loader)
            
            print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_results["loss"]:.4f} | Val Acc: {val_results["accuracy"]:.4f}')
            print(f'Val Precision: {val_results["precision"]:.4f} | Val Recall: {val_results["recall"]:.4f}')
            print(f'Val F1: {val_results["f1"]:.4f}')
            
            fold_metrics.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_results['loss'],
                'val_accuracy': val_results['accuracy'],
                'val_precision': val_results['precision'],
                'val_recall': val_results['recall'],
                'val_f1': val_results['f1'],
                'val_confusion_matrix': val_results['confusion_matrix']
            })
            
            if val_results['accuracy'] > best_val_acc:
                best_val_acc = val_results['accuracy']
                model.save_model(fold, fold_metrics, class_names, Config.SAVE_DIR)
        
        results.append(fold_metrics[-1])
    
    # Print and save final results
    print("\n\nFinal Results Across All Folds:")
    print("-" * 70)
    print(f"{'Fold':<10}{'Val Acc':<15}{'Val F1':<15}{'Val Precision':<15}{'Val Recall':<15}")
    print("-" * 70)
    
    for i, res in enumerate(results):
        print(f"{i+1:<10}{res['val_accuracy']:.4f}{'':<5}"
              f"{res['val_f1']:.4f}{'':<5}"
              f"{res['val_precision']:.4f}{'':<5}"
              f"{res['val_recall']:.4f}")
    
    avg_acc = np.mean([r['val_accuracy'] for r in results])
    avg_f1 = np.mean([r['val_f1'] for r in results])
    avg_precision = np.mean([r['val_precision'] for r in results])
    avg_recall = np.mean([r['val_recall'] for r in results])
    
    print("\nAverage Across All Folds:")
    print(f"Accuracy: {avg_acc:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    
    overall_results = {
        'class_names': class_names,
        'fold_results': results,
        'average_metrics': {
            'accuracy': avg_acc,
            'f1_score': avg_f1,
            'precision': avg_precision,
            'recall': avg_recall
        }
    }
    
    with open(os.path.join(Config.SAVE_DIR, 'overall_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=4)

if __name__ == '__main__':
    main()