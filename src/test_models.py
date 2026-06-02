"""
Test script for wildlife poaching risk prediction models.

This script loads all trained models and tests them on random images from the dataset.
Each time you run the script, it uses different random images for testing.
"""

import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder

# Add project root to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

# Import model architectures from notebook (we'll define them here)
from torchvision import models

# Set random seed for reproducibility (but different each run)
random.seed()
np.random.seed(random.randint(0, 10000))
torch.manual_seed(random.randint(0, 10000))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Model architectures (copied from notebook)
class BaselineCNN(nn.Module):
    """Baseline CNN model for poaching risk classification."""
    def __init__(self, num_classes=3, input_size=224):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        flattened_size = (input_size // 16) ** 2 * 256
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ResNetClassifier(nn.Module):
    """ResNet-based classifier using transfer learning."""
    def __init__(self, num_classes=3, pretrained=True, model_name='resnet18'):
        super(ResNetClassifier, self).__init__()
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unknown model: {model_name}")
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class EfficientNetClassifier(nn.Module):
    """EfficientNet-based classifier."""
    def __init__(self, num_classes=3, model_name='efficientnet_b0'):
        super(EfficientNetClassifier, self).__init__()
        try:
            from torchvision.models import efficientnet_b0, efficientnet_b1
            if model_name == 'efficientnet_b0':
                self.backbone = efficientnet_b0(pretrained=True)
                num_features = self.backbone.classifier[1].in_features
            elif model_name == 'efficientnet_b1':
                self.backbone = efficientnet_b1(pretrained=True)
                num_features = self.backbone.classifier[1].in_features
            else:
                raise ValueError(f"Unknown EfficientNet model: {model_name}")
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        except ImportError:
            raise ImportError("EfficientNet requires torchvision >= 0.13.0")

    def forward(self, x):
        return self.backbone(x)


class CustomCNN(nn.Module):
    """Custom CNN architecture with batch normalization."""
    def __init__(self, num_classes=3, input_size=224):
        super(CustomCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        flattened_size = (input_size // 16) ** 2 * 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_model_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint and return training state."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    val_acc = checkpoint.get('val_acc', None)
    return epoch, loss, val_acc


def load_model_weights(model, filepath, device):
    """Load model weights (state_dict)."""
    model.load_state_dict(torch.load(filepath, map_location=device))


def load_all_models(device):
    """Load all trained models."""
    models_dict = {}
    IMAGE_SIZE = 224

    # Define transforms (same as validation/test transforms)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Baseline CNN
    try:
        baseline_model = BaselineCNN(num_classes=3, input_size=IMAGE_SIZE).to(device)
        baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
        checkpoint_path = 'models/baseline_cnn_checkpoint.pth'
        if Path(checkpoint_path).exists():
            load_model_checkpoint(baseline_model, baseline_optimizer, checkpoint_path, device)
            baseline_model.eval()
            models_dict['Baseline CNN'] = baseline_model
            print(f"✓ Loaded Baseline CNN")
        else:
            print(f"⚠ Baseline CNN checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"⚠ Error loading Baseline CNN: {e}")

    # ResNet18
    try:
        resnet_model = ResNetClassifier(num_classes=3, pretrained=False, model_name='resnet18').to(device)
        resnet_optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.0001)
        checkpoint_path = 'models/resnet18_checkpoint.pth'
        if Path(checkpoint_path).exists():
            load_model_checkpoint(resnet_model, resnet_optimizer, checkpoint_path, device)
            resnet_model.eval()
            models_dict['ResNet18'] = resnet_model
            print(f"✓ Loaded ResNet18")
        else:
            print(f"⚠ ResNet18 checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"⚠ Error loading ResNet18: {e}")

    # EfficientNet-B0
    try:
        efficientnet_model = EfficientNetClassifier(num_classes=3, model_name='efficientnet_b0').to(device)
        efficientnet_optimizer = torch.optim.Adam(efficientnet_model.parameters(), lr=0.0001)
        checkpoint_path = 'models/efficientnet_b0_checkpoint.pth'
        if Path(checkpoint_path).exists():
            load_model_checkpoint(efficientnet_model, efficientnet_optimizer, checkpoint_path, device)
            efficientnet_model.eval()
            models_dict['EfficientNet-B0'] = efficientnet_model
            print(f"✓ Loaded EfficientNet-B0")
        else:
            print(f"⚠ EfficientNet-B0 checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"⚠ Error loading EfficientNet-B0: {e}")

    # Custom CNN
    try:
        custom_model = CustomCNN(num_classes=3, input_size=IMAGE_SIZE).to(device)
        custom_optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.001)
        checkpoint_path = 'models/custom_cnn_checkpoint.pth'
        if Path(checkpoint_path).exists():
            load_model_checkpoint(custom_model, custom_optimizer, checkpoint_path, device)
            custom_model.eval()
            models_dict['Custom CNN'] = custom_model
            print(f"✓ Loaded Custom CNN")
        else:
            print(f"⚠ Custom CNN checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"⚠ Error loading Custom CNN: {e}")

    return models_dict, transform


def predict_image(model, image_tensor, device, class_names):
    """Make prediction on a single image."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence, probabilities[0].cpu().numpy()


def visualize_predictions(image, true_label, predictions_dict, class_names, img_filename):
    """Visualize image with predictions from all models."""
    num_models = len(predictions_dict)
    fig, axes = plt.subplots(1, num_models + 1, figsize=(4 * (num_models + 1), 4))

    # Show original image
    axes[0].imshow(image)
    axes[0].set_title(f'True Label: {true_label}', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Show predictions for each model
    for idx, (model_name, pred_info) in enumerate(predictions_dict.items(), 1):
        predicted = pred_info['predicted']
        confidence = pred_info['confidence']
        probabilities = pred_info['probabilities']
        is_correct = pred_info['correct']

        # Create bar chart of probabilities
        colors = ['green' if i == np.argmax(probabilities) else 'gray' for i in range(len(class_names))]
        bars = axes[idx].bar(class_names, probabilities, color=colors, alpha=0.7)
        axes[idx].set_ylim([0, 1])
        axes[idx].set_ylabel('Probability', fontsize=10)
        axes[idx].set_title(f'{model_name}\nPredicted: {predicted}\nConfidence: {confidence:.1%}',
                           fontsize=10, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{prob:.1%}', ha='center', va='bottom', fontsize=8)

        # Highlight correct prediction
        if is_correct:
            axes[idx].spines['top'].set_color('green')
            axes[idx].spines['top'].set_linewidth(3)
            axes[idx].spines['bottom'].set_color('green')
            axes[idx].spines['bottom'].set_linewidth(3)
            axes[idx].spines['left'].set_color('green')
            axes[idx].spines['left'].set_linewidth(3)
            axes[idx].spines['right'].set_color('green')
            axes[idx].spines['right'].set_linewidth(3)

    plt.suptitle(f'Predictions for: {img_filename}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def test_models_on_random_images(num_images=5, show_plots=True):
    """Test all models on random images from the dataset."""
    print("=" * 80)
    print("WILDLIFE POACHING RISK PREDICTION - MODEL TESTING")
    print("=" * 80)
    print()

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('data.csv')
    print(f"✓ Loaded {len(df)} images from dataset")

    # Get class names from dataset
    label_encoder = LabelEncoder()
    label_encoder.fit(df['poaching_risk'])
    class_names = label_encoder.classes_
    print(f"✓ Classes: {class_names.tolist()}")
    print()

    # Load all models
    print("Loading models...")
    models_dict, transform = load_all_models(device)

    if not models_dict:
        print("\n❌ No models loaded! Please train models first.")
        return

    print(f"\n✓ Loaded {len(models_dict)} model(s)")
    print()

    # Pick random images (different each run)
    print("=" * 80)
    print(f"TESTING ON {num_images} RANDOM IMAGES")
    print("=" * 80)
    print()

    # Select random images ensuring we get different risk levels
    random_indices = df.sample(n=min(num_images, len(df)), random_state=None).index.tolist()

    all_results = []

    for idx, img_idx in enumerate(random_indices, 1):
        row = df.iloc[img_idx]
        img_path_str = row['image_path']
        img_filename = img_path_str.split('\\')[-1]  # Handle Windows path separator
        img_path = Path('grid_images') / img_filename
        true_label = row['poaching_risk']
        true_label_encoded = label_encoder.transform([true_label])[0]

        if not img_path.exists():
            print(f"⚠ Image not found: {img_path}")
            continue

        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
        except Exception as e:
            print(f"⚠ Error loading image {img_path}: {e}")
            continue

        print(f"\n{'='*80}")
        print(f"Image {idx}/{num_images}: {img_filename}")
        print(f"True Label: {true_label}")
        print(f"{'='*80}")

        # Test each model
        image_results = {
            'image': img_filename,
            'true_label': true_label,
            'predictions': {}
        }

        for model_name, model in models_dict.items():
            predicted_class, confidence, probabilities = predict_image(
                model, image_tensor, device, class_names
            )
            predicted_label = class_names[predicted_class]
            is_correct = predicted_label == true_label

            image_results['predictions'][model_name] = {
                'predicted': predicted_label,
                'confidence': confidence,
                'probabilities': probabilities,
                'correct': is_correct
            }

            # Color code based on correctness
            status = "✓" if is_correct else "✗"
            color_indicator = "🟢" if is_correct else "🔴"

            print(f"\n{model_name}:")
            print(f"  {status} Predicted: {predicted_label} (Confidence: {confidence:.2%})")
            print(f"  Probabilities: Low={probabilities[0]:.2%}, Medium={probabilities[1]:.2%}, High={probabilities[2]:.2%}")
            print(f"  {color_indicator} {'Correct!' if is_correct else 'Incorrect'}")

        all_results.append(image_results)

        # Visualize predictions
        if show_plots:
            try:
                fig = visualize_predictions(image, true_label, image_results['predictions'],
                                          class_names, img_filename)
                plt.show()
            except Exception as e:
                print(f"⚠ Could not display visualization: {e}")

        print()

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    # Calculate accuracy for each model
    model_accuracies = {model_name: [] for model_name in models_dict.keys()}

    for result in all_results:
        for model_name, pred_info in result['predictions'].items():
            model_accuracies[model_name].append(pred_info['correct'])

    print("Model Accuracy (on tested images):")
    print("-" * 80)
    for model_name, accuracies in model_accuracies.items():
        if accuracies:
            accuracy = sum(accuracies) / len(accuracies)
            correct_count = sum(accuracies)
            total_count = len(accuracies)
            print(f"{model_name:20s}: {accuracy:.1%} ({correct_count}/{total_count})")
        else:
            print(f"{model_name:20s}: N/A")

    print()
    print("=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print(f"\nNote: Each time you run this script, it uses different random images.")
    print(f"Run again to test on different samples!\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test wildlife poaching risk prediction models')
    parser.add_argument('--num-images', type=int, default=5,
                       help='Number of random images to test (default: 5)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable visualization plots')

    args = parser.parse_args()

    test_models_on_random_images(num_images=args.num_images, show_plots=not args.no_plots)
