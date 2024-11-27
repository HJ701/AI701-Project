import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OrthoAIDataset  # Ensure this is correctly implemented
from feature_extraction import (
    IntraoralFeatureExtractor,
    RadiographFeatureExtractor,
    TextFeatureExtractor
)
from fusion_layer import FusionLayer  # Adjust import based on your project structure
from classification_layer import ClassificationLayer  # Adjust import based on your project structure
from torchvision import transforms
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def train_epoch(epoch, train_loader, intraoral_extractor, radiograph_extractor, text_extractor,
               fusion_layer, classification_layer, criterion, optimizer, device, num_epochs):
    intraoral_extractor.train()
    radiograph_extractor.train()
    text_extractor.train()
    fusion_layer.train()
    classification_layer.train()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for batch_idx, (intraoral_imgs, radiograph_imgs, texts, labels) in enumerate(train_loader):
        try:
            # Move data to device
            intraoral_imgs = intraoral_imgs.to(device)  # Shape: [num_images, 3, 224, 224]
            radiograph_imgs = radiograph_imgs.to(device)  # Shape: [num_radiographs, 1, 224, 224]
            input_ids = texts['input_ids'].to(device)  # Shape: [batch_size, max_length]
            attention_mask = texts['attention_mask'].to(device)  # Shape: [batch_size, max_length]
            labels = labels.to(device).long()  # Shape: [batch_size]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass through feature extractors
            intraoral_feat = intraoral_extractor(intraoral_imgs)  # Shape: [num_images, 1536]
            radiograph_feat = radiograph_extractor(radiograph_imgs)  # Shape: [num_radiographs, 2048]
            text_feat = text_extractor(input_ids, attention_mask)  # Shape: [batch_size, 768]

            # Aggregate intraoral and radiograph features
            intraoral_feat = intraoral_feat.mean(dim=0, keepdim=True)  # Shape: [1, 1536]
            radiograph_feat = radiograph_feat.mean(dim=0, keepdim=True)  # Shape: [1, 2048]

            # Fusion
            fused_feat = fusion_layer(intraoral_feat, radiograph_feat, text_feat)  # Shape: [batch_size, 512]
            outputs = classification_layer(fused_feat)  # Shape: [batch_size, num_classes]

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update running loss and predictions
            running_loss += loss.item() * intraoral_imgs.size(0)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        except Exception as e:
            logging.error(f"Error during training at batch {batch_idx}: {e}")
            continue

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")

def validate_epoch(epoch, val_loader, intraoral_extractor, radiograph_extractor, text_extractor,
                  fusion_layer, classification_layer, criterion, device, num_epochs):
    intraoral_extractor.eval()
    radiograph_extractor.eval()
    text_extractor.eval()
    fusion_layer.eval()
    classification_layer.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (intraoral_imgs, radiograph_imgs, texts, labels) in enumerate(val_loader):
            try:
                # Move data to device
                intraoral_imgs = intraoral_imgs.to(device)
                radiograph_imgs = radiograph_imgs.to(device)
                input_ids = texts['input_ids'].to(device)
                attention_mask = texts['attention_mask'].to(device)
                labels = labels.to(device).long()

                # Forward pass through feature extractors
                intraoral_feat = intraoral_extractor(intraoral_imgs)
                radiograph_feat = radiograph_extractor(radiograph_imgs)
                text_feat = text_extractor(input_ids, attention_mask)

                # Aggregate intraoral and radiograph features
                intraoral_feat = intraoral_feat.mean(dim=0, keepdim=True)
                radiograph_feat = radiograph_feat.mean(dim=0, keepdim=True)

                # Fusion
                fused_feat = fusion_layer(intraoral_feat, radiograph_feat, text_feat)
                outputs = classification_layer(fused_feat)

                # Compute loss
                loss = criterion(outputs, labels)

                # Update running loss and predictions
                running_loss += loss.item() * intraoral_imgs.size(0)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            except Exception as e:
                logging.error(f"Error during validation at batch {batch_idx}: {e}")
                continue

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Epoch {epoch+1}/{num_epochs} Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")

    return epoch_loss

def test_model(test_loader, intraoral_extractor, radiograph_extractor, text_extractor,
               fusion_layer, classification_layer, device):
    try:
        # Load the best model
        checkpoint = torch.load('best_model.pth', map_location=device)
        intraoral_extractor.load_state_dict(checkpoint['intraoral_extractor'])
        radiograph_extractor.load_state_dict(checkpoint['radiograph_extractor'])
        text_extractor.load_state_dict(checkpoint['text_extractor'])
        fusion_layer.load_state_dict(checkpoint['fusion_layer'])
        classification_layer.load_state_dict(checkpoint['classification_layer'])

        intraoral_extractor.eval()
        radiograph_extractor.eval()
        text_extractor.eval()
        fusion_layer.eval()
        classification_layer.eval()

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch_idx, (intraoral_imgs, radiograph_imgs, texts, labels) in enumerate(test_loader):
                try:
                    # Move data to device
                    intraoral_imgs = intraoral_imgs.to(device)
                    radiograph_imgs = radiograph_imgs.to(device)
                    input_ids = texts['input_ids'].to(device)
                    attention_mask = texts['attention_mask'].to(device)
                    labels = labels.to(device).long()

                    # Forward pass through feature extractors
                    intraoral_feat = intraoral_extractor(intraoral_imgs)
                    radiograph_feat = radiograph_extractor(radiograph_imgs)
                    text_feat = text_extractor(input_ids, attention_mask)

                    # Aggregate intraoral and radiograph features
                    intraoral_feat = intraoral_feat.mean(dim=0, keepdim=True)
                    radiograph_feat = radiograph_feat.mean(dim=0, keepdim=True)

                    # Fusion
                    fused_feat = fusion_layer(intraoral_feat, radiograph_feat, text_feat)
                    outputs = classification_layer(fused_feat)

                    # Predictions
                    _, preds = torch.max(outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                except Exception as e:
                    logging.error(f"Error during testing at batch {batch_idx}: {e}")
                    continue

        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"Test Accuracy: {test_acc:.4f} F1 Score: {test_f1:.4f}")
    except Exception as e:
        logging.error(f"Error loading the model or during testing: {e}")

def main():
    # Hyperparameters
    NUM_CLASSES = 3  # Adjust based on your classification task
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 25
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Example normalization
                             std=[0.229, 0.224, 0.225]),
    ])

    radiograph_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
    ])

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    # Load the DataFrame
    df = pd.read_csv('/Users/hj/OrthoAI/Datasets/structured_data.csv')  

    # Convert 'clinical_notes' column to string type
    df['clinical_notes'] = df['clinical_notes'].astype(str)

    # Proceed with preprocessing (if any additional preprocessing is required)
    # If 'preprocess_and_extract_labels' handles label encoding and other preprocessing, ensure it's correctly implemented
    df = preprocess_and_extract_labels(df)  # Ensure this function handles necessary preprocessing

    # Create datasets
    train_df = df[df['set'] == 'train']
    val_df = df[df['set'] == 'validation']
    test_df = df[df['set'] == 'test']

    # Define directories for images
    image_dir = '/Users/hj/OrthoAI/Datasets/Intraoral_Images/'  # Adjust as per your directory structure
    radiograph_dir = '/Users/hj/OrthoAI/Datasets/Radiograph_Images/'  # Adjust as per your directory structure

    # Initialize datasets and data loaders
    train_dataset = OrthoAIDataset(
        train_df,
        image_dir=image_dir,
        radiograph_dir=radiograph_dir,
        transform=transform,
        radiograph_transform=radiograph_transform,
        tokenizer=tokenizer,
        max_length=128
    )
    val_dataset = OrthoAIDataset(
        val_df,
        image_dir=image_dir,
        radiograph_dir=radiograph_dir,
        transform=transform,
        radiograph_transform=radiograph_transform,
        tokenizer=tokenizer,
        max_length=128
    )
    test_dataset = OrthoAIDataset(
        test_df,
        image_dir=image_dir,
        radiograph_dir=radiograph_dir,
        transform=transform,
        radiograph_transform=radiograph_transform,
        tokenizer=tokenizer,
        max_length=128
    )

    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize feature extractors
    intraoral_extractor = IntraoralFeatureExtractor().to(DEVICE)
    radiograph_extractor = RadiographFeatureExtractor().to(DEVICE)
    text_extractor = TextFeatureExtractor().to(DEVICE)

    # Determine feature dimensions
    with torch.no_grad():
        # Dummy data for intraoral images
        dummy_intraoral = torch.randn(1, 3, 224, 224).to(DEVICE)
        intraoral_feature = intraoral_extractor(dummy_intraoral)
        intraoral_feature_dim = intraoral_feature.shape[1]  # Example: 1536

        # Dummy data for radiograph images
        dummy_radiograph = torch.randn(1, 1, 224, 224).to(DEVICE)
        radiograph_feature = radiograph_extractor(dummy_radiograph)
        radiograph_feature_dim = radiograph_feature.shape[1]  # Example: 2048

    # Assuming TextFeatureExtractor outputs 768-dimensional embeddings (e.g., BERT-base)
    text_feature_dim = 768

    FEATURE_DIMS = {
        'intraoral': intraoral_feature_dim,
        'radiograph': radiograph_feature_dim,
        'text': text_feature_dim
    }

    # Calculate input_dim for ClassificationLayer
    input_dim = sum(FEATURE_DIMS.values())  # Example: 1536 + 2048 + 768 = 4352

    # Initialize fusion and classification layers
    fusion_layer = FusionLayer(FEATURE_DIMS).to(DEVICE)
    classification_layer = ClassificationLayer(input_dim=input_dim, num_classes=NUM_CLASSES).to(DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        list(intraoral_extractor.parameters()) +
        list(radiograph_extractor.parameters()) +
        list(text_extractor.parameters()) +
        list(fusion_layer.parameters()) +
        list(classification_layer.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_epoch(epoch, train_loader, intraoral_extractor, radiograph_extractor, text_extractor,
                   fusion_layer, classification_layer, criterion, optimizer, DEVICE, NUM_EPOCHS)

        val_loss = validate_epoch(epoch, val_loader, intraoral_extractor, radiograph_extractor, text_extractor,
                                  fusion_layer, classification_layer, criterion, DEVICE, NUM_EPOCHS)

        # Step the scheduler
        scheduler.step()

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'intraoral_extractor': intraoral_extractor.state_dict(),
                'radiograph_extractor': radiograph_extractor.state_dict(),
                'text_extractor': text_extractor.state_dict(),
                'fusion_layer': fusion_layer.state_dict(),
                'classification_layer': classification_layer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
            print("Model saved.")

    print("Training complete.")

    # Test the model
    test_model(test_loader, intraoral_extractor, radiograph_extractor, text_extractor,
               fusion_layer, classification_layer, DEVICE)

if __name__ == '__main__':
    main()
