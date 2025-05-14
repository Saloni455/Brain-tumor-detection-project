import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import BrainTumorCNN  # Import from model.py instead of train_model.py

def load_model(model_path):
    # Initialize model
    model = BrainTumorCNN()
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    # Define the same transform used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    return input_tensor, image

def predict_image(model, image_tensor, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Move model and input to device
    model = model.to(device)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0]

def plot_prediction(image, predicted_class, confidence, probabilities, class_names):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot image
    ax1.imshow(image)
    ax1.set_title(f'Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.2%}')
    ax1.axis('off')
    
    # Plot probabilities
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probabilities.cpu().numpy())
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probabilities')
    
    plt.tight_layout()
    plt.show()

def main():
    # Define class names
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Load model
    model = load_model('final_model.pth')
    
    while True:
        # Get image path from user
        image_path = input("\nEnter the path to your brain MRI image (or 'q' to quit): ")
        
        if image_path.lower() == 'q':
            break
            
        try:
            # Preprocess image
            image_tensor, original_image = preprocess_image(image_path)
            
            # Make prediction
            predicted_class, confidence, probabilities = predict_image(model, image_tensor)
            
            # Display results
            print(f"\nPrediction Results:")
            print(f"Predicted Class: {class_names[predicted_class]}")
            print(f"Confidence: {confidence:.2%}")
            print("\nClass Probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"{class_names[i]}: {prob:.2%}")
            
            # Plot results
            plot_prediction(original_image, predicted_class, confidence, probabilities, class_names)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please make sure the image path is correct and the image is a valid brain MRI scan.")

if __name__ == "__main__":
    main() 
    