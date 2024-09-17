import os
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import tqdm

# Set up the Streamlit app title
st.title('Wildlife Species Image Classifier')

# Sidebar for user input (folder of images) and predict button
st.sidebar.header("Upload Images")

# Allow the user to upload multiple files
uploaded_files = st.sidebar.file_uploader("Upload a folder of images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

# Button to trigger prediction
if st.sidebar.button('Predict') and uploaded_files:

    st.write("**Prediction process started**")

    # Create progress bar
    progress_bar = st.progress(0)
    progress_step = 1 / len(uploaded_files)  # Calculate step for each image

    # Prepare the image dataset and transform
    class ImagesDataset(Dataset):
        def __init__(self, image_files):
            self.data = image_files
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        
        def __getitem__(self, index):
            image = Image.open(self.data[index]).convert("RGB")
            image = self.transform(image)
            return {"image": image, "filepath": self.data[index].name}

        def __len__(self):
            return len(self.data)

    # Define labels and load the model
    species_labels = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent']
    
    # Assuming the model is loaded from a file
    model_path = '/Users/alexandersimakov/Documents/condarun/streamlit/capybaras_streamlit/model.pth'
    model = torch.load(model_path)
    model.eval()
    
    # Load images into a DataLoader
    dataset = ImagesDataset(uploaded_files)
    dataloader = DataLoader(dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    preds_collector = []
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            filepaths = batch["filepath"]
            logits = model(images)
            preds = nn.functional.softmax(logits, dim=1)
            preds_labels = preds.argmax(dim=1)
            preds_collector.extend(list(zip(filepaths, preds_labels.cpu().numpy())))
            
            # Update progress bar
            progress_bar.progress((idx + 1) * progress_step)
    
    # Convert predictions to DataFrame
    preds_df = pd.DataFrame(preds_collector, columns=["filename", "predicted_label"])
    preds_df['predicted_species'] = preds_df['predicted_label'].apply(lambda x: species_labels[x])
    
    st.success("Prediction completed!")

    # Bar plot for species distribution
    species_distribution = preds_df['predicted_species'].value_counts()
    st.subheader('Species Distribution')

    fig, ax = plt.subplots()
    species_distribution.plot(kind='bar', ax=ax)
    ax.set_xlabel('Species')
    ax.set_ylabel('Number of Images')
    st.pyplot(fig)

    # Display predicted images with labels
    st.subheader('Predicted Images with Labels')
    num_images_to_display = st.slider("Select number of images to display", min_value=1, max_value=len(preds_df), value=5)
    
    for i in range(num_images_to_display):
        st.image(uploaded_files[i], caption=f"{uploaded_files[i].name}: {preds_df.iloc[i]['predicted_species']}")

    # Save predictions as CSV
    st.subheader("Download Predictions as CSV")
    csv = preds_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name="predictions.csv", mime="text/csv")
