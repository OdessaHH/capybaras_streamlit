import os
import zipfile
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO


# Set up the Streamlit app title and custom CSS for green jungle design
st.set_page_config(page_title="FAUNAVISION", page_icon="🌿")

# Set up the Streamlit app title and logo
logo_path = "FaunaVision_logo_edit1.png"  # Replace with the actual path to your logo
st.image(logo_path, width=800)  # Adjust the width as needed
st.markdown('<h2 style="color:#F0FFF0;">Wildlife Species Images Classifier</h2>', unsafe_allow_html=True)


# Define custom CSS for styling
custom_css = """
<style>
    body {
        background-color: #2E8B57; /* SeaGreen */
        color: #F0FFF0; /* HoneyDew */
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #2E8B57;
    }
    .sidebar .sidebar-content {
        background-color: #006400; /* DarkGreen */
        color: #F0FFF0; /* HoneyDew */
    }
    .css-1aumxhk {
        color: #F0FFF0; /* HoneyDew */
    }
    .stButton button {
        background-color: #228B22; /* ForestGreen */
        color: #F0FFF0; /* HoneyDew */
    }
    .stButton button:hover {
        background-color: #006400; /* DarkGreen */
    }
    .stImage {
        border-radius: 8px;
        margin-bottom: 10px; /* Add some space below each image */
    }
    .stMarkdown {
        color: #F0FFF0; /* HoneyDew */
    }
    .stTextInput input {
        background-color: #003300; /* Very Dark Green */
        color: #F0FFF0; /* HoneyDew */
    }
    .stTextInput input::placeholder {
        color: #A9A9A9; /* DarkGray */
    }
    .stDownloadButton {
        background-color: #228B22; /* ForestGreen */
        color: #F0FFF0; /* HoneyDew */
    }
    .stDownloadButton:hover {
        background-color: #006400; /* DarkGreen */
    }
    .predictions {
        margin-bottom: 20px; /* Space between predictions and next image */
    }
    .separator {
        border-top: 2px solid #F0FFF0; /* HoneyDew */
        margin-top: 20px; /* Space above the line */
        margin-bottom: 20px; /* Space below the line */
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar for user input (ZIP folder or multiple images)
st.sidebar.header("Upload Images")
uploaded_zip = st.sidebar.file_uploader("Upload a ZIP folder of images", type=['zip'])
uploaded_files = st.sidebar.file_uploader("Or upload multiple image files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

# Create a directory to store unzipped files if the user uploads a zip file
def extract_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        extract_dir = 'extracted_images'
        zip_ref.extractall(extract_dir)
        return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir)]

# List of all uploaded image files
image_files = []

if uploaded_zip:
    with BytesIO(uploaded_zip.getvalue()) as zip_buffer:
        image_files = extract_zip(zip_buffer)
elif uploaded_files:
    image_files = uploaded_files

# Button to trigger prediction
if st.sidebar.button('Predict') and image_files:
    st.write("**Predicting...**")

    class ImagesDataset(Dataset):
        def __init__(self, image_files):
            self.data = image_files
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        
        def __getitem__(self, index):
            if isinstance(self.data[index], str):  # If it's a file path from ZIP
                image = Image.open(self.data[index]).convert("RGB")
                filepath = os.path.basename(self.data[index])
            else:  # If it's an uploaded file
                image = Image.open(self.data[index]).convert("RGB")
                filepath = self.data[index].name
            
            image = self.transform(image)
            return {"image": image, "filepath": filepath}

        def __len__(self):
            return len(self.data)

    # Define species labels and their display names
    species_labels = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent']
    display_label_mapping = {
        'antelope_duiker': 'Antelope Duiker',
        'bird': 'Bird',
        'blank': 'Blank',
        'civet_genet': 'Civet Genet',
        'hog': 'Hog',
        'leopard': 'Leopard',
        'monkey_prosimian': 'Monkey/Prosimian',
        'rodent': 'Rodent'
    }

    model_path = 'model.pth'  # Replace with your model path
    model = torch.load(model_path)
    model.eval()

    dataset = ImagesDataset(image_files)
    dataloader = DataLoader(dataset, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Helper function to get top 3 predictions
    def get_top_3_predictions(preds, species_labels):
        top_3 = torch.topk(preds, 3)  # Get top 3
        top_3_probs = top_3.values.cpu().numpy() * 100  # Convert to percentages
        top_3_labels = [species_labels[i] for i in top_3.indices.cpu().numpy()]  # Get corresponding labels
        return list(zip(top_3_labels, top_3_probs))

    preds_collector = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            filepaths = batch["filepath"]
            logits = model(images)
            preds = nn.functional.softmax(logits, dim=1)
            
            for i in range(len(filepaths)):
                top_3 = get_top_3_predictions(preds[i], species_labels)  # Get top 3 predictions
                preds_collector.append((filepaths[i], top_3))

    # Convert predictions to DataFrame with separate columns for top 3 predictions
    preds_df = pd.DataFrame(preds_collector, columns=["filename", "top_3_predictions"])

    preds_df[['1st_label', '1st_prob']] = pd.DataFrame(preds_df['top_3_predictions'].apply(lambda x: [x[0][0], x[0][1]]).tolist(), index=preds_df.index)
    preds_df[['2nd_label', '2nd_prob']] = pd.DataFrame(preds_df['top_3_predictions'].apply(lambda x: [x[1][0], x[1][1]]).tolist(), index=preds_df.index)
    preds_df[['3rd_label', '3rd_prob']] = pd.DataFrame(preds_df['top_3_predictions'].apply(lambda x: [x[2][0], x[2][1]]).tolist(), index=preds_df.index)

    # Convert probabilities to percentages rounded to 2 decimals
    preds_df['1st_prob'] = preds_df['1st_prob'].round(2)
    preds_df['2nd_prob'] = preds_df['2nd_prob'].round(2)
    preds_df['3rd_prob'] = preds_df['3rd_prob'].round(2)

    # Keep only the relevant columns for CSV (original model labels)
    csv_df = preds_df[['filename', '1st_label', '1st_prob', '2nd_label', '2nd_prob', '3rd_label', '3rd_prob']]

    # Map the labels to their display names (for displaying in the app)
    preds_df['1st_label'] = preds_df['1st_label'].map(display_label_mapping)
    preds_df['2nd_label'] = preds_df['2nd_label'].map(display_label_mapping)
    preds_df['3rd_label'] = preds_df['3rd_label'].map(display_label_mapping)

    st.success("Prediction completed!")

    # Bar plot for species distribution
    all_predictions = preds_df['1st_label']  # First prediction (most probable)
    species_distribution = all_predictions.value_counts()

    st.subheader('Species Distribution')
    fig, ax = plt.subplots()
    species_distribution.plot(kind='bar', ax=ax, color='#228B22')  # ForestGreen color
    ax.set_xlabel('Species')
    ax.set_ylabel('Number of Images')
    st.pyplot(fig)

    st.write(f"**Total Images Analyzed**: {len(preds_df)}")
    for species, count in species_distribution.items():
        st.write(f"**{species}:** {count} images")

    # CSV Download Button
    st.subheader("Download Predictions as CSV")
    csv_filename = st.text_input("Enter CSV file name (without extension)", "predictions")
    csv_data = csv_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv_data, file_name=f"{csv_filename}.csv", mime="text/csv")

    # Display predicted images with labels
    st.subheader('Predicted Images with Labels')
    for i in range(len(preds_df)):
        if isinstance(image_files[0], str):  # From ZIP (file path)
            st.image(image_files[i], caption=f"{preds_df.iloc[i]['filename']}")
        else:  # Uploaded directly
            st.image(image_files[i], caption=f"{image_files[i].name}")

        # Display predictions with a line after the third prediction
        first_pred, first_prob = preds_df.iloc[i]['1st_label'], preds_df.iloc[i]['1st_prob']
        second_pred, second_prob = preds_df.iloc[i]['2nd_label'], preds_df.iloc[i]['2nd_prob']
        third_pred, third_prob = preds_df.iloc[i]['3rd_label'], preds_df.iloc[i]['3rd_prob']

        st.markdown(f'<div class="image-container"><p style="color:#FFD700; font-size:20px;"><strong>{first_pred} - {first_prob:.2f}%</strong></p></div>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:18px;">{second_pred} - {second_prob:.2f}%</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:18px;">{third_pred} - {third_prob:.2f}%</p>', unsafe_allow_html=True)
        
        # Add a horizontal line after the third prediction
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)