#pip install streamlit

import os
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import tqdm

# Set the title of the Streamlit app
st.title('Image Classification Model')

# Sidebar for uploading parameters and model
st.sidebar.header('Upload Configuration')

# Allow users to upload parameter CSV file
param_file = st.sidebar.file_uploader('Upload Parameters CSV', type=['csv'])
model_file = st.sidebar.file_uploader('Upload Model (model.pth)', type=['pth'])

if param_file and model_file:
    # Load the parameters
    df_param = pd.read_csv(param_file)
    df_param.set_index('parameter', inplace=True)
    
    # Convert 'value' column in the parameters
    def convert_value(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    
    df_param['value'] = df_param['value'].apply(convert_value)
    param = df_param.to_dict()['value']

    data_folder = param.get('data_folder', './data')
    label_csv = os.path.join(data_folder, param['label_csv'])
    train_features_csv = os.path.join(data_folder, param['train_features_csv'])
    test_features_csv = os.path.join(data_folder, param['test_features_csv'])
    
    # Load CSV files
    train_labels = pd.read_csv(label_csv, index_col="id")
    train_features = pd.read_csv(train_features_csv, index_col="id")
    test_features = pd.read_csv(test_features_csv, index_col="id")

    species_labels = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent']
    
    # Define the dataset class
    class ImagesDataset(Dataset):
        def __init__(self, x_df, y_df=None):
            self.data = x_df
            self.label = y_df
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        def __getitem__(self, index):
            image = Image.open(self.data.iloc[index]["filepath"]).convert("RGB")
            image = self.transform(image)
            image_id = self.data.index[index]
            if self.label is None:
                return {"image_id": image_id, "image": image}
            else:
                label = torch.tensor(self.label.iloc[index].values, dtype=torch.float)
                return {"image_id": image_id, "image": image, "label": label}

        def __len__(self):
            return len(self.data)

    # Load and display the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_file)
    model.to(device)

    # Create the evaluation dataset
    st.subheader('Dataset and Evaluation')
    st.write(f"Training Data: {len(train_features)} images")
    
    # Allow the user to load evaluation images from validation/test set
    st.write("Evaluating on test set")
    eval_dataset = ImagesDataset(test_features)  # No labels for test set
    eval_dataloader = DataLoader(eval_dataset, batch_size=32)

    # Start evaluation
    st.subheader("Predictions")
    st.write("Running model predictions...")

    preds_collector = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_dataloader, total=len(eval_dataloader)):
            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            logits = model(images)
            preds = nn.functional.softmax(logits, dim=1)
            preds_df = pd.DataFrame(preds.cpu().numpy(), index=image_ids, columns=species_labels)
            preds_collector.append(preds_df)
    
    eval_preds_df = pd.concat(preds_collector)
    
    # Display predictions
    eval_predictions = eval_preds_df.idxmax(axis=1)
    prediction_count = eval_predictions.value_counts()
    
    st.write("Prediction Count:")
    st.write(prediction_count)

    st.subheader("Class-wise Predictions")
    st.write(eval_preds_df)
    
    # Clear GPU cache
    model.cpu()
    torch.cuda.empty_cache()
else:
    st.sidebar.warning("Please upload both the parameter CSV and model file.")
