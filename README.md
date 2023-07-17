# Brain Tumour Classification Project

In this project, we developed an application for brain tumour classification based on histopathological data. We utilize the [Digital Brain Tumour Atlas](https://www.nature.com/articles/s41597-022-01157-0) dataset, which contains information about a variety of tumour types. The project includes data preprocessing, analysis, implementation of interpretability methods, the entire Godot code used for the frontend as well as the backend flask server.

## Installation

This project used Conda to manage the environment. Get it [here](https://www.anaconda.com/download).

You can create a new conda environment with the following command:

```bash
conda env create -f environment.yml
```

Activate using 

```bash
conda activate brain-tumour-classification
```

### Installation issues

If you have issues installing the environment, try installing the failed packages manually using pip.

## Project Structure
Here's a brief overview of the important files/folders:

`application/`: Contains the Godot frontend project, providing a user interface for image upload and classification result display.

`finetuning/`: This directory contains helper classes used for the VGG16 finetuning on our data.

`Data analysis.ipynb`: This Jupyter notebook is used for performing an exploratory data analysis on the preprocessed data, and training a convolutional neural network to classify tumor types.

`Data preprocessing.ipynb`: This Jupyter notebook is used for loading the annotation data, defining a script for downloading the histopathological data, and preprocessing it.

`Model_Interpretability.ipynb`: A notebook for visualizing key image areas used by the VGG16 model for predictions using techniques like Grad-CAM.

`Tensorboard_runs_analysis.ipynb`: Notebook for examining the model's performance during training using the data recorded with tensorboard.

`VGG16 Training.ipynb`: This notebook trains a VGG16 model on our dataset.

`annotation.csv`: This CSV file contains metadata about the patients and tumors.

`processed/`: This directory contains the processed histopathological data.

`flask_server.py`: A Python script that sets up a Flask server, creating an API for the VGG16 model. Users can submit images for classification.

`send_img.py`: A Python script for testing the API by sending test.png to it and saving the result in the /results directory.

## Wiki

For more information, please visit the [wiki](https://github.com/zebleck/Brain-Tumour-Analysis/wiki)!

## Download application

The executable for the frontend can be downloaded from [here](https://1drv.ms/f/s!Anworwr0CwTdg7EzzZppZPmq00KCtw?e=sp0FT0).

## Download data

To download and process the data, you need to get an authorization token. To get it, first [request access to the dataset](https://data-proxy.ebrains.eu/datasets/8fc108ab-e2b4-406f-8999-60269dc1f994). Then inspect the network traffic in the browser (F12 for Chrome) and search for the header `Authorization`. You can then insert the token value into the corresponding variable in `Data preprocessing.ipynb`.

To get preprocessed data at various sizes (256x256, 512x512, 1024x1024, 2048x2048), visit: https://1drv.ms/f/s!Anworwr0CwTdgeZTf9tN6CZCkzE39Q?e=S5YkaE

## Download model

The VGG-16 models, one pretrained on ImageNet and one trained from scratch can be found [here](https://1drv.ms/f/s!Anworwr0CwTdg7JPAVt2zCHYm_mV9w?e=ejehBW).
