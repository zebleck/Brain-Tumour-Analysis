import argparse
import json
from datetime import date, datetime
from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.metrics.road import ROADCombined
import time
import base64
import os
from os import listdir
from os.path import isfile, join
from skimage.segmentation import slic, mark_boundaries
from lime import lime_image

currentState_server = "idle"  # sry for global


def load_model(model_path):

    model = timm.create_model('vgg16', pretrained=False)

    # Output dimension, which is equal to the number of different tumour classes
    output_dim = 127

    # Replace the last fully connected layer to match the number of classes in the new data set
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, output_dim)

    # Add dropout
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, output_dim)
    )

    # Load the pretrained model weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    return model


def get_top_probs(model, input_tensor, top_n=10):
    model.eval()

    output = model(input_tensor.unsqueeze(0))  # unsqueeze single image into batch of 1
    top_probs, top_class_indices = torch.topk(output.softmax(dim=1), k=top_n)
    predictions = [(tumor_type_dict[idx], top_probs[0][i].item()) for i, idx in
                   enumerate(top_class_indices[0].tolist())]

    return predictions


def calcGradCAM(input_tensor):
    model.eval()

    input_tensor = input_tensor.unsqueeze(0)

    output = model(input_tensor)
    _, target_label = torch.max(output, dim=1)

    # In VGG, feature part is called 'features', the last layer is [-1]
    target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(target_label.item())]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Conver to 8-bit (range 0-255)
    grayscale_cam_8bit = np.squeeze(np.uint8(255 * grayscale_cam))

    # Convert the tensor image to numpy array, transpose from CxHxW to HxWxC format, and
    # reverse normalize using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    rgb_img = np.clip(input_tensor.squeeze().permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array(
        [0.485, 0.456, 0.406]), 0, 1)

    visualization = show_cam_on_image(rgb_img, grayscale_cam_8bit)

    # Calculate metrics
    cam_metric = CamMultImageConfidenceChange()
    scores, batch_visualizations = cam_metric(input_tensor,
                                              grayscale_cam,
                                              [ClassifierOutputTarget(target_label.item())],
                                              model,
                                              return_visualization=True)

    cam_metric_road = ROADCombined(percentiles=[20, 40, 60, 80])
    scores_road = cam_metric_road(input_tensor,
                                  grayscale_cam,
                                  [ClassifierOutputTarget(target_label.item())],
                                  model)

    # Print scores
    # print(f'CamMultImageConfidenceChange Score: {scores}')
    # print(f'ROAD Score: {scores_road}')

    return rgb_img, visualization, batch_visualizations, scores[0], scores_road[0]


tumor_type_dict = {0: 'Adamantinomatous craniopharyngioma', 1: 'Anaplastic astrocytoma, IDH-mutant',
                   2: 'Anaplastic astrocytoma, IDH-wildtype', 3: 'Anaplastic ependymoma', 4: 'Anaplastic ganglioglioma',
                   5: 'Anaplastic meningioma', 6: 'Anaplastic oligodendroglioma, IDH-mutant and 1p-19q codeleted',
                   7: 'Anaplastic pleomorphic xanthoastrocytoma', 8: 'Angiocentric glioma', 9: 'Angiomatous meningioma',
                   10: 'Angiosarcoma', 11: 'Astroblastoma', 12: 'Atypical choroid plexus papilloma',
                   13: 'Atypical meningioma', 14: 'Atypical teratoid-rhabdoid tumour', 15: 'Cellular schwannoma',
                   16: 'Central neurocytoma', 17: 'Cerebellar liponeurocytoma', 18: 'Chondrosarcoma',
                   19: 'Chordoid glioma of the third ventricle', 20: 'Chordoid meningioma', 21: 'Chordoma',
                   22: 'Choriocarcinoma', 23: 'Choroid plexus carcinoma', 24: 'Choroid plexus papilloma',
                   25: 'Clear cell meningioma', 26: 'CNS ganglioneuroblastoma', 27: 'Control',
                   28: 'Crystal-storing histiocytosis', 29: 'Desmoplastic infantile astrocytoma and ganglioglioma',
                   30: 'Diffuse astrocytoma, IDH-mutant', 31: 'Diffuse astrocytoma, IDH-wildtype',
                   32: 'Diffuse large B-cell lymphoma of the CNS', 33: 'Diffuse leptomeningeal glioneuronal tumour',
                   34: 'Diffuse midline glioma, H3 K27M-mutant', 35: 'Dysembryoplastic neuroepithelial tumour',
                   36: 'Dysplastic cerebellar gangliocytoma', 37: 'EBV-positive diffuse large B-cell lymphoma, NOS',
                   38: 'Embryonal carcinoma', 39: 'Embryonal tumour with multilayered rosettes, C19MC-altered',
                   40: 'Ependymoma', 41: 'Ependymoma, RELA fusion-positive', 42: 'Epitheloid MPNST',
                   43: 'Erdheim-Chester disease', 44: 'Ewing sarcoma', 45: 'Extraventricular neurocytoma',
                   46: 'Fibrosarcoma', 47: 'Fibrous meningioma', 48: 'Follicular lymphoma', 49: 'Gangliocytoma',
                   50: 'Ganglioglioma', 51: 'Ganglioneuroma', 52: 'Gemistocytic astrocytoma, IDH-mutant',
                   53: 'Germinoma', 54: 'Giant cell glioblastoma', 55: 'Glioblastoma, IDH-mutant',
                   56: 'Glioblastoma, IDH-wildtype', 57: 'Gliosarcoma', 58: 'Granular cell tumour of the sellar region',
                   59: 'Haemangioblastoma', 60: 'Haemangioma', 61: 'Haemangiopericytoma',
                   62: 'Hybrid nerve sheath tumours', 63: 'Immature teratoma',
                   64: 'Immunodeficiency-associated CNS lymphoma', 65: 'Inflammatory myofibroblastic tumour',
                   66: 'Intravascular large B-cell lymphoma', 67: 'Juvenile xanthogranuloma',
                   68: 'Langerhans cell histiocytosis', 69: 'Leiomyoma', 70: 'Leiomyosarcoma', 71: 'Lipoma',
                   72: 'Liposarcoma', 73: 'Low-grade B-cell lymphomas of the CNS',
                   74: 'Lymphoplasmacyte-rich meningioma', 75: 'Malignant peripheral nerve sheath tumour',
                   76: 'MALT lymphoma of the dura', 77: 'Mature teratoma', 78: 'Medulloblastoma, non-WNT-non-SHH',
                   79: 'Medulloblastoma, SHH-activated and TP53-mutant',
                   80: 'Medulloblastoma, SHH-activated and TP53-wildtype', 81: 'Medulloblastoma, WNT-activated',
                   82: 'Melanotic schwannoma', 83: 'Meningeal melanocytoma', 84: 'Meningeal melanoma',
                   85: 'Meningothelial meningioma', 86: 'Metaplastic meningioma', 87: 'Metastatic tumours',
                   88: 'Microcystic meningioma', 89: 'Mixed germ cell tumour', 90: 'Myxopapillary ependymoma',
                   91: 'Neurofibroma', 92: 'Olfactory neuroblastoma',
                   93: 'Oligodendroglioma, IDH-mutant and 1p-19q codeleted', 94: 'Osteochondroma', 95: 'Osteoma',
                   96: 'Osteosarcoma', 97: 'Papillary craniopharyngioma', 98: 'Papillary ependymoma',
                   99: 'Papillary glioneuronal tumour', 100: 'Papillary meningioma',
                   101: 'Papillary tumour of the pineal region', 102: 'Paraganglioma', 103: 'Perineurioma',
                   104: 'Pilocytic astrocytoma', 105: 'Pilomyxoid astrocytoma',
                   106: 'Pineal parenchymal tumour of intermediate differentiation', 107: 'Pineoblastoma',
                   108: 'Pineocytoma', 109: 'Pituicytoma', 110: 'Pituitary adenoma',
                   111: 'Pleomorphic xanthoastrocytoma', 112: 'Plexiform neurofibroma', 113: 'Psammomatous meningioma',
                   114: 'Rhabdoid meningioma', 115: 'Rhabdomyosarcoma', 116: 'Rosette-forming glioneuronal tumour',
                   117: 'Schwannoma', 118: 'Secretory meningioma', 119: 'Spindle cell oncocytoma',
                   120: 'Subependymal giant cell astrocytoma', 121: 'Subependymoma',
                   122: 'T-cell and NK-T-cell lymphomas of the CNS', 123: 'Tanycytic ependymoma',
                   124: 'Teratoma with malignant transformation', 125: 'Transitional meningioma',
                   126: 'Undifferentiated pleomorphic sarcoma'}

# Initialize Flask application
app = Flask(__name__)


@app.route('/histological_images', methods=['GET'])
def getAllAvailableFilesByName():
    allImages = [f for f in listdir("histo_images") if isfile(join("histo_images", f))]
    return {
        'images': allImages
    }


@app.route('/archive/folder', methods=['GET'])
def getAllAvailableFoldersInArchiveByName():
    allFolders = [f for f in listdir("archive") if os.path.isdir(join("archive", f))]
    return {
        'images': allFolders  # named images cuz of same handler function as histological_images in frontend
    }


@app.route('/state', methods=['GET'])
def getCurrentServerState():
    return {
        'state': currentState_server
    }


@app.route('/archive/<folder>', methods=['GET'])
def get_archive_with_path(folder):
    try:
        original = Image.open("archive/" + folder + "/original.png")
        visualizations = Image.open("archive/" + folder + "/visualization.png")
        batch = Image.open("archive/" + folder + "/batch_visualizations.png")
        lime = Image.open("archive/" + folder + "/lime_visualizations.png")

    except Exception as e:
        print(e)
        return 'Image not found', 400

    # Convert to base64
    byte_stream = io.BytesIO()
    original.save(byte_stream, format='PNG')
    byte_stream.seek(0)
    original_image_stream = base64.b64encode(byte_stream.read()).decode()

    byte_stream2 = io.BytesIO()
    visualizations.save(byte_stream2, format='PNG')
    byte_stream2.seek(0)
    visualizations_image_stream = base64.b64encode(byte_stream2.read()).decode()

    byte_stream3 = io.BytesIO()
    batch.save(byte_stream3, format='PNG')
    byte_stream3.seek(0)
    batch_image_stream = base64.b64encode(byte_stream3.read()).decode()

    # Convert the image to a byte stream with the desired colormap
    byte_stream = io.BytesIO()
    cmap = LinearSegmentedColormap.from_list('lime_cmap', ['red', 'white', 'blue'])
    plt.imshow(lime, cmap=cmap, vmin=-1.0, vmax=1.0)
    plt.axis('off')  # Remove axis
    plt.savefig(byte_stream, format='png', bbox_inches='tight', pad_inches=0)  # Save the image to the byte stream
    byte_stream.seek(0)
    lime_stream = base64.b64encode(byte_stream.getvalue()).decode()

    scores = read_number_from_file("archive/" + folder + "/scores.txt")
    scores_road = read_number_from_file("archive/" + folder + "/scores_road.txt")

    preds = read_json_file("archive/" + folder + "/predictions.json")

    response = {
        'original': original_image_stream,
        'predictions': preds,
        'visualization': visualizations_image_stream,
        'batch_visualizations': batch_image_stream,
        'lime_visualization': lime_stream,
        'scores': str(scores),
        'scores_road': str(scores_road),
    }

    return response


@app.route('/process_image_name/<file_name>', methods=['GET'])
def process_image_with_path(file_name):
    global currentState_server
    currentState_server = "load_image"

    try:
        image = Image.open("histo_images/" + file_name)
        # original_image_stream = base64.b64encode(image.tobytes()).decode()
        image_pil = image

    except Exception as e:
        print(e)
        return 'Image not found', 400

    currentState_server = "preprocess_image"
    image = preprocess_image(image)

    start_time = time.time()
    currentState_server = "prediction"
    preds = get_top_probs(model, image)
    print('Prediction time: {:.2f} seconds', time.time() - start_time)

    start_time = time.time()
    currentState_server = "interpretability"
    rgb_img, visualization, batch_visualizations, scores, scores_road = calcGradCAM(image)
    print('GradCAM time: {:.2f} seconds', time.time() - start_time)

    start_time = time.time()
    currentState_server = "convert_lime"
    lime_img = lime(image)
    print('LIME time: {:.2f} seconds', time.time() - start_time)

    start_time = time.time()
    currentState_server = "convert_visualization"
    # Convert numpy array to PIL Image and then to byte stream
    visualization_pil = Image.fromarray(visualization.astype(np.uint8))
    byte_stream1 = io.BytesIO()
    visualization_pil.save(byte_stream1, format='PNG')
    byte_stream1.seek(0)
    visualization_stream = base64.b64encode(byte_stream1.read()).decode()

    # Convert tensor to PIL Image and then to byte stream
    currentState_server = "convert_batch"
    batch_visualization_pil = transforms.ToPILImage()(batch_visualizations.squeeze(0))
    byte_stream2 = io.BytesIO()
    batch_visualization_pil.save(byte_stream2, format='PNG')
    byte_stream2.seek(0)
    batch_visualization_stream = base64.b64encode(byte_stream2.read()).decode()

    # Convert the image to a byte stream with the desired colormap
    byte_stream = io.BytesIO()
    cmap = LinearSegmentedColormap.from_list('lime_cmap', ['red', 'white', 'blue'])
    plt.imshow(lime_img, cmap=cmap, vmin=-1.0, vmax=1.0)
    plt.axis('off')  # Remove axis
    plt.savefig(byte_stream, format='png', bbox_inches='tight', pad_inches=0)  # Save the image to the byte stream
    byte_stream.seek(0)
    lime_stream = base64.b64encode(byte_stream.getvalue()).decode()

    currentState_server = "convert_original"
    # Convert original to base64
    byte_stream3 = io.BytesIO()
    image_pil.save(byte_stream3, format='PNG')
    byte_stream3.seek(0)
    original_image_stream = base64.b64encode(byte_stream3.read()).decode()

    print('Image conversion time: {:.2f} seconds', time.time() - start_time)

    response = {
        'original': original_image_stream,
        'predictions': preds,
        'visualization': visualization_stream,
        'batch_visualizations': batch_visualization_stream,
        'lime_visualization': lime_stream,
        'scores': str(scores),
        'scores_road': str(scores_road),
    }

    create_archive(response, file_name)

    currentState_server = "idle"

    return response


def preprocess_image(image):
    # Define the mean and standard deviation for normalization (same as ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define the image transformation pipeline with augmentation
    transform = transforms.Compose([
        transforms.Resize(512),  # Resize the image to 512x512 pixels
        transforms.CenterCrop(512),  # Perform a center crop of size 512x512 pixels
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=mean, std=std)  # Normalize the image using the specified mean and standard deviation
    ])

    processed_image = transform(image)
    return processed_image


def postprocess_image(image):
    # Implement your image postprocessing logic here
    # This could include converting the output to the desired format, applying filters, etc.
    # Return the postprocessed image
    return image


def custom_print(*args, **kwargs):
    if isPrint:
        print(*args, **kwargs)


def read_number_from_file(file_path):
    with open(file_path, 'r') as file:
        number = float(file.read().strip())
    return number


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data


def create_folder_with_date(custom_name):
    now = datetime.now()
    folder_name = f"{now.strftime('%Y-%m-%d_%H-%M')}_{custom_name}"
    os.makedirs("archive/" + folder_name)
    return folder_name


def get_file_name_without_extension(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return file_name


def create_archive(preds, file_name):
    global currentState_server
    currentState_server = "create_archive_folder"
    folder = create_folder_with_date(get_file_name_without_extension(file_name))

    currentState_server = "archive_original"
    # Decode and save original
    original_bytes = base64.b64decode(preds['original'])
    original_img = Image.open(io.BytesIO(original_bytes))
    original_img.save("archive/" + folder + "/original.png")

    currentState_server = "archive_visualization"
    # Decode and save visualization
    visualization_bytes = base64.b64decode(preds['visualization'])
    visualization_img = Image.open(io.BytesIO(visualization_bytes))
    visualization_img.save("archive/" + folder + "/visualization.png")

    currentState_server = "archive_batch"
    # Decode and save batch_visualizations
    batch_visualizations_bytes = base64.b64decode(preds['batch_visualizations'])
    batch_visualizations_img = Image.open(io.BytesIO(batch_visualizations_bytes))
    batch_visualizations_img.save("archive/" + folder + "/batch_visualizations.png")

    currentState_server = "archive_lime"
    # Decode and save lime_visualizations
    lime_visualizations_bytes = base64.b64decode(preds['lime_visualization'])
    lime_visualizations_img = Image.open(io.BytesIO(lime_visualizations_bytes))
    lime_visualizations_img.save("archive/" + folder + "/lime_visualizations.png")

    currentState_server = "archive_scores"
    # Convert string back to float and save scores
    scores = np.fromstring(preds['scores'][1:-1], sep=' ')
    np.savetxt("archive/" + folder + "/scores.txt", scores)

    currentState_server = "archive_scores_road"
    # Convert string back to float and save scores_road
    scores_road = np.fromstring(preds['scores_road'][1:-1], sep=' ')
    np.savetxt("archive/" + folder + "/scores_road.txt", scores_road)

    currentState_server = "archive_predictions"
    # Save predictions
    predictions = preds['predictions']
    with open("archive/" + folder + "/predictions.json", 'w') as f:
        json.dump(predictions, f)


import matplotlib.image as mpimg


def lime(image):
    image = image.numpy().transpose(1, 2, 0)

    explainer = lime_image.LimeImageExplainer()

    # Define the prediction function for the VGG16 model
    def batch_predict(images):
        images = torch.tensor(images).permute(0, 3, 1, 2)
        outputs = model(images)
        return outputs.detach().numpy()

    segments = slic(image, n_segments=100, compactness=10)

    # Generate explanations using LIME
    explanation = explainer.explain_instance(
        image,
        batch_predict,
        top_labels=1,
        num_samples=50,
        segmentation_fn=lambda x: segments,
        random_seed=42
    )

    # Retrieve the superpixel weights for the predicted class label
    predicted_label = np.argmax(batch_predict(np.expand_dims(image, 0)))
    superpixel_weights = explanation.local_exp[predicted_label]

    # Create a weighted mask based on superpixel importance
    weighted_mask = np.zeros_like(segments, dtype=np.float32)
    for idx, weight in superpixel_weights:
        weighted_mask[segments == idx] = weight

    return weighted_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='finetuning/models/pretrained_all_tumours.pt')
    args = parser.parse_args()

    if not os.path.exists('histo_images'):
        os.makedirs('histo_images')
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('archive'):
        os.makedirs('archive')
    model = load_model(args.model_path)
    isPrint = False

    app.run()