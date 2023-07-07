from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
from finetuning.ImageFolderDataset import ImageFolderDataset
from finetuning.ImageFolderDataset import loadAndPrepareData, countSamplesPerClass
from finetuning.TumorClassifier import TumorClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from matplotlib import pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.metrics.road import ROADCombined
import time
import base64

def load_model():# Imagenet pretrained vgg16 model
    vgg16_imagenet_path = "vgg16.tv_in1k/pytorch_model.bin"

    # Brain tumor trained vgg16 model
    model_path = "finetuning/models/pretrained_all_tumours.pt"

    model = timm.create_model('vgg16', pretrained=False)

    # load the pretrained model weights
    state_dict = torch.load(vgg16_imagenet_path, map_location=torch.device('cpu'))#timm.create_model('vgg19', pretrained=True)
    model.load_state_dict(state_dict)

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
    predictions = [(tumor_type_dict[idx], top_probs[0][i].item()) for i, idx in enumerate(top_class_indices[0].tolist())]

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
    rgb_img = np.clip(input_tensor.squeeze().permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

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
    #print(f'CamMultImageConfidenceChange Score: {scores}')
    #print(f'ROAD Score: {scores_road}')

    return rgb_img, visualization, batch_visualizations, scores[0], scores_road[0]

tumor_type_dict = {0: 'Adamantinomatous craniopharyngioma', 1: 'Anaplastic astrocytoma, IDH-mutant', 2: 'Anaplastic astrocytoma, IDH-wildtype', 3: 'Anaplastic ependymoma', 4: 'Anaplastic ganglioglioma', 5: 'Anaplastic meningioma', 6: 'Anaplastic oligodendroglioma, IDH-mutant and 1p-19q codeleted', 7: 'Anaplastic pleomorphic xanthoastrocytoma', 8: 'Angiocentric glioma', 9: 'Angiomatous meningioma', 10: 'Angiosarcoma', 11: 'Astroblastoma', 12: 'Atypical choroid plexus papilloma', 13: 'Atypical meningioma', 14: 'Atypical teratoid-rhabdoid tumour', 15: 'Cellular schwannoma', 16: 'Central neurocytoma', 17: 'Cerebellar liponeurocytoma', 18: 'Chondrosarcoma', 19: 'Chordoid glioma of the third ventricle', 20: 'Chordoid meningioma', 21: 'Chordoma', 22: 'Choriocarcinoma', 23: 'Choroid plexus carcinoma', 24: 'Choroid plexus papilloma', 25: 'Clear cell meningioma', 26: 'CNS ganglioneuroblastoma', 27: 'Control', 28: 'Crystal-storing histiocytosis', 29: 'Desmoplastic infantile astrocytoma and ganglioglioma', 30: 'Diffuse astrocytoma, IDH-mutant', 31: 'Diffuse astrocytoma, IDH-wildtype', 32: 'Diffuse large B-cell lymphoma of the CNS', 33: 'Diffuse leptomeningeal glioneuronal tumour', 34: 'Diffuse midline glioma, H3 K27M-mutant', 35: 'Dysembryoplastic neuroepithelial tumour', 36: 'Dysplastic cerebellar gangliocytoma', 37: 'EBV-positive diffuse large B-cell lymphoma, NOS', 38: 'Embryonal carcinoma', 39: 'Embryonal tumour with multilayered rosettes, C19MC-altered', 40: 'Ependymoma', 41: 'Ependymoma, RELA fusion-positive', 42: 'Epitheloid MPNST', 43: 'Erdheim-Chester disease', 44: 'Ewing sarcoma', 45: 'Extraventricular neurocytoma', 46: 'Fibrosarcoma', 47: 'Fibrous meningioma', 48: 'Follicular lymphoma', 49: 'Gangliocytoma', 50: 'Ganglioglioma', 51: 'Ganglioneuroma', 52: 'Gemistocytic astrocytoma, IDH-mutant', 53: 'Germinoma', 54: 'Giant cell glioblastoma', 55: 'Glioblastoma, IDH-mutant', 56: 'Glioblastoma, IDH-wildtype', 57: 'Gliosarcoma', 58: 'Granular cell tumour of the sellar region', 59: 'Haemangioblastoma', 60: 'Haemangioma', 61: 'Haemangiopericytoma', 62: 'Hybrid nerve sheath tumours', 63: 'Immature teratoma', 64: 'Immunodeficiency-associated CNS lymphoma', 65: 'Inflammatory myofibroblastic tumour', 66: 'Intravascular large B-cell lymphoma', 67: 'Juvenile xanthogranuloma', 68: 'Langerhans cell histiocytosis', 69: 'Leiomyoma', 70: 'Leiomyosarcoma', 71: 'Lipoma', 72: 'Liposarcoma', 73: 'Low-grade B-cell lymphomas of the CNS', 74: 'Lymphoplasmacyte-rich meningioma', 75: 'Malignant peripheral nerve sheath tumour', 76: 'MALT lymphoma of the dura', 77: 'Mature teratoma', 78: 'Medulloblastoma, non-WNT-non-SHH', 79: 'Medulloblastoma, SHH-activated and TP53-mutant', 80: 'Medulloblastoma, SHH-activated and TP53-wildtype', 81: 'Medulloblastoma, WNT-activated', 82: 'Melanotic schwannoma', 83: 'Meningeal melanocytoma', 84: 'Meningeal melanoma', 85: 'Meningothelial meningioma', 86: 'Metaplastic meningioma', 87: 'Metastatic tumours', 88: 'Microcystic meningioma', 89: 'Mixed germ cell tumour', 90: 'Myxopapillary ependymoma', 91: 'Neurofibroma', 92: 'Olfactory neuroblastoma', 93: 'Oligodendroglioma, IDH-mutant and 1p-19q codeleted', 94: 'Osteochondroma', 95: 'Osteoma', 96: 'Osteosarcoma', 97: 'Papillary craniopharyngioma', 98: 'Papillary ependymoma', 99: 'Papillary glioneuronal tumour', 100: 'Papillary meningioma', 101: 'Papillary tumour of the pineal region', 102: 'Paraganglioma', 103: 'Perineurioma', 104: 'Pilocytic astrocytoma', 105: 'Pilomyxoid astrocytoma', 106: 'Pineal parenchymal tumour of intermediate differentiation', 107: 'Pineoblastoma', 108: 'Pineocytoma', 109: 'Pituicytoma', 110: 'Pituitary adenoma', 111: 'Pleomorphic xanthoastrocytoma', 112: 'Plexiform neurofibroma', 113: 'Psammomatous meningioma', 114: 'Rhabdoid meningioma', 115: 'Rhabdomyosarcoma', 116: 'Rosette-forming glioneuronal tumour', 117: 'Schwannoma', 118: 'Secretory meningioma', 119: 'Spindle cell oncocytoma', 120: 'Subependymal giant cell astrocytoma', 121: 'Subependymoma', 122: 'T-cell and NK-T-cell lymphomas of the CNS', 123: 'Tanycytic ependymoma', 124: 'Teratoma with malignant transformation', 125: 'Transitional meningioma', 126: 'Undifferentiated pleomorphic sarcoma'}

# Initialize Flask application
app = Flask(__name__)

# Define a route for receiving image and returning the processed image
@app.route('/process_image', methods=['POST'])
def process_image():
    # Receive the image from the client

    try:
        image_file = request.files['image']
        image = Image.open(image_file)
    except:
        return 'Image not found', 400
    
    # Preprocess the image
    image = preprocess_image(image)

    start_time = time.time()
    preds = get_top_probs(model, image)
    print('Prediction time: {:.2f} seconds', time.time() - start_time)

    start_time = time.time()
    rgb_img, visualization, batch_visualizations, scores, scores_road = calcGradCAM(image)
    print(visualization)
    print(visualization.shape)
    print(batch_visualizations)
    print(batch_visualizations.shape)
    print('GradCAM time: {:.2f} seconds', time.time() - start_time)

    start_time = time.time()
    # Convert numpy array to PIL Image and then to byte stream
    visualization_pil = Image.fromarray(visualization.astype(np.uint8))
    byte_stream1 = io.BytesIO()
    visualization_pil.save(byte_stream1, format='PNG')
    byte_stream1.seek(0)
    visualization_stream = base64.b64encode(byte_stream1.read()).decode()

    # Convert tensor to PIL Image and then to byte stream
    batch_visualization_pil = transforms.ToPILImage()(batch_visualizations.squeeze(0))
    byte_stream2 = io.BytesIO()
    batch_visualization_pil.save(byte_stream2, format='PNG')
    byte_stream2.seek(0)
    batch_visualization_stream = base64.b64encode(byte_stream2.read()).decode()
    print('Image conversion time: {:.2f} seconds', time.time() - start_time)

    response = {
        'predictions': preds,
        'visualization': visualization_stream,
        'batch_visualizations': batch_visualization_stream,
        'scores': str(scores),
        'scores_road': str(scores_road)
    }

    return response
    
    # Pass the image to the machine learning model
    #output_image = model.predict(np.expand_dims(image, axis=0))[0]
    
    # Postprocess the output image
    #output_image = postprocess_image(output_image)
    
    # Convert the output image to bytes
    #output_image_bytes = io.BytesIO()
    #output_image.save(output_image_bytes, format='PNG')
    #output_image_bytes.seek(0)
    
    # Return the output image as a response
    return None#output_image_bytes.getvalue()

def preprocess_image(image):
    
    # Define the mean and standard deviation for normalization (same as ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define the image transformation pipeline with augmentation
    transform = transforms.Compose([
        transforms.Resize(512),                           # Resize the image to 512x512 pixels
        transforms.CenterCrop(512),                        # Perform a center crop of size 512x512 pixels
        transforms.ToTensor(),                              # Convert the image to a tensor
        transforms.Normalize(mean=mean, std=std)  # Normalize the image using the specified mean and standard deviation
    ])

    processed_image = transform(image)
    return processed_image

def postprocess_image(image):
    # Implement your image postprocessing logic here
    # This could include converting the output to the desired format, applying filters, etc.
    # Return the postprocessed image
    return image

if __name__ == '__main__':
    model = load_model()
    app.run()