import os
import io
import base64
from PIL import Image
import numpy as np
import requests
import json

url = 'http://localhost:5000/process_image'
image_path = 'test.png'

with open(image_path, 'rb') as file:
    files = {'image': file}
    response = requests.post(url, files=files)

if response.status_code == 200:
	if not os.path.exists('results'):
		os.makedirs('results')

	preds = response.json()
	# Decode and save visualization
	visualization_bytes = base64.b64decode(preds['visualization'])
	visualization_img = Image.open(io.BytesIO(visualization_bytes))
	visualization_img.save('results/visualization.png')

	# Decode and save batch_visualizations
	batch_visualizations_bytes = base64.b64decode(preds['batch_visualizations'])
	batch_visualizations_img = Image.open(io.BytesIO(batch_visualizations_bytes))
	batch_visualizations_img.save('results/batch_visualizations.png')

	# Convert string back to float and save scores
	scores = np.fromstring(preds['scores'][1:-1], sep=' ')
	np.savetxt('results/scores.txt', scores)

	# Convert string back to float and save scores_road
	scores_road = np.fromstring(preds['scores_road'][1:-1], sep=' ')
	np.savetxt('results/scores_road.txt', scores_road)

	# Save predictions
	predictions = preds['predictions']
	with open('results/predictions.json', 'w') as f:
		json.dump(predictions, f)
else:
    print('Error:', response.status_code)
