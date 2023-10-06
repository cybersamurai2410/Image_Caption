import os
import pickle # Store data
import re
from tqdm import tqdm # Progress bar

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model

base_dir = r"C:\ADITYA\Computer Science\Python\Image_Caption\flickr8k"
images_dir = os.path.join(base_dir, 'Images')
captions_file_path = os.path.join(base_dir, 'captions.txt')

# Load vgg16 model
print("Loading VGG16 model...")
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
print(model.summary())

# Extract features from image
print("Extracting image features...")
features = {}
for img_name in tqdm(os.listdir(images_dir)):
    img_path = os.path.join(images_dir, img_name)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature

# Load captions
print("Loading and processing captions...")
with open(captions_file_path, 'r') as f:
    next(f)
    captions_doc = f.read()

# Create mapping of image to captions
mapping = {}
for line in tqdm(captions_doc.split('\n')):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

# def clean(mapping):
#     for key, captions in mapping.items():
#         for i in range(len(captions)):
#             caption = captions[i]
#             caption = caption.lower()
#             caption = caption.replace('[^A-Za-z]', '')
#             caption = caption.replace('\s+', ' ')
#             caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
#             captions[i] = caption
# clean(mapping)

print("Cleaning captions...")
for key, captions in mapping.items():
    for i in range(len(captions)):
        caption = captions[i].lower()
        caption = re.sub('[^a-z\s]', '', caption)  # Only keep letters and spaces
        caption = re.sub('\s+', ' ', caption).strip()  # Remove extra spaces
        caption = 'startseq ' + caption + ' endseq'
        captions[i] = caption

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)

print(f"Vocab size: {vocab_size}") # Vocab size: 8485
print(f"Max caption length: {max_length}") # Max caption length: 35

# Save features
features_file_path = os.path.join(base_dir, 'features.pkl')
with open(features_file_path, 'wb') as f:
    pickle.dump(features, f)

# Save mapping
mapping_file_path = os.path.join(base_dir, 'mapping.pkl')
with open(mapping_file_path, 'wb') as f:
    pickle.dump(mapping, f)

# Save tokenizer
tokenizer_file_path = os.path.join(base_dir, 'tokenizer.pkl')
with open(tokenizer_file_path, 'wb') as f:
    pickle.dump(tokenizer, f)

# Save max length and vocab size
metadata = {
    'max_length': max_length,
    'vocab_size': vocab_size
}
metadata_file_path = os.path.join(base_dir, 'metadata.pkl')
with open(metadata_file_path, 'wb') as f:
    pickle.dump(metadata, f)
