import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

base_dir = r'C:\ADITYA\Computer Science\Python\Image_Caption\flickr8k'

# Load mapping
features_file_path = os.path.join(base_dir, 'features.pkl')
with open(features_file_path, 'rb') as f:
    features = pickle.load(f)

# Load mapping
mapping_file_path = os.path.join(base_dir, 'mapping.pkl')
with open(mapping_file_path, 'rb') as f:
    mapping = pickle.load(f)

# Load tokenizer
tokenizer_file_path = os.path.join(base_dir, 'tokenizer.pkl')
with open(tokenizer_file_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Load metadata
metadata_file_path = os.path.join(base_dir, 'metadata.pkl')
with open(metadata_file_path, 'rb') as f:
    metadata = pickle.load(f)
max_length = metadata['max_length']
vocab_size = metadata['vocab_size']

# Load the test set
test_file_path = os.path.join(base_dir, 'test_set.pkl')
with open(test_file_path, 'rb') as f:
    test = pickle.load(f)

# Load trained model
model = load_model(r'C:\ADITYA\Computer Science\Python\Image_Caption\models\best_model.h5')

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    # return in_text
    return in_text.replace('startseq', '').replace('endseq', '').strip()

def generate_caption(image_name):
    # load the image
    image_id = image_name.split('.')[0]
    img_path = os.path.join(base_dir, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)

if __name__ == "__main__":

    # validate with test data
    actual, predicted = list(), list()

    for key in tqdm(test):
        # get actual caption
        captions = mapping[key]
        # predict the caption for image
        y_pred = predict_caption(model, features[key], tokenizer, max_length)
        # split into words
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        # append to the list
        actual.append(actual_captions)
        predicted.append(y_pred)

    # calcuate BLEU score for text data
    print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

    generate_caption("1001773457_577c3a7d70.jpg")
