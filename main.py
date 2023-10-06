from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
import pickle
import os

from evaluation import predict_caption

base_dir = r'C:\ADITYA\Computer Science\Python\Image_Caption\flickr8k'

def load_data():
    with open(os.path.join(base_dir, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)

    with open(os.path.join(base_dir, 'mapping.pkl'), 'rb') as f:
        mapping = pickle.load(f)

    with open(os.path.join(base_dir, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(base_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    model = load_model(r'C:\ADITYA\Computer Science\Python\Image_Caption\models\best_model.h5')

    return features, mapping, tokenizer, metadata, model

def load_and_process_image(image_path):
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)

    return feature

if __name__ == "__main__":
    features, mapping, tokenizer, metadata, model = load_data()
    max_length = metadata['max_length']

    image_path = r'C:\ADITYA\Computer Science\Python\Image_Caption\flickr8k\images\667626_18933d713e.jpg'
    feature = load_and_process_image(image_path)
    caption = predict_caption(model, feature, tokenizer, max_length)
    print(caption)

    '''
    - vgg16 extracts features from image and trained model generates captions.
    '''
