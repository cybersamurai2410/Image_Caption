# Image Caption Generator using Deep Learning

## Objective
Combine computer vision and natural language processing techniques to automatically generate descriptive captions for input images by predicting the next word in a sentence given the previous words and the features extracted from both the image and the text captions.

## Dataset
- The dataset used for this project is Flickr8k and comprises 8,000 images, each associated with five captions.

## Model Architecture
- Employ a combination of Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) networks to achieve this objective.
- The CNN (VGG16 Network) is used to extract relevant features from the input images.
- The LSTM layers are utilized to process and generate text-based captions.
- The extracted features from both modalities (image and text) are concatenated to predict the next word in the caption sequence.

![model_architecture](https://github.com/cybersamurai2410/Image_Caption/assets/66138996/c33f43a3-d4ef-42c3-8695-caf9e0674b54)

## Evaluation Metrics
- The performance of the trained model is evaluated using validation loss/accuracy metrics and the BLEU Score and has achieved over 0.5 on average.
- BLEU (Bilingual Evaluation Understudy) Score is a metric commonly used to assess the quality of machine-generated text by comparing it to reference human-generated text.
