# Image Caption Generator using Deep Learning

## Objective
Combine computer vision and natural language processing techniques to automatically generate descriptive captions for input images by predicting the next word in a sentence given the previous words and the features extracted from both the image and the text captions.

## Dataset
- The dataset used for this project comprises 8,000 images, each associated with five captions.
- These images and their corresponding captions serve as the training data for our captioning model.

## Model Architecture
- Employ a combination of Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) networks to achieve this objective.
- The CNN (VGG16 Network) is used to extract relevant features from the input images.
- The LSTM layers are utilized to process and generate text-based captions.
- The extracted features from both modalities (image and text) are concatenated to predict the next word in the caption sequence.

## Metric for Evaluation
- The performance of the trained model is evaluated using the BLEU Score.
- BLEU (Bilingual Evaluation Understudy) Score is a metric commonly used to assess the quality of machine-generated text by comparing it to reference human-generated text.
