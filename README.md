# Image Caption Generator using Deep Learning

## Objective
Automatically generate descriptive captions for input images by predicting the next word in a sentence given the previous words and the features extracted from both the image and the text captions.

## Dataset
- The dataset used for this project comprises 8,000 images, each associated with five captions.
- These images and their corresponding captions serve as the training data for our captioning model.

## Model Architecture
- We employ a combination of Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) networks to achieve this objective.
- The CNN (VGG16 Network) is used to extract relevant features from the input images.
- The LSTM is utilized to process and generate text-based captions.
- The extracted features from both modalities (image and text) are concatenated to predict the next word in the caption sequence.

## Metric for Evaluation
- The performance of the trained model is evaluated using the BLEU Score.
- BLEU (Bilingual Evaluation Understudy) Score is a metric commonly used to assess the quality of machine-generated text by comparing it to reference human-generated text.
- It provides a measure of how closely the generated captions match the human-generated captions in the dataset.

## Model Variants
- Two model architectures have been implemented and evaluated in this project: the VGG16-based CNN-LSTM network.
- These architectures offer different approaches to image captioning, and their performance can be compared to determine the most effective approach for the task.

