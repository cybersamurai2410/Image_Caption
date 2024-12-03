# Image Caption Generator using Deep Learning

## Objective
Combine computer vision and natural language processing techniques to automatically generate descriptive captions for input images by predicting the next word in a sentence given the previous words and the features extracted from both the image and the text captions.

## Dataset
The dataset used for this project is Flickr8k and comprises 8,000 images, each associated with five captions.

## Model Architecture
- Employ a combination of Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) networks to achieve this objective.
- The CNN (VGG16 Network) is used to extract relevant features from the input images.
- The LSTM layers are utilized to process and generate text-based captions.
- The extracted features from both modalities (image and text) are concatenated to predict the next word in the caption sequence.

![model_architecture](https://github.com/cybersamurai2410/Image_Caption/assets/66138996/c33f43a3-d4ef-42c3-8695-caf9e0674b54)

## Evaluation Metrics
- The performance of the trained model is evaluated using validation loss/accuracy metrics and the BLEU Score and has achieved over 0.5 on average.
- BLEU (Bilingual Evaluation Understudy) Score is a metric commonly used to assess the quality of machine-generated text by comparing it to reference human-generated text.

## Screenshots
<img width="571" alt="imgcaption" src="https://github.com/cybersamurai2410/Image_Caption/assets/66138996/b4f898dd-24aa-427c-98de-d5a37540ea33">
<img width="468" alt="imgcaption2" src="https://github.com/cybersamurai2410/Image_Caption/assets/66138996/2c974f55-73be-46b8-9030-93fff6456851">

# Vision-Language 
Application open-source models from Hugging Face for the vision-language tasks:
- Image captioning
- Image retrieval
- Visual Q&A

[Click to watch demo]()

## Screenshots
<img width="536" alt="img_caption" src="https://github.com/user-attachments/assets/5f4bcf2a-cad6-41fa-b81f-4d4eacc908cb">
<img width="644" alt="img_retrieval" src="https://github.com/user-attachments/assets/bb4852c3-8c7e-4af0-88e6-9b6d33e86c13">
<img width="489" alt="visqa" src="https://github.com/user-attachments/assets/55b107d6-9a9a-4701-9ca4-b73273b158f4">

