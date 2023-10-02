import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False) # Load the Inception v3 model pretrained on ImageNet
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size) # Replace the final fully connected layer of Inception v3 to have output of size 'embed_size'

        # Activation and Dropout layers for regularization and non-linearity
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images) # Pass the images through the Inception model to get the features

        # Loop through all parameters of the inception model and set their 'requires_grad' attribute based on whether we're training the CNN
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                # Always allow gradients for the final fully connected layer
                param.requires_grad = True
            else:
                # For other layers, allow gradients only if train_CNN is True
                param.requires_grad = self.train_CNN

        # Apply ReLU activation and dropout to the features and return
        return self.dropout(self.relu(features))
