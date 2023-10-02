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

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # Embedding layer that turns words (numbers) into vectors of a specific size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers) # LSTM layer for processing the embedded word vectors and produce hidden states
        self.linear = nn.Linear(hidden_size, vocab_size) # Fully connected layer to transform the LSTM's hidden state output into word scores
        self.dropout = nn.Dropout(0.5) # Dropout layer for regularization

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions)) # Turn captions into embeddings. Captions are word indices, and this turns them into vectors

        # Concatenate the features from the encoder (context) with the embeddings.
        # The features act as the initial words before the actual caption tokens.
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)

        hiddens, _ = self.lstm(embeddings) # Pass the sequence of embedded word vectors (along with the image features) through the LSTM
        outputs = self.linear(hiddens) # Get the word scores for each position in the caption sequence

        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size) # Encoder part which uses a CNN architecture to extract features from the image
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers) # Decoder part which uses an RNN architecture to generate captions from the features

    def forward(self, images, captions):
        features = self.encoderCNN(images) # Extract features from the image using the encoder
        outputs = self.decoderRNN(features, captions) # Generate captions from the features using the decoder

        return outputs

    # This function generates a caption for a given image
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = [] # List to hold the generated caption

        with torch.no_grad(): # Ensure no gradients are calculated during caption generation
            x = self.encoderCNN(image).unsqueeze(0) # Extract features from the image using the encoder
            states = None # Initial states for the LSTM

            # Generate caption up to a maximum length
            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states) # Pass the previous word (or image feature for the first iteration) through the LSTM

                # Get the most probable next word
                output = self.decoderRNN.linear(hiddens.unsqueeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item()) # Append the predicted word's index to the result_caption list
                x = self.decoderRNN.embed(predicted).unsqueeze(0) # Get the embedding of the predicted word to be the input for the next timestep

                # If the End-Of-Sequence token is predicted, stop generation
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

            # Convert the list of word indices to actual words and return
            return [vocabulary.itos[idx] for idx in result_caption]
