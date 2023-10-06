import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

base_dir = r'C:\ADITYA\Computer Science\Python\Image_Caption\flickr8k'

# Load features
print('Loading features...')
features_file_path = os.path.join(base_dir, 'features.pkl')
with open(features_file_path, 'rb') as f:
    features = pickle.load(f)

# Load mapping
print('Loading mapping...')
mapping_file_path = os.path.join(base_dir, 'mapping.pkl')
with open(mapping_file_path, 'rb') as f:
    mapping = pickle.load(f)

# Load tokenizer
print('Loading tokenizer...')
tokenizer_file_path = os.path.join(base_dir, 'tokenizer.pkl')
with open(tokenizer_file_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Load metadata
print('Loading metadata...')
metadata_file_path = os.path.join(base_dir, 'metadata.pkl')
with open(metadata_file_path, 'rb') as f:
    metadata = pickle.load(f)
max_length = metadata['max_length']
vocab_size = metadata['vocab_size']

print('Splitting dataset...')
image_ids = list(mapping.keys())
train_split = int(len(image_ids) * 0.80)
val_split = train_split + int(len(image_ids) * 0.10)
train = image_ids[:train_split]
val = image_ids[train_split:val_split]
test = image_ids[val_split:]

# Save the test set
test_file_path = os.path.join(base_dir, 'test_set.pkl')
with open(test_file_path, 'wb') as f:
    pickle.dump(test, f)

# Create data generator to get data in batch avoid high memory consumption
# def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
#     # loop over images
#     X1, X2, y = list(), list(), list()
#     n = 0
#     while 1:
#         for key in data_keys:
#             n += 1
#             captions = mapping[key]
#             # process each caption
#             for caption in captions:
#                 # encode the sequence
#                 seq = tokenizer.texts_to_sequences([caption])[0]
#                 # split the sequence into X, y pairs
#                 for i in range(1, len(seq)):
#                     # split into input and output pairs
#                     in_seq, out_seq = seq[:i], seq[i]
#                     # pad input sequence
#                     in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
#                     # encode output sequence
#                     out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
#
#                     # store the sequences
#                     X1.append(features[key][0])
#                     X2.append(in_seq)
#                     y.append(out_seq)
#             if n == batch_size:
#                 X1, X2, y = np.array(X1), np.array(X2), np.array(y)
#                 yield [X1, X2], y
#                 X1, X2, y = list(), list(), list()
#                 n = 0

def preprocess_data(data_keys, mapping, features, tokenizer, max_length, vocab_size):
    X1, X2, y = list(), list(), list()
    for key in tqdm(data_keys):
        captions = mapping[key]
        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                X1.append(features[key][0])
                X2.append(in_seq)
                y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)

print('Preprocessing training set...')
trainX1, trainX2, trainY = preprocess_data(train, mapping, features, tokenizer, max_length, vocab_size)
print('Preprocessing validation set...')
valX1, valX2, valY = preprocess_data(val, mapping, features, tokenizer, max_length, vocab_size)

print('Building model...')
# Encoder model
# image feature layers
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# Decoder model
decoder1 = fe2 + se3
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# plot_model(model, show_shapes=True)

# Define the checkpoint file path where the model weights will be saved
checkpoint_filepath = "model_checkpoint.h5"
early_stopping_patience = 10  # Number of epochs with no improvement to wait before stopping


# Create a ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    checkpoint_filepath,   # Specify the file path to save the checkpoint
    save_best_only=True,   # Save only the best model based on validation loss
    save_weights_only=True,  # Save only the model weights, not the entire model
    monitor='val_loss',    # Monitor validation loss
    mode='min',            # Minimize the monitored quantity (validation loss)
    verbose=1               # Display progress during checkpoint saving
)

# Create an EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=early_stopping_patience,
    mode='min',
    verbose=1,
    restore_best_weights=True  # Restore model weights to the best checkpoint on early stopping
)

# Load the model weights and optimizer state from the last checkpoint
try:
    model.load_weights(checkpoint_filepath)
    print("Resuming training from the last checkpoint.")
except (OSError, ValueError):
    print("No checkpoint found. Training from scratch.")

print('Training...')
# train the model
epochs = 20
batch_size = 32

history = model.fit([trainX1, trainX2], trainY, validation_data=([valX1, valX2], valY),
                    epochs=epochs, verbose=1, callbacks=[checkpoint, early_stopping])

# steps = len(train) // batch_size
# for i in range(epochs):
#     print('epoch - ', i)
#     generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
#     history = model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1,
#                         callbacks=[checkpoint, early_stopping])

print('Metrics:', history.history.keys())
print('Plotting training results...')
def plot_training_metrics(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Plot training results
plot_training_metrics(history)

print('Saving model...')
model_path = os.path.join(r'C:\ADITYA\Computer Science\Python\Image_Caption\models', 'best_model.h5')
model.save(model_path)
