from utils.generate_dataset import dataset_generator
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import  tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, Activation, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    # Note: You'll need to adjust the index depending on your dataset's directory structure
    return parts[-2]

def process_path(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    audio = decode_audio(audio_binary)
    return audio, label

class BasicMFCC(tf.keras.layers.Layer):
    def __init__(self, sample_rate=8000, mfcc_features=26, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.mfcc_features = mfcc_features
        # Define the parameters for the STFT, Mel spectrogram, and MFCC
        self.frame_length = 0.025  # Window length for the STFT (25 ms)
        self.frame_step = 0.010    # Frame step for the STFT (10 ms)
        self.mel_bins = 40         # Number of Mel bins
        self.lower_edge_hertz = 20 # Lower bound of frequency
        self.upper_edge_hertz = 4000 # Upper bound of frequency, can adjust based on Nyquist theorem and your application

    def call(self, inputs):
        # Ensure the input waveform is properly shaped and has the correct dtype
        waveforms = tf.cast(inputs, tf.float32)

        # Compute the STFT of the audio
        stfts = tf.signal.stft(waveforms, 
                               frame_length=int(self.sample_rate * self.frame_length), 
                               frame_step=int(self.sample_rate * self.frame_step), 
                               fft_length=int(self.sample_rate * self.frame_length))
        spectrograms = tf.abs(stfts)

        # Compute Mel spectrogram
        num_spectrogram_bins = stfts.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.mel_bins, num_spectrogram_bins, self.sample_rate, 
            self.lower_edge_hertz, self.upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        # Compute log Mel spectrogram
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        # Compute MFCCs from log Mel spectrograms
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :self.mfcc_features]

        return mfccs

class train():
    # constant
    settings = {"batchsize":8,
                "epochs":100, 
                "validation_ratio":0.1, 
                "testing_ratio":0.05, 
                "generate_data_flag":False,
                "learning_rate":1e-3,
                "cnn_dropout":0.2,
                "num_classes": 2,
                "dataset_path": "data/dataset/sounds/",
                "model_path": "data/models/vad_v1.h5",
                "tf-lite_path": "data/models/vad_v1.tflite"}
    sample_rate = 8000 # 8khz
    audio_length = 1 # 1 second
    val_flag = True
    test_flag = True

    def __init__(self):
        pass
    
    def begin_training(self, flag=True, save_flag=False):
        # load data
        self.data_train, self.data_val, self.data_test = self.load_dataset()
        # exit(0)
        # model architecture
        self.model = self.model_v1()
        # summary
        if flag: self.model.summary()
        # compile model
        self.model.compile(optimizer='adam', 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy'])
        # train the model
        self.begin_fiting()
        # saving
        if save_flag:
            self.model.save(self.settings["model_path"])
            self.convert_to_tflite()
        return
    
    def begin_fiting(self):
        # checkpoint for saving at each epoch when validation loss is minimum
        checkpoint = ModelCheckpoint(self.settings["model_path"], monitor='val_loss', verbose=1,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', save_frequency="epoch")
        if self.val_flag and self.test_flag:
            # fit and validate
            self.model.fit(self.data_train,epochs=self.settings["epochs"],validation_data=self.data_val, verbose=2, callbacks=[checkpoint])
            # evaluate
            self.model.evaluate(self.data_test,verbose=2)
        elif self.val_flag:
            # fit and validate
            self.model.fit(self.data_train,epochs=self.settings["epochs"],validation_data=self.data_val, verbose=2, callbacks=[checkpoint])
        elif self.test_flag:
            # fit only
            self.model.fit(self.data_train,epochs=self.settings["epochs"], verbose=2)
            # evaluate
            self.model.evaluate(self.data_test,verbose=2)
        else:
            # fit only
            self.model.fit(self.data_train,epochs=self.settings["epochs"], verbose=2)
        return
    
    def convert_to_tflite(self):
        # Load the trained TensorFlow model
        model = tf.keras.models.load_model(self.settings["model_path"])

        # Initialize the TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # (Optional) Set the optimization strategy for the converter if needed
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Convert the model
        tflite_model = converter.convert()

        # Save the converted TFLite model to the specified path
        with open(self.settings["tf-lite_path"], 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model successfully converted to TFLite format and saved to {self.settings['tf-lite_path']}")
        return

    
    def model_v1(self):
        # Define the input layer
        input_layer = Input(shape=(8000,))
        # Add the custom MFCC layer
        mfcc_layer = BasicMFCC()(input_layer)
        # Conv1D Block 1
        x = self.conv1d_block(mfcc_layer, 32, 16)
        # Conv1D Block 2
        x = self.conv1d_block(x, 32, 16)
        # Conv1D Block 3
        x = self.conv1d_block(x, 32, 16)
        # Flatten the output of the last Conv1D block
        x = Flatten()(x)
        # Add the output layer with softmax activation for binary classification
        output_layer = Dense(2, activation='softmax')(x)
        # Create the model
        model = Model(inputs=input_layer, outputs=output_layer)
        return model
        
    def conv1d_block(self, prev_layer, filters, kernel_size):
        # Conv1D Block
        conv = Conv1D(filters=filters, kernel_size=kernel_size, strides=2, padding='same')(prev_layer)
        conv = Dropout(self.settings['cnn_dropout'])(conv)
        conv = Activation('relu')(conv)
        return conv
    
    def load_dataset(self):
        # check or generate dataset
        DataGen = dataset_generator()
        # configure
        DataGen.audio_length = self.audio_length
        DataGen.sample_rate = self.sample_rate
        # generate audio data
        status = DataGen.generate_dataset(reset=False) # only create it if none existant
        if not status:
            print("Can not train without dataset, please follow the instructions in readme to add dataset")
            os._exit(0)
            
        # all is good load dataset
        train_dataset = tf.data.Dataset.list_files(str(f"{self.settings['dataset_path']}/train/*/*.wav"), shuffle=True)
        train_dataset = train_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.list_files(str(f"{self.settings['dataset_path']}/val/*/*.wav"), shuffle=False)
        val_dataset = val_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train_dataset.batch(self.settings['batchsize']).prefetch(AUTOTUNE)
        val_dataset = val_dataset.batch(self.settings['batchsize']).prefetch(AUTOTUNE)
        return train_dataset, val_dataset, val_dataset

if __name__ == "__main__":
    main = train()
    main.begin_training()