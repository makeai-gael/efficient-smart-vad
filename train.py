from utils.generate_dataset import dataset_generator
import os, time, sys, librosa
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np


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
                "model_path": "data/model/vad_v1/vad_v1.h5"}
    sample_rate = 8000 # 8khz
    audio_length = 1 # 1 second
    val_flag = True
    test_flag = True

    def __init__(self):
        pass
    
    def begin_training(self, flag=True, save_flag=False):
        # load data
        self.data_train, self.data_val, self.data_test = self.load_dataset()
        exit(0)
        if self.data_val=="": self.val_flag=False
        if self.data_test=="": self.test_flag=False

        # model architecture
        self.model = self.model_v1()
        # summary
        if flag: self.model.summary()
        # compile model
        self.model.compile(
                    loss= keras.losses.CategoricalCrossentropy(),
                    optimizer=keras.optimizers.Adam(lr=self.settings["learning_rate"]),
                    metrics=["accuracy"]
                    )
        # train the model
        self.begin_fiting()
        # saving
        if save_flag:
            self.model.save(self.settings["model_path"])
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

    def model_v1(self):
        Input = keras.Input(shape=(51, 26), dtype=tf.float32)

        # ... TBD
        x = Input

        output = layers.Softmax()(x)
        model = keras.Model(inputs=Input, outputs=output)
        return model
    
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
        # TBD
        return None, None, None
        
    def process_audio(self, src:str):
        wave, sr = librosa.load(src, mono=True, sr=self.sample_rate)
        strip_h = 26
        window_width = int(0.025*sr) # 25ms
        stride = int(0.020*sr)  #20ms
        mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=strip_h, hop_length=stride, n_fft=window_width)
        # reduce mfcc values close to 1
        mfcc = (mfcc - np.mean(mfcc))/np.std(mfcc)

        # reshape to proper 3D
        mfcc = np.reshape(mfcc,[26,-1])
        mfcc = np.transpose(mfcc,[1,0])
        return mfcc



if __name__ == "__main__":
    main = train()
    main.begin_training()