from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
import numpy as np
from pyaudio import paInt16, PyAudio
import time

class stream_processor():
    # constant

    def __init__(self):
        pass

    def pcm2float(self, sig, dtype='float32'):
        """
        Convert PCM signal to floating point with a range from -1 to 1.
        """
        sig = np.asarray(sig)
        dtype = np.dtype(dtype)
        i = np.iinfo(sig.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig.astype(dtype) - offset) / abs_max

    def process_stream(self, y):
        # convert string of format int16 to float32
        decoded = np.fromstring(y, 'int16')
        decoded = self.pcm2float(decoded)
        return decoded
    def batch_signal(self, data, sr):
        wave = np.array(data)
        len_ = wave.shape[0]
        if len_< sr:
            wav = np.full((sr), 0, dtype=np.float32)
            wav[0:len_] = wave
        else: wav = wave[0:sr]
        waves = np.reshape(wav,[1,-1]) # 1x8000
        return waves

class vad_inference():
    # constants
    settings = {
        "model_path": "data/models/vad_v1.h5",
        "sample_rate": 8000,
        "chunk": 1024,
        "format": paInt16,
        "channels": 1,
        "swidth": 2
    }
    Labels = {1:'key_word', 0:'noise'}
    DT = 0
    dt_count = 0
    def __init__(self):
        self.initiate()

    def initiate(self):
        # stream
        self.init_stream()
        # load model
        self.model = load_model(self.settings["model_path"])
        # load stream processor
        self.process_stream = stream_processor()
        return
    
    def listen(self):
        prev_process = []
        pred_time_frame = 1 # 1 sec
        key_count = 0
        space_count = 0
        while True:
            sound_input = self.stream.read(self.settings["chunk"])
            processed = self.process_stream.process_stream(sound_input)
            # add the detected noise to list
            prev_process.extend(processed)
            # clip the list to the last one second for frame of 1
            prev_process = prev_process[-int(self.settings["sample_rate"]*pred_time_frame):]
            pred = self.run_inference(prev_process)[0]
            self.dt_count += 1
            if pred == 'key_word':
                key_count += 1
            elif pred == 'noise':
                space_count += 1
            if space_count == 2:
                key_count = 0
                space_count = 0
            elif key_count == 7:
                return 'key_word', f"dt: {self.DT*1000/self.dt_count}"

    
    def run_inference(self, data):
        t = time.time()
        input_data = self.process_stream.batch_signal(data, self.settings["sample_rate"])
        output = self.model.predict(x=input_data, verbose=0)
        dt = time.time() - t
        self.DT += dt
        # self.dt_count += 1
        #print(output)
        prediction = self.process_output_prediction(output)
        return prediction
    def process_output_prediction(self, output):
        pred_class = np.argmax(output,axis=1)
        pred_class = [self.Labels[i] for i in pred_class]
        return pred_class

    def init_stream(self):
        self.p = PyAudio()
        self.stream = self.p.open(rate=self.settings["sample_rate"],
                                    channels=self.settings["channels"],
                                    format=self.settings["format"],
                                    input=True,
                                    output=True,
                                    frames_per_buffer=self.settings["chunk"])
        return
    def close_stream(self):
        self.p.terminate()
        return

if __name__ == "__main__":
    main = vad_inference()
    print("ready to listen...")
    while True:
        word, dt = main.listen()
        if word == 'key_word':
            print(f'Key_word detected=> {dt}')