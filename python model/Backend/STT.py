from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

class voice():

    # Function to record audio
    def record_audio_and_transcript(self):
        sample_rate = 16000  # Required by Whisper
        record_duration = 10  # Duration in seconds
        model_size = "medium.en"
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        trans = ""
        print("Recording...")
        audio = sd.rec(int(record_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording complete.")

        output_file = "temp_audio.wav"
        wav.write(output_file, sample_rate, (audio * 32767).astype(np.int16))  # Convert to 16-bit PCM format

        print(f"Temporary audio file saved as {output_file}")

        segments, info = model.transcribe("/home/greatness-within/PycharmProjects/Minchu/temp_audio.wav", beam_size=5)
        for segment in segments:
            print(type(segment))
            trans += str(segment)
        print(trans)
        return trans