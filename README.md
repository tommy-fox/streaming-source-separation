# open-unmix-real-time-stream

<b>Overview:</b><br>
This project utilizes the Open-Unmix source separation modeling architecture <br>
to produce separated audio to the loudspeakers in real-time. <br>

This was accomplished by training unidirectional LSTM models <br>
and implementing a producer-consumer multithreading system in Python. <br>

Included in the 'models' folder are models for sung vocals and spoken speech targets.

The model for sung vocals was trained using the MUSDB dataset <br>
and the spoken speech model was trained using a subset of 7000 examples <br>
from Mozilla's Common Voice dataset and 7000 samples from the UrbanSound8k dataset of environmental urban noise. 

<b>Usage:</b><br>
The repository must be obtained using git LFS, as the model sizes required large file storage.

Given the provided models, the program can separate sung vocals from a musical mix or speech from environmental noise.
When evaluating music files, either sung vocals or the backing instruments may be extracted.

<b>Examples:</b><br>
```
python3 unmix_stream.py path_to_music_wav_file acapelella
```
```
python3 unmix_stream.py path_to_music_wav_file instrumental
```
```
python3 unmix_stream.py path_to_noisy_speech_wav_file speech
```
