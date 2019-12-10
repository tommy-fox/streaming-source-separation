# open-unmix-real-time-stream

<b>Overview:</b><br>
This project utilizes the Open-Unmix source separation modeling architecture <br>
to produce separated audio to the loudspeakers in real-time. <br>

This was accomplished by training unidirectional LSTM models <br>
and implementing a producer-consumer multithreading system in Python. <br>

Included in the 'models' folder are trained models for sung vocals and spoken speech targets.

The model for sung vocals was trained using the MUSDB dataset <br>
and the spoken speech model was trained using a subset of 7000 examples <br>
from Mozilla's Common Voice dataset and 7000 samples from the UrbanSound8k dataset of urban noise. 

<b>Usage:</b><br>
The repository must be obtained using git LFS, as the model sizes required large file storage.

Given the provided models, the program can separate sung vocals from a musical mix or speech from environmental noise.<br>
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
<b>References:</b><br>
Stöter, F.R., Uhlich, S., Liutkus, A., Mitsufuji, Y. (2019). Open-Unmix - A Reference Implementation for Music Source Separation. Journal of Open Source Software, Open Journals, 4(41), 1667.<br>
[Open-Unmix Repository](https://github.com/sigsep/open-unmix-pytorch/blob/master/README.md)

Mozilla (2017). Mozilla Common Voice.<br> 
[Common Voice Dataset](https://voice.mozilla.org/en)


Salamon, J., Jacoby, C., & Bello, J. P. (2014, November). A dataset and taxonomy for urban sound research. In Proceedings of the 22nd ACM international conference on Multimedia (pp. 1041-1044). ACM.<br>
[UbanSound Dataset Paper](https://www.researchgate.net/profile/Justin_Salamon/publication/267269056_A_Dataset_and_Taxonomy_for_Urban_Sound_Research/links/544936af0cf2f63880810a84/A-Dataset-and-Taxonomy-for-Urban-Sound-Research.pdf)
