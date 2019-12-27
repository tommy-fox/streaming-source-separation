# streaming-source-separation

<b>Overview:</b><br>
This project utilizes the Open-Unmix source separation modeling architecture <br>
to produce separated audio to the loudspeakers as it is being processed. <br><br>

<img src = "https://camo.githubusercontent.com/b5a867eb9b7c0a0fa0d8dc90962628ea41a8a374/68747470733a2f2f646f63732e676f6f676c652e636f6d2f64726177696e67732f642f652f32504143582d317654506f516950776d6466455434705a68756531527647376f45554a7a376555655176437536767a59654b5277486c366279345252546e7068496d534b4d306b354b587739725a316949466e7047572f7075623f773d39353926683d333038">
<br>
The original Open-Unmix repository can be found [here](https://github.com/sigsep/open-unmix-pytorch/blob/master/README.md).

Open-Unmix utilizes 3 bidirectional LSTM layers to generate a spectral mask of its targeted source.<br>
The final separation is produced by Wiener filtering the original mixed signal with the estimated spectral mask.

The online, streaming version was accomplished by training unidirectional LSTM models <br>
and implementing a producer-consumer multithreading system in Python. <br>

Included in the 'models' folder are trained models for sung vocals and spoken speech targets.<br>
These were uploaded using git lfs and may require lfs in order to obtain them locally.

The model for sung vocals was trained using the MUSDB dataset <br>
and the spoken speech model was trained using a subset of 7000 examples <br>
from Mozilla's Common Voice dataset and 7000 samples from the UrbanSound8k dataset of urban noise. 

<b>Examples:</b><br>
Given the provided models, the program can separate sung vocals from a musical mix <br>
or speech from environmental noise.<br>
When evaluating music files, either sung vocals or the backing instruments may be extracted.
```
python3 unmix_stream.py path_to_music_file.wav acapella
```
```
python3 unmix_stream.py path_to_music_file.wav instrumental
```
```
python3 unmix_stream.py path_to_noisy_speech_file.wav speech
```
<b>References:</b><br>
Stöter, F.R., Uhlich, S., Liutkus, A., Mitsufuji, Y. (2019). Open-Unmix - A Reference Implementation for Music Source Separation. Journal of Open Source Software, Open Journals, 4(41), 1667.<br>
[Open-Unmix Repository](https://github.com/sigsep/open-unmix-pytorch/blob/master/README.md)

Mozilla (2017). Mozilla Common Voice.<br> 
[Common Voice Dataset](https://voice.mozilla.org/en)


Salamon, J., Jacoby, C., & Bello, J. P. (2014, November). A dataset and taxonomy for urban sound research. In Proceedings of the 22nd ACM international conference on Multimedia (pp. 1041-1044). ACM.<br>
[UbanSound Dataset Paper](https://www.researchgate.net/profile/Justin_Salamon/publication/267269056_A_Dataset_and_Taxonomy_for_Urban_Sound_Research/links/544936af0cf2f63880810a84/A-Dataset-and-Taxonomy-for-Urban-Sound-Research.pdf)
