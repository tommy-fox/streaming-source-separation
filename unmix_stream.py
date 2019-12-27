import test_stream
import torch
import numpy as np
import soundfile as sf
import pyaudio
import struct
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
import wave
import _thread as thread
from queue import Queue
import time
import sys

# Load audio blocks from file
def load_audio_blocks(filename, frame_length, block_length, hop_length):
    
    # Separate audio file into overlapping blocks
    blocks = sf.blocks(filename,
                       blocksize  = frame_length + (block_length - 1) * hop_length,
                       overlap    = frame_length - hop_length,
                       fill_value = None,
                       start      = 0,
                       frames     = -1,
                       dtype      = np.float32,
                       always_2d  = False)
    return blocks

# Read audio from the input queue and put in the separated audio queue
def unmix_separate_streamer(unmix_model, blocks, mode):

    # Initialize LSTM hidden state and cell state
    h_t_minus1 = None
    c_t_minus1 = None

    # Separate input buffer with Unmix
    for audio_block in blocks:
        # If audio is mono, force to false stereo
        if audio_block.ndim != 2:
            tensor_list = [torch.from_numpy(audio_block[:]), torch.from_numpy(audio_block[:])]
            audio_block = torch.stack(tensor_list).T
            
        estimate_buffer, h_t_minus1, c_t_minus1 = test_stream.separate(audio          = audio_block,
                                                                       softmask       = True,
                                                                       alpha          = 1.0,
                                                                       targets        = ['vocals'],
                                                                       residual_model = False,
                                                                       niter          = 1,
                                                                       device         = 'cpu',
                                                                       unmix_target   = unmix_model,
                                                                       h_t_minus1     = h_t_minus1,
                                                                       c_t_minus1     = c_t_minus1)
        
        # If seeking the instrumental, subtract vocal estimate from mix
        if(mode == 'instrumental'):
            original = np.asarray(audio_block[FRAME_LENGTH-HOP_LENGTH:,:])
            separated_output = original - estimate_buffer['vocals'][FRAME_LENGTH-HOP_LENGTH:,:]
            
        else:
            separated_output = estimate_buffer['vocals'][FRAME_LENGTH-HOP_LENGTH:,:]

        # Send separated audio to output
        audio_queue.put(separated_output)
        
def write_audio_to_speakers(stream):
    while True:
        try:
            out_block = audio_queue.get()

        except audio_queue.Empty:
            pass

        else:
            # Get audio from output queue and interleave
            left_out = out_block[:,0]
            right_out = out_block[:,1]

            output  = []
            for i in range(len(left_out)):
                output.append(left_out[i])
                output.append(right_out[i])

            # Convert output value to binary data
            out_binary = struct.pack('f' *len(output), *output)

            # Write to stream
            stream.write(out_binary)

if ((len(sys.argv) != 3)):
    print('Usage: unmix_stream_run.py, filepath (.wav file), model type (instrumental, acapella, or speech)')
    sys.exit()

# Read mode input by user (speech, acapella, or instrumental)
mode = sys.argv[2]

# Audio queue for pipelining separated audio to output stream
audio_queue = Queue()

# Stream settings match Unmix's STFT settings
RATE         = 44100
HOP_LENGTH   = 1024
FRAME_LENGTH = 4096
BLOCK_LENGTH = 128

# Read wavefile file information
filename    = sys.argv[1]
wf          = wave.open(filename, 'rb')
frames      = wf.getnframes()
duration    = frames / RATE

# Separate audio file into blocks
blocks = load_audio_blocks(filename, FRAME_LENGTH, BLOCK_LENGTH, HOP_LENGTH)

# Create and open PyAudio stream
p = pyaudio.PyAudio()

stream = p.open(
                format    = pyaudio.paFloat32,
                channels  = 2,
                rate      = RATE,
                input     = False,
                output    = True)

# Load model
if(mode == 'instrumental' or 'acapella'):
    unmix_model = test_stream.load_model(target     = 'sung_vocals',
                                         device     = device)

elif(mode == 'speech'):
    unmix_model = test_stream.load_model(target     = 'vocals',
                                         device     = device)

if __name__ == '__main__':
       
    print('Playing ' + filename + '\npress Ctrl+C to quit')
    try:
        
        # Run producer of separated audio
        thread.start_new_thread(unmix_separate_streamer, (unmix_model,blocks,mode))
        
        # Run consumer of separated audio to play to speakers
        thread.start_new_thread(write_audio_to_speakers, (stream,))
             
        # Wait until file is finished playing
        time.sleep(duration)
        
    # Allow user to stop audio playback with CTRL+C
    except KeyboardInterrupt:
        print('\nQuitting...')
        
    finally:
        thread.alive = False
        sys.exit()
