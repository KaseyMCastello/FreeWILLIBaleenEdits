"""
Functions to stream entire xwav directory over UDP to simulate a real-time audio stream.
Author: Kasey Castello
Time: June 14, 2025

Credit for sub-fucntions:

get_datetime: Joshua Zingale (June 3rd, 2024)
"""

from AudioStreamDescriptor import XWAVhdr
from datetime import timedelta
import numpy as np


def get_datetime(xwav_time: float, xwav):
    '''Given the time in seconds from the beginning of an XWAV along with
    the XWAV, returns the corresponding absolute datetime'''
    
    # The length of each block in seconds
    BLOCK_LEN = 75

    # Allow xwav to be a string or XWAVhdr
    if type(xwav) != XWAVhdr:
        xwav = XWAVhdr(xwav)

    # Most recent block for the starttime
    block_offset = xwav_time % BLOCK_LEN # calculate remainder
    block_idx = int(xwav_time // BLOCK_LEN)

    return xwav.raw['dnumStart'][block_idx] + timedelta(seconds = block_offset)

# hepler function uses WAVhdr to read wav file header info and extract wav file start time as a datetime object
def extract_wav_start(path):
    
    if path.endswith('.x.wav'):
        xwav_hdr= XWAVhdr(path)
        return xwav_hdr.dtimeStart
    if path.endswith('.wav'):
        wav_hdr = WAVhdr(path)
        return wav_hdr.start

#Helper function to read xwav audio data from file. Dynamically determines the number of channels and bits per sample from the header.
def read_xwav_audio(filepath):
    # Parse header
    header = XWAVhdr(filepath)
    start_byte = header.xhd["byte_loc"][0]
    n_channels = header.xhd["NumChannels"]
    bits_per_sample = header.xhd["BitsPerSample"]
    dtype = {16: np.int16, 32: np.int32}[bits_per_sample]
    
    # Read binary audio starting from first data block
    with open(filepath, "rb") as f:
        f.seek(start_byte)
        raw = f.read()
    
    audio = np.frombuffer(raw, dtype=dtype)
    audio = audio.reshape(-1, n_channels).T  #remove interleaving for ease of testing shape: [channels, samples]
    return audio, header.xhd["SampleRate"]