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
import struct
import scipy.io.wavfile as wav


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

def reconstruct_wav_from_packet(packet_bytes, num_channels=1, sample_rate=200000, header_size=12, output_path="reconstructed.wav"):
    """
    Reconstruct a .wav file from a UDP data packet containing interleaved audio bytes.
    
    Parameters:
        packet_bytes (bytes): Full UDP packet, including header + audio
        num_channels (int): Number of interleaved channels (default 4)
        sample_rate (int): Original sampling rate in Hz (e.g., 200000)
        header_size (int): Size of timestamp header at start of packet (default 8 bytes)
        output_path (str): Where to save the reconstructed .wav file
    """

    # 1. Remove time header from the start
    audio_bytes = packet_bytes[header_size:]

    # 2. Unpack bytes to uint16
    num_samples = len(audio_bytes) // 2  # 2 bytes per uint16
    audio_uint16 = struct.unpack(f'>{num_samples}H', audio_bytes)
    audio_uint16 = np.array(audio_uint16, dtype=np.uint16)

    # 3. Convert to int16 by reversing +32768 offset
    audio_int16 = (audio_uint16.astype(np.int32) - 32768).astype(np.int16)

    # 4. Reshape into (samples, channels)
    if num_samples % num_channels != 0:
        raise ValueError(f"Data length {num_samples} is not divisible by {num_channels} channels.")

    audio_int16 = audio_int16.reshape(-1, num_channels)

    # 5. Save as .wav
    wav.write(output_path, sample_rate, audio_int16)
    print(f"WAV written to: {output_path}")