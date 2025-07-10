from encodings.punycode import T
from pyexpat.errors import XML_ERROR_UNKNOWN_ENCODING
from re import X
import struct
import socket
import numpy as np
import time
from datetime import timedelta
import argparse
import os
import sys
import multiprocessing
from AudioStreamDescriptor import XWAVhdr
import scipy.io.wavfile as wav
import math

#Added from npy version for xwavs
import soundfile as sf

#Local utilities (assumes these functions exist in utils_xWavSim.py)
from firmware_config.firmware_1240 import NUM_CHAN
from utils_xWavSim import get_datetime, extract_wav_start, read_xwav_audio, reconstruct_wav_from_packet

# Local utilities (assumes these functions exist in utils.py)
from utils import SetHighPriority, ReadBinaryData,  InterleaveData, ScaleData, ConvertToBytes, Sleep

sys.argv.extend([
    "--port", "1045",
    "--ip", "192.168.137.2",
    "--data_dir", r"E:\Lab Work\RealtimeWork"
])

# Ensure this process runs with high priority.
SetHighPriority(15)  # nice value = -15

class ArgumentParserService:
    """Handles command-line argument parsing, providing a single responsibility."""
    @staticmethod
    def parseArguments():
        #Modified to remove fully simulated data and use xwav files instead.
        parser = argparse.ArgumentParser(description='Program command line arguments')
        parser.add_argument('--port', default=1045, type=int, help='UDP port to send data to')
        parser.add_argument('--ip', default="192.168.7.2", type=str, help='IP address to send data to')
        parser.add_argument('--data_dir', default="", type=str, help='Directory containing .x.wav data files')
        parser.add_argument('--loop', action='store_true', help='Enable looping over the data')
        parser.add_argument('--stretch', action='store_true', help='Normalize data values to the min/max range of 16-bit unsigned int')
        parser.add_argument('--imu', action='store_true', help='Read in IMU data from file')
        parsedArgs = parser.parse_args()
        return parsedArgs

class XWAVFileProcessor:
    """
    Responsible for loading and transforming .xwav files into a dictionary of 75s data chunks, each with a timestamp. 
    This will later be converted to a UDP packet for transmission.


    return: chunk_dict (format (chunk start time: 75s of audio data.))
    
    NOTE: If there are any gaps in the data, they will be padded on the end of chunks as zeros.
    Added by K.C.
    Adapts .npy processor to  handle .xwav files, including reading headers and data.
    """
    @staticmethod
    def processXwavFile(xwav_file_path, return_dict, expected_channels, fw, packet_time):
        """
        Load the .xwav file and prepare it for UDP transmission.
        The processed bytes are stored in a shared returnDict under 'dataBytes'.
        """
        print("Loading file:", xwav_file_path)
        # Load the xwav file header to extract metadata
        xwav = XWAVhdr(xwav_file_path)
        sr = xwav.xhd['SampleRate']
        n_channels = xwav.xhd['NumChannels']
        bps = xwav.xhd['BitsPerSample']
        bytes_per_sample = bps / 8

        # Compute number of samples per raw file
        samples_per_raw = [int(b / (n_channels * bytes_per_sample)) for b in xwav.xhd['byte_length']]
        dnum_start = xwav.raw['dnumStart'] # datetime list of raw file starts
        chunk_time = 75
        chunk_samples = chunk_time * sr

        current_time = None
        prev_end_time = None
        sample_ptr = 0

        chunk_dict = {}
        
        waveform, sr = read_xwav_audio(xwav_file_path)
        buffer = np.empty((waveform.shape[0], 0), dtype=waveform.dtype)
        
        for i, n_samples in enumerate(samples_per_raw):
            raw_time = dnum_start[i]
            raw_audio = waveform[:, sample_ptr:sample_ptr + n_samples]
            sample_ptr += n_samples
            # Check if the raw audo has any time gaps in it. If so, fill the gaps with zeros.
            if(prev_end_time and (raw_time - prev_end_time).total_seconds() > 1):
                gap_sec = (raw_time - prev_end_time).total_seconds()
                # Pad and save buffer if it has any data
                if buffer.shape[1] > 0:
                    pad_len = chunk_samples - buffer.shape[1]
                    padded = np.pad(buffer, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
                    chunk_dict[current_time] = padded
                # Insert silent chunks to fill the time gap
                num_silent_chunks = int(gap_sec // chunk_time)
                for g in range(num_silent_chunks):
                    zero_chunk = np.zeros((waveform.shape[0], chunk_samples), dtype=waveform.dtype)
                    silent_time = prev_end_time + timedelta(seconds=g * chunk_time)
                    chunk_dict[silent_time] = zero_chunk
                # Reset buffer and current time
                buffer = raw_audio
                current_time = raw_time
            #Otherwise, no gap and we can append the raw audio to the buffer
            else:
                # No gap, continue accumulating
                if buffer.shape[1] == 0:
                    current_time = raw_time
                buffer = np.concatenate((buffer, raw_audio), axis=1)
            while( buffer.shape[1] >= chunk_samples):
                chunk = buffer[:, :chunk_samples]
                chunk_dict[current_time] = chunk
                buffer = buffer[:, chunk_samples:]
                current_time += timedelta(seconds=chunk_time)
            prev_end_time = raw_time + timedelta(seconds=n_samples / sr)
        # Final chunk (pad if needed)
        if buffer.shape[1] > 0:
            padded = torch.nn.functional.pad(buffer, (0, chunk_samples - buffer.shape[1]), 'constant', 0)
            chunk_dict[current_time] = padded
       
        #Check if we should interleave the data.
        #1240: 4-Channel. -1:Harp 1 Channel (Change this after discuss with Joe)
        if fw == 1240 or fw==-1:
            for start_time in chunk_dict:
                chunk = chunk_dict[start_time]              # shape: (channels, chunk_samples)
                interleaved_chunk = InterleaveData(chunk)   # flatten column-wise, shape: (channels * chunk_samples,)
                chunk_dict[start_time] = interleaved_chunk  # replace original chunk with interleaved vector
        else:
            print("Unsupported firmware version")
            sys.exit()
       
        #Allows user to choose to use multiprocessing manager or not. If using multiple xwavs in folder, multiprocessing is recommended.
        if return_dict is not None:
            return_dict["chunk_dict"] = chunk_dict  # store raw chunks here
        else:
            return chunk_dict

class DataSimulator:
    """
    Manages the main simulation loop, sending data packets via UDP, handling IMU data (if any), 
    and orchestrating time or data glitches. 
    """
    def __init__(self, arguments):
        self.arguments = arguments

        # Handle IP override
        if self.arguments.ip == "self":
            self.arguments.ip = "127.0.0.1"
        
        # Prepare list of .x.wav data files
        self.xwavFiles = [ os.path.join(self.arguments.data_dir, f) for f in os.listdir(self.arguments.data_dir) if f.endswith('.x.wav')]

        if(len(self.xwavFiles) == 0):
            print(f"ERROR: No .x.wav files found in directory: {self.arguments.data_dir}. Exiting.")
            sys.exit()

        header = XWAVhdr(self.xwavFiles[0])
        num_channels = header.xhd["NumChannels"]
        sr = header.xhd['SampleRate']

        # Determine firmware based on number of channels
        if num_channels == 4:
            import firmware_config.firmware_1240 as fwConfig
            self.fw = 1240
            print(f"Simulating firmware version: 1240")
        elif num_channels == 1:
            import firmware_config.firmware_testH1C as fwConfig
            self.fw = -1  # Harp 1 Channel (Change this to the real version after discussing with Joe)
            print(f"Simulating firmware version: KC FAKE (REPLACE AFTER JOE TALK")
        else:
            print(f"ERROR: Unsupported channel count: {num_channels}")
            sys.exit()

        # Use inferred firmware constants
        self.packetSize = fwConfig.PACKET_SIZE 
        self.numChannels = fwConfig.NUM_CHAN
        self.dataSize = fwConfig.DATA_SIZE
        self.bytesPerSample = fwConfig.BYTES_PER_SAMP
        self.microIncrement = fwConfig.MICRO_INCR
        self.sr = fwConfig.SAMPLE_RATE
        self.headSize = fwConfig.HEAD_SIZE

        #Check that the firmware the user wants to simulate is compatible with the xwav file
        if num_channels != self.numChannels:
           print(f"ERROR: {num_channels} XWAV Channel(s) detected from file but chosen FW expects {self.numChannels} channels. Change firmware or xwav directory. Exiting.")
           sys.exit() 
        if sr != self.sr:
           print(f"ERROR: {sr} Hz detected from file but chosen FW expects {self.sr} Hz. Change firmware or xwav directory. Exiting.")
           sys.exit() 

        # Print initial settings
        print(f"Sending data to {self.arguments.ip} on port {self.arguments.port}")

        #### Initialize multiprocessing manager and dictionary for data exchange
        # spin up a small “manager server” process under the hood to hold Python objects in a shared memory space, 
        # and hands out proxy objects back to your processes.
        self.manager = multiprocessing.Manager() 
        # This asks the manager server for a new, empty dictionary proxy.. This serves as a shared buffer between processes
        self.returnDict = self.manager.dict()

        # Initialize socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def preloadFirstFile(self):
        """
        Preload the first file's data synchronously.
        """
        XWAVFileProcessor.processXwavFile( xwav_file_path=self.xwavFiles[0], return_dict=self.returnDict,expected_channels=self.numChannels, fw=self.fw,  packet_time=self.microIncrement)
        return self.returnDict["chunk_dict"]

    def startFilePreloadProcess(self, fileIndex, returnDict):
        """
        Launch a separate process to preload the next file's data.
        """
        processNext = multiprocessing.Process( target=XWAVFileProcessor.processXwavFile,
            args=(
                self.xwavFiles[fileIndex],
                returnDict,
                self.numChannels,
                self.fw,
                self.microIncrement
            )
        )
        processNext.start()
        return processNext

    def run(self):
        """
        Main loop that streams data from .npy files over UDP, handles optional looping, 
        time/data glitches, and IMU data concatenation.
        """
        currentChunkHolder = self.preloadFirstFile()
        
        # Start preloading the next file if available
        nextProcess = None
        nextReturnDict = None
        currentFileIndex = 0

        firstPacketSent = False
        packet_interval = self.microIncrement / 1e6 # Convert microseconds to seconds for sleep timing
        packetIndex = 0;
        realWorldStartTime = time.time()
        while True: 
            
            for start_time, chunk_data in currentChunkHolder.items():
                currentDataBytes = ConvertToBytes(chunk_data)
                dataChunkIndex = 0
                time1 = time.time()
                while True:
                    startByte = dataChunkIndex * self.dataSize
                    endByte = (dataChunkIndex + 1) * self.dataSize

                    if startByte >= len(currentDataBytes):
                        # If we've reached the end of the current data, break to load next chunk
                        break

                    #Send the right time in the packet. (Data chunk time start plus the increment times its packet number.)
                    packet_time = start_time + timedelta(seconds=dataChunkIndex * (self.microIncrement / 1e6))
                    
                    # Prepare timestamp fields
                    year = packet_time.year - 2000
                    month = packet_time.month
                    day = packet_time.day
                    hour = packet_time.hour
                    minute = packet_time.minute
                    second = packet_time.second
                    microseconds = packet_time.microsecond

                    # Build the time header
                    timePack = struct.pack("BBBBBB", year, month, day, hour, minute, second)
                    microPack = microseconds.to_bytes(4, byteorder='big')
                    zeroPack = struct.pack("BB", 0, 0)
                    timeHeader = timePack + microPack + zeroPack

                    #Make the packet
                    dataPacket = currentDataBytes[startByte:endByte]
                    packet = timeHeader + dataPacket

                    # Send the UDP packet
                    self.socket.sendto(packet, (self.arguments.ip, self.arguments.port))

                    if not firstPacketSent:
                        print("First packet sent.")
                        firstPacketSent = True
                    
                    # Attempt to maintain real-time pacing
                    # --- Drift correction sleep ---
                    time_to_sleep = packet_interval * 0.8000
                    if time_to_sleep > 0:
                        Sleep(time_to_sleep)
                    # else: we're running late — don't sleep, just send the next packet immediately
                    
                    dataChunkIndex += 1
                    packetIndex +=1 
                    
            # ensures that, before we switch buffers, the preload worker has definitely finished writing into
            if nextProcess is not None:
                nextProcess.join()

            currentFileIndex += 1
            if currentFileIndex >= len(self.xwavFiles):
                break  # No more files to process

            # Move to the newly loaded data
            currentChunkHolder = nextReturnDict["chunk_dict"] if nextReturnDict else None

            # Start preloading the following file (if any)
            nextFileIndex = currentFileIndex + 1
            if nextFileIndex < len(self.xwavFiles):
                nextReturnDict = self.manager.dict()
                nextProcess = self.startFilePreloadProcess(nextFileIndex, nextReturnDict)
            else:
                nextProcess = None

        # Clean up
        self.socket.close()


if __name__ == "__main__":
    # Parse command-line arguments
    arguments = ArgumentParserService.parseArguments()
    dataSimulator = DataSimulator(arguments)
    dataSimulator.run()
    
    
    
