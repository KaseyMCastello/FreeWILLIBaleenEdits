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


#Added from npy version for xwavs
import soundfile as sf


#Local utilities (assumes these functions exist in utils_xWavSim.py)
from utils_xWavSim import get_datetime, extract_wav_start, read_xwav_audio

# Local utilities (assumes these functions exist in utils.py)
from utils import SetHighPriority, ReadBinaryData,  InterleaveData, ScaleData, ConvertToBytes, Sleep

sys.argv.extend([
    "--port", "1045",
    "--ip", "192.168.1.235",
    "--data_dir", r"C:\Users\kasey\Desktop\socalsim"
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
        parser.add_argument('--data_dir', default="simulator_data/track132_5minchunks/", type=str, help='Directory containing .npy data files')
        parser.add_argument('--loop', action='store_true', help='Enable looping over the data')
        parser.add_argument('--stretch', action='store_true', help='Normalize data values to the min/max range of 16-bit unsigned int')
        parser.add_argument('--imu', action='store_true', help='Read in IMU data from file')
        parsedArgs = parser.parse_args()
        return parsedArgs

class XWAVFileProcessor:
    """
    Responsible for loading and transforming .xwav files into byte data for streaming.
    
    Added by K.C.
    Adapts .npy processor to  handle .xwav files, including reading headers and data.
    """
    @staticmethod
    def processXwavFile(xwav_file_path, return_dict, expected_channels, data_scale, stretch, fw):
        """
        Load the .xwav file and prepare it for UDP transmission.
        The processed bytes are stored in a shared returnDict under 'dataBytes'.
        """
        print("Loading file:", xwav_file_path)
        # Load the xwav file header to extract metadata
        wav_start_time = extract_wav_start(xwav_file_path)
        header = XWAVhdr(xwav_file_path)
        
        #print(f"Loaded waveform with shape: {waveform.shape} and sample rate: {sr}")
        data, sr = read_xwav_audio(xwav_file_path)

        #Check that the firmware the user wants to simulate is compatible with the xwav file
        if data.shape[0] != expected_channels:
           print(f"ERROR: {data.shape[0]} XWAV Channel(s) detected but chosen FW [{expected_channels}]. Change firmware or xwav directory. Exiting.")
           sys.exit()

        #Check if we should interleave the data.
        #1240: 4-Channel. -1:Harp 1 Channel (Change this after discuss with Joe)
        if fw == 1240 or fw==-1:
            interleavedData = InterleaveData(data)
        else:
            print("Unsupported firmware version")
            sys.exit()

        scaledData = ScaleData(interleavedData, data_scale, stretch)
        processedDataBytes = ConvertToBytes(scaledData)

        #Allows user to choose to use multiprocessing manager or not. If using multiple xwavs in folder, multiprocessing is recommended.
        if return_dict is not None:
            return_dict["dataBytes"] = processedDataBytes #multiprocessing
        else:
            return processedDataBytes #not multiprocessing

        return processedDataBytes

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

        # If certain firmware files define additional constants, reference them conditionally:
        # E.g. firmware_1550 or firmware_1240 might define `highAmplitudeIndex`
        if hasattr(fwConfig, 'highAmplitudeIndex'):
            self.highAmplitudeIndex = fwConfig.highAmplitudeIndex
        else:
            self.highAmplitudeIndex = 0  # or define another sensible default if not present

        # For scaling data
        self.DATA_SCALE = 2**15

        # Print initial settings
        print(f"Sending data to {self.arguments.ip} on port {self.arguments.port}")

        # If requested, read IMU data and update PACKET_SIZE accordingly
        self.imuByteData = None
        if self.arguments.imu:
            self.imuByteData = ReadBinaryData("../../IMU_matlab/202518_stationary.imu")
            print("Packet size before IMU:", self.packetSize)
            self.packetSize += len(self.imuByteData[0])
            print("Packet size after IMU:", self.packetSize)
            print("IMU samples loaded:", len(self.imuByteData))

        #### Initialize multiprocessing manager and dictionary for data exchange
        # spin up a small “manager server” process under the hood to hold Python objects in a shared memory space, 
        # and hands out proxy objects back to your processes.
        self.manager = multiprocessing.Manager() 
        # This asks the manager server for a new, empty dictionary proxy.. This serves as a shared buffer between processes
        self.returnDict = self.manager.dict()

        # Initialize socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Start date/time 
        self.currentDateTime = header.raw["dnumStart"][0]

    def preloadFirstFile(self):
        """
        Preload the first file's data synchronously.
        """
        XWAVFileProcessor.processXwavFile(
            xwav_file_path=self.xwavFiles[0],
            return_dict=self.returnDict,
            expected_channels=self.numChannels,
            data_scale=self.DATA_SCALE,
            stretch=self.arguments.stretch,
            fw=self.fw
        )
        return self.returnDict['dataBytes']

    def startFilePreloadProcess(self, fileIndex, returnDict):
        """
        Launch a separate process to preload the next file's data.
        """
        processNext = multiprocessing.Process(
            target=XWAVFileProcessor.processXwavFile,
            args=(
                self.xwavFiles[fileIndex],
                returnDict,
                self.numChannels,
                self.DATA_SCALE,
                self.arguments.stretch,
                self.fw
            )
        )
        processNext.start()
        return processNext

    def run(self):
        """
        Main loop that streams data from .npy files over UDP, handles optional looping, 
        time/data glitches, and IMU data concatenation.
        """
        currentDataBytes = self.preloadFirstFile()

        # Start preloading the next file if available
        nextProcess = None
        nextReturnDict = None
        currentFileIndex = 0

        while True:
            isFirstRead = True
            
            dataChunkIndex = 0  # renamed 'flag' -> 'dataChunkIndex'

            while True:
                startByte = dataChunkIndex * self.dataSize
                endByte = (dataChunkIndex + 1) * self.dataSize
                startTime = time.time()
                if startByte >= len(currentDataBytes):
                    # If we've reached the end of the current data, break to load next file
                    break

                # Prepare timestamp fields
                year = self.currentDateTime.year - 2000
                month = self.currentDateTime.month
                day = self.currentDateTime.day
                hour = self.currentDateTime.hour
                minute = self.currentDateTime.minute
                second = self.currentDateTime.second
                microseconds = self.currentDateTime.microsecond

                # Build the time header
                timePack = struct.pack("BBBBBB", year, month, day, hour, minute, second)
                microPack = microseconds.to_bytes(4, byteorder='big')
                zeroPack = struct.pack("BB", 0, 0)
                timeHeader = timePack + microPack + zeroPack

                #Make the packet
                dataPacket = currentDataBytes[startByte:endByte]
                packet = timeHeader + dataPacket

                #Handle IMU data if enabled
                if self.arguments.imu and self.imuByteData is not None:
                    imuSample = self.imuByteData[dataChunkIndex % len(self.imuByteData)]
                    packet += struct.pack(f"{len(imuSample)}B", *imuSample)

                # Send the UDP packet
                self.socket.sendto(packet, (self.arguments.ip, self.arguments.port))

                if isFirstRead:
                    print("First packet sent.")
                    isFirstRead = False

                # Increment time for next packet
                self.currentDateTime += timedelta(microseconds=int(self.microIncrement))

                # Attempt to maintain real-time pacing
                sleepTime = (self.microIncrement * 1e-6)
                Sleep(sleepTime)

                dataChunkIndex += 1
                
            # ensures that, before we switch buffers, the preload worker has definitely finished writing into
            if nextProcess is not None:
                nextProcess.join()

            currentFileIndex += 1
            if currentFileIndex >= len(self.npyFiles):
                break  # No more files to process

            # Move to the newly loaded data
            currentDataBytes = nextReturnDict['dataBytes'] if nextReturnDict else None

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
    
    
    
