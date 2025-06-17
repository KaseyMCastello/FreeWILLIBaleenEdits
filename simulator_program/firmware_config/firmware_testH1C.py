SAMPLE_RATE = 200000                # Sample rate (Hz)
HEAD_SIZE = 12                     # packet head size (bytes)
MICRO_INCR = 1240               # time between packets
NUM_CHAN = 1;                      # number of channels per packet
SAMPS_PER_CHANNEL = 248;           #1240 ms (200000 Hz * 0.001 s) *1.240 ms /sample
BYTES_PER_SAMP = 2;                                             # bytes per sample
DATA_SIZE = SAMPS_PER_CHANNEL * NUM_CHAN * BYTES_PER_SAMP;   # packet data size (bytes) = 1240
PACKET_SIZE = HEAD_SIZE + DATA_SIZE;                            # packet size (bytes) = 1252

REQUIRED_BYTES = DATA_SIZE + HEAD_SIZE
DATA_BYTES_PER_CHANNEL = SAMPS_PER_CHANNEL * BYTES_PER_SAMP       # number of data bytes per channel (REQUIRED_BYTES - 12) / 4 channels
