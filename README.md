# CLDNN-P7-AAU
On the Generalisation Ability of Unsupervised and Supervised Voice Activity Detection Methods


Data Sets:

--- The Apollo-11 data organized in the archive ---

```
./Fearless_Steps
├── Data                       -- a tar.gz File (unzip to see the folder contents)
│   ├── Audio                  -- the audio as 8kHz wav files
│   │   └── Tracks             -- audio files for VAD
│   │       ├── Dev               (.wav Files)
│   └── Transcripts            -- file level transcriptions
│       ├── SAD                   (.TXT Files)
│       │   └── Dev
```


--- The Aurora-2 data organized in the archive ---
```
./aurora2
├── SPEECHDATA                    -- the audio as 8kHz wav files
│   ├── TESTA                        (.08 Files)
│   ├── TESTB                        (.08 Files)
│   ├── TESTC                        (.08 Files)
│   └── TRAIN                     -- Multi-condition training data
│                                    (.08 Files)
├── Aurora2TestSet-ReferenceVAD   -- file level transcriptions
├── Aurora2TrainSet-ReferenceVAD  -- file level transcriptions
```

Scripts:

CLDNN_MAIN.py
	- Main script for training and testing convolutional long short-term memory fully connected deep neural networks.

MODULE.py
	- Module script containing functionality used in CLDNN_MAIN.

