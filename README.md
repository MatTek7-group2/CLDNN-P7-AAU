# On the Generalisation Ability of Unsupervised and Supervised Voice Activity Detection Methods
Scripts accompanying the paper:

A. A. Andersen, D. B. van Diepen, M. V. Vejling and M. S. Kaaber "On the Generalisation Ability of Unsupervised and Supervised Voice Activity Detection Methods", SEMCON, Aalborg University, Denmark, 2020

Internally published for the seventh semester conference (SEMCON) at Department of Electronic Systems, Aalborg University.

## Data Sets:
In order to run the scripts, the data sets should be organised as follows:

#### --- The Apollo-11 data organised in the archive ---

```
./Fearless_2019
├── Fearless_Steps
│   ├── Data                       -- unzipped from the tar.gz file
│   │   ├── Audio                  -- the audio as 8kHz wav files
│   │   │   └── Tracks             -- audio files for VAD
│   │   │       ├── Dev               (.wav Files)
│   │   └── Transcripts            -- file level transcriptions
│   │       ├── SAD                   (.TXT Files)
│   │       │   └── Dev
```


#### --- The Aurora-2 data organised in the archive ---
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

## Scripts:

`CLDNN_MAIN.py`
	- Main script for training and testing convolutional long short-term memory fully connected deep neural networks.

`MODULE.py`
	- Module script containing functionality used in CLDNN_MAIN.

