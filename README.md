# LIP-TRAC

LIP-TRAC (Lipreading through a Temporal Recurrent And Convolutional network) is a visual speech recognition model that transcribes speech from video of a speaker's mouth. It is a 3D-convolutional and bidirectional-GRU network trained with CTC loss on the Oxford-BBC LRS2 dataset, and it runs on a Raspberry Pi 5 through TensorFlow Lite.

Author: Monish Saravana Kumar Divya Sundari ([monishsaravana.com](https://monishsaravana.com/))

## Repository layout

```
assets/       Haar cascade used for face detection
docs/         Project notes and the specification document
models/       Trained weights (Keras) and TensorFlow Lite exports
notebooks/
  data/       Building the training folders and dataset statistics
  training/   Training notebooks (model definition, data pipeline, CTC loss)
  export/     Saving the model and converting it to TensorFlow Lite
  scratch/    Small experiments
pi/           Raspberry Pi 5 capture, preprocessing and inference scripts
```

## Model

Input frames are grayscale 40×120 mouth crops, normalized per clip by subtracting the clip's mean pixel value and dividing by its standard deviation.

| # | Layer | Configuration | Parameters |
|---|---|---|---:|
| 1 | Conv3D | 32 filters, 3×3×3, same padding, ReLU | 896 |
| 2 | MaxPool3D | 1×2×2 | 0 |
| 3 | Conv3D | 64 filters, 3×3×3, same padding, ReLU | 55,360 |
| 4 | MaxPool3D | 1×2×2 | 0 |
| 5 | Conv3D | 96 filters, 3×3×3, same padding, ReLU | 165,984 |
| 6 | MaxPool3D | 1×2×2 | 0 |
| 7 | TimeDistributed(Flatten) | 7,200 features per frame | 0 |
| 8 | Bidirectional GRU | 256 units per direction | 11,455,488 |
| 9 | Dropout | 0.5 | 0 |
| 10 | Bidirectional GRU | 256 units per direction | 1,182,720 |
| 11 | Dropout | 0.5 | 0 |
| 12 | Dense, softmax | 29 classes (A to Z, space, padding, CTC blank) | 14,877 |
| | **Total** | | **12,875,325** |

## Data

LRS2 is not included. Its license does not allow redistribution, so request access from the [Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) and download it yourself.

The notebooks in `notebooks/data/` prepare the data:

- `fileRearrange.ipynb` copies every clip from LRS2's `mvlrs_v1/main` folders into one flat folder (`allFiles`), renaming each file to `<folder>_<clip>`.
- `trainFileGen.ipynb` copies a random sample of clips and transcripts from that folder into a training folder (`trainFiles1`, and so on).
- `dataStats.ipynb` computes frame and character counts; its output is in `dataStats.txt`.
- `NumToWord.ipynb` converts numerals in transcripts to words.

The notebooks use absolute Windows paths (for example `A:\Lip Reading\Potential Datasets\BBC LRS2\...`). Change `rootDir`, `rootDir2` and the `list_files` patterns to match where your copy of LRS2 lives.

## Training

The notebooks in `notebooks/training/` define the data pipeline, the model (`createModel`) and the CTC loss, then train with Adam. They are development runs kept for reference:

- `Lipreading.ipynb` and `LipreadingMOREEPOCHS.ipynb` train on the `trainFiles1` sample.
- `Lipreading_MOREDATA.ipynb` and `Lipreading_MOREDATA2.ipynb` train on a larger sample (`allFiles2`). The second one uses a smaller model.

Training was done on an NVIDIA RTX 3060 Ti (8 GB).

## Export to TensorFlow Lite

`notebooks/export/` saves the trained model and converts it to TensorFlow Lite with default (dynamic-range) quantization. `tfLiteFinal.ipynb` and `tflitetest.ipynb` run the converted model and include a NumPy implementation of CTC decoding.

## Raspberry Pi 5

The scripts in `pi/` run the deployed model:

| Script | Purpose |
|---|---|
| `lipread.py` | Entry point. Records 75 frames, builds the cropped mouth video, runs the model and prints the time taken by each step. |
| `vidProcessing.py` | Records from the camera with `picamera2` and writes the cropped face video. |
| `cropFuncs.py` | Haar-cascade face detection and the relative mouth crop. |
| `modelPredictions.py` | Preprocesses a clip, runs the TensorFlow Lite model and decodes the output with greedy CTC decoding. |
| `recordVideoFromCam.py`, `processVideoFromCam.py` | Earlier capture and processing scripts. |

The scripts expect the files laid out as they were on the device:

- `cropFuncs.py` loads the cascade from `util/haarcascade_frontalface_default.xml`. Copy `assets/haarcascade_frontalface_default.xml` there, or next to the scripts for the older capture scripts.
- `modelPredictions.py` loads the model from `GDRIVE/finalModelCUSTOM.tflite`. Point it at `pi/finalModel.tflite` or a model from `models/`.

## Setup

```
pip install -r requirements.txt        # training and export
pip install -r pi/requirements.txt     # on the Raspberry Pi
```
