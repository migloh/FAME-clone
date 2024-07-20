# FAME-Net Clone
A study on the code [FAME-Net](https://github.com/alexhe101/FAME-Net)

## Necessary dependencies
```
torch 2.3.1
cuda 11.8
python 3.10
opencv-python
scipy
numpy
matplotlib
pyyaml
tqdm
tensorboardX
```
## Data folder structure
```
dataset/
    |- train
        |- mask
            |- image10.extension // sample 1, shape (1, W, H)
            |- image20.extension // sample 2, shape (1, W, H)
        |- ms
            |- image11.extension // sample 1, channel 1, shape (1, W, H)
            |- image12.extension // sample 1, channel 2, shape (1, W, H)
            |- image13.extension // sample 1, channel 3, shape (1, W, H)
            |- image14.extension // sample 1, channel 4, shape (1, W, H)
            |- image15.extension // sample 1, channel 5, shape (1, W, H)
            |- image21.extension // sample 1, channel 1, shape (1, W, H)
            |- image22.extension // sample 1, channel 2, shape (1, W, H)
            |- image23.extension // sample 1, channel 3, shape (1, W, H)
            |- image24.extension // sample 1, channel 4, shape (1, W, H)
            |- image25.extension // sample 1, channel 5, shape (1, W, H)
        |- pan
            |- image10.extension // sample 1, shape (1, W, H)
            |- image20.extension // sample 2, shape (1, W, H)
    |- eval
        |- ms
        |- pan
    |- test
        |- ms
        |- pan
```
## How to run the project
```
cd src/
python main.py
```