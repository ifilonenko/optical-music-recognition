# Optical Music Recognition
## Getting Started
This framework allows for a user to input an image of an arbitrary image.
It then handles detecting the pitch and duration of music notes.

#### Virtualenv
This application runs in Python3.

```bash
> virtualenv venv -p python3
> source venv/bin/activate
> pip install git+https://github.com/vishnubob/python-midi@feature/python3
Collecting git+https://github.com/vishnubob/python-midi@feature/python3
  Cloning https://github.com/vishnubob/python-midi (to revision feature/python3)
Installing collected packages: midi
  Running setup.py install for midi ... done
Successfully installed midi-0.2.3
> pip install -r requirements.txt
...
> brew install poppler
```

Make sure to install LillyPond found [here](http://lilypond.org/download.html)
 and MuseScrore 2 found [here](https://musescore.org/en).

## Step-by-Step
### Downloading files
```bash
> python download_score.py --api-key <API_KEY>
```
### Training
```bash
> python trainer.py --trainer trainer --epoch 100 --model model
```

### Inference
```bash
> python inference.py --trainer trainer --model model --image <INPUT_IMG_PATH>
```
