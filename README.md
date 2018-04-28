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

## Quick guide
We have included the `config.p`, `data_util.p`, and `model.h5` files that were
trained across all data files. We also have included the `encoder_example.h5`
and `decoder_example.h5` files so that you may immediately run inference.
An example inference command that can be immediately run upon
cloning this repo is provided below:
```bash
> python inference.py --configs configs --data-util data_util \
--model model --encoder encoder_example --decoder decoder_example \
--output output_example --image data/evaluation/images/144269.jpg
```

## Step-by-Step Breakdown
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
> python inference.py --configs configs --data-util data_util \
--model model --encoder encoder --decoder decoder \
--output output --image <YOUR_IMAGE_HERE>
```
