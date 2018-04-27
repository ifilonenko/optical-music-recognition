import argparse
import csv
import json
import os
import shutil
import urllib.request

import midi
import music21
from music21 import *

"""
    This script will populate data/ with all the necessary pdfs,images, and
    notes to begin training.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run script to download ' + \
    'music scores, pdf, mxls, and notes.')
    parser.add_argument('--api-key', type=str, help='an API key for MuseScore')
    args = vars(parser.parse_args())
    if not args['api_key']:
        raise argparse.ArgumentTypeError('Must supply a MuseScore API key')
    # We need to download 4 components:
    # - a json with the score information, containing the secret id
    # - the score mid file, using the public and secret id
    # - the score pdf file, using hte public and secret id
    score_json_url = \
    'http://api.musescore.com/services/rest/score/{}.json?oauth_consumer_key='
    score_json_url += args["api_key"]
    score_file_mid = 'http://static.musescore.com/{}/{}/score.mid'
    score_file_pdf = 'http://static.musescore.com/{}/{}/score.pdf'
    # We are now making a training, evaluation, and validation set
    DIRS = ["training", "evaluation", "validation"]
    # Reading the ids which list monotonic music scores
    with open('data/train_keys.txt', 'r') as f:
        train_ids = f.read().splitlines()
    with open('data/validation_keys.txt', 'r') as f:
        validation_ids = f.read().splitlines()
    with open('data/evaluation_keys.txt', 'r') as f:
        evaluation_ids = f.read().splitlines()
    dir_ids = {DIRS[0]: train_ids, DIRS[1]:validation_ids, \
    DIRS[2]:evaluation_ids}

    for parent_dir in DIRS:
        print("Working on %s" % parent_dir)
        print("==========================")
        dir_pdf = 'data/%s/pdfs' % parent_dir
        if os.path.exists(dir_pdf):
            shutil.rmtree(dir_pdf)
        os.makedirs(dir_pdf)
        dir_mxl = 'data/%s/mids' % parent_dir
        if os.path.exists(dir_mxl):
            shutil.rmtree(dir_mxl)
        os.makedirs(dir_mxl)
        dir_notes = 'data/%s/notes' % parent_dir
        if os.path.exists(dir_notes):
            shutil.rmtree(dir_notes)
        os.makedirs(dir_notes)
        curr_ids = dir_ids[parent_dir]
        good_ids = []
        for i, score_id in enumerate(curr_ids):
            # We download the first score
            print('%d/%d' % (i, len(curr_ids)))
            try:
                # First download score JSON to get secret
                r = urllib.request.urlopen(score_json_url.format(score_id))
                score_json = json.loads(r.read().decode(\
                r.info().get_param('charset') or 'utf-8'))
                score_secret = score_json['secret']
                # Define save location
                filename_mid = \
                    'data/{}/mids/{}.mid'.format(parent_dir, score_id)
                filename_pdf = \
                    'data/{}/pdfs/{}.pdf'.format(parent_dir, score_id)
                filename_notes = \
                    'data/{}/notes/{}.csv'.format(parent_dir, score_id)
                # Download score
                urllib.request.urlretrieve(\
                score_file_mid.format(score_id, score_secret), filename_mid)
                urllib.request.urlretrieve(\
                score_file_pdf.format(score_id, score_secret), filename_pdf)
                # pattern = midi.read_midifile(filename_mid)
                # try:
                piece = converter.parse(filename_mid)
                all_parts = []
                for part in piece.parts:
                    for event in part:
                        if getattr(event, 'isNote', None) and event.isNote:
                            all_parts.append(\
                            (event.pitch.ps, event.quarterLength))
                        if getattr(event, 'isRest', None) and event.isRest:
                            all_parts.append((144.0, event.quarterLength))
                with open(filename_notes, 'w') as out:
                    csv_out = csv.writer(out)
                    for row in all_parts:
                        csv_out.writerow(row)
            except urllib.error.HTTPError as e:
                print('API_key: %s is not valid or bad URL' % args['api_key'])
            except:
                print('failed: %s' % score_id)
            else:
                continue
    print("Done!")
