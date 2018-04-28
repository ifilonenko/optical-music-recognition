# pylint: disable=R1702
import os
import shutil

import numpy as np
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image as KImage

from configs import Configs
from models import build_res_net

def convert_dir_pdf2img(config: Configs):
    for p_d in config.directories:
        print("Converting to images for %s" % p_d)
        print("==========================")
        dir_pdf = 'data/%s/pdfs' % p_d
        dir_notes = 'data/%s/notes' % p_d
        dir_images = 'data/%s/images' % p_d
        if os.path.exists(dir_images):
            shutil.rmtree(dir_images)
        os.makedirs(dir_images)
        id_list = set()
        for subdir, _, files in os.walk(dir_notes):
            for file in files:
                id_list.add((file[:-4]))
        for subdir, _, files in os.walk(dir_pdf):
            for file in files:
                if file[:-4] in id_list:
                    images = convert_from_path(\
                    os.path.join(subdir, file), output_folder=dir_images, \
                    dpi=300, fmt='jpg')
                    temp_files = [image.filename for image in images]
                    for in_indx, f in enumerate(temp_files):
                        if len(temp_files) == 1:
                            os.rename(f, "{}/{}.jpg".format(\
                            dir_images, file[:-4]))
                        else:
                            os.rename(f, "{}/{}-{}.jpg".format(\
                            dir_images, file[:-4], in_indx+1))

def load_images(image_list):
    images = []
    for indx in image_list:
        c_img = np.expand_dims(KImage.img_to_array(\
        KImage.load_img(indx, target_size=(224, 224))), axis=0)
        images.append(c_img)
    return preprocess_input(np.vstack(images))

def image_generator(fnames, batch_size):
    while True: #Keras generators need to loop forever for some reason...
        cfns = []
        for p in fnames:
            cfns.append(p)
            if len(cfns) == batch_size:
                yield load_images(cfns)
                cfns = []
        if cfns:
            yield load_images(cfns)
            cfns = []

def build_vector_features(base_model, o_dir, file_list, config: Configs):
    all_ids = []
    all_paths = []
    print(file_list)
    for subdir, _, files in os.walk(file_list):
        for f in files:
            if not bool(config.file_re.search(f)):
                all_ids.append(f[:-4])
                all_paths.append(os.path.join(subdir, f))
    fpath = "{}/vectors.out".format(o_dir)
    print("Extracting from {} files".format(len(all_paths)))
    print('Saving to {}'.format(fpath))
    gen = image_generator(all_paths, 1)
    feats = base_model.predict_generator(gen,\
    len(all_paths), use_multiprocessing=True, verbose=1)
    np.savetxt(fpath, feats)
    return all_ids

def grab_ids(file_list, config: Configs):
    all_ids = []
    for _, _, files in os.walk(file_list):
        for f in files:
            if not bool(config.file_re.search(f)):
                all_ids.append(f[:-4])
    return all_ids

"""
    This class will take all pdfs that are a single page that have notes and
    convert them into images. Then they will build the vector representations
    of the images using ResNet50
"""
class TransferLearning:
    def __init__(self, c: Configs, skip_image=False):
        if not skip_image:
            # Convert pdf to images
            convert_dir_pdf2img(c)
        # Build ResNet50
        image_model = build_res_net()
        # Compute vectors from images
        id_dir = {}
        for i, parent_dir in enumerate(c.directories):
            print("Working on %s" % parent_dir)
            print("==========================")
            output_dir = 'data/%s' % parent_dir
            dir_imgs = '%s/images' % output_dir
            id_dir[c.directories[i]] = build_vector_features(\
            image_model, output_dir, dir_imgs, c)
        self.dir_of_ids = id_dir
        self.data_util = None
