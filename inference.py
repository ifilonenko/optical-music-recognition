# pylint: disable=W0201
import argparse
import pickle
import os.path
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import load_model, Model
from keras.preprocessing import image as KImage
from music21.midi import translate
from music21 import pitch, note
from music21.articulations import Fingering, Articulation
from music21.stream import Stream

from configs import Configs
from data_utils import DataUtil
from image_transfer_learning import image_generator
from models import build_res_net
from utils import pickle_load

class Inference:
    def __init__(self, c: Configs, data_util: DataUtil):
        self.configs = c
        self.data_util = data_util
        self.padded_start_state = np.tile(data_util.middle_c, (c.L, 1))\
            .reshape(1, c.L, data_util.encoding_size + 1)
        self.indx = data_util.INDX

    def build_encoder_decoder(self, loaded_model, encoder_name, decoder_name):
        encoder_model = \
            Model(loaded_model.layers[0].output, loaded_model.layers[4].output)
        decoder_state_input_h = Input(shape=(1, \
            self.configs.lstm_input_size))
        decoder_state_input_c = Input(shape=(1, \
            self.configs.lstm_input_size))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        img_dense_output = loaded_model.layers[3](loaded_model.layers[1].input)
        decoder_outputs, state_h_new, state_c_new = \
            loaded_model.layers[5](\
                img_dense_output, initial_state=decoder_states_inputs)
        decoder_states = [state_h_new, state_c_new]
        decoder_dense_output = loaded_model.layers[6](decoder_outputs)
        decoder_model = \
            Model([loaded_model.layers[1].output] + decoder_states_inputs,\
                [decoder_dense_output] + decoder_states)
        self.encoder_model = encoder_model
        encoder_model.save('cache/{}.h5'.format(encoder_name))
        self.decoder_model = decoder_model
        decoder_model.save('cache/{}.h5'.format(decoder_name))

    def decode_sequence(self, input_vec):
        lstm_size = self.configs.lstm_input_size
        s_v = self.encoder_model.predict(\
            input_vec.reshape(1, 1, self.configs.image_vec_dim))
        output_tokens, _, _ = self.decoder_model.predict(
            [self.padded_start_state] + \
            [s_v[0].reshape(1, 1, lstm_size), s_v[1].reshape(1, 1, lstm_size)])
        output_result = []
        for row in output_tokens[0]:
            if (row == self.data_util.stop_state).any():
                break
            else:
                pitch_indx = np.argmax(row[:-1][:self.indx])
                duration_indx = np.argmax(row[:-1][self.indx:])
                c_result = np.zeros((self.data_util.encoding_size+1,))
                c_result[pitch_indx] = 1.0
                c_result[self.indx+duration_indx] = 1.0
                output_result.append(\
                    self.data_util.decode_pitch_duration(c_result[:-1]))
        return output_result

    def output_stream(self, note_pd_lst):
        stream = Stream()
        for note_pair in note_pd_lst:
            n = note.Note(pitch.Pitch(\
                note_pair[0][0]), quarterLength=note_pair[1][0])
            stream.append(n)
        return stream

    def output_pdf(self, input_stream, file_name):
        location = str(input_stream.write("lily.pdf"))
        os.rename(location, '{}.pdf'.format(file_name))
        mf = translate.music21ObjectToMidiFile(input_stream)
        mf.open('{}.mid'.format(file_name), 'wb')
        mf.write()
        mf.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running inference' + \
    'on an example image or a random sample of the evaluation data')
    parser.add_argument('--configs', type=str, help='name of pickled Configs')
    parser.add_argument('--data-util', type=str, help='name pickled DataUtil')
    parser.add_argument('--model', type=str, help='name of Keras Saved Model')
    parser.add_argument('--image', type=str, help='file extension of the image')
    parser.add_argument('--encoder', type=str, help='name of Keras Saved Model')
    parser.add_argument('--decoder', type=str, help='name of Keras Saved Model')
    parser.add_argument('--output', type=str, help='name of output PDF')
    args = vars(parser.parse_args())
    if not args['configs']:
        raise argparse.ArgumentTypeError('Must supply picked Configs')
    if not args['data_util']:
        raise argparse.ArgumentTypeError('Must supply picked DataUtil')
    if not args['model']:
        raise argparse.ArgumentTypeError('Must supply a Keras Saved Model')
    encoder_f_name = "encoder"
    decoder_f_name = "decoder"
    configs_load: Configs = \
        pickle_load('cache/{}.p'.format(args['configs']))
    data_util_load: DataUtil = \
        pickle_load('cache/{}.p'.format(args['data_util']))
    inference = Inference(configs_load, data_util_load)
    if args["image"]:
        resNet: ResNet50 = build_res_net()
        gen = image_generator([args["image"]], 1)
        input_img_vector = resNet.predict_generator(gen, 1, verbose=1)
    else:
        sample_vec = np.loadtxt("data/evaluation/vectors.out")
        input_img_vector = sample_vecs[np.random.choice(sample_vec.shape[0], 1)]
    if args["encoder"] and args["decoder"]:
        encoder_f_name = args["encoder"]
        decoder_f_name = args["decoder"]
        if os.path.exists(args["encoder"]) and os.path.exists(args["decoder"]):
            inference.encoder_model = \
                load_model('cache/{}.h5'.format(args['encoder']))
            inference.decoder_model = \
                load_model('cache/{}.h5'.format(args['decoder']))
        else:
            l_m = load_model('cache/{}.h5'.format(args['model']))
            inference.build_encoder_decoder(l_m, encoder_f_name, decoder_f_name)
    else:
        l_m = load_model('cache/{}.h5'.format(args['model']))
        inference.build_encoder_decoder(l_m, encoder_f_name, decoder_f_name)
    s = inference.output_stream(inference.decode_sequence(input_img_vector))
    output_pdf_fn = "output"
    if args["output"]:
        output_pdf_fn = args["output"]
    inference.output_pdf(s, output_pdf_fn)
