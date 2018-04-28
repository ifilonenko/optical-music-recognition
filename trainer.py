# pylint: disable=W0201
import argparse
import csv
from fractions import Fraction
import os.path
import pickle

from keras.models import load_model
import numpy as np

from data_utils import DataUtil
from configs import Configs
from image_transfer_learning import TransferLearning
from models import EncoderDecoderModel
from utils import MacOSFile, pickle_dump, pickle_load

class TrainerInstance:
    def __init__(self, c: Configs):
        self.configs = c

    def set_data_util_values(self, dataUtil: DataUtil):
        self.data_util = dataUtil
        self.stop_state = np.zeros((1, dataUtil.encoding_size+1))
        self.stop_state[0][dataUtil.encoding_size] = 1.0
        self.middle_c = np.append(dataUtil.encode_pitch_duration([48., 1.0]), 0)

class TrainerSetup(TrainerInstance):
    def __init__(self, c: Configs):
        super().__init__(c)
        self.evaluation_data = np.loadtxt("data/evaluation/vectors.out")
        self.training_data = np.loadtxt("data/training/vectors.out")
        self.validation_data = np.loadtxt("data/validation/vectors.out")

    def build_data(self, data_type, tf: TransferLearning):
        def parse_val(v):
            try:
                new_v = float(v)
            except ValueError:
                new_v = float(Fraction(v))
            return new_v
        flatten = lambda l: [item for sublist in l for item in sublist]
        t_data = []
        good_ids = []
        for file_name in tf.dir_of_ids[data_type]:
            with open(\
            'data/{}/notes/{}.csv'.format(data_type, file_name), 'r') as f:
                intermediate = [[float(tuple(line)[0]),\
                parse_val((tuple(line)[1]))] for line in csv.reader(f)]
                if intermediate:
                    t_data.append(intermediate)
                    good_ids.append(file_name)
        return (t_data, np.array(flatten(t_data)), good_ids)

    def set_results(self, tf: TransferLearning):
        training_results, flattened_training_results, training_good_ids = \
            self.build_data("training", tf)
        validation_results, flattened_validation_results, validation_good_ids =\
            self.build_data("validation", tf)
        evaluation_results, flattened_evaluation_results, evaluation_good_ids =\
            self.build_data("evaluation", tf)
        self.training_results = training_results
        self.flattened_training_results = flattened_training_results
        self.training_good_ids = training_good_ids
        self.validation_results = validation_results
        self.flattened_validation_results = flattened_validation_results
        self.validation_good_ids = validation_good_ids
        self.evaluation_results = evaluation_results
        self.flattened_evaluation_results = flattened_evaluation_results
        self.evaluation_good_ids = evaluation_good_ids

    def one_hot_conversion(self):
        self.one_hot_training_data = \
        self.one_hot_conversion_func(self.training_results)
        self.one_hot_validation_data = \
            self.one_hot_conversion_func(self.validation_results)
        self.one_hot_evaluation_data = \
            self.one_hot_conversion_func(self.evaluation_results)

    def one_hot_conversion_func(self, input_result):
        result = []
        for s in input_result:
            result.append(np.array([\
                np.append(self.data_util.encode_pitch_duration(tup), 0) \
                for tup in s]))
        return np.array(result)

class Trainer(TrainerInstance):
    def __init__(self, c: Configs, dir_dict):
        super().__init__(c)
        self.id_dir = dir_dict

    def padding(self, input_mat):
        (num_rows, vec_size) = input_mat.shape
        num_to_pad = self.configs.L - num_rows
        if num_to_pad == 0:
            return input_mat
        elif num_to_pad > 0:
            return np.vstack((input_mat, np.zeros((num_to_pad, vec_size))))
        return input_mat[:self.configs.L]

    def build_inputs(self, ts: TrainerSetup):
        self.build_training_input(ts, "training")
        self.build_training_input(ts, "validation")
        self.build_training_input(ts, "evaluation")

    def build_training_input(self, ts: TrainerSetup, data_type):
        inv_id_dict = {k:i for i, k in enumerate(self.id_dir[data_type])}
        new_data_vectors = \
            getattr(ts, '{}_data'.format(data_type))[np.ix_([inv_id_dict[k] \
            for k in getattr(ts, '{}_good_ids'.format(data_type))],)]
        setattr(self, 'encoder_input_data_{}'.format(data_type), \
            np.array([k.reshape(1, self.configs.image_vec_dim) \
                for k in new_data_vectors]))
        setattr(self, 'decoder_input_data_{}'.format(data_type), \
            np.array([self.padding(np.vstack((k[1:], self.stop_state))) \
                for k in getattr(ts, 'one_hot_{}_data'.format(data_type))]))
        setattr(self, 'decoder_output_data_{}'.format(data_type), \
            np.array([self.padding(np.vstack((self.middle_c, k[:-1]))) \
                for k in getattr(ts, 'one_hot_{}_data'.format(data_type))]))

    def fit_model(self, edm: EncoderDecoderModel, model_name):
         edm.model.fit(\
                [self.encoder_input_data_training, \
                self.decoder_input_data_training], \
            self.decoder_output_data_training,\
            batch_size=self.configs.batch_size,\
            epochs=self.configs.number_of_epochs,\
            verbose=1,\
            validation_data=\
                ([self.encoder_input_data_validation, \
                self.decoder_input_data_validation],\
                self.decoder_output_data_validation))
         print("Finished the training of Encoder-Decoder")
         print('Saving model at: {}'.format(model_name))
         edm.model.save(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run script to train the' + \
    'Encoder Decoder Model and output the trained model for inference.')
    parser.add_argument('--configs', \
        type=str, help='name of where to store Configs')
    parser.add_argument('--data-util', \
        type=str, help='name of where to save DataUtil')
    parser.add_argument('--model', type=str, help='name of Keras Saved Model')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--no-image', \
        action='store_true', help='skip the pdf2img conversion process')
    args = vars(parser.parse_args())
    print("Building Trainer object")
    print("=======================")
    print("Setting up configs")
    configs = Configs()
    configs_file_name = "cache/configs.p"
    if args["configs"]:
        configs_file_name = 'cache/{}.p'.format(args["configs"])
    pickle_dump(configs, configs_file_name)
    print("Running Transfer Learning")
    skip_image_value = False
    if args["no_image"]:
        skip_image_value = args["no_image"]
    transfer_learning = TransferLearning(configs, skip_image_value)
    print("Processing data for Encoder-Decoder Training")
    trainerSetup = TrainerSetup(configs)
    trainerSetup.set_results(transfer_learning)
    print("Fitting decoding and encoding for LabelEncoding")
    data_util = DataUtil(trainerSetup.flattened_training_results,\
                         trainerSetup.flattened_validation_results,\
                         trainerSetup.flattened_evaluation_results)
    data_util.fit_labelers()
    data_util.set_INDX()
    data_util.data_util_global_vals()
    data_util_file_name = "cache/data_util.p"
    if args["data_util"]:
        data_util_file_name = 'cache/{}.p'.format(args["data_util"])
    assert(data_util.decode_pitch_duration(\
    data_util.encode_pitch_duration([44., 0.25])) == \
        [np.array([44.]), np.array([0.25])])
    pickle_dump(data_util, data_util_file_name)
    trainerSetup.set_data_util_values(data_util)
    print("Doing One Hot Encodings of data")
    trainerSetup.one_hot_conversion()
    trainer = Trainer(configs, transfer_learning.dir_of_ids)
    trainer.set_data_util_values(data_util)
    trainer.build_inputs(trainerSetup)
    print("Finished data formating for Encoder-Decoder Training")
    model_file_name = "cache/model.h5"
    if args["model"]:
        model_file_name = 'cache/{}.h5'.format(args["model"])
    if not os.path.exists(model_file_name):
        if args["epochs"]:
            trainer.configs.number_of_epochs = args["epochs"]
        print("Beginning the training of Encoder-Decoder")
        model = EncoderDecoderModel(\
            trainer.configs, trainer.data_util.encoding_size)
        model.build_initial_model()
        model.compile()
        trainer.fit_model(model, model_file_name)
        new_model = load_model(model_file_name)
        print('Evaluation score: %f' % \
        new_model.evaluate([trainer.encoder_input_data_evaluation,\
            trainer.decoder_input_data_evaluation],\
                        trainer.decoder_output_data_evaluation,\
                        batch_size=trainer.configs.batch_size,\
                        verbose=1))
    else:
        print('Model exists at {}'.format(model_file_name))
