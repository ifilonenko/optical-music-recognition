# pylint: disable=W0201
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

class DataUtil:
    def __init__(self,
                 flattened_training,
                 flattened_validation,
                 flattened_evaluation):
        self.flattened_training = flattened_training
        self.flattened_validation = flattened_validation
        self.flattened_evaluation = flattened_evaluation

    def fit_labelers(self):
        lb_pitch = LabelBinarizer()
        lb_pitch.fit(np.concatenate(\
        (np.concatenate((self.flattened_training[:, 0],
                         self.flattened_validation[:, 0])),\
                         self.flattened_evaluation[:, 0])))

        self.lb_pitch = lb_pitch
        le_duration = LabelEncoder()
        le_duration.fit(np.concatenate(\
        (np.concatenate((self.flattened_training[:, 1],
                         self.flattened_validation[:, 1])),\
                         self.flattened_evaluation[:, 1])))
        self.le_duration = le_duration

        lb_duration = LabelBinarizer()
        lb_duration.fit([self.le_duration.transform([r])[0] for r in \
                 np.concatenate(\
                 (np.concatenate((self.flattened_training[:, 1],\
                                  self.flattened_validation[:, 1])),\
                                  self.flattened_evaluation[:, 1]))])
        self.lb_duration = lb_duration

    def encode_pitch_duration(self, pd):
        return np.concatenate((self.lb_pitch.transform([pd[0]])[0],\
                               self.lb_duration.transform(\
                               self.le_duration.transform([pd[1]]))[0]))
    def set_INDX(self):
        self.INDX = self.lb_pitch.transform([44.]).shape[1]
        self.encoding_size = self.encode_pitch_duration([44., 0.5]).shape[0]

    def decode_pitch_duration(self, one_hot):
        import warnings
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        return [self.lb_pitch.inverse_transform(\
        np.array([one_hot[:self.INDX]])),\
        self.le_duration.inverse_transform(\
        self.lb_duration.inverse_transform(np.array([one_hot[self.INDX:]])))]

    def data_util_global_vals(self):
        self.stop_state = np.zeros((1, self.encoding_size+1))
        self.stop_state[0][self.encoding_size] = 1.0
        self.middle_c = np.append(self.encode_pitch_duration([48., 1.0]), 0)
