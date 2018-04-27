# pylint: disable=W0102
import re
class Configs:
    def __init__(self,
                 N: int = 1,
                 L: int = 400,
                 image_vec_dim: int = 2048,
                 lstm_input_size: int = 256,
                 batch_size: int = 334,
                 number_of_epochs: int = 5,
                 activation: str = 'softmax',
                 optimizer: str = 'rmsprop',
                 loss: str = 'categorical_crossentropy',
                 dirs=["training", "evaluation", "validation"],
                 file_re=re.compile('([0-9]+)-([0-9]+).jpg')):
        self.N = N
        self.L = L
        self.image_vec_dim = image_vec_dim
        self.lstm_input_size = lstm_input_size
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.directories = dirs
        self.file_re = file_re
