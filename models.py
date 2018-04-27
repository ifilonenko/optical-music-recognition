# pylint: disable=W0201
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Input, LSTM, TimeDistributed
from keras.models import Model, Sequential


from configs import Configs
class RNNLayer:
    def __init__(self, layer_size, seq_return, state_return):
        self.layer_size = layer_size
        self.return_sequences = seq_return
        self.return_state = state_return

class LSTMLayer(RNNLayer):
    def __init__(self, layer_size, seq_return, state_return, name):
        super().__init__(layer_size, seq_return, state_return)
        self.layer = LSTM(layer_size, \
                          return_sequences=seq_return, \
                          return_state=state_return,
                          name=name)

class EncoderDecoderModel:
    def __init__(self, configs: Configs, decoder_size):
        self.encoder_input_shape = (configs.N, configs.image_vec_dim)
        self.lstm_input_size = configs.lstm_input_size
        self.decoder_input_size = decoder_size+1
        self.decoder_input_shape = (configs.L, decoder_size+1)
        self.activation = configs.activation
        self.optimizer = configs.optimizer
        self.loss = configs.loss

    def build_initial_model(self):
        encoder_inputs = Input(shape=self.encoder_input_shape, \
                name="encoder_input")
        # Input into Dense(256) layer which will turn (n,2048) into (n,256)
        image_dense = TimeDistributed(Dense(self.lstm_input_size, \
                name="encoder_td_dense"))
        image_dense_input = image_dense(encoder_inputs)
        encoder_lstm = LSTMLayer(self.lstm_input_size, False, True, "e_lstm")
        _, state_h, state_c = encoder_lstm.layer(image_dense_input)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        decoder_inputs = Input(shape=self.decoder_input_shape, \
                name="decoder_input")
        inv_image_dense = TimeDistributed(Dense(self.lstm_input_size, \
                name="decoder_td_dense"))
        decoder_lstm = LSTMLayer(self.lstm_input_size, True, True, "d_lstm")
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_outputs, _, _ = decoder_lstm.layer(\
                                inv_image_dense(decoder_inputs),\
                                initial_state=encoder_states)
        # Softmax because it is multi-class classification
        decoder_dense = Dense(self.decoder_input_size, \
                              activation=self.activation, \
                              name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

def build_res_net():
    base_model = ResNet50()
    base_model.trainable = False
    res_model = Sequential()
    # pylint: disable=E1123,E1120
    res_model.add(Model(input=base_model.input,\
                    output=base_model.get_layer('avg_pool').output))
    res_model.add(Flatten())
    return res_model
