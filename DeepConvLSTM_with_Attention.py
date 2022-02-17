import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Activation, Permute
from tensorflow.keras.layers import Multiply, Lambda, Reshape, Dot, Concatenate, RepeatVector, TimeDistributed, Permute, Bidirectional

class Attention(Layer):
    """
    Layer for implementing two common types of attention mechanisms, i) global (soft) attention
    and ii) local (hard) attention, for two types of sequence tasks, i) many-to-one and
    ii) many-to-many.
    The setting use_bias=False converts the Dense() layers into annotation weight matrices. Softmax
    activation ensures that all weights sum up to 1. Read more here to make more sense of the code
    and implementations:
    i)   https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention
    ii)  https://github.com/philipperemy/keras-attention-mechanism/issues/14
    iii) https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
    SUGGESTION: If model doesn't converge or the test accuracy is lower than expected, try playing
    around with the hidden size of the recurrent layers, the batch size in training process, or the
    param @window_width if using a 'local' attention.
    NOTE: This implementation takes the hidden states associated with the last timestep of the input
    sequence as the target hidden state (h_t) as suggested by @felixhao28 in i) for many-to-one
    scenarios. Hence, when trying to predict what word (token) comes after sequence ['I', 'love',
    'biscuits', 'and'], we take h('and') with shape (1, H) as the target hidden state. For
    many-to-many scenarios, it takes the hidden state associated with the timestep that is being
    currently iterated in the target sequence, usually by a decoder-like architecture.
    @param (str) context: the context of the problem at hand, specify 'many-to-many' for
           sequence-to-sequence tasks such as machine translation and question answering, or
           specify 'many-to-one' for tasks such as sentiment classification and language modelling
    @param (str) alignment_type: type of attention mechanism to be applied, 'local-m' corresponds to
           monotonic alignment where we take the last @window_width timesteps, 'local-p' corresponds
           to having a Gaussian distribution around the predicted aligned position, whereas
           'local-p*' corresponds to the newly proposed method to adaptively learning the unique
           timesteps to give attention (currently only works for many-to-one scenarios)
    @param (int) window_width: width for set of source hidden states in 'local' attention
    @param (str) score_function: alignment score function config; current implementations include
           the 'dot', 'general', and 'location' both by Luong et al. (2015), 'concat' by Bahdanau et
           al. (2015), and 'scaled_dot' by Vaswani et al. (2017)
    @param (str) model_api: specify to use TF's Sequential OR Functional API, note that attention
           weights are not outputted with the former as it only accepts single-output layers
    """
    def __init__(self, context='many-to-many', alignment_type='global', window_width=None,
                 score_function='general', model_api='functional', **kwargs):
        if context not in ['many-to-many', 'many-to-one']:
            raise ValueError("Argument for param @context is not recognized")
        if alignment_type not in ['global', 'local-m', 'local-p', 'local-p*']:
            raise ValueError("Argument for param @alignment_type is not recognized")
        if alignment_type == 'global' and window_width is not None:
            raise ValueError("Can't use windowed approach with global attention")
        if context == 'many-to-many' and alignment_type == 'local-p*':
            raise ValueError("Can't use local-p* approach in many-to-many scenarios")
        if score_function not in ['dot', 'general', 'location', 'concat', 'scaled_dot']:
            raise ValueError("Argument for param @score_function is not recognized")
        if model_api not in ['sequential', 'functional']:
            raise ValueError("Argument for param @model_api is not recognized")
        super(Attention, self).__init__(**kwargs)
        self.context = context
        self.alignment_type = alignment_type
        self.window_width = window_width  # D
        self.score_function = score_function
        self.model_api = model_api

    def get_config(self):
        base_config = super(Attention, self).get_config()
        base_config['alignment_type'] = self.alignment_type
        base_config['window_width'] = self.window_width
        base_config['score_function'] = self.score_function
        base_config['model_api'] = self.model_api
        return base_config

    def build(self, input_shape):
        # Declare attributes for easy access to dimension values
        if self.context == 'many-to-many':
            self.input_sequence_length, self.hidden_dim = input_shape[0][1], input_shape[0][2]
            self.target_sequence_length = input_shape[1][1]
        elif self.context == 'many-to-one':
            self.input_sequence_length, self.hidden_dim = input_shape[0][1], input_shape[0][2]

        # Build weight matrices for different alignment types and score functions
        if 'local-p' in self.alignment_type:
            self.W_p = Dense(units=self.hidden_dim, use_bias=False)
            self.W_p.build(input_shape=(None, None, self.hidden_dim))                               # (B, 1, H)
            self._trainable_weights += self.W_p.trainable_weights

            self.v_p = Dense(units=1, use_bias=False)
            self.v_p.build(input_shape=(None, None, self.hidden_dim))                               # (B, 1, H)
            self._trainable_weights += self.v_p.trainable_weights

        if 'dot' not in self.score_function:  # weight matrix not utilized for 'dot' function
            self.W_a = Dense(units=self.hidden_dim, use_bias=False)
            self.W_a.build(input_shape=(None, None, self.hidden_dim))                               # (B, S*, H)
            self._trainable_weights += self.W_a.trainable_weights

        if self.score_function == 'concat':  # define additional weight matrices
            self.U_a = Dense(units=self.hidden_dim, use_bias=False)
            self.U_a.build(input_shape=(None, None, self.hidden_dim))                               # (B, 1, H)
            self._trainable_weights += self.U_a.trainable_weights

            self.v_a = Dense(units=1, use_bias=False)
            self.v_a.build(input_shape=(None, None, self.hidden_dim))                               # (B, S*, H)
            self._trainable_weights += self.v_a.trainable_weights

        super(Attention, self).build(input_shape)

    def call(self, inputs):
        # Pass decoder output (prev. timestep) alongside encoder output for all scenarios
        if not isinstance(inputs, list):
            raise ValueError("Pass a list=[encoder_out (Tensor), decoder_out (Tensor)," +
                             "current_timestep (int)] for all scenarios")

        # Specify source and target states (and timestep if applicable) for easy access
        if self.context == 'many-to-one':
            # Get h_t, the current (target) hidden state as the last timestep of input sequence
            target_hidden_state = inputs[1]                                                         # (B, H)
            source_hidden_states = inputs[0]                                                        # (B, S, H)
        elif self.context == 'many-to-many':
            # Get h_t, the current (target) hidden state from the previous decoded hidden state
            target_hidden_state = inputs[1]                                                         # (B, H)
            current_timestep = inputs[2]
            source_hidden_states = inputs[0]                                                        # (B, S, H)

        # Add time axis to h_t
        target_hidden_state = tf.expand_dims(input=target_hidden_state, axis=1)                     # (B, 1, H)

        # Get h_s, source hidden states through specified attention mechanism
        if self.alignment_type == 'global':                                                         # Global Approach
            source_hidden_states = source_hidden_states                                             # (B, S, H)

        elif 'local' in self.alignment_type:                                                        # Local Approach
            # Automatically set window width to default value (8 -> no real logic behind this value)
            self.window_width = 8 if self.window_width is None else self.window_width

            # Get aligned position (between inputs & targets) and derive a context window to focus
            if self.alignment_type == 'local-m':                                                    # Monotonic Alignment
                # Set alignment position
                if self.context == 'many-to-one':
                    aligned_position = self.input_sequence_length
                elif self.context == 'many-to-many':
                    aligned_position = current_timestep
                # Get window borders
                left = int(aligned_position - self.window_width
                           if aligned_position - self.window_width >= 0
                           else 0)
                right = int(aligned_position + self.window_width
                            if aligned_position + self.window_width <= self.input_sequence_length
                            else self.input_sequence_length)
                # Extract window window
                source_hidden_states = Lambda(lambda x: x[:, left:right, :])(source_hidden_states)  # (B, S*=(D, 2xD), H)

            elif self.alignment_type == 'local-p':                                                  # Predictive Alignment
                aligned_position = self.W_p(target_hidden_state)                                    # (B, 1, H)
                aligned_position = Activation('tanh')(aligned_position)                             # (B, 1, H)
                aligned_position = self.v_p(aligned_position)                                       # (B, 1, 1)
                aligned_position = Activation('sigmoid')(aligned_position)                          # (B, 1, 1)
                aligned_position = aligned_position * self.input_sequence_length                    # (B, 1, 1)

            elif self.alignment_type == 'local-p*':                                                 # Completely Predictive Alignment
                aligned_position = self.W_p(source_hidden_states)                                   # (B, S, H)
                aligned_position = Activation('tanh')(aligned_position)                             # (B, S, H)
                aligned_position = self.v_p(aligned_position)                                       # (B, S, 1)
                aligned_position = Activation('sigmoid')(aligned_position)                          # (B, S, 1)
                # Only keep top D values out of the sigmoid activation, and zero-out the rest
                aligned_position = tf.squeeze(aligned_position, axis=-1)                            # (B, S)
                top_probabilities = tf.nn.top_k(input=aligned_position,                             # (values:(B, D), indices:(B, D))
                                                k=self.window_width,
                                                sorted=False)
                onehot_vector = tf.one_hot(indices=top_probabilities.indices,
                                           depth=self.input_sequence_length)                        # (B, D, S)
                onehot_vector = tf.reduce_sum(onehot_vector, axis=1)                                # (B, S)
                aligned_position = Multiply()([aligned_position, onehot_vector])                    # (B, S)
                aligned_position = tf.expand_dims(aligned_position, axis=-1)                        # (B, S, 1)
                initial_source_hidden_states = source_hidden_states                                 # (B, S, 1)
                source_hidden_states = Multiply()([source_hidden_states, aligned_position])         # (B, S*=S(D), H)
                # Scale back-to approximately original hidden state values
                aligned_position += tf.keras.backend.epsilon()                                      # (B, S, 1)
                source_hidden_states /= aligned_position                                            # (B, S*=S(D), H)
                source_hidden_states = initial_source_hidden_states + source_hidden_states          # (B, S, H)

        # Compute alignment score through specified function
        if 'dot' in self.score_function:                                                            # Dot Score Function
            attention_score = Dot(axes=[2, 2])([source_hidden_states, target_hidden_state])         # (B, S*, 1)
            if self.score_function == 'scaled_dot':
                attention_score *= 1 / np.sqrt(float(source_hidden_states.shape[2]))                # (B, S*, 1)

        elif self.score_function == 'general':                                                      # General Score Function
            weighted_hidden_states = self.W_a(source_hidden_states)                                 # (B, S*, H)
            attention_score = Dot(axes=[2, 2])([weighted_hidden_states, target_hidden_state])       # (B, S*, 1)

        elif self.score_function == 'location':                                                     # Location-based Score Function
            weighted_target_state = self.W_a(target_hidden_state)                                   # (B, 1, H)
            attention_score = Activation('softmax')(weighted_target_state)                          # (B, 1, H)
            attention_score = RepeatVector(source_hidden_states.shape[1])(attention_score)          # (B, S*, H)
            attention_score = tf.reduce_sum(attention_score, axis=-1)                               # (B, S*)
            attention_score = tf.expand_dims(attention_score, axis=-1)                              # (B, S*, 1)

        elif self.score_function == 'concat':                                                       # Concat Score Function
            weighted_hidden_states = self.W_a(source_hidden_states)                                 # (B, S*, H)
            weighted_target_state = self.U_a(target_hidden_state)                                   # (B, 1, H)
            weighted_sum = weighted_hidden_states + weighted_target_state                           # (B, S*, H)
            weighted_sum = Activation('tanh')(weighted_sum)                                         # (B, S*, H)
            attention_score = self.v_a(weighted_sum)                                                # (B, S*, 1)

        # Compute attention weights
        attention_weights = Activation('softmax')(attention_score)                                  # (B, S*, 1)

        # Distribute weights around aligned position for local-p approach only
        if self.alignment_type == 'local-p':                                                        # Gaussian Distribution
            gaussian_estimation = lambda s: tf.exp(-tf.square(s - aligned_position) /
                                                   (2 * tf.square(self.window_width / 2)))
            gaussian_factor = gaussian_estimation(0)
            for i in range(1, self.input_sequence_length):
                gaussian_factor = Concatenate(axis=1)([gaussian_factor, gaussian_estimation(i)])    # (B, S*, 1)
            attention_weights = attention_weights * gaussian_factor                                 # (B, S*, 1)

        # Derive context vector
        context_vector = source_hidden_states * attention_weights                                   # (B, S*, H)

        if self.model_api == 'functional':
            return context_vector, attention_weights
        elif self.model_api == 'sequential':
            return context_vector


class SelfAttention(Layer):
    """
    Layer for implementing self-attention mechanism. Weight variables were preferred over Dense()
    layers in implementation because they allow easier identification of shapes. Softmax activation
    ensures that all weights sum up to 1.
    @param (int) size: a.k.a attention length, number of hidden units to decode the attention before
           the softmax activation and becoming annotation weights
    @param (int) num_hops: number of hops of attention, or number of distinct components to be
           extracted from each sentence.
    @param (bool) use_penalization: set True to use penalization, otherwise set False
    @param (int) penalty_coefficient: the weight of the extra loss
    @param (str) model_api: specify to use TF's Sequential OR Functional API, note that attention
           weights are not outputted with the former as it only accepts single-output layers
    """
    def __init__(self, size, num_hops=8, use_penalization=True,
                 penalty_coefficient=0.1, model_api='functional', batch_size = 1, **kwargs):
        if model_api not in ['sequential', 'functional']:
            raise ValueError("Argument for param @model_api is not recognized")
        self.size = size
        self.num_hops = num_hops
        self.use_penalization = use_penalization
        self.penalty_coefficient = penalty_coefficient
        self.model_api = model_api
        self.batch_size = batch_size
        super(SelfAttention, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        base_config['size'] = self.size
        base_config['batch_size'] = self.batch_size
        base_config['num_hops'] = self.num_hops
        base_config['use_penalization'] = self.use_penalization
        base_config['penalty_coefficient'] = self.penalty_coefficient
        base_config['model_api'] = self.model_api
        return base_config

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.size, int(input_shape[2])),                                # (size, H)
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.num_hops, self.size),                                 # (num_hops, size)
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):  # (B, S, H)
        # Expand weights to include batch size through implicit broadcasting
        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]
        W1, W2 = tf.tile(W1, [self.batch_size, 1, 1]), tf.tile(W2, [self.batch_size, 1, 1])
        #W1, W2 = tf.compat.v1.repeat(W1, repeats = [self.batch_size], axis=0), tf.compat.v1.repeat(W2, repeats = [self.batch_size], axis=0)
        hidden_states_transposed = Permute(dims=(2, 1))(inputs)                                     # (B, H, S)
        attention_score = tf.matmul(W1, hidden_states_transposed)                                   # (B, size, S)
        attention_score = Activation('tanh')(attention_score)                                       # (B, size, S)
        attention_weights = tf.matmul(W2, attention_score)                                          # (B, num_hops, S)
        attention_weights = Activation('softmax')(attention_weights)                                # (B, num_hops, S)
        embedding_matrix = tf.matmul(attention_weights, inputs)                                     # (B, num_hops, H)
        embedding_matrix_flattened = Flatten()(embedding_matrix)                                    # (B, num_hops*H)

        if self.use_penalization:
            attention_weights_transposed = Permute(dims=(2, 1))(attention_weights)                  # (B, S, num_hops)
            product = tf.matmul(attention_weights, attention_weights_transposed)                    # (B, num_hops, num_hops)
            identity = tf.eye(self.num_hops, batch_shape=(inputs.shape[0],))                        # (B, num_hops, num_hops)
            frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(product - identity)))  # distance
            self.add_loss(self.penalty_coefficient * frobenius_norm)  # loss

        if self.model_api == 'functional':
            return embedding_matrix_flattened, attention_weights
        elif self.model_api == 'sequential':
            return embedding_matrix_flattened

import numpy as np
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
import scipy.stats as st
import sys
import os
import random
import numpy as np
import tensorflow as tf

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,  device_count = {'GPU': 0})
session_conf.gpu_options.allow_growth = True

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, Conv2D, Lambda, Input, Bidirectional, CuDNNLSTM, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.python.keras import backend as K
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import keras

def model(x_train, num_labels, LSTM_units, num_conv_filters, batch_size, F, D):
    """
    The proposed model with CNN layer, LSTM RNN layer and self attention layers.
    Inputs:
    - x_train: required for creating input shape for RNN layer in Keras
    - num_labels: number of output classes (int)
    - LSTM_units: number of RNN units (int)
    - num_conv_filters: number of CNN filters (int)
    - batch_size: number of samples to be processed in each batch
    - F: the attention length (int)
    - D: the length of the output (int)
    Returns
    - model: A Keras model
    """
    cnn_inputs = Input(shape=(x_train.shape[1], x_train.shape[2], 1), batch_size=batch_size, name='rnn_inputs')
    cnn_layer = Conv2D(num_conv_filters, kernel_size = (1, x_train.shape[2]), strides=(1, 1), padding='valid', data_format="channels_last")
    cnn_out = cnn_layer(cnn_inputs)

    sq_layer = Lambda(lambda x: K.squeeze(x, axis = 2))
    sq_layer_out = sq_layer(cnn_out)

    rnn_layer = LSTM(LSTM_units, return_sequences=True, name='lstm', return_state=True) #return_state=True
    rnn_layer_output, _, _ = rnn_layer(sq_layer_out)

    encoder_output, attention_weights = SelfAttention(size=F, num_hops=D, use_penalization=False, batch_size = batch_size)(rnn_layer_output)
    dense_layer = Dense(num_labels, activation = 'softmax')
    dense_layer_output = dense_layer(encoder_output)

    model = Model(inputs=cnn_inputs, outputs=dense_layer_output)
    print (model.summary())

    return model

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


EPOCH = 10
BATCH_SIZE = 16
LSTM_UNITS = 32
CNN_FILTERS = 3
NUM_LSTM_LAYERS = 1
LEARNING_RATE = 1e-4
PATIENCE = 20
SEED = 0
F = 32
D = 10
#DATA_FILES = ['WISDM.npz']
#DATA_FILES = ['ubicomp_deepconv_attn.npz']
DATA_FILES = ['ubicomp_deepconv_attn_topic.npz']

MODE = 'LOTO'
BASE_DIR = './data/' + MODE + '/'
SAVE_DIR = './model_with_self_attn_' + MODE + '_results'

if not os.path.exists(os.path.join(SAVE_DIR)):
    os.mkdir(os.path.join(SAVE_DIR))

if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    tf.compat.v1.set_random_seed(0)

    for DATA_FILE in DATA_FILES:
        data_input_file = os.path.join(BASE_DIR, DATA_FILE)
        tmp = np.load(data_input_file, allow_pickle=True)
        X = tmp['X']
        #print(X.shape)
        X = np.squeeze(X, axis = 1)
        #print(X.shape)
        #break
        y_one_hot = tmp['y']
        folds = tmp['folds']

        NUM_LABELS = y_one_hot.shape[1]

        avg_acc = []
        avg_recall = []
        avg_f1 = []
        early_stopping_epoch_list = []
        y = np.argmax(y_one_hot, axis=1)

        for i in range(0, len(folds)):
            train_idx = folds[i][0]
            test_idx = folds[i][1]

            X_train, y_train, y_train_one_hot = X[train_idx], y[train_idx], y_one_hot[train_idx]
            X_test, y_test, y_test_one_hot = X[test_idx], y[test_idx], y_one_hot[test_idx]

            X_train_ = np.expand_dims(X_train, axis = 3)
            X_test_ = np.expand_dims(X_test, axis = 3)

            train_trailing_samples =  X_train_.shape[0]%BATCH_SIZE
            test_trailing_samples =  X_test_.shape[0]%BATCH_SIZE


            if train_trailing_samples!= 0:
                X_train_ = X_train_[0:-train_trailing_samples]
                y_train_one_hot = y_train_one_hot[0:-train_trailing_samples]
                y_train = y_train[0:-train_trailing_samples]
            if test_trailing_samples!= 0:
                X_test_ = X_test_[0:-test_trailing_samples]
                y_test_one_hot = y_test_one_hot[0:-test_trailing_samples]
                y_test = y_test[0:-test_trailing_samples]

            print (y_train.shape, y_test.shape)

            rnn_model = model(x_train = X_train_, num_labels = NUM_LABELS, LSTM_units = LSTM_UNITS, \
                num_conv_filters = CNN_FILTERS, batch_size = BATCH_SIZE, F = F, D= D)

            model_filename = SAVE_DIR + '/best_model_with_self_attn_' + str(DATA_FILE[0:-4]) + '_fold_' + str(i) + '.h5'
            callbacks = [ModelCheckpoint(filepath=model_filename, monitor = 'val_accuracy', save_weights_only=True, save_best_only=True), EarlyStopping(monitor='val_accuracy', patience=PATIENCE)]#, LearningRateScheduler()]

            opt = optimizers.Adam(clipnorm=1.)

            rnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            history = rnn_model.fit(X_train_, y_train_one_hot, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks, validation_data=(X_test_, y_test_one_hot))

            early_stopping_epoch = callbacks[1].stopped_epoch - PATIENCE + 1
            print('Early stopping epoch: ' + str(early_stopping_epoch))
            early_stopping_epoch_list.append(early_stopping_epoch)

            if early_stopping_epoch <= 0:
                early_stopping_epoch = -100

            # Evaluate model and predict data on TEST
            print("******Evaluating TEST set*********")
            rnn_model.load_weights(model_filename)

            y_test_predict = rnn_model.predict(X_test_, batch_size = BATCH_SIZE)
            y_test_predict = np.array(y_test_predict)
            y_test_predict = np.argmax(y_test_predict, axis=1)

            #all_trainable_count = int(np.sum([K.count_params(p) for p in set(rnn_model.trainable_weights)]))

            MAE = metrics.mean_absolute_error(y_test, y_test_predict, sample_weight=None, multioutput='uniform_average')

            acc_fold = accuracy_score(y_test, y_test_predict)
            avg_acc.append(acc_fold)

            recall_fold = recall_score(y_test, y_test_predict, average='macro')
            avg_recall.append(recall_fold)

            f1_fold  = f1_score(y_test, y_test_predict, average='macro')
            avg_f1.append(f1_fold)

            with open(SAVE_DIR + '/results_model_with_self_attn_' + MODE + '.csv', 'a') as out_stream:
                out_stream.write(str(SEED) + ', ' + str(DATA_FILE[0:-4]) + ', ' + str(i) + ', ' + str(early_stopping_epoch) + ', ' + str(acc_fold) + ', ' + str(MAE) + ', ' + str(recall_fold) + ', ' + str(f1_fold) + '\n')


            print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold, i))
            print('______________________________________________________')
            K.clear_session()

    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
    ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc = np.mean(avg_f1), scale=st.sem(avg_f1))

    print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))
