import numpy as np
np.random.seed(666)

import threading
import itertools
import scipy.sparse as ss

from keras.layers import (
    Input,
    Dense,
    RepeatVector,
    LSTM, Permute, Multiply, Reshape, Lambda, Concatenate, multiply, Flatten, Dropout, Activation,
)
from keras.regularizers import l2, l1, l1_l2
from keras.models import Model
from keras.optimizers import Nadam, Adam, Optimizer, Adadelta
from keras import backend as K
from keras import regularizers
from tqdm import tqdm

from .utils import get_random_state, weighted_binary_crossentropy, \
    get_rnn_unit, w_bin_xentropy
from mlearn.criteria import (
    reweight_pairwise_f1_score,
    reweight_pairwise_rank_loss,
    reweight_pairwise_accuracy_score,
    sparse_reweight_pairwise_f1_score,
    sparse_reweight_pairwise_rank_loss,
    sparse_reweight_pairwise_accuracy_score,
)


def get_data_recurrent(n, time_steps, input_dim, attention_column=None):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    if attention_column is None:
        attention_column = np.random.randint(low=0, high=input_dim)
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    """
    Now let's define get_activations so we can extract out relevant information from the layers to analyze our results.
    For a given Model and inputs, find all the activations in specified layer
    If no layer then use all layers

    Returns:
    activations from all the layer(s)
    """
    # print("modelnet"+str(model))
    print("---- activations ----")
    activations = []
    inp = model.input
    # inp = model.layers.Input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


# if True, the attention vector is shared across the input_dimensions where the attention is applied.
def attention_3d_block(inputs, single_attention_vector=False):
    print(inputs.shape)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    # time_steps = int(inputs.shape[1])
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, time_steps))(a)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    attention_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, attention_probs])
    return output_attention_mul


def attention_3d_block_m(inputs, single_attention_vector=False):
    """
     Implement the attention block applied between two layers

     Argument:
     inputs -- output of the previous layer, set of hidden states

     Returns:
     output_attention_mul -- inputs weighted with attention probabilities
     """
    # inputs.shape = (batch_size, time_steps, input_dim)
    # time_step, input_dim
    print(inputs.shape)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]

    a = Permute((2, 1))(inputs)

    # kidney
    # a = Dense(time_steps, activation='relu', use_bias=True,
    #           kernel_initializer='random_uniform', kernel_regularizer=l2(0.00001))(a)

    # liver
    a = Dense(time_steps, activation='relu', use_bias=True, kernel_initializer='random_uniform',
              kernel_regularizer=l2(1e-6))(a)

    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a = Activation('softmax')(a)

    # Permute the 2nd and the first column of 'a' to get a probability vector of attention.
    a_probs = Permute((2, 1), name='attention_vec1')(a)
    # Apply the attention probabilities to the "inputs" by multiplying element-wise.
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def arch_002(input_shape, n_labels, weight_input_shape, l2w, rnn_unit='lstm'):
    # L2 Regularization, a method to reduce overfitting for neural networks
    if l2w is None:
        regularizer = None
    else:
        regularizer = l2(l2w)

    inputs = Input(shape=input_shape[1:])
    x = RepeatVector(input_shape[0])(inputs)  # RepeatVector():Repeat the vector "inputs" input_shape[0] times

    print(x)

    x = attention_3d_block_m(x)

    x = Dense(128, kernel_regularizer=regularizer, activation='relu')(x)  # Dense():实现全连接层,units=128代表该层的输出维度

    x = get_rnn_unit(rnn_unit, 128, x, activation='sigmoid', l2w=regularizer, recurrent_dropout=0.1)

    outputs = Dense(n_labels, activation='sigmoid', name='outputs')(x)

    weight_input = Input(shape=weight_input_shape)

    model = Model(inputs=[inputs, weight_input], outputs=[outputs])
    get_output = Model(inputs=model.input, outputs=model.get_layer('outputs').output)

    return model, weight_input, get_output


# ---------------------------Everything before this line is the modified model---------------------------


def arch_001(input_shape, n_labels, weight_input_shape, l2w=1e-5, rnn_unit='lstm'):
    # L2 Regularization, a method to reduce overfitting for neural networks
    if l2w is None:
        regularizer = None
    else:
        regularizer = l2(l2w)

    print(input_shape)
    print(n_labels)
    print(weight_input_shape)

    inputs = Input(shape=input_shape[1:])
    x = RepeatVector(input_shape[0])(inputs)
    x = Dense(128, kernel_regularizer=regularizer, activation='relu')(x)

    x = get_rnn_unit(rnn_unit, 128, x, activation='sigmoid', l2w=regularizer, recurrent_dropout=0.1)

    outputs = Dense(n_labels, activation='sigmoid', name='arch001_outputs')(x)
    weight_input = Input(shape=weight_input_shape)

    model = Model(inputs=[inputs, weight_input], outputs=[outputs])
    get_output = Model(inputs=model.input, outputs=model.get_layer('arch001_outputs').output)

    return model, weight_input, get_output


class RethinkNet(object):
    """
    RethinkNet model

    Parameters
    ----------
    n_features: int
    n_labels: int
    scoring_fn:
    reweight: ['balanced', 'None', 'hw', 'vw']
        'hw': horizontal reweighting
        'vw': vertical reweighting
        'balanced':
        'None':
    b: int, optional, default=3
        number of rethinking iteration to perform
    nb_epochs: int
        number of epochs to train, to save time, we just train 100 epochs
        train the training data n times
    batch_size: int, optional, default=256

    Attributes
    ----------
    model : keras.models.Model instance

    References
    ----------
    Yao-Yuan Yang, Yi-An Lin, Hong-Min Chu, Hsuan-Tien Lin. "Deep Learning
    with a Rethinking Structure ffor Multi-label Classification."
    https://arxiv.org/abs/1802.01697, (2018).
    """

    # Hyper Parameters
    # liver b=5, nb_epochs=368, l2w=10 ** -6; kidney b=3, nb_epochs=151, l2w=10 ** -5
    def __init__(self, n_features: int, n_labels: int, scoring_fn, l2w, architecture: str,
                 learning_rate=0.001, decay=0., b: int = 5, batch_size: int = 256,
                 nb_epochs: int = 2, reweight: str = 'hw', optimizer='Adam',
                 random_state=2020, predict_period: int = 1):
        self.random_state = get_random_state(random_state)
        self.batch_size = batch_size
        self.b = b
        self.scoring_fn = scoring_fn
        self.predict_period = predict_period

        if reweight in ['balanced', 'None']:
            self.reweight_scoring_fn = None
        elif reweight in ['hw', 'vw']:
            # if 'pairwise_hamming' in self.scoring_fn.__str__():
            #    self.reweight_scoring_fn = reweight_pairwise_hamming
            # To convert Python objects into strings by using __str__
            if 'pairwise_rank_loss' in self.scoring_fn.__str__():
                self.reweight_scoring_fn = sparse_reweight_pairwise_rank_loss
            elif 'pairwise_accuracy_score' in self.scoring_fn.__str__():
                self.reweight_scoring_fn = sparse_reweight_pairwise_accuracy_score
            elif 'pairwise_f1_score' in self.scoring_fn.__str__():
                self.reweight_scoring_fn = sparse_reweight_pairwise_f1_score
            else:
                raise ValueError(self.scoring_fn, "not supported")

        self.nb_epochs = nb_epochs
        self.reweight = reweight
        self.l2w = l2w

        self.n_labels = n_labels
        self.n_features = n_features
        self.input_shape = ((self.b,) + (n_features,))
        self.weight_input_shape = (self.b, self.n_labels,)
        model, weight_input, get_output = \
            globals()[architecture](self.input_shape, self.n_labels,
                                    self.weight_input_shape, self.l2w)
        model.summary()
        self.nb_params = int(model.count_params())

        # An optimizer is one of the two arguments required for compiling a Keras model.
        if optimizer is None:
            optimizer = Nadam()
        elif optimizer == 'Adam':
            optimizer = Adam(lr=learning_rate, decay=decay)
        elif optimizer == 'Adadelta':
            optimizer = Adadelta(lr=learning_rate, decay=decay)
        if not isinstance(optimizer, Optimizer):
            raise ValueError("optimizer should be keras.optimizers.Optimizer."
                             "got :", optimizer)

        self.loss = weighted_binary_crossentropy(weight_input)  # binary_crossentropy(二元交叉熵)

        # model.compile(loss='binary_crossentropy', optimizer=optimizer)
        model.compile(loss=self.loss, optimizer=optimizer)
        self.model = model
        self.get_output = get_output

    def _prep_X(self, X):
        X = X.toarray()
        return X

    def _prep_Y(self, Y):
        Y = Y.toarray()
        Y = np.repeat(Y[:, np.newaxis, :], self.b, axis=1)
        return Y

    def _prep_weight(self, trn_pred, trnY):
        weight = np.ones((trnY.shape[0], self.b, self.n_labels),
                         dtype='float32')
        i_start = 1
        if 'vw' in self.reweight:
            i_start = 0
        for i in range(i_start, self.b):
            if self.reweight == 'balanced':
                weight[:, i, :] = trnY.astype('float32') * (
                        1. / self.ones_weight - 1.)
                weight[:, i, :] += 1.
            elif self.reweight == 'None':
                pass
            elif self.reweight_scoring_fn in [
                sparse_reweight_pairwise_accuracy_score,
                sparse_reweight_pairwise_f1_score,
                sparse_reweight_pairwise_rank_loss]:
                trn_pre = trn_pred[i - 1]
                if 'vw' in self.reweight:
                    trn_pre = trn_pred[i]
                # weight[:, i, :] = self.reweight_scoring_fn(
                #     trnY, trn_pre,
                #     use_true=('truth' in self.reweight))
                weight[:, i, :] = self.reweight_scoring_fn(
                    trnY, trn_pre)
            elif self.reweight_scoring_fn is not None:
                trn_pre = trn_pred[i - 1]
                if 'vw' in self.reweight:
                    trn_pre = trn_pred[i]
                # w = self.reweight_scoring_fn(
                #     trnY,
                #     trn_pre.toarray(),
                #     use_true=('truth' in self.reweight))
                w = self.reweight_scoring_fn(
                    trnY,
                    trn_pre.toarray())
                weight[:, i, :] = np.abs(w[:, :, 0] - w[:, :, 1])
            else:
                raise NotImplementedError()
            weight[:, i, :] *= weight[:, i, :].size / weight[:, i, :].sum()

        return weight.astype('float32')

    def train(self, X, Y, callbacks=None):
        self.history = []
        nb_epochs = self.nb_epochs
        X = ss.csr_matrix(X).astype('float32')
        Y = ss.csr_matrix(Y).astype(np.int8)

        if self.reweight == 'balanced':
            self.ones_weight = Y.astype(np.int32).sum() / Y.shape[0] / Y.shape[1]

        trn_pred = []
        for _ in range(self.b):
            trn_pred.append(
                ss.csr_matrix((X.shape[0], self.n_labels), dtype=np.int8))

        predict_period = self.predict_period
        for epoch_i in range(0, nb_epochs, predict_period):
            input_generator = InputGenerator(
                self, X, Y, trn_pred, shuffle=False,
                batch_size=self.batch_size, random_state=self.random_state)
            input_generator.next()
            _ = self.model.fit_generator(
                input_generator,
                steps_per_epoch=((X.shape[0] - 1) // self.batch_size) + 1,
                epochs=epoch_i + predict_period,
                max_queue_size=32,
                workers=1,
                use_multiprocessing=False,
                initial_epoch=epoch_i,
                validation_data=None,
                verbose=1,
                callbacks=callbacks)

            # import tensorflow as tf
            # tf.compat.v1.disable_v2_behavior()
            # init = tf.compat.v1.initialize_all_variables()
            # with tf.compat.v1.Session() as sess:
            #     print(sess.run(init))

            # workers=8, use_multiprocessing=True, verbose=1,

            trn_scores = []

            trn_pred, temp = self.predict_chain(X)
            for j in range(self.b):
                trn_scores.append(np.mean(self.scoring_fn(Y, trn_pred[j])))
            print("[epoch %6d] trn:" % (epoch_i + predict_period), trn_scores)

            self.history.append({
                'epoch_nb': epoch_i,
                'trn_scores': trn_scores,
            })

    def predict_chain(self, X):
        ret = [[] for i in range(self.b)]
        features_return = [[] for i in range(self.b)]
        batches = range(X.shape[0] // self.batch_size
                        + ((X.shape[0] % self.batch_size) > 0))
        _ = np.ones((self.batch_size, self.b, self.n_labels))

        for bs in tqdm(batches, desc="Predicting"):
            if (bs + 1) * self.batch_size > X.shape[0]:
                batch_idx = np.arange(X.shape[0])[bs * self.batch_size: X.shape[0]]
            else:
                batch_idx = np.arange(X.shape[0])[bs * self.batch_size: (bs + 1) * self.batch_size]

            pred_chain = self.model.predict([self._prep_X(X[batch_idx]), _])
            features = self.get_output.predict([self._prep_X(X[batch_idx]), _])
            pred_chain = pred_chain > 0.5

            for i in range(self.b):
                # print("pred_chain:"+str(pred_chain[:, i, :]))
                ret[i].append(ss.csr_matrix(pred_chain[:, i, :], dtype=np.int8))

            for j in range(self.b):
                # print("features_return:" + str(features[:, j, :]))
                features_return[j].append(ss.csr_matrix(features[:, j, :], dtype=np.float64))

        for i in range(self.b):
            ret[i] = ss.vstack(ret[i])

        for i in range(self.b):
            features_return[i] = ss.vstack(features_return[i])

        return ret, features_return

    def predict(self, X):
        X = ss.csr_matrix(X)
        pred_chain, features_temp = self.predict_chain(X)
        pred = pred_chain[-1]
        return pred

    def model_get_output_features(self, X):
        X = ss.csr_matrix(X)
        pred_chain, feature_vis = self.predict_chain(X)

        features = feature_vis[-1]
        features = features.toarray()
        # print(features)
        # print(features.shape)
        return features

    def predict_probability(self, X):
        X = ss.csr_matrix(X)

        # predict_chain
        ret = [[] for i in range(self.b)]
        batches = range(X.shape[0] // self.batch_size
                        + ((X.shape[0] % self.batch_size) > 0))
        _ = np.ones((self.batch_size, self.b, self.n_labels))

        for bs in tqdm(batches, desc="Predicting"):
            if (bs + 1) * self.batch_size > X.shape[0]:
                batch_idx = np.arange(X.shape[0])[bs * self.batch_size: X.shape[0]]
            else:
                batch_idx = np.arange(X.shape[0])[bs * self.batch_size: (bs + 1) * self.batch_size]

            pred_chain = self.model.predict([self._prep_X(X[batch_idx]), _])

            for i in range(self.b):
                # print(pred_chain[:, i, :])
                ret[i].append(ss.csr_matrix(pred_chain[:, i, :], dtype=np.float32))

        for i in range(self.b):
            ret[i] = ss.vstack(ret[i])

        pred = ret[-1]
        return pred

    def predict_topk(self, X, k=5):
        ret = np.zeros((self.b, X.shape[0], k), np.float32)
        batches = range(X.shape[0] // self.batch_size \
                        + ((X.shape[0] % self.batch_size) > 0))
        _ = np.ones((self.batch_size, self.b, self.n_labels))

        for bs in batches:
            if (bs + 1) * self.batch_size > X.shape[0]:
                batch_idx = np.arange(X.shape[0])[bs * self.batch_size: X.shape[0]]
            else:
                batch_idx = np.arange(X.shape[0])[bs * self.batch_size: (bs + 1) * self.batch_size]

            pred_chain = self.model.predict([self._prep_X(X[batch_idx]), _])

            for i in range(self.b):
                ind = np.argsort(pred_chain[:, i, :], axis=1)[:, -k:][:, ::-1]
                ret[i, batch_idx, :] = ind

        return ret

    def get_att(self, testing_inputs_1):
        attention_vectors = []
        for i in range(1):  # range(1) can skip data generation, range(k) generate k training example (x, y).
            # Generate one training example (x, y), the attention column can be on any time-step.
            # testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM, attention_column=3)
            # Extract the attention vector predicted by the model "m" on the training example "x".
            attention_vector = np.mean(get_activations(self.model,
                                                       testing_inputs_1,
                                                       print_shape_only=True,
                                                       layer_name='attention_vec1')[0], axis=2).squeeze()

            # append the attention vector to the list of attention vectors
            # attention_vectors.append(attention_vector)
            attention_vector = attention_vector.tolist()
            print("attention1 = ", attention_vector)
            # assert (np.sum(attention_vector) - 1.0) < 1e-5
            attention_vectors = attention_vector
            print("attention_vectors = "+str(attention_vectors))

        # Compute the average attention on every time-step
        attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
        # attention_vector_final = attention_vector_final.reshape(-1, 1)
        print("attention_vector_final = " + str(attention_vector_final))

        # plot part
        # pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar', rot=0,
        #                                                                      title='Attention Mechanism as a function'
        #                                                                            ' of input dimensions.')
        # plt.show()

        return attention_vector_final


class InputGenerator(object):
    def __init__(self, model, X, Y=None, pred=None, shuffle=False,
                 batch_size=256, random_state=None):
        self.model = model
        self.X = X
        self.Y = Y
        self.lock = threading.Lock()
        if random_state is None:
            self.random_state = np.random.RandomState()

        self.index_generator = self._flow_index(X.shape[0], batch_size, shuffle,
                                                random_state)
        self.dummy_weight = np.ones((batch_size, self.model.b, self.model.n_labels),
                                    dtype=float)

        self.pred = pred

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _flow_index(self, n, batch_size, shuffle, random_state):
        index = np.arange(n)
        for epoch_i in itertools.count():
            if shuffle:
                random_state.shuffle(index)
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                yield epoch_i, index[batch_start: batch_end]

    def next(self):
        with self.lock:
            epoch_i, index_array = next(self.index_generator)
        batch_X = self.X[index_array]
        preped_X = self.model._prep_X(batch_X)

        if self.Y is None:
            return [preped_X, self.dummy_weight]
        else:
            batch_Y = self.Y[index_array]
            preped_Y = self.model._prep_Y(batch_Y)
            pred = [self.pred[j][index_array] for j in range(self.model.b)]

            if self.model.reweight_scoring_fn in [
                sparse_reweight_pairwise_accuracy_score,
                sparse_reweight_pairwise_f1_score,
                sparse_reweight_pairwise_rank_loss]:
                lbl_weight = self.model._prep_weight(pred, batch_Y)
            else:
                lbl_weight = self.model._prep_weight(pred, preped_Y[:, 0, :])
            return [preped_X, lbl_weight], preped_Y
