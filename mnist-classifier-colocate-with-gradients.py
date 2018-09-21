from keras.layers import Conv2D, Dense, Input, Flatten
from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True

set_session(tf.Session(config=config))

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255

with tf.device('/gpu:0'):
    input = Input(shape=(train_data.shape[1],
                         train_data.shape[2], train_data.shape[3]))
    x = Conv2D(64, (5, 5), padding='same', activation='relu')(input)
    x = Flatten()(x)

with tf.device('/gpu:1'):
    x = Dense(128)(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(input, output)


class MyAdamOptimizer(tf.train.AdamOptimizer):
    def compute_gradients(self,
                          loss,
                          var_list=None,
                          gate_gradients=tf.train.Optimizer.GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=True,
                          grad_loss=None):
        return super(MyAdamOptimizer, self).compute_gradients(
            loss,
            var_list=None,
            gate_gradients=tf.train.Optimizer.GATE_OP,
            aggregation_method=None,
            colocate_gradients_with_ops=True,
            grad_loss=None)

    def minimize(
            loss,
            global_step=None,
            var_list=None,
            gate_gradients=tf.train.Optimizer.GATE_OP,
            aggregation_method=None,
            colocate_gradients_with_ops=True,
            name=None,
            grad_loss=None):
        return super(MyAdamOptimizer, self).minimize(
            loss,
            global_step=None,
            var_list=None,
            gate_gradients=tf.train.Optimizer.GATE_OP,
            aggregation_method=None,
            colocate_gradients_with_ops=True,
            name=None,
            grad_loss=None)


print(model.summary())
model.compile(optimizer=MyAdamOptimizer(learning_rate=0.001),
              # model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_labels, batch_size=64, epochs=5,
          validation_data=(test_data, test_labels))
