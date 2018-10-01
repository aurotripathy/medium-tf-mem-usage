from keras.layers import Conv2D, Dense, Input, Flatten
from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from time import time
from keras.utils.training_utils import multi_gpu_model

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255

input = Input(shape=(train_data.shape[1],
                     train_data.shape[2], train_data.shape[3]))
x = Conv2D(64, (3, 3), padding='same', activation='relu')(input)
x = Flatten()(x)
output = Dense(10, activation='softmax')(x)
model = Model(input, output)

model = multi_gpu_model(model, gpus=4)
print(model.summary())
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir="/root/data/logs/{}".format(time()))
model.fit(train_data, train_labels, batch_size=64, epochs=20,
          validation_data=(test_data, test_labels),
          callbacks=[tensorboard])
