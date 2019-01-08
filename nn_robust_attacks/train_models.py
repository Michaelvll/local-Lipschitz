## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os
from keras import backend as K

def gradient_penalty_loss(correct, scores, inputs, gradient_penalty_weight):
    gradients = K.gradients(scores, inputs)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight *  gradient_l2_norm
    return K.mean(gradient_penalty)

def make_model(data, params):
    model = Sequential()

    print(data.train_data.shape)
    
    model.add(Conv2D(params[0], (3, 3),
                            input_shape=data.train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(10))
    return model

def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None, lip=False, lip_lambda=1.0):
    """
    Standard neural network training procedure.
    """
    print(data.train_data.shape[1:])
    model = make_model(data, params)
    inputs = Input(data.train_data.shape[1:])
    scores = model(inputs)
    model = Model(inputs=inputs, outputs=scores)

    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    def lip_loss(correct, scores):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=scores/train_temp) + gradient_penalty_loss(correct, scores, inputs=inputs, gradient_penalty_weight=lip_lambda)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    

    if lip:
        model.compile(loss=lip_loss,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    else:
        model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              nb_epoch=num_epochs,
              shuffle=True)
    

    if file_name != None:
        model.save(file_name)

    return model



def train_distillation(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1):
    """
    Train a network using defensive distillation.

    Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
    Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
    IEEE S&P, 2016.
    """
    if not os.path.exists(file_name+"_init"):
        # Train for one epoch to get a good starting point.
        train(data, file_name+"_init", params, 1, batch_size)
    
    # now train the teacher at the given temperature
    teacher = train(data, file_name+"_teacher", params, num_epochs, batch_size, train_temp,
                    init=file_name+"_init")

    # evaluate the labels at temperature t
    predicted = teacher.predict(data.train_data)
    with tf.Session() as sess:
        y = sess.run(tf.nn.softmax(predicted/train_temp))
        print(y)
        data.train_labels = y

    # train the student model at temperature t
    student = train(data, file_name, params, num_epochs, batch_size, train_temp,
                    init=file_name+"_init")

    # and finally we predict at temperature 1
    predicted = student.predict(data.train_data)

    print(predicted)
    
if not os.path.isdir('models'):
    os.makedirs('models')

train(CIFAR(), "models/cifar", [64, 64, 128, 128, 256, 256], num_epochs=50)
# train(MNIST(), "models/mnist", [32, 32, 64, 64, 200, 200], num_epochs=50)
train(CIFAR(), "models/cifar_lip1", [64, 64, 128, 128, 256, 256], num_epochs=50, lip=True)
# train(MNIST(), "models/mnist_lip1", [32, 32, 64, 64, 200, 200], num_epochs=50, lip=True)

# train_distillation(MNIST(), "models/mnist-distilled-100", [32, 32, 64, 64, 200, 200],
#                    num_epochs=50, train_temp=100)
# train_distillation(CIFAR(), "models/cifar-distilled-100", [64, 64, 128, 128, 256, 256],
                #    num_epochs=50, train_temp=100)
