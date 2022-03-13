import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization
from typing import List

from .layer import LRPLayer


class BatchNormalizationLRP(LRPLayer):
    def __init__(self, layer, *args, name: str = 'batchnorm_lrp', **kwargs):
        assert isinstance(layer, BatchNormalization), \
            ('BatchNormalizationLRP should only be called with a '
             'BatchNormalization layer')

        super().__init__(layer, *args, name=name, **kwargs)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        a, R = inputs

        gammas = self.layer.gamma
        means = self.layer.moving_mean
        betas = self.layer.beta
        vars = self.layer.moving_variance
        eps = self.layer.epsilon

        vars = tf.add(vars, eps, name=f'{self.name}/variance/epsilon')
        stddev = tf.sqrt(vars, name=f'{self.name}/stddev')
        offset = tf.multiply(gammas, means, name=f'{self.name}/offset')
        relevances = tf.subtract(R, betas, name=f'{self.name}/relevances')
        relevances = tf.multiply(relevances, stddev,
                                 name=f'{self.name}/relevances/scaled')
        relevances = tf.add(offset, relevances, name=f'{self.name}/nominator')
        relevances = tf.divide(relevances, gammas, name=self.name)

        return relevances
