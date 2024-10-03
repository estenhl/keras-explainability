from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Conv3D, Dense,
                                     GlobalAveragePooling3D,
                                     GlobalMaxPooling3D, MaxPooling3D, Input)

from explainability import LRP, LRPStrategy
from explainability.layers import PoolingLRPLayer


def test_lrp_strategy_wrong_number_of_layers():
    inputs = Input((16, 16, 16, 1))

    x = Conv3D(2, (3, 3, 3), padding='SAME')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(5)(x)

    model = Model(inputs, x)

    strategy = LRPStrategy(
        layers=[
            {'alpha': 1, 'beta': 0},
            {'epsilon': 0.25}
        ]
    )

    exception = False

    try:
        LRP(model, layer=11, idx=3, strategy=strategy)
    except Exception:
        exception = True

    assert exception, \
        ('Creating an LRP with a strategy which does not have the right '
         'number of layers does not raise an exception')


def test_lrp_strategy_and_epsilon():
    inputs = Input((16, 16, 16, 1))

    x = Conv3D(2, (3, 3, 3), padding='SAME')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(5)(x)

    model = Model(inputs, x)

    strategy = LRPStrategy(
        layers=[
            {'alpha': 1, 'beta': 0},
            {'epsilon': 0.25},
            {'alpha': 2, 'beta': 1},
            {'epsilon': 0.5}
        ]
    )

    exception = False

    try:
        LRP(model, layer=11, idx=3, strategy=strategy, epsilon=0.25)
    except Exception:
        exception = True

    assert exception, \
        ('Creating an LRP with a strategy and epsilon does not raise an '
         'exception')

def test_lrp_strategy_and_gamma():
    inputs = Input((16, 16, 16, 1))

    x = Conv3D(2, (3, 3, 3), padding='SAME')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(5)(x)

    model = Model(inputs, x)

    strategy = LRPStrategy(
        layers=[
            {'alpha': 1, 'beta': 0},
            {'epsilon': 0.25},
            {'alpha': 2, 'beta': 1},
            {'epsilon': 0.5}
        ]
    )

    exception = False

    try:
        LRP(model, layer=11, idx=3, strategy=strategy, gamma=0.25)
    except Exception:
        exception = True

    assert exception, \
        ('Creating an LRP with a strategy and gamma does not raise an '
         'exception')

def test_lrp_strategy_and_alpha():
    inputs = Input((16, 16, 16, 1))

    x = Conv3D(2, (3, 3, 3), padding='SAME')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(5)(x)

    model = Model(inputs, x)

    strategy = LRPStrategy(
        layers=[
            {'alpha': 1, 'beta': 0},
            {'epsilon': 0.25},
            {'alpha': 2, 'beta': 1},
            {'epsilon': 0.5}
        ]
    )

    exception = False

    try:
        LRP(model, layer=11, idx=3, strategy=strategy, alpha=1, beta=0)
    except Exception as e:
        exception = True

    assert exception, \
        ('Creating an LRP with a strategy and alpha/beta does not raise an '
         'exception')

def test_lrp_strategy_sets_values():
    inputs = Input((16, 16, 16, 1))

    x = Conv3D(2, (3, 3, 3), padding='SAME')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(5)(x)

    model = Model(inputs, x)

    strategy = LRPStrategy(
        layers=[
            {'alpha': 1, 'beta': 0},
            {'gamma': 0.25},
            {'alpha': 2, 'beta': 1},
            {'epsilon': 0.5}
        ]
    )

    explainer = LRP(model, layer=11, idx=3, strategy=strategy)

    conv1_lrp = explainer.layers[23]
    conv2_lrp = explainer.layers[20]
    conv3_lrp = explainer.layers[17]
    dense_lrp = explainer.layers[13]

    assert conv1_lrp.epsilon == None, \
        'LRPStrategy does not set properties of layers correctly'
    assert conv1_lrp.gamma == None, \
        'LRPStrategy does not set properties of layers correctly'
    assert conv1_lrp.alpha == 1, \
        'LRPStrategy does not set properties of layers correctly'
    assert conv1_lrp.beta == 0, \
        'LRPStrategy does not set properties of layers correctly'

    assert conv2_lrp.epsilon == None, \
        'LRPStrategy does not set properties of layers correctly'
    assert conv2_lrp.gamma == 0.25, \
        'LRPStrategy does not set properties of layers correctly'
    assert conv2_lrp.alpha == None, \
        'LRPStrategy does not set properties of layers correctly'
    assert conv2_lrp.beta == None, \
        'LRPStrategy does not set properties of layers correctly'

    assert conv3_lrp.epsilon == None, \
        'LRPStrategy does not set properties of layers correctly'
    assert conv3_lrp.gamma == None, \
        'LRPStrategy does not set properties of layers correctly'
    assert conv3_lrp.alpha == 2, \
        'LRPStrategy does not set properties of layers correctly'
    assert conv3_lrp.beta == 1, \
        'LRPStrategy does not set properties of layers correctly'


    assert dense_lrp.epsilon == 0.5, \
        'LRPStrategy does not set properties of layers correctly'
    assert dense_lrp.gamma == None, \
        'LRPStrategy does not set properties of layers correctly'
    assert dense_lrp.alpha == None, \
        'LRPStrategy does not set properties of layers correctly'
    assert dense_lrp.beta == None, \
        'LRPStrategy does not set properties of layers correctly'

def test_lrp_strategy_pooling():
    inputs = Input((16, 16, 16, 1))

    x = Conv3D(2, (3, 3, 3), padding='SAME')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='SAME')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2))(x)

    x = GlobalMaxPooling3D()(x)
    x = Dense(5)(x)

    model = Model(inputs, x)

    strategy = LRPStrategy(
        pooling = [
            {'strategy': 'flat'},
            {'strategy': 'redistribute'},
            {'strategy': 'winner-takes-all'},
            {'strategy': 'winner-takes-all'}
        ]
    )

    explainer = LRP(model, layer=11, idx=3, strategy=strategy)

    assert explainer.layers[21].strategy == PoolingLRPLayer.Strategy.FLAT
    assert explainer.layers[18].strategy == \
           PoolingLRPLayer.Strategy.REDISTRIBUTE
    assert explainer.layers[15].strategy == \
           PoolingLRPLayer.Strategy.WINNER_TAKES_ALL
    assert explainer.layers[14].strategy == \
           PoolingLRPLayer.Strategy.WINNER_TAKES_ALL
