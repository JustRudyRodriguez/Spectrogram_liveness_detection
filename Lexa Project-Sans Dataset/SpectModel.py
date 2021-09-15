import tensorflow as tf
from tensorflow.keras import layers, models, applications


def build_model(hp):
    # So I will begin by creating an instance of the mobilenet2 model here, with he example dataset above.
    # ok, so according to the reading I did, removing top means it's not using the fully connected layer.
    # This stack overflow comment simplifies it. https://stackoverflow.com/a/46745897
    # the short of it. I may need to add my own "top" the section that makes the final classifications.

    # I need to
    mobile = applications.MobileNetV2(
        input_shape=(224, 224, 3), alpha=hp.Choice('alpha', values=[.5, .75, 1.0, 1.3, 1.4]), include_top=False,
        weights="imagenet",
        input_tensor=None, classes=2, pooling=hp.Choice('pooling', values=['avg', 'max'])
    )

    # 156 layers in mobilenetv2
    print(len(mobile.layers))
    # we freeze these layers, before adding our own. [:-n] if we want to include more off the top. 155 is the total net.
    #:-hp.Int('unfrozen layers',) add this as a parameter. # try per "block"

    for x in mobile.layers:
        x.trainable = False

    # I create a "new" model here.
    newmod = models.Sequential()
    # Adding mobilenetv2 without top to the new model.
    newmod.add(mobile)

    for i in range(hp.Int('num_Blayers', min_value=0, max_value=3)):
        newmod.add(layers.Dense(units=hp.Int('units_B_' + str(i),
                                             min_value=32,
                                             max_value=512,
                                             step=32),
                                activation='relu',
                                trainable=True))

    # there are 2 possible outputs, so the final layer needs to be 1.
    # Because of the use of a BCE loss(), the output layer needs to be a single node with sigmoid activation.
    for i in range(hp.Int('num_Slayers', min_value=1, max_value=1)):
        newmod.add(layers.Dense(units=hp.Int('units_S_' + str(i),
                                             min_value=16,
                                             max_value=128,
                                             step=16),
                                activation='relu',
                                trainable=True))

        newmod.add(layers.Dense(1, activation='sigmoid', trainable=True))

    # Using SGD, because my training layers are shallow, adam may still be better alternative.
    # actually I left it with adam, SGD had poor results.


    tf.keras.optimizers.Adam(
        hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]), name="Adam")

    # I'm gathering the confusion table here with the true/false positives/negatives, and the AUC for validation but I can remove that later. This is enough data to calculate ROC graph, which I SHOULD be able to calculate EER with.
    newmod.compile(optimizer="Adam",
                   loss='binary_crossentropy',
                   metrics=['binary_accuracy',
                            tf.keras.metrics.FalsePositives(),
                            tf.keras.metrics.FalseNegatives(),
                            tf.keras.metrics.AUC(),
                            tf.keras.metrics.TruePositives(),
                            tf.keras.metrics.TrueNegatives()]
                   )
    return newmod
