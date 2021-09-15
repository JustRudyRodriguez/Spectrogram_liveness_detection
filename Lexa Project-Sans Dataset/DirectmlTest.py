import matplotlib.pyplot as plt
import numpy as np
# Runninger tensorflow-directml version.
import tensorflow as tf
from tensorflow.keras import layers, models, applications

import DatasetBuilder as db

print(tf.__version__)

# so this dataset in particular has the images imported as 28x28 whiteness value arrays.

# the dataset does not contain "names" for the categories, so we add those here.
directory = "C:/Users/rudy_/PycharmProjects/pythonProject/Classified"

# these pre-processing settings yielded very good results .


print("loading datasets.")
trainingdata, labels = db.databuilder("Core", "CoreMeta.csv", x=224, y=224, size=1000)  # /1.5 scale = x=788, y=398
print("Training set Loaded.")
evaldata, elabels = db.databuilder("Eval", "EvalMeta.csv", x=224, y=224, size=1000)
print("Evaluation set Loaded.")


# I am supposed to use this data preprocess function, but for now it seems to work without.
trainingdata = tf.keras.applications.mobilenet_v2.preprocess_input(trainingdata)
evaldata = tf.keras.applications.mobilenet_v2.preprocess_input(evaldata)


# So I will begin by creating an instance of the mobilenet2 model here, with he example dataset above.
# ok, so according to the reading I did, removing top means it's not using the fully connected layer.
# This stack overflow comment simplifies it. https://stackoverflow.com/a/46745897
# the short of it. I may need to add my own "top" the section that makes the final classifications.

mobile = applications.MobileNetV2(
    input_shape=(224, 224, 3), alpha=1.0, include_top=False, weights="imagenet",
    input_tensor=None, classes=2  # ,pooling='avg' #adding this in a layer manually.
)
# 156 layers in mobilenetv2
print(len(mobile.layers))
# we freeze these layers, before adding our own. [:-n] if we want to include more off the top.
for x in mobile.layers:
    x.trainable = False
# I create a "new" model here.

newmod = models.Sequential()
# Adding mobilenetv2 without top to the new model.
newmod.add(mobile)
# creating a new "top" layer, to classify the 2 possible objects.
newmod.add(layers.GlobalAveragePooling2D())
newmod.add(layers.Dense(128, activation='relu', trainable=True))
# there are 2 possible outputs, so the final layer needs to be 1.
# Because of the use of a BCE loss(), the output layer needs to be a single node with sigmoid activation.
newmod.add(layers.Dense(1, activation='sigmoid', trainable=True))

print(newmod.summary())
# Using SGD, because my training layers are shallow, adam may still be better alternative.
opt = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.1 #, nesterov=True
)

# I'm gathering the confusion table here with the true/false positives/negatives,  and the AUC for validation but I can remove that later. This is enough data to calculate ROC graph, which I SHOULD be able to calculate EER with.
newmod.compile(optimizer="Adam",
               loss='binary_crossentropy',
               metrics=['binary_accuracy',
                        tf.keras.metrics.FalsePositives(),
                        tf.keras.metrics.FalseNegatives(),
                        tf.keras.metrics.AUC(),
                        tf.keras.metrics.TruePositives(),
                        tf.keras.metrics.TrueNegatives()]
               )
print("model compiled.")

# may put dataset in x, labels in y value. I should check if I can add more workers.
# need to
print("Beginning Training.")
history = newmod.fit(trainingdata, labels,
                     epochs=20,
                     use_multiprocessing=True,
                     validation_split=0.1,
                     batch_size=32
                     )
# Some values I think I can plot; loss, binary_accuracy, val_loss, val_binary_accuracy
# This should save the history after training, to a file.
np.save("Training_History.npy", history.history)
# plots data.
plt.plot(history.history['binary_accuracy'], label='accuracy')
plt.plot(history.history['val_binary_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
results = newmod.evaluate(evaldata, elabels, verbose=1)

newmod.save("first_model")

print(results)

from sklearn.metrics import roc_curve
predictions = newmod.predict(evaldata).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(elabels, predictions)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# the threshold of fnr == fpr
fnr = 1 - tpr_keras
eer_threshold = thresholds_keras[np.nanargmin(np.absolute((fnr - fpr_keras)))]

# theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
eer_1 = fpr_keras[np.nanargmin(np.absolute((fnr - fpr_keras)))]
eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr_keras)))]

# return the mean of eer from fpr and from fnr
print("EER: ")
eer = (eer_1 + eer_2) / 2
print(eer)