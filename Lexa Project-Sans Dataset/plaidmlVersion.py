import matplotlib.pyplot as plt
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#import tensorflow as tf
# delete this after testing

import numpy as np
from keras import layers, models, applications, losses
import DatasetBuilder as db
# from tensorflow import losses
PLAIDML_VERBOSE = 1
# print(tf.__version__)


# so this dataset in particular has the images imported as 28x28 whiteness value arrays.

# the dataset does not contain "names" for the categories, so we add those here.
directory = "C:/Users/rudy_/PycharmProjects/pythonProject/Classified"

print("loss function test")

print("loss function created")


print("loading datasets. ")
trainingdata, labels =  db.databuilder("Core", "CoreMeta.csv", size = 10000)
print("Training set Loaded.")
evaldata, elabels =  db.databuilder("Core", "CoreMeta.csv", size = 10000)
print("Evaluation set Loaded.")
# this should display some of the data as test to make sure it's looking good.
'''
plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
'''
# I am supposed to use this data preprocess function, but for now it seems to work without.
# x = tf.keras.applications.mobilenet_v2.preprocess_input(x)


# So I will begin by creating an instance of the mobilenet2 model here, with he example dataset above.
# ok, so according to the reading I did, removing top means it's not using the fully connected layer.
# This stack overflow comment simplifies it. https://stackoverflow.com/a/46745897
# the short of it. I may need to add my own "top" the section that makes the final classifications.
model = applications.MobileNetV2(
    input_shape=(250, 150, 3), alpha=1.0, include_top=False, weights='imagenet',
    input_tensor=None, pooling=None, classes=2
    # ,classifier_activation='softmax'
)
# I create a "new" model here.
newmod = models.Sequential()
# Adding mobilenetv2 without top to the new model.
newmod.add(model)
# creating a new "top" layer, to classify the 2 possible objects.
newmod.add(layers.Flatten())
newmod.add(layers.Dense(64, activation='relu'))
# there are 2 possible outputs, so the final layer needs to be 2. the highest valued node will be the "winner".
# this layer may actually only need 1 node, depending on the activation being used.
newmod.add(layers.Dense(1, activation='relu'))

print(model.summary())


newmod.compile(optimizer='adam',
              loss = lambda y_true, y_pred: losses.binary_crossentropy(
    y_true, y_pred)
,
              metrics =  ['binary_accuracy'])
print("model compiled.")
# may put dataset in x, labels in y value. I should check if I can add more workers.
# need to
print("Beginning Training.")
history = newmod.fit(trainingdata, labels,  epochs=10, validation_split=0.1
                     )
# Some values I think I can plot; loss, binary_accuracy, val_loss, val_binary_accuracy
# This should save the history after training, to a file.
np.save("Training_History.npy", history.history)
# plots data.
plt.plot(history.history['binary_accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = newmod.evaluate(evaldata, verbose=2)

print(test_acc)
