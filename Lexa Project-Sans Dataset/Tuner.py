import matplotlib.pyplot as plt
import numpy as np
# Runninger tensorflow-directml version.
import tensorflow as tf
import DatasetBuilder as db
from SpectModel import build_model
import kerastuner

print(tf.__version__)


directory = "C:/Users/rudy_/PycharmProjects/pythonProject/Classified"


print("loading datasets.")
trainingdata, labels = db.databuilder("Core", "CoreMeta.csv", x=224, y=224, environment='2')  # /1.5 scale = x=788, y=398

env2,labl2 = db.databuilder("Core", "CoreMeta.csv", x=224, y=224, environment='3')
# to train all indoor examples we load in set 2 & 3, so we load both environments individually

trainingdata = np.concatenate((trainingdata,env2))
del env2
labels = np.concatenate((labels,labl2))
del labl2
print("Training set Loaded.")




# I am supposed to use this data preprocess function, but for now it seems to work without.
trainingdata = tf.keras.applications.mobilenet_v2.preprocess_input(trainingdata)
# labels = tf.keras.applications.mobilenet_v2.preprocess_input(labels)
# elabels = tf.keras.applications.mobilenet_v2.preprocess_input(elabels)

# not working currently, need to finish import.
tuner = kerastuner.RandomSearch(
    build_model,
    objective='binary_accuracy',
    max_trials=20,
    executions_per_trial=3,
    directory='tuner_results',
    project_name='indoor_testing_224'#This needs to be changed to re-test

)



print("model compiled.")

# may put dataset in x, labels in y value. I should check if I can add more workers.
# need to
tuner.search(trainingdata, labels,
             epochs =5,
             batch_size = 32,
             validation_split = .1
              )
print(tuner.results_summary())
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("best HyperParameters:")
print(best_hps)
# training model to find best epochs
# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)

history = model.fit(trainingdata, labels, epochs=25, validation_split=0.2)

np.save('modelTraining_History.npy',history.history)
#checking for best epoch
val_acc_per_epoch = history.history['val_binary_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# recreate model based on best parameters.
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model with best epochs.
hypermodel.fit(trainingdata, labels, epochs=best_epoch, validation_split=0.1)

evaldata, elabels = db.databuilder("Eval", "EvalMeta.csv", x=224, y=224, environment='2')

env2,labl2 = db.databuilder("Eval", "EvalMeta.csv", x=224, y=224, environment='3')
# to train all indoor examples we load in set 2 & 3, so we load both environments individually and concatenate

evaldata = np.concatenate((evaldata,env2))
del env2
elabels = np.concatenate((elabels,labl2))
del labl2
print("Evaluation set Loaded.")
evaldata = tf.keras.applications.mobilenet_v2.preprocess_input(evaldata)

eval_result = hypermodel.evaluate(evaldata, elabels)

hypermodel.save('Indoor_Tuned_model')

print("[test loss, test accuracy]:", eval_result)

from sklearn.metrics import roc_curve
predictions = hypermodel.predict(evaldata).ravel()
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
plt.savefig("saved Figure")