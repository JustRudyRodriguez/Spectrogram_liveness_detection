import numpy as np
import DatasetBuilder as db
import matplotlib.pyplot as plt
import tensorflow as tf

hypermodel = tf.keras.models.load_model('C:/Users/rudy_/PycharmProjects/pythonProject/Indoor_Tuned_model3')


evaldata, elabels = db.databuilder("Eval", "EvalMeta.csv", x=224, y=224, environment='2')

env2,labl2 = db.databuilder("Eval", "EvalMeta.csv", x=224, y=224, environment='3')
# to train all indoor examples we load in set 2 & 3, so we load both environments individually and concatenate

evaldata = np.concatenate((evaldata,env2))
del env2
elabels = np.concatenate((elabels,labl2))
del labl2
print("Evaluation set Loaded.")
evaldata = tf.keras.applications.mobilenet_v2.preprocess_input(evaldata)

#eval_result = hypermodel.evaluate(evaldata, elabels)


#print("[test loss, test accuracy]:", eval_result)




from sklearn.metrics import roc_curve
predictions = hypermodel.predict(evaldata).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(elabels, predictions)
plabels = np.asarray(elabels)

#np.asarray(eval_result).save("Stats/stats3.npy")

np.save("predictionsLabels3", plabels)
np.save("predictions3", predictions)

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
