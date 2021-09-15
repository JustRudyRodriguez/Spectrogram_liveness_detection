import tensorflow as tf
import os



print(os.getcwd())
model = tf.keras.models.load_model("indoor_Tuned_model3")

converter = tf.lite.TFLiteConverter.from_saved_model("indoor_Tuned_model3")
tflite_model = converter.convert()

open("tfliteIndoorModel3.tflite","wb").write(tflite_model)

#model.save("indoor_tuned_model.h5")
print(model.summary())