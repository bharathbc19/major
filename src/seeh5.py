import tensorflow as tf

model = tf.keras.models.load_model(r"D:\fraud-detection-federated\docs\global_model.h5")

model.summary()