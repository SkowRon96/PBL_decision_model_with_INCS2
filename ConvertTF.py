from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf

# model_file = "Keras_Model/model.json"
# weights_file = "Keras_Model/weights.h5"
model_file = "Keras_Model/model_real.json"
weights_file = "Keras_Model/weights_real.h5"

with open(model_file, "r") as file:
    config = file.read()

K.set_learning_phase(0)
model = model_from_json(config)
model.load_weights(weights_file)

saver = tf.train.Saver()
sess = K.get_session()
saver.save(sess, "./TF_Model/tf_model_real")

fw = tf.summary.FileWriter('logs', sess.graph)
fw.close()