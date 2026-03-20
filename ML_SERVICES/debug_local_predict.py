from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("best_acne_model_updated.keras")

img_path = "/Users/unmeshachar/Desktop/CUSTOM_CNN/Acne_Image.jpeg"

img = load_img(img_path, target_size=(256, 256))
img_array = img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

score = model.predict(img_batch, verbose=0)[0][0]

print("Local debug score:", score)
print("Predicted label:", "Clear" if score >= 0.20 else "Acne")