from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model once when the application starts
model = tf.keras.models.load_model('best_acne_model_V2.keras')
print("CNN model loaded and ready for inference")


def preprocess_image(image_file):
    # Open the uploaded image file
    image = Image.open(image_file)

    # Convert the image to RGB to ensure 3 channels
    image = image.convert('RGB')

    # Resize the image to match model input size of (256, 256)
    image = image.resize((256, 256))

    # Convert image to numpy array
    image_array = np.array(image)

    # Add batch dimension: (256,256,3) -> (1,256,256,3)
    image_array = np.expand_dims(image_array, axis=0)

    print("Processed image shape:", image_array.shape)
    return image_array


@app.route('/')
def hello_world():
    return 'ML Service is running'


@app.route('/predict', methods=['POST'])
def predict():
    # Check if file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['file']

    # Check if user selected a file
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Preprocess the image
        processed_image = preprocess_image(file)

        # Predict
        predictions = model.predict(processed_image, verbose=0)

        # Extract sigmoid score as normal python float
        score = float(predictions[0][0])

        # Apply threshold to determine class label
        threshold = 0.45
        label = 'Clear' if score >= threshold else 'Acne'

        # Return the json response
        return jsonify({
            "label": label,
            "score": score,
            "threshold": threshold
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)