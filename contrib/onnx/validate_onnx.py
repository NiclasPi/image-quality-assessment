import argparse

import numpy as np
import onnxruntime as ort
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src.handlers.model_builder import Nima
from src.utils.utils import calc_mean_score


def load_image(image_path, preprocessing_function):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = preprocessing_function(x)
    return np.expand_dims(x, axis=0).astype(np.float32)


def main(base_model_name, weights_file, onnx_file, image_path):
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    x = load_image(image_path, nima.preprocessing_function())

    keras_pred = nima.nima_model.predict(x)[0]

    session = ort.InferenceSession(onnx_file)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_pred = session.run([output_name], {input_name: x})[0][0]

    max_abs_diff = np.max(np.abs(keras_pred - onnx_pred))

    print('Keras prediction:      ', np.round(keras_pred, 5))
    print('ONNX prediction:       ', np.round(onnx_pred, 5))
    print('Max abs diff:           ', max_abs_diff)
    print('Keras mean score:      ', calc_mean_score(keras_pred))
    print('ONNX mean score:       ', calc_mean_score(onnx_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    parser.add_argument('-o', '--onnx-file', help='path of onnx model', required=True)
    parser.add_argument('-i', '--image-path', help='path of test image', required=True)

    args = parser.parse_args()

    main(args.base_model_name, args.weights_file, args.onnx_file, args.image_path)
