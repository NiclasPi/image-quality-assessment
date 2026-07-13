import argparse

import tensorflow.keras.backend as K
import tf2onnx

from src.handlers.model_builder import Nima


def main(base_model_name, weights_file, output_path, opset):
    K.set_learning_phase(0)

    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    onnx_model, _ = tf2onnx.convert.from_keras(nima.nima_model, opset=opset)

    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    print(f'ONNX model exported to: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    parser.add_argument('-o', '--output-path', help='path to save the .onnx model', required=True)
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')

    args = parser.parse_args()

    main(args.base_model_name, args.weights_file, args.output_path, args.opset)
