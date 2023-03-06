import argparse
import json

from explainability import LRPStrategy, LRP

from tensorflow.keras.models import load_model


def convert_to_explainable_pipeline(model: str, strategy: str, layer: int,
                                    index: int, include_prediction: bool,
                                    destination: str):
    model = load_model(model)

    with open(strategy, 'r') as f:
        strategy = json.load(f)

    strategy = LRPStrategy(strategy['layers'])
    #explainer = LRP(model, layer=layer, idx=index, strategy=strategy)

    import nibabel as nib
    import numpy as np
    img = nib.load('/Users/esten/Downloads/data/image.nii.gz').get_fdata()
    img = img / 255.
    img = np.expand_dims(img, 0)
    print(img.shape)
    #print(explainer.input.shape)
    outputs = model.predict(img)
    print(outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Creates an explainable pipeline '
                                      'producing relevance maps from a '
                                      'tensorflow model'))
    parser.add_argument('-m', '--model', required=True,
                        help='Path to tensorflow model')
    parser.add_argument('-s', '--strategy', required=True,
                        help='Path to a JSON file containing the LRP strategy')
    parser.add_argument('-l', '--layer', required=False, default=-1, type=int,
                        help='The layer to extract explanations for')
    parser.add_argument('-i', '--index', required=False, default=0, type=int,
                        help='Index of the node to extract explanations for')
    parser.add_argument('-p', '--include_prediction', action='store_true',
                        help=('If set, pipeline also includes prediction as '
                              'output'))
    parser.add_argument('-d', '--destination', required=True,
                        help='Path where pipeline is written')

    args = parser.parse_args()

    convert_to_explainable_pipeline(model=args.model,
                                    strategy=args.strategy,
                                    layer=args.layer,
                                    index=args.index,
                                    include_prediction=args.include_prediction,
                                    destination=args.destination)
