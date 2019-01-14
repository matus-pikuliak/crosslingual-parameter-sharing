import ast
import glob
import os
import sys

import h5py
import numpy as np

log_folder = sys.argv[1]
h5_folder = sys.argv[2]
output_folder = sys.argv[3]

def additional_stats(name, datum):
    h5_filepath = os.path.join(
        h5_folder,
        f'{name}-{datum["epoch"]}-{datum["task"]}-{datum["language"]}-{datum["role"]}.h5')
    output = dict()
    if os.path.isfile(h5_filepath):
        with h5py.File(h5_filepath, 'r') as h5:
            print(h5_filepath)
            output['gradient_norm'] = h5['gradient_norm'].value

            output['cont_repr_weights_norm'] = np.linalg.norm(h5['cont_repr_weights'])
            output['cont_repr_weights_input_norm'] = np.linalg.norm(h5['cont_repr_weights'], axis=0)
            output['cont_repr_weights_input_norm_avg'] = np.mean(output['cont_repr_weights_input_norm'])
            output['cont_repr_weights_input_norm_std'] = np.std(output['cont_repr_weights_input_norm'])
            output['cont_repr_weights_output_norm'] = np.linalg.norm(h5['cont_repr_weights'], axis=1)
            output['cont_repr_weights_output_norm_avg'] = np.mean(output['cont_repr_weights_output_norm'])
            output['cont_repr_weights_output_norm_std'] = np.std(output['cont_repr_weights_output_norm'])

            output['cont_repr_weights_grad_norm'] = np.linalg.norm(h5['cont_repr_weights_grad'])
            output['cont_repr_weights_grad_input_norm'] = np.linalg.norm(h5['cont_repr_weights_grad'], axis=0)
            output['cont_repr_weights_grad_input_norm_avg'] = np.mean(output['cont_repr_weights_grad_input_norm'])
            output['cont_repr_weights_grad_input_norm_std'] = np.std(output['cont_repr_weights_grad_input_norm'])
            output['cont_repr_weights_grad_output_norm'] = np.linalg.norm(h5['cont_repr_weights_grad'], axis=1)
            output['cont_repr_weights_grad_output_norm_avg'] = np.mean(output['cont_repr_weights_grad_output_norm'])
            output['cont_repr_weights_grad_output_norm_std'] = np.std(output['cont_repr_weights_grad_output_norm'])

            output['cont_repr_norm_avg'] = np.mean(np.linalg.norm(h5['cont_repr'], axis=1))
            output['cont_repr_norm_std'] = np.std(np.linalg.norm(h5['cont_repr'], axis=1))

    return output


log_files = glob.glob(log_folder)

for log_file in log_files:
    data = {}
    data['results'] = []
    with open(log_file) as f:
        for line in f:
            if 'optimizer' in line:
                data['hparams'] = ast.literal_eval(line[2:])
            if not line.startswith('#'):
                datum = ast.literal_eval(line)
                datum.update(additional_stats(os.path.split(log_file)[-1], datum))
                data['results'].append(datum)
        with open(os.path.join(output_folder, os.path.split(log_file)[-1]), 'w') as f:
            f.write(data)
