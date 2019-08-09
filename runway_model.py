import argparse, importlib

import torch
import runway

from vaesampler import BPEmbVaeSampler

model_path = "./models/poetry_500k_sample/2019-08-09T08:27:43.289493-011.pt"

@runway.setup
def setup():
    use_gpu = True if torch.cuda.is_available() else False
    config_file = "config.config_poetry_500k_sample"
    params = argparse.Namespace(**importlib.import_module(config_file).params)
    model = BPEmbVaeSampler(lang='en', vs=10000, dim=100,
            decode_from=model_path, params=params, cuda=use_gpu)
    return model

@runway.command('generate',
        inputs={
            'z': runway.vector(length=32),
            'temperature': runway.number(default=0.5, min=0.05, max=2.0,
                step=0.05)
        },
        outputs={'out': runway.text})
def generate(model, inputs):
    z = torch.from_numpy(inputs['z']).float().unsqueeze(0)
    temperature = inputs['temperature']
    with torch.no_grad():
        return model.sample(z, temperature)[0]

@runway.command('reconstruct',
        inputs={
            'in': runway.text,
            'temperature': runway.number(default=0.5, min=0.05, max=2.0,
                step=0.05)
        },
        outputs={'out': runway.text})
def reconstruct(model, inputs):
    with torch.no_grad():
        return model.sample(
                model.z([inputs['in']]), inputs['temperature'])[0]

if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000)

