"""
An interface for loading each model.

All models are for a sample_rate of 44.1kHz
Luckily for us, open_unmix has a STFT built in.
"""
import os
from demucs.utils import load_model as load_demucs_or_tasnet
from open_unmix import test as unmix_test
from wave_u_net.waveunet import Waveunet
from wave_u_net.utils import load_model as load_waveunet, DataParallel

checkpoints = "checkpoints"

def demucs():
    return load_demucs_or_tasnet(os.path.join(checkpoints, "demucs_extra.th"))

def tasnet():
    return load_demucs_or_tasnet(os.path.join(checkpoints, "tasnet_extra.th"))

def open_unmix():
    targets = ['bass', 'drums', 'other', 'vocals']
    models = {}
    for target in targets:
        models[target] = unmix_test.load_model(
            target, os.path.join(checkpoints, "open-unmix"), device='cuda'
        )
    return models

def wav_u_net():
    instruments = ["bass", "drums", "other", "vocals"]
    features = 32
    load_model = 'checkpoints/waveunet/model'
    levels = 6
    depth = 1
    sr = 44100
    channels = 2
    kernel_size = 5
    output_size = 2.0
    strides = 4
    conv_type = 'gn'
    res = 'fixed'
    feature_growth = 'double'
    num_samples = int(output_size * sr)
    separate = 1
    num_features = [features*2**i for i in range(0, levels)]
    model = Waveunet(channels, num_features, channels,
                     instruments, kernel_size=kernel_size,
                     target_output_size=num_samples, depth=depth,
                     strides=strides, conv_type=conv_type,
                     res=res, separate=separate)

    model = DataParallel(model)
    model.cuda()
    load_waveunet(model, None, load_model, True)
    return model

if __name__ == "__main__":
    # This is basically a weak test
    demucs()
    tasnet()
    open_unmix()
    wav_u_net()
    print("loaded all models sucessfully :D")