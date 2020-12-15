"""
An interface for loading each model.

All models are for a sample_rate of 44.1kHz
Luckily for us, open_unmix has a STFT built in.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import ModuleDict
from demucs.utils import load_model as load_demucs_or_tasnet, apply_model
from demucs.model import Demucs
from demucs.tasnet import ConvTasNet
from open_unmix import test as unmix_test
from wave_u_net.waveunet import Waveunet
from wave_u_net.utils import load_model as load_waveunet, DataParallel
from nussl import AudioSignal, STFTParams
from norbert import wiener

torch.no_grad()

checkpoints = "checkpoints"

class ModelWrapper(nn.Module):
    """
    Parent Wrapper Class for turning model output into a
    source dictionary.

    Parent Class used for type checking.
    """

class FB_Wrapper(ModelWrapper):
    """
    Wraps models that were trained for the Demucs paper.
    Returns a dictionary of audio signals
    model (Demucs or Tasnet): Either a Demucs or Conv-TasNet model from FAIR.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        if isinstance(model, Demucs):
            self.name = "Demucs"
        elif isinstance(model, ConvTasNet):
            self.name = "ConvTasNet"
        else:
            ValueError("A non-facebook model was passed to this wrapper!")

    def forward(self, mix):
        # Whitening was found in the evaluation script...
        ref = mix.mean(dim=0)
        mix = (mix - ref.mean()) / ref.std()
        x = apply_model(self.model, mix)
        x = x * ref.std() + ref.mean()
        x = self.demucs_to_audiosignal(x)
        return x

    def demucs_to_audiosignal(self, out):
        out = out.detach().cpu().numpy()
        est = {
            'drums': AudioSignal(audio_data_array=out[0], sample_rate=44_100),
            'bass':  AudioSignal(audio_data_array=out[1], sample_rate=44_100),
            'other': AudioSignal(audio_data_array=out[2], sample_rate=44_100),
            'vocals':AudioSignal(audio_data_array=out[3], sample_rate=44_100)
        }
        return est

class OpenUnmixWrapper(ModelWrapper):
    """
    Wraps OpenUnmix.
    Returns a dictionary of audio signals
    model_path (str): Path to OpenUnmix model state dictionary
    """
    def __init__(self, model_path):
        super().__init__()
        self.targets = ['bass', 'drums', 'other', 'vocals']
        self.model_dict = ModuleDict()
        self.stft_params = None
        self.name = "OpenUnmix"
        for target in self.targets:
            self.model_dict[target] = unmix_test.load_model(
                target, model_path, device='cuda'
            )
            if self.stft_params is None:
                self.stft_params = STFTParams(
                    window_length=self.model_dict[target].stft.n_fft,
                    hop_length=self.model_dict[target].stft.n_hop,
                    window_type='hann'
                )

    def forward(self, mix):
        mix = mix.unsqueeze(0)
        mask_dict = OrderedDict()
        for key, model in self.model_dict.items():
            est = model(mix)
            est = est[:, 0, ...].permute(0, 2, 1)
            est = est.detach().cpu().numpy()
            mask_dict[key] = est

        mix = model.stft(mix).cpu().numpy()
        mix = mix.squeeze(0)
        mix = mix[..., 0] + mix[..., 1]*1j # OpenUnmix was made during torch<=1.5.0
        mix = mix.transpose(2, 1, 0)

        masks = np.stack(list(mask_dict.values()), axis=-1)
        est_stft = torch.tensor(wiener(
            masks, mix.astype(np.complex128), eps=1e-7, use_softmask=False
        )).cuda()
        est_signal = model.stft.inverse(est_stft.permute(3, 2, 1, 0))
        estimated = {}
        for idx, source in enumerate(mask_dict.keys()):
            audio_data = est_signal[idx, ...].cpu().numpy()
            estimated[source] = AudioSignal(
                audio_data_array=audio_data, sample_rate=44_100)
        return estimated

class WaveUNetWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.name = "Wave-U-Net"

    def forward(self, mix):
        """
        Taken from Wave-U-Net-Pytorch/test.py

        # NOTE: WaveUNet does not use Whitening
        """
        def compute_model_output(inputs):
            all_outputs = {}

            if self.model.separate:
                for inst in self.model.instruments:
                    output = self.model(inputs, inst)
                    all_outputs[inst] = output[inst].detach().clone()
            else:
                all_outputs = self.model(inputs)

            return all_outputs
        expected_outputs = mix.shape[1]

        # Pad input if it is not divisible in length by the frame shift number
        output_shift = self.model.shapes["output_frames"]
        pad_back = mix.shape[1] % output_shift
        pad_back = 0 if pad_back == 0 else output_shift - pad_back

        target_outputs = mix.shape[-1] + pad_back
        outputs = {key: np.zeros((2, mix.shape[-1] + pad_back), np.float32) for key in self.model.instruments}

        # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
        pad_front_context = self.model.shapes["output_start_frame"]
        pad_back_context = self.model.shapes["input_frames"] - self.model.shapes["output_end_frame"]
        # mix = np.pad(mix, [(0,0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)
        padder = torch.nn.ConstantPad1d((pad_front_context, pad_back_context + pad_back), 0.0)
        padder.cuda()
        mix = padder(mix)

        # Iterate over mixture magnitudes, fetch network prediction
        for target_start_pos in range(0, target_outputs, self.model.shapes["output_frames"]):
            # Prepare mixture excerpt by selecting time interval
            curr_input = mix[:, target_start_pos:target_start_pos + self.model.shapes["input_frames"]] # Since mix was front-padded input of [targetpos:targetpos+inputframes] actually predicts [targetpos:targetpos+outputframes] target range

            # Convert to Pytorch tensor for model prediction
            curr_input = curr_input.unsqueeze(0)

            # Predict
            for key, curr_targets in compute_model_output(curr_input).items():
                outputs[key][:,target_start_pos:target_start_pos+self.model.shapes["output_frames"]] = curr_targets.squeeze(0).cpu().numpy()

        # Crop to expected length (since we padded to handle the frame shift)
        output_audio_data = {key : outputs[key][:,:expected_outputs] for key in outputs.keys()}
        output_audio_signal = {k: AudioSignal(audio_data_array=d, sample_rate=44_100)
                               for k, d in output_audio_data.items()}

        return output_audio_signal

"""
############################################################
# These last four are to be called from outside the class. #
############################################################
"""
def demucs():
    model = load_demucs_or_tasnet(os.path.join(checkpoints, "demucs_extra.th"))
    return FB_Wrapper(model)

def tasnet():
    model = load_demucs_or_tasnet(os.path.join(checkpoints, "tasnet_extra.th"))
    return FB_Wrapper(model)

def open_unmix():
    return OpenUnmixWrapper(os.path.join(checkpoints, "open-unmix"))

def wav_u_net(num_samples):
    instruments = ["bass", "drums", "other", "vocals"]
    features = 32
    load_model = 'checkpoints/waveunet/model'
    levels = 6
    depth = 1
    sr = 44100
    channels = 2
    kernel_size = 5
    strides = 4
    conv_type = 'gn'
    res = 'fixed'
    feature_growth = 'double'
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
    return WaveUNetWrapper(model)

if __name__ == "__main__":
    # This is basically a weak test
    demucs()
    tasnet()
    o = open_unmix()
    o(torch.zeros(1, 2, 20000).cuda())
    wav_u_net(44_100)
    print("loaded all models sucessfully :D")