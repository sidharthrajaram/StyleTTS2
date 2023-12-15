from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

import importlib.resources
from pathlib import Path
import librosa
import scipy
import torch
import torchaudio
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import yaml

from . import models
from . import utils
from .phoneme import PhonemeConverterFactory
from .text_utils import TextCleaner
from .Utils.PLBERT.util import load_plbert
from .Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


class StyleTTS2:
    def __init__(self, model_checkpoint_path, config_path, phoneme_converter='gruut'):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.phoneme_converter = PhonemeConverterFactory.load_phoneme_converter(phoneme_converter)
        self.config = None
        self.model_params = None
        self.model = self.load_model(model_checkpoint_path, config_path)

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )


    def load_model(self, model_path, config_path):

        if not Path(model_path).exists():
            raise ValueError("Invalid model checkpoint path. Provide a valid path to a StyleTTS2 LibriTTS checkpoint. If needed, download from https://huggingface.co/yl4579/StyleTTS2-LibriTTS.")

        self.config = yaml.safe_load(open(config_path))

        # load pretrained ASR model
        ASR_config = self.config.get('ASR_config', 'Utils/ASR/config.yml')
        if not Path(ASR_config).exists():
            ASR_config = importlib.resources.files('styletts2') / 'Utils/ASR/config.yml'
        ASR_path = self.config.get('ASR_path', 'Utils/ASR/epoch_00080.pth')
        if not Path(ASR_path).exists():
            ASR_path = importlib.resources.files('styletts2') / 'Utils/ASR/epoch_00080.pth'
        text_aligner = models.load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = self.config.get('F0_path', 'Utils/JDC/bst.t7')
        if not Path(F0_path).exists():
            F0_path = importlib.resources.files('styletts2') / 'Utils/JDC/bst.t7'
        pitch_extractor = models.load_F0_models(F0_path)

        # load BERT model
        BERT_path = self.config.get('PLBERT_dir', 'Utils/PLBERT/')
        if not Path(BERT_path).exists():
            BERT_path = importlib.resources.files('styletts2') / 'Utils/PLBERT/'
        plbert = load_plbert(BERT_path)

        self.model_params = utils.recursive_munch(self.config['model_params'])
        model = models.build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        _ = [model[key].eval() for key in model]
        _ = [model[key].to(self.device) for key in model]

        params_whole = torch.load(model_path, map_location='cpu')
        params = params_whole['net']

        for key in model:
            if key in params:
                print('%s loaded' % key)
                try:
                    model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [model[key].eval() for key in model]

        return model


    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)


    def inference(self,
                  text: str,
                  target_voice_path='ReferenceAudio/f-us-demo.wav',
                  output_wav_file=None,
                  output_sample_rate=24000,
                  phonemes=False,
                  alpha=0.3,
                  beta=0.7,
                  diffusion_steps=5,
                  embedding_scale=1):

        # default to clone 696_92939_000016_000006 voice
        if (not target_voice_path) or (not Path(target_voice_path).exists()):
            target_voice_path = importlib.resources.files('styletts2') / 'ReferenceAudio/696_92939_000016_000006.wav'

        text = text.strip()
        text = text.replace('"', '')
        phonemized_text = self.phoneme_converter.phonemize(text)
        ps = word_tokenize(phonemized_text)
        phoneme_string = ' '.join(ps)

        textcleaner = TextCleaner()
        tokens = textcleaner(phoneme_string)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        ref_s = self.compute_style(target_voice_path)  # target style vector

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                  embedding=bert_dur,
                                  embedding_scale=embedding_scale,
                                  features=ref_s, # reference from the same speaker as the embedding
                                  num_steps=diffusion_steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            # duration prediction
            d = self.model.predictor.text_encoder(d_en,
                                                  s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,
                                     F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        output = out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later
        if output_wav_file:
            scipy.io.wavfile.write(output_wav_file, rate=24000, data=output)
        return output

