# StyleTTS 2: The Python Package

This package makes StyleTTS2, an approach to human-level text-to-speech, accessible with an inference module that uses strictly MIT licensed libraries. See ***Conditions and Terms of Use***, ***Common Issues***, and ***Notes*** below.

## Quick Start
1. Ensure you are running Python >= 3.9 (currently supports 3.9, 3.10 due to some other library dependencies)
2. [Optional] Downloaded the StyleTTS2 LibriTTS checkpoint and corresponding config file. Both are available to download at https://huggingface.co/yl4579/StyleTTS2-LibriTTS. You can also provide paths to your own checkpoint and config file (just ensure it is the same format as the [original one](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/blob/main/Models/LibriTTS/config.yml)).
3. Install the package using pip:
```bash
pip install styletts2
```
4. Try it out either in Python shell or in your code: 
```python
from styletts2 import tts

# No paths provided means default checkpoints/configs will be downloaded/cached.
my_tts = tts.StyleTTS2()

# Optionally create/write an output WAV file.
out = my_tts.inference("Hello there, I am now a python package.", output_wav_file="test.wav")

# Specific paths to a checkpoint and config can also be provided.
other_tts = tts.StyleTTS2(model_checkpoint_path='/PATH/TO/epochs_2nd_00020.pth', config_path='/PATH/TO/config.yml')

# Specify target voice to clone. When no target voice is provided, a default voice will be used.
other_tts.inference("Hello there, I am now a python package.", target_voice_path="/PATH/TO/some_voice.wav", output_wav_file="another_test.wav")
```

## Inference function reference

```
def inference(self,
              text: str,
              target_voice_path=None,
              output_wav_file=None,
              output_sample_rate=24000,
              alpha=0.3,
              beta=0.7,
              diffusion_steps=5,
              embedding_scale=1,
              ref_s=None)
```
**text**: Input text to turn into speech.

**target_voice_path**: Path to audio file of target voice to clone.

**output_wav_file**: Name of output audio file (if output WAV file is desired).

**output_sample_rate**: Output sample rate (default 24000).

**alpha**: Determines timbre of speech, higher means style is more suitable to text than to the target voice.

**beta**: Determines prosody of speech, higher means style is more suitable to text than to the target voice.

**diffusion_steps**: The more the steps, the more diverse the samples are, with the cost of speed.

**embedding_scale**: Higher scale means style is more conditional to the input text and hence more emotional.

**ref_s**: Pre-computed style vector to pass directly.

**return**: audio data as a Numpy array (will also create the WAV file if output_wav_file was set).


### *Note: I'm not affiliated with the original authors. StyleTTS2 is a neat, open source, state-of-the-art approach to TTS. Pass your kudos to the authors at the model repo:*

# [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models](https://github.com/yl4579/StyleTTS2)

### Original authors: Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani

> In this paper, we present StyleTTS 2, a text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis. StyleTTS 2 differs from its predecessor by modeling styles as a latent random variable through diffusion models to generate the most suitable style for the text without requiring reference speech, achieving efficient latent diffusion while benefiting from the diverse speech synthesis offered by diffusion models. Furthermore, we employ large pre-trained SLMs, such as WavLM, as discriminators with our novel differentiable duration modeling for end-to-end training, resulting in improved speech naturalness. StyleTTS 2 surpasses human recordings on the single-speaker LJSpeech dataset and matches it on the multispeaker VCTK dataset as judged by native English speakers. Moreover, when trained on the LibriTTS dataset, our model outperforms previous publicly available models for zero-shot speaker adaptation. This work achieves the first human-level TTS synthesis on both single and multispeaker datasets, showcasing the potential of style diffusion and adversarial training with large SLMs.

Paper: [https://arxiv.org/abs/2306.07691](https://arxiv.org/abs/2306.07691)

Audio samples: [https://styletts2.github.io/](https://styletts2.github.io/)

Online demo: [Hugging Face](https://huggingface.co/spaces/styletts2/styletts2) (thank [@fakerybakery](https://github.com/fakerybakery) for the wonderful online demo)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/) [![Slack](https://img.shields.io/badge/Join%20Our%20Community-Slack-blue)](https://join.slack.com/t/styletts2/shared_invite/zt-2805io6cg-0ROMhjfW9Gd_ix_FJqjGmQ)

## Conditions and Terms of Use
***Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.*** 

## Common Issues
- **[MacOS] ImportError due to incompatible architecture for pycrfsuite**: This is caused by a dependency on python-crfsuite by the [gruut](https://github.com/rhasspy/gruut) phoneme converter. If you are operating on a conda environment, try the following:
    ```bash
    conda install -c conda-forge python-crfsuite
    ```
    Another option is adding another phoneme converter with the abstraction detailed in `phoneme.py` and using that instead.

- **Voice quality**: This is more of a catch-all issue for voice quality related issues. In most cases, strange annunciations are the result of the phoneme converter. The hope is that the field of MIT licensed phoneme converters (i.e Gruut, DeepPhonemizer, etc.) will eventually become incredibly competitive with the legacy converters such as `espeak`. However, in the meantime here are some potential avenues for quality improvement:

    - **More phonetically diverse target voice samples for cloning**: The WAV file passed as the target/reference voice should preferably have a good range of pronunciations and be of good audio quality. In experimenting with cloning, I've noticed that the speech output quality does improve alongside the quality of the target/reference voice sample.

- **High-pitched background noise**: This is caused by numerical float differences in older GPUs. For more details, please refer to issue [#13](https://github.com/yl4579/StyleTTS2/issues/13). Basically, you will need to use more modern GPUs or do inference on CPUs.

- **Pre-trained model license**: You only need to abide by the above rules if you use **the pre-trained models** and the voices are **NOT** in the training set, i.e., your reference speakers are not from any open access dataset. For more details of rules to use the pre-trained models, please see [#37](https://github.com/yl4579/StyleTTS2/issues/37).

## TODO
- [x] Inference support for LibriTTS (voice cloning) model
- [x] Option to provide style vector directly
- [ ] Inference support for LJSpeech model
- [ ] Support for DeepPhonemizer in `phoneme.py`
- [ ] Automatic updates to StyleTTS2 model source code via GitHub Actions
- [x] Documentation for inference and load methods
- [ ] Caching style vector of a reference/target voice (even faster inference)

## Notes
- If specific checkpoint paths are not provided, default checkpoints and sub-module checkpoints are downloaded from the [HuggingFace repo](https://huggingface.co/yl4579/StyleTTS2-LibriTTS) and the [original GitHub repo](https://github.com/yl4579/StyleTTS2/tree/main/Utils), respectively, and then cached (similar behavior to HuggingFace Transformers API).

- This package currently only supports inference capabilities. Dependencies and scripts related to training and fine-tuning have been pruned out. Check the [original repository](https://github.com/yl4579/StyleTTS2) for training/fine-tuning needs.

- Currently using MIT-licensed [gruut](https://github.com/rhasspy/gruut) as the IPA phoneme converter. Found it to be the best alternative to phoneme converters based on [espeak](https://github.com/espeak-ng/espeak-ng)