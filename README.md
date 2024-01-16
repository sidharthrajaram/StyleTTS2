# StyleTTS 2: The Python Package

This package makes StyleTTS2, an approach to human-level text-to-speech, accessible with an inference module. It is based on the [original StyleTTS2 implementation](https://github.com/yl4579/StyleTTS2) and the [orginal StyleTTS2 Package](https://github.com/sidharthrajaram/StyleTTS2) with the following changes:

* This package allows the espeak backend and the gruut backend to be used interchangeably. 
* It also allows the user to specify a local directory to store downloaded files to use for air-gapped systems.

## Quick Start
1. Ensure you are running Python >= 3.9 (currently supports 3.9, 3.10 due to some other library dependencies)

2. Install the package using pip:
```bash
pip install styletts2
```
3. Try it out either in Python shell or in your code: 
```python
from styletts2 import tts

model = tts.StyleTTS2(
    phoneme_converter="espeak",  # "espeak", "gruut
    local="models/",  # where cached_path will store/load downloaded files
)


out = model.inference(
    "Hello there, I am now a python package. Hello.",
    target_voice_path="voices/m-us-2.wav",
    output_wav_file="test.wav",
    alpha=0.3,
    beta=0.7,
    diffusion_steps=5,
    embedding_scale=1,
    speed=1.0,
)
```

## Function References

```
class StyleTTS2(model_checkpoint_path=None, 
                config_path=None, 
                phoneme_converter=None,     
                local=None):
```    

| Parameter           | Description                                                            |
|----------------------|------------------------------------------------------------------------|
| model_checkpoint_path | **Not Needed when local is set**. Path to StyleTTS2 checkpoint file. If None, will download the LibriTTS checkpoint. |
| config_path          | **Not Needed when local is set**. Path to StyleTTS2 config file. If None, will download the LibriTTS config file. |
| phoneme_converter    | Phoneme converter to use. Currently supports "espeak" and "gruut".     |
| local                | Path to local directory to store downloaded files. If None, will use default cache directory. |

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
| Parameter           | Description                                                            |
|----------------------|------------------------------------------------------------------------|
| text                 | Input text to turn into speech.                                        |
| target_voice_path   | Path to audio file of target voice to clone.                           |
| output_wav_file     | Name of output audio file (if output WAV file is desired).             |
| output_sample_rate  | Output sample rate (default 24000).                                    |
| alpha               | Determines timbre of speech, higher means style is more suitable to text than to the target voice. |
| beta                | Determines prosody of speech, higher means style is more suitable to text than to the target voice. |
| diffusion_steps     | The more the steps, the more diverse the samples are, with the cost of speed. |
| embedding_scale     | Higher scale means style is more conditional to the input text and hence more emotional. |
| ref_s              | Pre-computed style vector to pass directly.                             |
| speed               | Speed of speech, 1.0 is normal. Range .75 to 1.75 is recommended               |
| return              | Audio data as a Numpy array (will also create the WAV file if output_wav_file was set). |


### *Note: I'm not affiliated with the original authors. StyleTTS2 is an open source, state-of-the-art approach to TTS. Pass your kudos to the authors at the model repo:*

# [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models](https://github.com/yl4579/StyleTTS2)