from gruut import sentences
from collections.abc import Iterable


class PhonemeConverter:
    def phonemize(self, text):
        pass


class GruutPhonemizer(PhonemeConverter):
    def phonemize(self, text, lang='en-us'):
        phonemized = []
        for sent in sentences(text, lang=lang):
            for word in sent:
                if isinstance(word.phonemes, Iterable):
                    phonemized.append(''.join(word.phonemes))
                elif isinstance(word.phonemes, str):
                    phonemized.append(word.phonemes)
        phonemized_text = ' '.join(phonemized)
        return phonemized_text


# class YourPhonemizer(Phonemizer):
#     def phonemize(self, text):
#         ...


class PhonemeConverterFactory:
    @staticmethod
    def load_phoneme_converter(name: str, **kwargs):
        if name == 'gruut':
            return GruutPhonemizer()
        else:
            raise ValueError("Invalid phoneme converter.")