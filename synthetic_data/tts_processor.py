import torchaudio

from whisperspeech.pipeline import Pipeline
from whisperspeech.t2s_up_wds_mlang_enclm import TSARTransformer

def convert_text_to_audio(pipe: Pipeline, text: str):
    """Convert text to audio.

    Args:
        pipe (Pipeline): The pipeline to use for text-to-speech.
        text (str): The text to convert to audio.

    Returns:
        torch.Tensor: The generated audio.
    """
    return pipe.generate(text)


def convert_text_to_audio_file(pipe: Pipeline, text: str, output_path: str):
    """Convert text to audio and save it to a file.

    Args:
        pipe (Pipeline): The pipeline to use for text-to-speech.
        text (str): The text to convert to audio.
        output_path (str): The path to save the audio file.
    """
    pipe.generate_to_file(output_path, text)


class TTSProcessor:
    def __init__(self, device: str):
        
        """Initialize the TTS Processor with a specified device."""
        self.pipe = Pipeline(
            t2s_ref="collabora/whisperspeech:t2s-v1.1-small-en+pl.model",
            s2a_ref="collabora/whisperspeech:s2a-q4-tiny-en+pl.model", device=device, torch_compile=True
        )

    def get_reference_voice_embedding(self, path: str):
        """Get the reference voice embedding from the given audio file.

        Args:
            path (str): The path to the audio file.
        Returns:
            torch.Tensor: The reference voice embedding."""
        return self.pipe.extract_spk_emb(path).cpu()

    def convert_text_to_audio(self, text: str, speaker=None):
        """Convert text to audio.

        Args:
            text (str): The text to convert to audio.

        Returns:
            torch.Tensor: The generated audio.
        """
        return self.pipe.generate(text, speaker=speaker)

    def convert_text_to_audio_file(self, text: str, output_path: str, speaker=None):
        """Convert text to audio and save it to a file.

        Args:
            text (str): The text to convert to audio.
            output_path (str): The path to save the audio file.
        """
        self.pipe.generate_to_file(output_path, text, speaker=speaker)