import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="vector_quantize_pytorch"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="You are using `torch.load`"
)
import torch
import torchaudio
import yaml
from huggingface_hub import hf_hub_download

from ichigo.asr.arch.quantizer import Quantizer
from ichigo.asr.arch.r2t import Rep2Text
from ichigo.asr.arch.s2r import Speech2Rep


def load_quantizer(ref, config):
    if ":" in ref:
        repo_id, filename = ref.split(":", 1)
        local_filename = hf_hub_download(repo_id=repo_id, filename=filename)
    else:
        local_filename = ref

    spec = torch.load(local_filename)
    model_state_dict = {
        k.replace("model.", ""): v for k, v in spec["state_dict"].items()
    }

    quantizer = Quantizer(config)
    quantizer.load_state_dict(model_state_dict, strict=False)
    quantizer.eval()

    return quantizer


class IchigoASR:
    def __init__(
        self,
        config: str = "merge-2560d",
    ):
        # Load config
        config_path = Path(__file__).parent / "config" / f"{config}.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        model_path = f"{self.config['model_hub']}:{self.config['model_name']}.pth"

        self.s2r = Speech2Rep(self.config)
        self.quantizer = load_quantizer(ref=model_path, config=self.config)
        self.r2t = Rep2Text(self.config)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.s2r.to(self.device)
        self.quantizer.to(self.device)
        self.r2t.to(self.device)

    def preprocess(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        return audio.to(self.device)

    def get_stoks(self, input_path: Union[str, Path]):
        """Support return stoks for a single file"""
        input_path = Path(input_path)
        wav, sr = torchaudio.load(str(input_path))
        wav = self.preprocess(wav, sr)

        embs, n_frames = self.s2r(wav)
        stoks = self.quantizer(embs, n_frames, return_stoks=True)
        return stoks

    def transcribe(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = "transcription.txt",
        extensions: tuple = (".wav", ".mp3", ".flac"),
    ) -> Union[str, Dict[str, str]]:
        """Transcribe audio file or folder of audio files.

        Args:
            input_path: Path to audio file or folder containing audio files
            output_path: Path to save transcript(s). If input is folder, creates 'transcripts' subfolder
            extensions: Tuple of valid audio file extensions to process (only used for folder input)

        Returns:
            For single file: transcript string and metadata dict
            For folder: dictionary mapping filenames to their transcripts
        """
        input_path = Path(input_path)

        # Handle single file
        if input_path.is_file():
            if not input_path.suffix.lower() in extensions:
                raise ValueError(f"Unsupported file type: {input_path.suffix}")

            start_time = time.time()
            wav, sr = torchaudio.load(str(input_path))
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = self.preprocess(wav, sr)
            duration = wav.shape[1] / 16000

            # ! Inference
            embs, n_frames = self.s2r(wav)
            dequantize_embed = self.quantizer(embs, n_frames)
            result = self.r2t(dequantize_embed)
            transcript = result[0].text

            process_time = time.time() - start_time
            metadata = {
                "duration": duration,
                "process_time": process_time,
                "rtf": process_time / duration if duration > 0 else 0,
            }

            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(transcript)

            return transcript, metadata

        # Handle folder
        elif input_path.is_dir():
            if output_path is None:
                output_path = input_path / "transcription.txt"
            else:
                output_path = Path(output_path)
                if output_path.is_dir():
                    output_path = output_path / "transcription.txt"

            audio_files = [
                f
                for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            ]
            non_audio = [
                f
                for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() not in extensions
            ]

            if non_audio:
                warnings.warn(
                    f"Found {len(non_audio)} non-audio files that will be skipped:\n"
                    f"{', '.join(f.name for f in non_audio)}"
                )

            if not audio_files:
                warnings.warn(
                    f"No audio files found with extensions {extensions} in {input_path}"
                )
                return {}

            results = {}
            # Create or open the transcription file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for audio_file in sorted(audio_files):
                    try:
                        transcript, _ = self.transcribe(audio_file, None)
                        results[audio_file.name] = transcript
                        f.write(f"{audio_file.name}\t{transcript}\n")
                        print(f"Successfully transcribed: {audio_file.name}")
                    except Exception as e:
                        error_msg = f"Error processing {audio_file.name}: {str(e)}"
                        print(f"ERROR: {error_msg}")
                        results[audio_file.name] = f"ERROR: {error_msg}"
                        f.write(f"{audio_file.name}\tERROR: {error_msg}\n")

            success = sum(1 for v in results.values() if not v.startswith("ERROR"))
            failed = sum(1 for v in results.values() if v.startswith("ERROR"))
            print(f"\nTranscription Summary:")
            print(f"- Total files in directory: {len(audio_files) + len(non_audio)}")
            print(f"- Audio files processed: {len(audio_files)}")
            print(f"- Non-audio files skipped: {len(non_audio)}")
            print(f"- Successful transcriptions: {success}")
            print(f"- Failed transcriptions: {failed}")

            return results

        else:
            raise ValueError(f"Input path does not exist: {input_path}")
