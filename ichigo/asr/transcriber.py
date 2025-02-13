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

        model_path = f"{self.config['model_hub']}:{self.config['model_name']}"

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

    def decode_stoks(self, stoks: torch.Tensor):
        embeds = self.quantizer.dequantize(stoks)
        output = self.r2t(embeds)[0].text
        return output

    def _find_split_point(self, wav: torch.Tensor, start_idx: int, end_idx: int) -> int:
        """Find the best point to split audio by looking for silence or low amplitude.

        Args:
            wav: Audio tensor [1, T]
            start_idx: Start index of search region
            end_idx: End index of search region

        Returns:
            Index of best splitting point
        """
        segment = wav.abs().squeeze(0)[start_idx:end_idx].cpu().numpy()

        # Calculate RMS energy in small windows
        window_size = 1600  # 100ms windows at 16kHz
        energies = []
        for i in range(0, len(segment) - window_size, window_size):
            window = segment[i : i + window_size]
            energy = (window**2).mean() ** 0.5
            energies.append((i + start_idx, energy))

        quietest_idx, _ = min(energies, key=lambda x: x[1])
        return quietest_idx

    def transcribe(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = "transcription.txt",
        extensions: tuple = (".wav", ".mp3", ".flac"),
        chunk: float = 20.0,
        overlap: float = 1.0,
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
        chunk_size = int(chunk * 16000)
        overlap_size = int(overlap * 16000)

        # Handle single file
        if input_path.is_file():
            if not input_path.suffix.lower() in extensions:
                raise ValueError(f"Unsupported file type: {input_path.suffix}")

            start_time = time.time()
            wav, sr = torchaudio.load(str(input_path))
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = self.preprocess(wav, sr)
            audio_length = wav.shape[1] / 16000

            # Split long audio into chunks
            if wav.shape[1] <= chunk_size:
                chunks = [wav]
            else:
                chunks = []
                i = 0
                while i < wav.shape[1]:
                    if i + chunk_size >= wav.shape[1]:
                        # Handle the last chunk
                        chunk = wav[:, i:]
                        chunks.append(chunk)
                        break

                    # Find the best split point in the overlap region
                    search_start = i + chunk_size - overlap_size
                    search_end = min(i + chunk_size + overlap_size, wav.shape[1])
                    split_point = self._find_split_point(wav, search_start, search_end)

                    # Extract chunk up to the split point
                    chunk = wav[:, i:split_point]
                    chunks.append(chunk)
                    i = split_point

            results = []
            for chunk in chunks:
                embs, n_frames = self.s2r(chunk)
                dequantize_embed = self.quantizer(embs, n_frames)
                result = self.r2t(dequantize_embed)
                results.append(result[0].text.strip())
            transcript = " ".join(results)

            process_time = time.time() - start_time

            metadata = {
                "stok_rate": self.get_stoks(input_path).shape[-1] / audio_length,
                "audio_length": audio_length,
                "process_time": process_time,
                "rtf": process_time / audio_length if audio_length > 0 else 0,
            }

            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(transcript)

            return transcript, metadata

        # Handle folder
        elif input_path.is_dir():
            if output_path is None:
                output_path = input_path / "transcription.csv"
            else:
                output_path = Path(output_path)
                if output_path.is_dir():
                    output_path = output_path / "transcription.csv"

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
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write("filename,transcription\n")  # CSV header
                for audio_file in sorted(audio_files):
                    try:
                        transcript, _ = self.transcribe(audio_file, None)
                        results[audio_file.name] = transcript
                        safe_transcript = (
                            f'"{transcript}"' if "," in transcript else transcript
                        )
                        f.write(f"{audio_file.name},{safe_transcript}\n")
                        print(f"Successfully transcribed: {audio_file.name}")
                    except Exception as e:
                        error_msg = f"Error processing {audio_file.name}: {str(e)}"
                        print(f"ERROR: {error_msg}")
                        results[audio_file.name] = f"ERROR: {error_msg}"
                        f.write(f'{audio_file.name},"ERROR: {error_msg}"\n')

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

    def transcribe_tensor(
        self, wav, chunk_sec: float = 20.0, overlap_sec: float = 1.0
    ) -> str:
        chunk_size = int(chunk_sec * 160000)
        overlap_size = int(overlap_sec * 16000)

        if wav.shape[1] <= chunk_size:
            chunks = [wav]
        else:
            chunks = []
            i = 0
            while i < wav.shape[1]:
                if i + chunk_size >= wav.shape[1]:
                    chunk = wav[:, i:]
                    chunks.append(chunk)
                    break

                search_start = i + chunk_size - overlap_size
                search_end = min(i + chunk_size + overlap_size, wav.shape[1])
                split_point = self._find_split_point(wav, search_start, search_end)

                chunk = wav[:, i:split_point]
                chunks.append(chunk)
                i = split_point

        results = []
        for chunk in chunks:
            embs, n_frames = self.s2r(chunk)
            dequantize_embed = self.quantizer(embs, n_frames)
            result = self.r2t(dequantize_embed)
            results.append(result[0].text.strip())
        transcript = " ".join(results)

        return transcript
