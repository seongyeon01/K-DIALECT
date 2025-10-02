import os
from dataclasses import dataclass
import math
import librosa
import torch
import torch.nn.functional as F
import torchaudio
from coqpit import Coqpit
import torch.nn as nn
import insightface
import cv2
import numpy as np
from typing import Tuple, Union, List


from TTS.tts.layers.xtts.gpt import GPT
from TTS.tts.layers.xtts.dialect_pitch import DialectPitchPredictor, PitchEmbedding
from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
from TTS.tts.layers.xtts.stream_generator import init_stream_support
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, split_sentence
from TTS.tts.layers.xtts.xtts_manager import SpeakerManager, LanguageManager
from TTS.tts.models.base_tts import BaseTTS
from TTS.utils.io import load_fsspec

init_stream_support()

class PrecomputedImageEncoder(nn.Module):
    """
    이미지 경로 혹은 전처리된 Tensor를 받아
    (512,) 혹은 (B,512) 형태의 embedding을 로드/입력 후
    xtts_embedding_dim 차원으로 projection.
    """

    def __init__(
        self,
        emb_dir: str,
        xtts_embedding_dim: int = 1024,
        device: str = None,
    ):
        super().__init__()
        self.emb_dir = emb_dir
        self.projection = nn.Linear(512, xtts_embedding_dim)

    def forward(self, image_input: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        device = next(self.projection.parameters()).device

        if torch.is_tensor(image_input):
            emb = image_input.to(device)
            # (512,) → (1,512)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            return self.projection(emb)

        paths = image_input if isinstance(image_input, (list, tuple)) else [image_input]
        out_list = []
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0]
            npy_path = os.path.join(self.emb_dir, f"{base}.npy")
            if not os.path.isfile(npy_path):
                raise FileNotFoundError(f"Embedding not found: {npy_path}")
            emb_np = np.load(npy_path)               
            emb = torch.from_numpy(emb_np).unsqueeze(0).to(device)
            out_list.append(self.projection(emb))       

        return torch.cat(out_list, dim=0)
class FusionMLP(nn.Module):
    def __init__(self, in_dim=1024, out_dim=512, hidden_dim=1024):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, arc_emb, clip_emb):
        x = torch.cat([arc_emb, clip_emb], dim=-1)
        return F.normalize(self.fusion(x), dim=-1)

def wav_to_mel_cloning(
    wav,
    mel_norms_file="../experiments/clips_mel_norms.pth",
    mel_norms=None,
    device=torch.device("cpu"),
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    power=2,
    normalized=False,
    sample_rate=22050,
    f_min=0,
    f_max=8000,
    n_mels=80,
):
    """
    Convert waveform to mel-spectrogram with hard-coded parameters for cloning.

    Args:
        wav (torch.Tensor): Input waveform tensor.
        mel_norms_file (str): Path to mel-spectrogram normalization file.
        mel_norms (torch.Tensor): Mel-spectrogram normalization tensor.
        device (torch.device): Device to use for computation.

    Returns:
        torch.Tensor: Mel-spectrogram tensor.
    """
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def load_audio(audiopath, sampling_rate):
    # better load setting following: https://github.com/faroit/python_audio_loading_benchmark

    # torchaudio should chose proper backend to load audio depending on platform
    audio, lsr = torchaudio.load(audiopath)

    # stereo to mono if needed
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '10' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 10) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    # clip audio invalid values
    audio.clip_(-1, 1)
    return audio


def pad_or_truncate(t, length):
    """
    Ensure a given tensor t has a specified sequence length by either padding it with zeros or clipping it.

    Args:
        t (torch.Tensor): The input tensor to be padded or truncated.
        length (int): The desired length of the tensor.

    Returns:
        torch.Tensor: The padded or truncated tensor.
    """
    tp = t[..., :length]
    if t.shape[-1] == length:
        tp = t
    elif t.shape[-1] < length:
        tp = F.pad(t, (0, length - t.shape[-1]))
    return tp


@dataclass
class XttsAudioConfig(Coqpit):
    """
    Configuration class for audio-related parameters in the XTTS model.

    Args:
        sample_rate (int): The sample rate in which the GPT operates.
        output_sample_rate (int): The sample rate of the output audio waveform.
    """

    sample_rate: int = 22050
    output_sample_rate: int = 24000


@dataclass
class XttsArgs(Coqpit):
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
        clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
        decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

        For GPT model:
        gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
        gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
        gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
        gpt_use_perceiver_resampler (bool, optional):  If True, it will use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198. Defaults to False.
    """

    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    gpt_checkpoint: str = None
    clvp_checkpoint: str = None
    decoder_checkpoint: str = None
    num_chars: int = 255

    # XTTS GPT Encoder params
    tokenizer_file: str = ""
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None
    gpt_start_text_token: int = None
    gpt_stop_text_token: int = None
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193
    gpt_code_stride_len: int = 1024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = False

    # HifiGAN Decoder params
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True

    # constants
    duration_const: int = 102400


class Xtts(BaseTTS):
    """ⓍTTS model implementation.

    ❗ Currently it only supports inference.

    Examples:
        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> from TTS.tts.models.xtts import Xtts
        >>> config = XttsConfig()
        >>> model = Xtts.inif_from_config(config)
        >>> model.load_checkpoint(config, checkpoint_dir="paths/to/models_dir/", eval=True)
    """

    def __init__(self, config: Coqpit):
        super().__init__(config, ap=None, tokenizer=None)
        self.mel_stats_path = None
        self.config = config
        self.gpt_checkpoint = self.args.gpt_checkpoint
        self.decoder_checkpoint = self.args.decoder_checkpoint  # TODO: check if this is even needed
        self.models_dir = config.model_dir
        self.gpt_batch_size = self.args.gpt_batch_size
        self.dialect_embedding = nn.Embedding(6, config['model_args']['gpt_n_model_channels'])
        self.dialect_pitch_predictor = DialectPitchPredictor(
            dialect_emb_module=self.dialect_embedding,
            hidden_dim=config['model_args']['gpt_n_model_channels'],
        )
        self.pitch_embedding = PitchEmbedding(
            num_bins=256,
            embed_dim=config['model_args']['gpt_n_model_channels'],
            f0_min=0,
            f0_max= 382,
        )
        self.film_gamma = nn.Linear(config['model_args']['gpt_n_model_channels'], config['model_args']['gpt_n_model_channels'])
        self.film_beta  = nn.Linear(config['model_args']['gpt_n_model_channels'], config['model_args']['gpt_n_model_channels'])
        self.tokenizer = VoiceBpeTokenizer()
        self.gpt = None
        self.init_models()
        self.register_buffer("mel_stats", torch.ones(80))

    def init_models(self):
        """Initialize the models. We do it here since we need to load the tokenizer first."""
        d_vec_dim = int(getattr(self.args, "d_vector_dim", 256))
        secs_dim  = int(getattr(self.args, "secs_dim", d_vec_dim))
        freeze_fusion = bool(getattr(self.args, "freeze_fusion", True))
        if self.tokenizer.tokenizer is not None:
            self.args.gpt_number_text_tokens = self.tokenizer.get_number_tokens()
            self.args.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id("[START]")
            self.args.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]")

        if self.args.gpt_number_text_tokens:
            self.gpt = GPT(
                layers=self.args.gpt_layers,
                model_dim=self.args.gpt_n_model_channels,
                start_text_token=self.args.gpt_start_text_token,
                stop_text_token=self.args.gpt_stop_text_token,
                heads=self.args.gpt_n_heads,
                max_text_tokens=self.args.gpt_max_text_tokens,
                max_mel_tokens=self.args.gpt_max_audio_tokens,
                max_prompt_tokens=self.args.gpt_max_prompt_tokens,
                number_text_tokens=self.args.gpt_number_text_tokens,
                num_audio_tokens=self.args.gpt_num_audio_tokens,
                start_audio_token=self.args.gpt_start_audio_token,
                stop_audio_token=self.args.gpt_stop_audio_token,
                use_perceiver_resampler=self.args.gpt_use_perceiver_resampler,
                code_stride_len=self.args.gpt_code_stride_len,
            )

        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=self.args.input_sample_rate,
            output_sample_rate=self.args.output_sample_rate,
            output_hop_length=self.args.output_hop_length,
            ar_mel_length_compression=self.args.gpt_code_stride_len,
            decoder_input_dim=self.args.decoder_input_dim,
            d_vector_dim=self.args.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer,
        )
        self.dialect_embedding = nn.Embedding(6, self.config['model_args']['gpt_n_model_channels'])
        self.arcface_encoder = PrecomputedImageEncoder(
            emb_dir="/root/datasets/data_22050/embs",
            xtts_embedding_dim=512,
        ).to(self.device)

        self.clip_encoder = PrecomputedImageEncoder(
            emb_dir="/root/datasets/data_22050/clip_embs",
            xtts_embedding_dim=512,
        ).to(self.device)

        self.image_spkr_proj = nn.Linear(512, d_vec_dim).to(self.device)

        self.concat_fusion = FusionMLP(in_dim=1024, out_dim=512, hidden_dim=1024).to(self.device)
        ckpt = torch.load("/root/workspace/best_fusion_model.pth", map_location=self.device)
        fusion_state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
        self.concat_fusion.load_state_dict(fusion_state, strict=True)
        if freeze_fusion:
            for p in self.concat_fusion.parameters():
                p.requires_grad = False
            self.concat_fusion.eval()

        self.clip_proj = nn.Linear(512, 80).to(self.device)
        self.style_adapter = nn.Sequential(
            nn.Linear(80, 256),
            nn.ReLU(),
            nn.Linear(256, 80)
        ).to(self.device)

        self.image_spkr_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, d_vec_dim)
        )
        self.image_spkr_head[-1] = nn.utils.weight_norm(self.image_spkr_head[-1])
        self.image_spkr_head = self.image_spkr_head.to(self.device)

        self.secs_dim = secs_dim
        self.secs_align = nn.Linear(d_vec_dim, secs_dim, bias=False).to(self.device)
        nn.init.orthogonal_(self.secs_align.weight)
        self.style_stopgrad_from_arc = bool(getattr(self.args, "style_stopgrad_from_arc", False))
        self.spk_feat_drop_p         = float(getattr(self.args, "spk_feat_drop_p", 0.0))
        self.style_mel_norm          = bool(getattr(self.args, "style_mel_norm", True))
        self.style_time_jitter_std   = float(getattr(self.args, "style_time_jitter_std", 0.0))
        self.spk_to_logf0 = nn.Sequential(
            nn.Linear(d_vec_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Compute the conditioning latents for the GPT model from the given audio.

        Args:
            audio (tensor): audio tensor.
            sr (int): Sample rate of the audio.
            length (int): Length of the audio in seconds. If < 0, use the whole audio. Defaults to 30.
            chunk_length (int): Length of the audio chunks in seconds. When `length == chunk_length`, the whole audio
                is being used without chunking. It must be < `length`. Defaults to 6.
        """
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        if self.args.gpt_use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i : i + 22050 * chunk_length]

                # if the chunk is too short ignore it 
                if audio_chunk.size(-1) < 22050 * 0.33:
                    continue

                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )
                style_emb = self.gpt.get_style_emb(mel_chunk.to(self.device), None)
                style_embs.append(style_emb)

            # mean style embedding
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = wav_to_mel_cloning(
                audio,
                mel_norms=self.mel_stats.cpu(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            cond_latent = self.gpt.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.device)
        )

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path=None,
        audio_tensor=None,
        max_ref_length=30,
        gpt_cond_len=6,
        gpt_cond_chunk_len=6,
        librosa_trim_db=None,
        sound_norm_refs=False,
        load_sr=22050,
    ):
        """
        Get the conditioning latents for the GPT model from the given audio.

        Args:
            audio_path (str or List[str], optional): Path to reference audio file(s).
            audio_tensor (torch.Tensor, optional): Pre-loaded audio tensor [1, T].
            max_ref_length (int): Maximum length of each reference audio in seconds.
            gpt_cond_len (int): Length of the audio used for gpt latents.
            gpt_cond_chunk_len (int): Chunk length used for gpt latents.
            librosa_trim_db (int, optional): Trim the audio using this value. If None, not trimming.
            sound_norm_refs (bool): Whether to normalize the audio.
            load_sr (int): Sample rate to load the audio.
        """
        if audio_tensor is not None:
            audio = audio_tensor.to(self.device)
            audio = audio[:, : load_sr * max_ref_length]
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            gpt_cond_latents = self.get_gpt_cond_latents(
                audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
            )
            return gpt_cond_latents, speaker_embedding

        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        speaker_embeddings = []
        audios = []
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to(self.device)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            spk_emb = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(spk_emb)
            audios.append(audio)

        # merge and compute GPT latents
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )

        # average speaker embeddings
        if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings).mean(dim=0)
        else:
            speaker_embedding = None

        return gpt_cond_latents, speaker_embedding

    def get_conditioning_latents_image(
        self,
        arcface_input: torch.Tensor,
        clip_input:    torch.Tensor,
        gpt_cond_len:  int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        arc_emb  = self.arcface_encoder(arcface_input) 
        clip_emb = self.clip_encoder(clip_input)       
        arc_for_style = arc_emb.detach() if self.style_stopgrad_from_arc else arc_emb
        fused_512 = self.concat_fusion(arc_for_style, clip_emb)
        spk_src = arc_emb
        spk_latent = self.image_spkr_head(spk_src)      # [B, D_spk]
        if self.training and self.spk_feat_drop_p > 0.0:
            p = float(self.spk_feat_drop_p)
            mask = torch.empty_like(spk_latent).bernoulli_(1.0 - p)
            spk_latent = spk_latent * mask / (1.0 - p)
        spk_latent = F.normalize(spk_latent, dim=1)
        speaker_embedding = spk_latent.unsqueeze(-1) 

        style_mel = self.clip_proj(fused_512)     
        style_mel = self.style_adapter(style_mel)     
        if self.style_mel_norm:
            mu  = style_mel.mean(dim=1, keepdim=True)
            std = style_mel.std(dim=1, keepdim=True).clamp_min(1e-5)
            style_mel = (style_mel - mu) / std

        T = gpt_cond_len or self.args.gpt_cond_len
        style_seq = style_mel.unsqueeze(-1).expand(-1, -1, T)  
        if self.training and self.style_time_jitter_std > 0.0:
            style_seq = style_seq + torch.randn_like(style_seq) * float(self.style_time_jitter_std)

        cond       = self.gpt.get_style_emb(style_seq, return_latent=False)
        gpt_latent = cond.transpose(1, 2).contiguous()            
        return gpt_latent, speaker_embedding


    def synthesize(self, text, config, language, image_path, speaker_id=None, **kwargs):
        """Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            config (XttsConfig): Config with inference parameters.
            speaker_wav (list): List of paths to the speaker audio files to be used for cloning.
            language (str): Language ID of the speaker.
            **kwargs: Inference settings. See `inference()`.

        Returns:
            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,
            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`
            as latents used at inference.

        """
        assert image_path is not None or speaker_id is not None, "You must provide either image_path or speaker_id."
        assert (
            "zh-cn" if language == "zh" else language in self.config.languages
        ), f" ❗ Language {language} is not supported. Supported languages are {self.config.languages}"
        # Use generally found best tuning knobs for generation.
        settings = {
            "temperature": config.temperature,
            "length_penalty": config.length_penalty,
            "repetition_penalty": config.repetition_penalty,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        settings.update(kwargs)  # allow overriding of preset settings with kwargs
        if speaker_id is not None:
            gpt_cond_latent, speaker_embedding = self.speaker_manager.speakers[speaker_id].values()
            return self.inference(text, language, gpt_cond_latent, speaker_embedding, **settings)
        settings.update({
            "gpt_cond_len": config.gpt_cond_len,
            "gpt_cond_chunk_len": config.gpt_cond_chunk_len,
            "max_ref_len": config.max_ref_len,
            "sound_norm_refs": config.sound_norm_refs,
        })
        return self.full_inference(text, image_path, language, **settings)

    @torch.inference_mode()
    def full_inference(
        self,
        text,
        image_path,
        language,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        # Cloning
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
        max_ref_len=10,
        sound_norm_refs=False,
        dialect_id=None,
        **hf_generate_kwargs,
    ):
        """
        This function produces an audio clip of the given text being spoken with the given reference voice.

        Args:
            text: (str) Text to be spoken.

            ref_audio_path: (str) Path to a reference audio file to be used for cloning. This audio file should be >3
                seconds long.

            language: (str) Language of the voice to be generated.

            temperature: (float) The softmax temperature of the autoregressive model. Defaults to 0.65.

            length_penalty: (float) A length penalty applied to the autoregressive decoder. Higher settings causes the
                model to produce more terse outputs. Defaults to 1.0.

            repetition_penalty: (float) A penalty that prevents the autoregressive decoder from repeating itself during
                decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc. Defaults to 2.0.

            top_k: (int) K value used in top-k sampling. [0,inf]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 50.

            top_p: (float) P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 0.8.

            gpt_cond_len: (int) Length of the audio used for cloning. If audio is shorter, then audio length is used
                else the first `gpt_cond_len` secs is used. Defaults to 30 seconds.

            gpt_cond_chunk_len: (int) Chunk length used for cloning. It must be <= `gpt_cond_len`.
                If gpt_cond_len == gpt_cond_chunk_len, no chunking. Defaults to 6 seconds.

            hf_generate_kwargs: (**kwargs) The huggingface Transformers generate API is used for the autoregressive
                transformer. Extra keyword args fed to this function get forwarded directly to that API. Documentation
                here: https://huggingface.co/docs/transformers/internal/generation_utils

        Returns:
            Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
            Sample rate is 24kHz.
        """
        # (gpt_cond_latent, speaker_embedding) = self.get_conditioning_latents(
        #     audio_path=ref_audio_path,
        #     gpt_cond_len=gpt_cond_len,
        #     gpt_cond_chunk_len=gpt_cond_chunk_len,
        #     max_ref_length=max_ref_len,
        #     sound_norm_refs=sound_norm_refs,
        # )
        (gpt_cond_latent, speaker_embedding) = self.get_conditioning_latents_image(
            image_path,
            gpt_cond_len=gpt_cond_len,
        )

        return self.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            dialect_id=dialect_id,
            **hf_generate_kwargs,
        )

    @torch.inference_mode()
    def inference(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        dialect_id=None,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        num_beams=1,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        B, T, C = gpt_cond_latent.shape
        if dialect_id is None:
            raise ValueError("dialect_id must be provided for dialect-conditioned inference")

        dids = torch.full((B,), int(dialect_id), device=self.device, dtype=torch.long)

        style_emb = self.dialect_embedding(dids)            
        gamma     = self.film_gamma(style_emb).unsqueeze(-1) 
        beta      = self.film_beta(style_emb).unsqueeze(-1)  

        spk_lat = speaker_embedding.squeeze(-1)             
        base_logf0 = self.spk_to_logf0(spk_lat).squeeze(-1) 
        base_logf0 = base_logf0.unsqueeze(1).expand(-1, T) 

        delta_semi = self.dialect_pitch_predictor(dids, T)  
        delta_semi = torch.clamp(delta_semi, -5.0, 5.0)   

        delta_log = delta_semi * (math.log(2.0) / 12.0)     
        pred_logf0 = base_logf0 + delta_log   
        pred_f0    = torch.exp(pred_logf0) 

        pitch_e = self.pitch_embedding(pred_f0).transpose(1, 2) 

        cond_t = gpt_cond_latent.transpose(1, 2)        
        fused  = gamma * cond_t + beta + pitch_e  
        gpt_cond_latent = fused.transpose(1, 2).contiguous()

        if enable_text_splitting:
            text = split_sentence(text, language, self.tokenizer.char_limits[language])
        else:
            text = [text]

        wavs = []
        gpt_latents_list = []
        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            with torch.no_grad():
                gpt_codes = self.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    input_tokens=None,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.gpt_batch_size,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    **hf_generate_kwargs,
                )
                print("▶ gpt_codes.shape:", gpt_codes.shape)
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device
                )

                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                )

                if length_scale != 1.0:
                    gpt_latents = F.interpolate(
                        gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                    ).transpose(1, 2)

                gpt_latents_list.append(gpt_latents.cpu())
                wavs.append(self.hifigan_decoder(gpt_latents, g=speaker_embedding).cpu().squeeze())

        return {
            "wav": torch.cat(wavs, dim=0).numpy(),
            "gpt_latents": torch.cat(gpt_latents_list, dim=1).numpy(),
            "speaker_embedding": speaker_embedding,
        }

    def handle_chunks(self, wav_gen, wav_gen_prev, wav_overlap, overlap_len):
        """Handle chunk formatting in streaming mode"""
        wav_chunk = wav_gen[:-overlap_len]
        if wav_gen_prev is not None:
            wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) : -overlap_len]
        if wav_overlap is not None:
            # cross fade the overlap section
            if overlap_len > len(wav_chunk):
                # wav_chunk is smaller than overlap_len, pass on last wav_gen
                if wav_gen_prev is not None:
                    wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) :]
                else:
                    # not expecting will hit here as problem happens on last chunk
                    wav_chunk = wav_gen[-overlap_len:]
                return wav_chunk, wav_gen, None
            else:
                crossfade_wav = wav_chunk[:overlap_len]
                crossfade_wav = crossfade_wav * torch.linspace(0.0, 1.0, overlap_len).to(crossfade_wav.device)
                wav_chunk[:overlap_len] = wav_overlap * torch.linspace(1.0, 0.0, overlap_len).to(wav_overlap.device)
                wav_chunk[:overlap_len] += crossfade_wav

        wav_overlap = wav_gen[-overlap_len:]
        wav_gen_prev = wav_gen
        return wav_chunk, wav_gen_prev, wav_overlap

    @torch.inference_mode()
    def inference_stream(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # Streaming
        stream_chunk_size=20,
        overlap_wav_len=1024,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        speed=1.0,
        enable_text_splitting=False,
        dialect_id=None,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        if enable_text_splitting:
            text = split_sentence(text, language, self.tokenizer.char_limits[language])
        else:
            text = [text]

        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."
            if hasattr(self, "dialect_embedding") and dialect_id is not None:
                dialect_emb = self.dialect_embedding(torch.tensor([dialect_id], device=self.device))
    
            fake_inputs = self.gpt.compute_embeddings(
                gpt_cond_latent.to(self.device),
                text_tokens,
                dialect_emb=dialect_emb,
            )
            gpt_generator = self.gpt.get_generator(
                fake_inputs=fake_inputs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=1,
                num_return_sequences=1,
                length_penalty=float(length_penalty),
                repetition_penalty=float(repetition_penalty),
                output_attentions=False,
                output_hidden_states=True,
                **hf_generate_kwargs,
            )

            last_tokens = []
            all_latents = []
            wav_gen_prev = None
            wav_overlap = None
            is_end = False

            while not is_end:
                try:
                    x, latent = next(gpt_generator)
                    last_tokens += [x]
                    all_latents += [latent]
                except StopIteration:
                    is_end = True

                if is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size):
                    gpt_latents = torch.cat(all_latents, dim=0)[None, :]
                    if length_scale != 1.0:
                        gpt_latents = F.interpolate(
                            gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                        ).transpose(1, 2)
                    wav_gen = self.hifigan_decoder(gpt_latents, g=speaker_embedding.to(self.device))
                    wav_chunk, wav_gen_prev, wav_overlap = self.handle_chunks(
                        wav_gen.squeeze(), wav_gen_prev, wav_overlap, overlap_wav_len
                    )
                    last_tokens = []
                    yield wav_chunk

    def forward(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    def eval_step(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    @staticmethod
    def init_from_config(config: "XttsConfig", **kwargs):  # pylint: disable=unused-argument
        return Xtts(config)

    def eval(self):  # pylint: disable=redefined-builtin
        """Sets the model to evaluation mode. Overrides the default eval() method to also set the GPT model to eval mode."""
        self.gpt.init_gpt_for_inference()
        super().eval()

    def get_compatible_checkpoint_state_dict(self, model_path):
        checkpoint = load_fsspec(model_path, map_location=torch.device("cpu"))["model"]
        # remove xtts gpt trainer extra keys
        ignore_keys = ["torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"]
        for key in list(checkpoint.keys()):
            # check if it is from the coqui Trainer if so convert it
            if key.startswith("xtts."):
                new_key = key.replace("xtts.", "")
                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]
                key = new_key

            # remove unused keys
            if key.split(".")[0] in ignore_keys:
                del checkpoint[key]

        return checkpoint

    def load_checkpoint(
        self,
        config,
        checkpoint_dir=None,
        checkpoint_path=None,
        vocab_path=None,
        eval=True,
        strict=True,
        use_deepspeed=False,
        speaker_file_path=None,
    ):
        """
        Loads a checkpoint from disk and initializes the model's state and tokenizer.

        Args:
            config (dict): The configuration dictionary for the model.
            checkpoint_dir (str, optional): The directory where the checkpoint is stored. Defaults to None.
            checkpoint_path (str, optional): The path to the checkpoint file. Defaults to None.
            vocab_path (str, optional): The path to the vocabulary file. Defaults to None.
            eval (bool, optional): Whether to set the model to evaluation mode. Defaults to True.
            strict (bool, optional): Whether to strictly enforce that the keys in the checkpoint match the keys in the model. Defaults to True.

        Returns:
            None
        """

        model_path = checkpoint_path or os.path.join(checkpoint_dir, "model.pth")
        vocab_path = vocab_path or os.path.join(checkpoint_dir, "vocab.json")

        if speaker_file_path is None and checkpoint_dir is not None:
            speaker_file_path = os.path.join(checkpoint_dir, "speakers_xtts.pth")

        self.language_manager = LanguageManager(config)
        self.speaker_manager = None
        if speaker_file_path is not None and os.path.exists(speaker_file_path):
            self.speaker_manager = SpeakerManager(speaker_file_path)

        if os.path.exists(vocab_path):
            self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)

        self.init_models()

        checkpoint = self.get_compatible_checkpoint_state_dict(model_path)
        print("🔍 dialect_embedding key in checkpoint:", any("dialect_embedding" in k for k in checkpoint.keys()))
        # deal with v1 and v1.1. V1 has the init_gpt_for_inference keys, v1.1 do not
        try:
            self.load_state_dict(checkpoint, strict=strict)
        except:
            if eval:
                self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache)
            self.load_state_dict(checkpoint, strict=strict)

        if eval:
            self.hifigan_decoder.eval()
            self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=use_deepspeed)
            self.gpt.eval()

    def train_step(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )
