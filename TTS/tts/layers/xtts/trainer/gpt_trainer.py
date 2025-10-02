from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import random
import torchaudio
from coqpit import Coqpit
from torch.nn import functional as F
from torch.utils.data import DataLoader
from trainer.torch import DistributedSampler
from trainer.trainer_utils import get_optimizer, get_scheduler
import librosa
import numpy as np
import math

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.classifier import (
    diag_info_nce,
    orthogonal_loss
)
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
from TTS.utils.io import load_fsspec


@dataclass
class GPTTrainerConfig(XttsConfig):
    lr: float = 5e-06
    training_seed: int = 1
    optimizer_wd_only_on_weights: bool = False
    weighted_loss_attrs: dict = field(default_factory=lambda: {})
    weighted_loss_multipliers: dict = field(default_factory=lambda: {})
    test_sentences: List[dict] = field(default_factory=lambda: [])
    d_vector_dim: int = 512


@dataclass
class XttsAudioConfig(XttsAudioConfig):
    dvae_sample_rate: int = 22050


@dataclass
class GPTArgs(XttsArgs):
    min_conditioning_length: int = 66150
    max_conditioning_length: int = 132300
    gpt_loss_text_ce_weight: float = 0.01
    gpt_loss_mel_ce_weight: float = 1.0
    gpt_num_audio_tokens: int = 8194
    debug_loading_failures: bool = False
    max_wav_length: int = 255995  # ~11.6 seconds
    max_text_length: int = 200
    tokenizer_file: str = ""
    mel_norm_file: str = "https://coqui.gateway.scarf.sh/v0.14.0_models/mel_norms.pth"
    dvae_checkpoint: str = ""
    xtts_checkpoint: str = ""
    gpt_checkpoint: str = ""  # if defined it will replace the gpt weights on xtts model
    vocoder: str = ""  # overide vocoder key on the config to avoid json write issues
    c_contrast: float = 0.7   # contrastive loss weight
    c_adv:     float = 0.3   # adversarial loss weight
    c_spk_distill = 1.0
    c_mse = 0.5
    c_mp_nce = 0.5
    tau = 0.07
    tau_style = 0.07
    tau_spk = 0.07
    c_ortho = 0.05
    c_arc = 0.5
    arc_margin = 0.2
    arc_scale = 30.0
    mix_ratio = 0.4
    pitch_loss_weight = 0.3 


def callback_clearml_load_save(operation_type, model_info):
    # return None means skip the file upload/log, returning model_info will continue with the log/upload
    # you can also change the upload destination file name model_info.upload_filename or check the local file size with Path(model_info.local_model_path).stat().st_size
    assert operation_type in ("load", "save")
    # print(operation_type, model_info.__dict__)

    if "similarities.pth" in model_info.__dict__["local_model_path"]:
        return None

    return model_info


class GPTTrainer(BaseTTS):
    def __init__(self, config: Coqpit):
        """
        Tortoise GPT training class
        """
        super().__init__(config, ap=None, tokenizer=None)
        self.config = config
        # init XTTS model
        self.xtts = Xtts(self.config)
        # create the tokenizer with the target vocabulary
        self.xtts.tokenizer = VoiceBpeTokenizer(self.args.tokenizer_file)
        # init gpt encoder and hifigan decoder
        self.xtts.init_models()

        if self.args.xtts_checkpoint:
            self.load_checkpoint(self.config, self.args.xtts_checkpoint, eval=False, strict=False)

        # set mel stats
        if self.args.mel_norm_file:
            self.xtts.mel_stats = load_fsspec(self.args.mel_norm_file)

        # load GPT if available
        if self.args.gpt_checkpoint:
            gpt_checkpoint = torch.load(self.args.gpt_checkpoint, map_location=torch.device("cpu"))
            # deal with coqui Trainer exported model
            if "model" in gpt_checkpoint.keys() and "config" in gpt_checkpoint.keys():
                print("Coqui Trainer checkpoint detected! Converting it!")
                gpt_checkpoint = gpt_checkpoint["model"]
                states_keys = list(gpt_checkpoint.keys())
                for key in states_keys:
                    if "gpt." in key:
                        new_key = key.replace("gpt.", "")
                        gpt_checkpoint[new_key] = gpt_checkpoint[key]
                        del gpt_checkpoint[key]
                    else:
                        del gpt_checkpoint[key]

            # edit checkpoint if the number of tokens is changed to ensures the better transfer learning possible
            if (
                "text_embedding.weight" in gpt_checkpoint
                and gpt_checkpoint["text_embedding.weight"].shape != self.xtts.gpt.text_embedding.weight.shape
            ):
                num_new_tokens = (
                    self.xtts.gpt.text_embedding.weight.shape[0] - gpt_checkpoint["text_embedding.weight"].shape[0]
                )
                print(f" > Loading checkpoint with {num_new_tokens} additional tokens.")

                # add new tokens to a linear layer (text_head)
                emb_g = gpt_checkpoint["text_embedding.weight"]
                new_row = torch.randn(num_new_tokens, emb_g.shape[1])
                start_token_row = emb_g[-1, :]
                emb_g = torch.cat([emb_g, new_row], axis=0)
                emb_g[-1, :] = start_token_row
                gpt_checkpoint["text_embedding.weight"] = emb_g

                # add new weights to the linear layer (text_head)
                text_head_weight = gpt_checkpoint["text_head.weight"]
                start_token_row = text_head_weight[-1, :]
                new_entry = torch.randn(num_new_tokens, self.xtts.gpt.text_head.weight.shape[1])
                text_head_weight = torch.cat([text_head_weight, new_entry], axis=0)
                text_head_weight[-1, :] = start_token_row
                gpt_checkpoint["text_head.weight"] = text_head_weight

                # add new biases to the linear layer (text_head)
                text_head_bias = gpt_checkpoint["text_head.bias"]
                start_token_row = text_head_bias[-1]
                new_bias_entry = torch.zeros(num_new_tokens)
                text_head_bias = torch.cat([text_head_bias, new_bias_entry], axis=0)
                text_head_bias[-1] = start_token_row
                gpt_checkpoint["text_head.bias"] = text_head_bias

            self.xtts.gpt.load_state_dict(gpt_checkpoint, strict=True)
            print(">> GPT weights restored from:", self.args.gpt_checkpoint)

        # Mel spectrogram extractor for conditioning
        if self.args.gpt_use_perceiver_resampler:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=2048,
                hop_length=256,
                win_length=1024,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )
        else:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=4096,
                hop_length=1024,
                win_length=4096,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )

        # Load DVAE
        self.dvae = DiscreteVAE(
            channels=80,
            normalization=None,
            positional_dims=1,
            num_tokens=self.args.gpt_num_audio_tokens - 2,
            codebook_dim=512,
            hidden_dim=512,
            num_resnet_blocks=3,
            kernel_size=3,
            num_layers=2,
            use_transposed_convs=False,
        )

        self.dvae.eval()
        if self.args.dvae_checkpoint:
            dvae_checkpoint = torch.load(self.args.dvae_checkpoint, map_location=torch.device("cpu"))
            self.dvae.load_state_dict(dvae_checkpoint, strict=False)
            print(">> DVAE weights restored from:", self.args.dvae_checkpoint)
        else:
            raise RuntimeError(
                "You need to specify config.model_args.dvae_checkpoint path to be able to train the GPT decoder!!"
            )

        # Mel spectrogram extractor for DVAE
        self.torch_mel_spectrogram_dvae = TorchMelSpectrogram(
            mel_norm_file=self.args.mel_norm_file, sampling_rate=config.audio.dvae_sample_rate
        )
        n_model = self.xtts.args.gpt_n_model_channels
        self.dialect_embedding = nn.Embedding(num_embeddings=6, embedding_dim=n_model)
        nn.init.xavier_uniform_(self.dialect_embedding.weight)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens, dialect_labels):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, num_samples, 80,t_m)
        cond_idxs: cond start and end indexs, (b, 2)
        cond_lens: long tensor, (b,)
        """
        d_emb = self.dialect_embedding(dialect_labels)            # (B, D)

        losses = self.xtts.gpt(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            dialect_emb=d_emb,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
        )
        return losses

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:  # pylint: disable=W0613
        test_audios = {}
        if self.config.test_sentences:
            # init gpt for inference mode
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.xtts.gpt.eval()
            print(" | > Synthesizing test sentences.")
            for idx, s_info in enumerate(self.config.test_sentences):
                wav = self.xtts.synthesize(
                    s_info["text"],
                    self.config,
                    # s_info["speaker_wav"],
                    s_info["language"],
                    s_info["image_path"],
                    gpt_cond_len=3,
                )["wav"]
                test_audios["{}-audio".format(idx)] = wav

            # delete inference layers
            del self.xtts.gpt.gpt_inference
            del self.xtts.gpt.gpt.wte
        return {"audios": test_audios}

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.args.output_sample_rate)

    def format_batch(self, batch: Dict) -> Dict:
        return batch

    @torch.no_grad()  # torch no grad to avoid gradients from the pre-processing and DVAE codes extraction
    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        batch["text_lengths"] = batch["text_lengths"]
        batch["wav_lengths"] = batch["wav_lengths"]
        batch["text_inputs"] = batch["padded_text"]
        batch["cond_idxs"] = batch["cond_idxs"]
        # compute conditioning mel specs
        # transform waves from torch.Size([B, num_cond_samples, 1, T] to torch.Size([B * num_cond_samples, 1, T] because if is faster than iterate the tensor
        B, num_cond_samples, C, T = batch["conditioning"].size()
        conditioning_reshaped = batch["conditioning"].view(B * num_cond_samples, C, T)
        paired_conditioning_mel = self.torch_mel_spectrogram_style_encoder(conditioning_reshaped)
        # transform torch.Size([B * num_cond_samples, n_mel, T_mel]) in torch.Size([B, num_cond_samples, n_mel, T_mel])
        n_mel = self.torch_mel_spectrogram_style_encoder.n_mel_channels  # paired_conditioning_mel.size(1)
        T_mel = paired_conditioning_mel.size(2)
        paired_conditioning_mel = paired_conditioning_mel.view(B, num_cond_samples, n_mel, T_mel)
        # get the conditioning embeddings
        batch["cond_mels"] = paired_conditioning_mel
        # compute codes using DVAE
        if self.config.audio.sample_rate != self.config.audio.dvae_sample_rate:
            dvae_wav = torchaudio.functional.resample(
                batch["wav"],
                orig_freq=self.config.audio.sample_rate,
                new_freq=self.config.audio.dvae_sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        else:
            dvae_wav = batch["wav"]
        dvae_mel_spec = self.torch_mel_spectrogram_dvae(dvae_wav)
        codes = self.dvae.get_codebook_indices(dvae_mel_spec)

        batch["audio_codes"] = codes
        labels = batch["dialect_labels"].to(self.device)
        batch["dialect_emb"] = self.dialect_embedding(labels)  # (B,128)
        batch["image_emb"] = batch["image_emb"]
        batch["f0_emb"] = batch["f0_emb"]
        # delete useless batch tensors
        del batch["padded_text"]
        # del batch["wav"]
        del batch["conditioning"]
        return batch

    def train_step(self, batch, criterion):
        device = self.device
        dialect_labels = batch["dialect_labels"].to(self.device)
        loss_dict = {}

        # ----- 필수 텐서 -----
        text_inputs  = batch["text_inputs"]
        text_lengths = batch["text_lengths"]
        audio_codes  = batch["audio_codes"]
        wav_lengths  = batch["wav_lengths"]
        cond_idxs    = batch["cond_idxs"]
        cond_lens    = batch["cond_lens"]
        dialect_lbls = batch["dialect_labels"].to(device)

        with torch.no_grad():
            a_lat, a_spk_3d = self.xtts.get_conditioning_latents(
                audio_tensor=batch["wav"],
                max_ref_length=self.args.max_conditioning_length,
                gpt_cond_len=12,
                gpt_cond_chunk_len=4,
            )
            a_style = F.normalize(a_lat.mean(dim=2), dim=1)  
            a_spk   = F.normalize(a_spk_3d.squeeze(-1), dim=1) 

            secs_teacher = F.normalize(self.xtts.secs_encoder(batch["wav"], sr=22050), dim=1)  # [B, secs_dim]

        i_lat, i_spk_3d = self.xtts.get_conditioning_latents_image(
            batch["image_emb"], clip_input=batch["clip_emb"], gpt_cond_len=12
        )
        i_style = F.normalize(i_lat.mean(dim=2), dim=1) 
        i_spk   = F.normalize(i_spk_3d.squeeze(-1), dim=1) 

        i_spk_proj = F.normalize(self.xtts.secs_align(i_spk), dim=1) 

        loss_contrast = self.args.c_contrast * (
            diag_info_nce(i_style, a_style, tau=self.args.tau_style) +
            diag_info_nce(i_spk,   a_spk,   tau=self.args.tau_spk)
        )

        loss_mse_style   = self.args.c_mse * F.mse_loss(i_style, a_style)

        loss_spk_distill = self.args.c_spk_distill * (
            1.0 - F.cosine_similarity(i_spk_proj, secs_teacher, dim=1)
        ).mean()

        loss_ortho = self.args.c_ortho * orthogonal_loss(i_style, i_spk)

        loss_dict.update(
            loss_contrastive=loss_contrast,
            loss_mse_style=loss_mse_style,
            loss_spk_distill=loss_spk_distill,
            loss_orthogonal=loss_ortho,
        )
        cond_lat = a_lat.clone()    
        speaker_embedding = a_spk        
                
        B, T, C = cond_lat.shape
        style_emb = self.xtts.dialect_embedding(dialect_labels)    

        delta_semi_pred = self.xtts.dialect_pitch_predictor(dialect_labels, T) 

        cond_t = cond_lat.transpose(1, 2)  
        gamma  = self.xtts.film_gamma(style_emb).unsqueeze(-1) 
        beta   = self.xtts.film_beta(style_emb).unsqueeze(-1)  

        pitch_emb = self.xtts.pitch_embedding(delta_semi_pred).transpose(1, 2)  # [B, C, T]
        fused = gamma * cond_t + beta + pitch_emb
        cond_mels = fused.transpose(1, 2)  

        gt_f0 = batch["f0_emb"].to(self.device) 
        T_gt = min(delta_semi_pred.size(1), gt_f0.size(1))


        gt_logf0 = torch.log(gt_f0[:, :T_gt].clamp(min=1e-6)) 

        with torch.no_grad():
            spk_lat = speaker_embedding.squeeze(-1)           
            base_logf0 = self.xtts.spk_to_logf0(spk_lat).squeeze(-1)  
        base_logf0 = base_logf0.unsqueeze(1).expand(-1, T_gt)  

        cond_mels = i_lat if (random.random() < float(self.args.mix_ratio)) else a_lat
        delta_logf0 = gt_logf0 - base_logf0
        delta_semi_gt = delta_logf0 * (12.0 / math.log(2.0))
        loss_pitch = F.mse_loss(delta_semi_pred[:, :T_gt], delta_semi_gt)
        loss_dict["loss_pitch"] = loss_pitch * self.args.pitch_loss_weight


        loss_text, loss_mel, mel_logits = self.forward(
            text_inputs, text_lengths, audio_codes, wav_lengths,
            cond_mels, cond_idxs, cond_lens, dialect_lbls
        )
        loss_dict["loss_text_ce"] = loss_text * self.args.gpt_loss_text_ce_weight
        loss_dict["loss_mel_ce"]  = loss_mel   * self.args.gpt_loss_mel_ce_weight

        total = (
            loss_dict["loss_text_ce"] + loss_dict["loss_mel_ce"]
        + loss_dict["loss_contrastive"] + loss_dict["loss_mse_style"]
        + loss_dict["loss_spk_distill"] + loss_dict["loss_orthogonal"] + loss_dict["loss_pitch"]
        )
        loss_dict["loss"] = total

        return {"model_outputs": None}, loss_dict


    def eval_step(self, batch, criterion):
        # ignore masking for more consistent evaluation
        batch["cond_idxs"] = None
        return self.train_step(batch, criterion)

    def on_train_epoch_start(self, trainer):
        trainer.model.eval()  # the whole model to eval
        # put gpt model in training mode
        if hasattr(trainer.model, "module") and hasattr(trainer.model.module, "xtts"):
            trainer.model.module.xtts.gpt.train()
        else:
            trainer.model.xtts.gpt.train()

    def on_init_end(self, trainer):  # pylint: disable=W0613
        # ignore similarities.pth on clearml save/upload
        if self.config.dashboard_logger.lower() == "clearml":
            from clearml.binding.frameworks import WeightsFileHandler

            WeightsFileHandler.add_pre_callback(callback_clearml_load_save)

    @torch.no_grad()
    def inference(
        self,
        x,
        aux_input=None,
    ):  # pylint: disable=dangerous-default-value
        return None

    @staticmethod
    def get_criterion():
        return None

    def get_sampler(self, dataset: TTSDataset, num_gpus=1):
        # sampler for DDP
        batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        return batch_sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":  # pylint: disable=W0613
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = XTTSDataset(self.config, samples, self.xtts.tokenizer, config.audio.sample_rate, is_eval)

            # wait all the DDP process to be ready
            if num_gpus > 1:
                torch.distributed.barrier()

            # sort input sequences from short to long
            # dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(dataset, num_gpus)

            # ignore sampler when is eval because if we changed the sampler parameter we will not be able to compare previous runs
            if sampler is None or is_eval:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                loader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size = config.eval_batch_size if is_eval else config.batch_size,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
        return loader

    def get_optimizer(self) -> List:
        """Initiate and return the optimizer based on the config parameters."""
        # ToDo: deal with multi GPU training
        if self.config.optimizer_wd_only_on_weights:
            # parameters to only GPT model
            net = self.xtts.gpt

            # normalizations
            norm_modules = (
                nn.BatchNorm2d,
                nn.InstanceNorm2d,
                nn.BatchNorm1d,
                nn.InstanceNorm1d,
                nn.BatchNorm3d,
                nn.InstanceNorm3d,
                nn.GroupNorm,
                nn.LayerNorm,
            )
            # nn.Embedding
            emb_modules = (nn.Embedding, nn.EmbeddingBag)

            param_names_notweights = set()
            all_param_names = set()
            param_map = {}
            for mn, m in net.named_modules():
                for k, v in m.named_parameters():
                    v.is_bias = k.endswith(".bias")
                    v.is_weight = k.endswith(".weight")
                    v.is_norm = isinstance(m, norm_modules)
                    v.is_emb = isinstance(m, emb_modules)

                    fpn = "%s.%s" % (mn, k) if mn else k  # full param name
                    all_param_names.add(fpn)
                    param_map[fpn] = v
                    if v.is_bias or v.is_norm or v.is_emb:
                        param_names_notweights.add(fpn)

            params_names_notweights = sorted(list(param_names_notweights))
            params_notweights = [param_map[k] for k in params_names_notweights]
            params_names_weights = sorted(list(all_param_names ^ param_names_notweights))
            params_weights = [param_map[k] for k in params_names_weights]
            params_weights += [self.dialect_embedding.weight]
            params_weights += list(self.xtts.arcface_encoder.projection.parameters())
            params_weights += list(self.xtts.clip_encoder.projection.parameters())
            params_weights += list(self.xtts.image_spkr_proj.parameters())
            params_weights += list(self.xtts.clip_proj.parameters())
            params_weights += list(self.xtts.style_adapter.parameters())
            params_weights += list(self.xtts.concat_fusion.parameters())
            params_weights += list(self.xtts.image_spkr_head.parameters())
            params_weights += list(self.xtts.secs_align.parameters())
            groups = [
                {"params": params_weights, "weight_decay": self.config.optimizer_params["weight_decay"]},
                {"params": params_notweights, "weight_decay": 0},
            ]
            # torch.optim.AdamW
            opt = get_optimizer(
                self.config.optimizer,
                self.config.optimizer_params,
                self.config.lr,
                parameters=groups,
            )
            opt._group_names = [params_names_weights, params_names_notweights]
            return opt

        return get_optimizer(
            self.config.optimizer,
            self.config.optimizer_params,
            self.config.lr,
            # optimize only for the GPT model
            parameters=self.xtts.gpt.parameters(),
        )

    def get_scheduler(self, optimizer) -> List:
        """Set the scheduler for the optimizer.

        Args:
            optimizer: `torch.optim.Optimizer`.
        """
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)

    def load_checkpoint(
        self,
        config,
        checkpoint_path,
        eval=False,
        strict=True,
        cache_storage="/tmp/tts_cache",
        target_protocol="s3",
        target_options={"anon": True},
    ):  # pylint: disable=unused-argument, disable=W0201, disable=W0102, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""

        state = self.xtts.get_compatible_checkpoint_state_dict(checkpoint_path)

        # load the model weights
        self.xtts.load_state_dict(state, strict=strict)

        if eval:
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.eval()
            assert not self.training

    @staticmethod
    def init_from_config(config: "GPTTrainerConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (GPTTrainerConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        return GPTTrainer(config)
