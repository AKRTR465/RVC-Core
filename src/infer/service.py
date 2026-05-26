import logging
import os

import torch

from src.infer.model_utils import (
    get_index_path_from_model,
    get_model_path_from_sid,
    load_hubert,
)
from src.infer.pipeline import Pipeline
from src.models.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from src.utils.audio import clean_path

logger = logging.getLogger(__name__)


class VoiceConversionService:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.hubert_model = None
        self.config = config

    def clear_model_cache(self):
        for name in ("net_g", "n_spk", "hubert_model", "tgt_sr", "pipeline", "cpt"):
            setattr(self, name, None)
        self.version = None
        self.if_f0 = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def has_loaded_state(self):
        return any(
            getattr(self, name, None) is not None
            for name in ("hubert_model", "net_g", "cpt", "pipeline")
        )

    def resolve_synthesizer_class(self):
        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }.get((self.version, self.if_f0))
        if synthesizer_class is None:
            raise ValueError(
                f"Unsupported checkpoint metadata: version={self.version}, f0={self.if_f0}"
            )
        return synthesizer_class

    def load_model(self, sid):
        person = get_model_path_from_sid(sid, self.config.ckpt_root)
        if person == "":
            raise FileNotFoundError(f"Model not found under {self.config.ckpt_root}: {sid}")
        logger.info("Loading: %s", person)

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        self.net_g = self.resolve_synthesizer_class()(
            *self.cpt["config"], is_half=self.config.is_half
        )
        del self.net_g.enc_q
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        self.net_g = self.net_g.half() if self.config.is_half else self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        self.n_spk = self.cpt["config"][-3]
        index_path = get_index_path_from_model(sid, self.config.ckpt_root)
        logger.info("Select index: %s", index_path)
        return self.n_spk, index_path

    def resolve_index_path(self, file_index, file_index2):
        if file_index:
            index_path = os.path.abspath(clean_path(file_index))
            filename = os.path.basename(index_path)
            if filename.startswith("trained_"):
                filename = f"added_{filename[len('trained_'):]}"
            return os.path.join(
                os.path.dirname(index_path),
                filename,
            )
        if file_index2:
            return file_index2
        return ""

    def convert_audio(
        self,
        sid,
        audio,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        if self.hubert_model is None:
            self.hubert_model = load_hubert(self.config)
        file_index = self.resolve_index_path(file_index, file_index2)
        times = [0, 0, 0]
        audio_opt = self.pipeline.pipeline(
            self.hubert_model,
            self.net_g,
            sid,
            audio,
            times,
            int(f0_up_key),
            f0_method,
            file_index,
            index_rate,
            self.if_f0,
            filter_radius,
            self.tgt_sr,
            resample_sr,
            rms_mix_rate,
            self.version,
            protect,
            f0_file,
        )
        tgt_sr = resample_sr if self.tgt_sr != resample_sr >= 16000 else self.tgt_sr
        return tgt_sr, audio_opt, file_index, times
