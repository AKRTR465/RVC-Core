import logging
import os

logger = logging.getLogger(__name__)

import numpy as np

from src.infer.batch import resolve_batch_input_paths, save_converted_audio
from src.infer.service import VoiceConversionService
from src.utils.audio import clean_path, load_audio


class VC(VoiceConversionService):
    def _protect_updates(self, *to_return_protect):
        has_f0 = self.if_f0 == 1
        return (
            {
                "visible": has_f0,
                "value": to_return_protect[0] if has_f0 and to_return_protect else 0.5,
                "__type__": "update",
            },
            {
                "visible": has_f0,
                "value": to_return_protect[1] if has_f0 and to_return_protect else 0.33,
                "__type__": "update",
            },
        )

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        if sid == "" or sid == []:
            if self.has_loaded_state():
                logger.info("Clean model cache")
                self.clear_model_cache()
            to_return_protect0, to_return_protect1 = self._protect_updates(
                *to_return_protect
            )
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        n_spk, index_path = self.load_model(sid)
        to_return_protect0, to_return_protect1 = self._protect_updates(
            *to_return_protect
        )
        index = {
            "value": index_path,
            "__type__": "update",
        }

        return (
            (
                {"visible": True, "maximum": max(n_spk - 1, 0), "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": max(n_spk - 1, 0), "__type__": "update"}
        )

    def vc_single(
        self,
        sid,
        input_audio_path,
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
        if input_audio_path is None:
            return "You need to upload an audio", None
        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            tgt_sr, audio_opt, file_index, times = self.convert_audio(
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
            )
            index_info = (
                "Index:\n%s." % file_index
                if os.path.exists(file_index)
                else "Index not used."
            )
            return (
                "Success.\n%s\nTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs."
                % (index_info, *times),
                (tgt_sr, audio_opt),
            )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("Single conversion failed")
            return f"Failed: {exc}", (None, None)

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = clean_path(dir_path)
            opt_root = clean_path(opt_root)
            os.makedirs(opt_root, exist_ok=True)
            try:
                paths = resolve_batch_input_paths(dir_path, paths)
            except (OSError, TypeError, AttributeError) as exc:
                logger.warning(
                    "Failed to enumerate input directory %s: %s", dir_path, exc
                )
                paths = resolve_batch_input_paths("", paths)
            infos = []
            if not paths:
                yield ""
                return
            for path in paths:
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        save_converted_audio(opt_root, path, audio_opt, tgt_sr, format1)
                    except (OSError, RuntimeError, ValueError, TypeError) as exc:
                        logger.exception("Failed to save converted file %s", path)
                        info += f"\nFailed to save output: {exc}"
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            logger.exception("Batch conversion failed")
            yield f"Failed: {exc}"

