import contextlib
import gc
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
from nms import ssc

__all__ = [
    "ImageMatcher",
]


@dataclass
class LAFFeatures:
    lafs: torch.Tensor  # 1 x num kpts x 2 x 3
    resps: torch.Tensor  # 1 x num kpts
    descs: torch.Tensor  # 1 x num kpts x 128

    def __len__(self):
        return self.lafs.shape[1]

    @cached_property
    def cv2_kpts(
        self,
    ) -> List[cv2.KeyPoint]:
        mkpts = kornia.feature.get_laf_center(self.lafs).squeeze()  # num kpts x 2
        scales = kornia.feature.get_laf_scale(self.lafs).flatten()  # num kpts
        orientations = kornia.feature.get_laf_orientation(self.lafs).flatten()  # num kpts
        responses = self.resps.flatten()  # num kpts

        cv2_kpts = []
        for (x, y), scale, orientation, response in zip(mkpts, scales, orientations, responses):
            cv2_kpts.append(
                cv2.KeyPoint(
                    x=float(x),
                    y=float(y),
                    _size=float(scale),
                    _angle=float(orientation),
                    _response=float(response),
                )
            )

        return cv2_kpts

    def extract_indices(self, _idxs):
        return LAFFeatures(
            self.lafs[:, _idxs, :, :],
            self.resps[:, _idxs],
            self.descs[:, _idxs, :],
        )

    def to_cpu(self):
        return LAFFeatures(
            self.lafs.detach().cpu(),
            self.resps.detach().cpu(),
            self.descs.detach().cpu(),
        )


class ImageMatcher:
    __DEFAULT_CONFIG: Dict[str, Any] = {
        "num_features": 4000,
        "use_nms": True,
        "grid_size": 30,
        "max_nms_matches": 200,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device_str: str = "cpu",
    ):
        self._config = self.__DEFAULT_CONFIG.copy()
        if config is not None:
            self._config.update(config)

        if "cuda" in device_str and not torch.cuda.is_available():
            device_str = "cpu"
        self._device = torch.device(device_str)

        with contextlib.redirect_stdout(None):
            self._light_glue_matcher = (
                kornia.feature.LightGlueMatcher(
                    "keynet_affnet_hardnet",
                )
                .to(self._device)
                .eval()
            )

        self._keypoint_detector = kornia.feature.KeyNetAffNetHardNet(
            num_features=self._config["num_features"],
            device=self._device,
        ).eval()

    @torch.no_grad()
    def run(
        self,
        query_image: np.ndarray,
        ref_image: np.ndarray,
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint]]:
        query_f = self.extract_features(query_image)
        ref_f = self.extract_features(ref_image)

        _, idxs = self._light_glue_matcher(
            query_f.descs.squeeze(0),
            ref_f.descs.squeeze(0),
            query_f.lafs,
            ref_f.lafs,
        )

        if len(idxs) == 0:
            return [], []

        query_f = query_f.extract_indices(idxs[:, 0]).to_cpu()
        ref_f = ref_f.extract_indices(idxs[:, 1]).to_cpu()

        query_kpts = query_f.cv2_kpts
        ref_kpts = ref_f.cv2_kpts
        del query_f, ref_f

        if self._config["use_nms"]:
            height, width = query_image.shape
            indices_to_extract = ssc(
                query_kpts,
                max(1, min(self._config["max_nms_matches"], height * width / self._config["grid_size"] ** 2)),
                width,
                height,
            )
            query_kpts = [query_kpts[idx] for idx in indices_to_extract]
            ref_kpts = [ref_kpts[idx] for idx in indices_to_extract]

        return query_kpts, ref_kpts

    @torch.no_grad()
    def extract_features(
        self,
        image: np.ndarray,
    ) -> LAFFeatures:
        # lafs: 1 x num kpts x 2 x 3
        # resps: 1 x num kpts
        # descs: 1 x num kpts x 128

        return LAFFeatures(
            *self._keypoint_detector(
                torch.from_numpy(image[None, None, ...]).to(self._device).float() / 255.0,
            )
        )

    def __del__(self):
        del self._light_glue_matcher
        del self._keypoint_detector
        gc.collect()
        torch.cuda.empty_cache()
