from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class FeatureProfile:
    version: str
    feature_dir_name: str
    feature_dim: int
    index_uses_n_cpu: bool
    index_build_kwargs: MappingProxyType


_VERSION_PROFILES = {
    "v1": FeatureProfile(
        version="v1",
        feature_dir_name="3_feature256",
        feature_dim=256,
        index_uses_n_cpu=False,
        index_build_kwargs=MappingProxyType({"nprobe": 9}),
    ),
    "v2": FeatureProfile(
        version="v2",
        feature_dir_name="3_feature768",
        feature_dim=768,
        index_uses_n_cpu=True,
        index_build_kwargs=MappingProxyType(
            {
                "shuffle": True,
                "reduce_large": True,
                "nprobe": 1,
                "add_batch_size": 8192,
            }
        ),
    ),
}
_DIMENSION_PROFILES = {
    profile.feature_dim: profile for profile in _VERSION_PROFILES.values()
}


def get_feature_profile(version: str) -> FeatureProfile:
    key = str(version).strip().lower()
    try:
        return _VERSION_PROFILES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported RVC version: {version!r}. Expected one of: {', '.join(sorted(_VERSION_PROFILES))}"
        ) from exc


def get_feature_profile_by_dim(feature_dim: int) -> FeatureProfile:
    try:
        resolved_dim = int(feature_dim)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature_dim must be an integer, got: {feature_dim!r}") from exc
    try:
        return _DIMENSION_PROFILES[resolved_dim]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported feature_dim: {feature_dim!r}. Expected one of: {', '.join(map(str, sorted(_DIMENSION_PROFILES)))}"
        ) from exc
