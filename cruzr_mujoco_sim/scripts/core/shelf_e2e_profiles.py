#!/usr/bin/env python3
"""Versioned collection-camera profiles for the dual-material task."""

from __future__ import annotations

from cruzr_s2_sdk_contract import (
    SDK_COLLECTION_PROFILE,
    SDK_POLICY_IMAGE_MAP,
)
from shelf_e2e_contract import POLICY_IMAGE_MAP


STRICT_COLLECTION_PROFILE = "strict_v1"

_PROFILE_IMAGE_MAPS = {
    STRICT_COLLECTION_PROFILE: dict(POLICY_IMAGE_MAP),
    SDK_COLLECTION_PROFILE: dict(SDK_POLICY_IMAGE_MAP),
}


def normalize_collection_profile(value: str | None) -> str:
    """Map missing legacy metadata to strict_v1 and reject unknown profiles."""
    profile = (value or STRICT_COLLECTION_PROFILE).strip()
    if profile not in _PROFILE_IMAGE_MAPS:
        raise ValueError(
            f"unsupported collection profile {profile!r}; expected one of "
            f"{sorted(_PROFILE_IMAGE_MAPS)}"
        )
    return profile


def policy_image_map(profile: str | None) -> dict[str, str]:
    profile = normalize_collection_profile(profile)
    return dict(_PROFILE_IMAGE_MAPS[profile])


def collection_cameras(profile: str | None) -> tuple[str, ...]:
    return tuple(
        value.rsplit(".", 1)[-1]
        for value in policy_image_map(profile).values()
    )


def supported_collection_profiles() -> tuple[str, ...]:
    return tuple(_PROFILE_IMAGE_MAPS)
