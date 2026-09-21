"""Empirical Resource Profiles and Subsystem Budget Allocator.

Sections 12 & 13:
Derives profile dynamically from measurements:
- MINIMAL: <= 8GB RAM, CPU-only
- LOW: 8-16GB RAM, integrated GPU or low VRAM
- BALANCED: 16-32GB RAM, integrated GPU (e.g. AMD 680M) or >= 6GB dedicated VRAM
- PERFORMANCE: >= 32GB RAM, >= 8GB dedicated VRAM
- MAXIMUM: >= 64GB RAM, >= 16GB dedicated VRAM

Computes bounded budgets with configurable safety margins:
- os_reserve
- renderer_reserve
- physics_reserve
- brain_reserve
- interactive_ai_reserve
- background_ai_reserve
- model_cache
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any


class ProfileTier(str, Enum):
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    BALANCED = "BALANCED"
    PERFORMANCE = "PERFORMANCE"
    MAXIMUM = "MAXIMUM"


@dataclass
class SubsystemBudgets:
    os_reserve_mb: float
    renderer_reserve_mb: float
    physics_reserve_mb: float
    brain_reserve_mb: float
    interactive_ai_reserve_mb: float
    background_ai_reserve_mb: float
    model_cache_mb: float
    vram_budget_mb: float
    safe_ram_target_mb: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "os_reserve_mb": self.os_reserve_mb,
            "renderer_reserve_mb": self.renderer_reserve_mb,
            "physics_reserve_mb": self.physics_reserve_mb,
            "brain_reserve_mb": self.brain_reserve_mb,
            "interactive_ai_reserve_mb": self.interactive_ai_reserve_mb,
            "background_ai_reserve_mb": self.background_ai_reserve_mb,
            "model_cache_mb": self.model_cache_mb,
            "vram_budget_mb": self.vram_budget_mb,
            "safe_ram_target_mb": self.safe_ram_target_mb
        }


def derive_profile(probe: Dict[str, Any]) -> ProfileTier:
    """Derives empirical profile tier based on real probed hardware."""
    ram_gb = probe["memory"]["ram_total_gb"]
    gpu = probe["gpu"]
    has_gpu = gpu.get("vulkan_available", False)
    vram_mb = gpu.get("dedicated_vram_mb", 0.0) + gpu.get("shared_memory_mb", 0.0) * 0.25

    if ram_gb <= 8.5:
        return ProfileTier.MINIMAL
    elif ram_gb <= 16.5:
        return ProfileTier.LOW
    elif ram_gb <= 32.5:
        return ProfileTier.BALANCED
    elif ram_gb <= 64.5 and has_gpu and vram_mb >= 6000:
        return ProfileTier.PERFORMANCE
    elif ram_gb > 64.5 and has_gpu and vram_mb >= 12000:
        return ProfileTier.MAXIMUM
    else:
        return ProfileTier.BALANCED


def compute_budgets(probe: Dict[str, Any], tier: ProfileTier) -> SubsystemBudgets:
    """Computes bounded memory budgets with mandatory safety margins."""
    ram_total_mb = probe["memory"]["ram_total_bytes"] / (1024 ** 2)
    gpu = probe["gpu"]
    has_vulkan = gpu.get("vulkan_available", False)
    dedicated_vram = gpu.get("dedicated_vram_mb", 0.0)

    # 15% OS safety reserve minimum
    os_reserve = max(1024.0, ram_total_mb * 0.15)
    usable_ram = max(512.0, ram_total_mb - os_reserve)

    if tier == ProfileTier.MINIMAL:
        # Strict low-memory allocation
        renderer = 256.0
        physics = 128.0
        brain = 256.0
        interactive_ai = 512.0
        bg_ai = 0.0
        model_cache = 1024.0
        vram = 0.0 if not has_vulkan else 512.0
    elif tier == ProfileTier.LOW:
        renderer = 512.0
        physics = 256.0
        brain = 512.0
        interactive_ai = 1536.0
        bg_ai = 512.0
        model_cache = 2048.0
        vram = dedicated_vram if has_vulkan else 0.0
    elif tier == ProfileTier.BALANCED:
        # Standard desktop / AMD 680M workstation configuration
        renderer = 1024.0
        physics = 512.0
        brain = 1024.0
        interactive_ai = 2560.0
        bg_ai = 1536.0
        model_cache = 4096.0
        vram = max(dedicated_vram, 2048.0) if has_vulkan else 0.0
    elif tier == ProfileTier.PERFORMANCE:
        renderer = 2048.0
        physics = 1024.0
        brain = 2048.0
        interactive_ai = 4096.0
        bg_ai = 3072.0
        model_cache = 8192.0
        vram = dedicated_vram if has_vulkan else 2048.0
    else:  # MAXIMUM
        renderer = 4096.0
        physics = 2048.0
        brain = 4096.0
        interactive_ai = 8192.0
        bg_ai = 6144.0
        model_cache = 16384.0
        vram = dedicated_vram if has_vulkan else 4096.0

    safe_ram_target = usable_ram * 0.85

    return SubsystemBudgets(
        os_reserve_mb=round(os_reserve, 1),
        renderer_reserve_mb=round(renderer, 1),
        physics_reserve_mb=round(physics, 1),
        brain_reserve_mb=round(brain, 1),
        interactive_ai_reserve_mb=round(interactive_ai, 1),
        background_ai_reserve_mb=round(bg_ai, 1),
        model_cache_mb=round(model_cache, 1),
        vram_budget_mb=round(vram, 1),
        safe_ram_target_mb=round(safe_ram_target, 1)
    )
