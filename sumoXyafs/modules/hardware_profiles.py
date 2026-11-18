"""Hardware specifications and profiles for Edge, OBU, and Cloud computing nodes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwareSpec:
    name: str
    device_type: str  # "obu", "rsu_mid", "rsu_high", "cloud"
    gflops: float  # Processing capacity in GFLOPs/sec (1 TFLOPS = 1000 GFLOPS)
    ram_mb: int  # RAM memory in Megabytes
    max_buffer_tasks: int  # Maximum concurrent queued tasks

    @property
    def gflops_per_ms(self) -> float:
        return self.gflops / 1000.0


# Pre-configured realistic hardware profiles
JETSON_NANO = HardwareSpec(
    name="NVIDIA Jetson Nano",
    device_type="obu",
    gflops=472.0,  # ~0.47 TFLOPS
    ram_mb=4096,
    max_buffer_tasks=2,
)

SIEMENS_RS22X = HardwareSpec(
    name="Siemens Sitraffic RS22X",
    device_type="rsu_mid",
    gflops=3200.0,  # ~3.2 TFLOPS (32 TOPS INT8)
    ram_mb=16384,
    max_buffer_tasks=15,
)

JETSON_AGX_ORIN = HardwareSpec(
    name="NVIDIA Jetson AGX Orin",
    device_type="rsu_high",
    gflops=20000.0,  # ~20 TFLOPS FP32 (275 TOPS INT8)
    ram_mb=65536,
    max_buffer_tasks=50,
)

CLOUD_DATACENTER = HardwareSpec(
    name="Cloud Data Center Cluster",
    device_type="cloud",
    gflops=1000000.0,  # 1 PFLOPS
    ram_mb=524288,
    max_buffer_tasks=10000,
)

HARDWARE_PROFILES: dict[str, HardwareSpec] = {
    "jetson_nano": JETSON_NANO,
    "obu_default": JETSON_NANO,
    "siemens_rs22x": SIEMENS_RS22X,
    "rsu_mid": SIEMENS_RS22X,
    "jetson_agx_orin": JETSON_AGX_ORIN,
    "rsu_high": JETSON_AGX_ORIN,
    "cloud": CLOUD_DATACENTER,
    "cloud_datacenter": CLOUD_DATACENTER,
}


def get_hardware_spec(profile_key: str, default: HardwareSpec = SIEMENS_RS22X) -> HardwareSpec:
    key = str(profile_key).strip().lower()
    return HARDWARE_PROFILES.get(key, default)
