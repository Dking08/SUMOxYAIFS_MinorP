"""Unit tests for hardware_profiles module."""
import unittest

from modules.hardware_profiles import (
    JETSON_AGX_ORIN,
    JETSON_NANO,
    SIEMENS_RS22X,
    CLOUD_DATACENTER,
    get_hardware_spec,
)


class TestHardwareProfiles(unittest.TestCase):
    def test_predefined_profiles(self):
        self.assertEqual(JETSON_NANO.device_type, "obu")
        self.assertEqual(SIEMENS_RS22X.ram_mb, 16384)
        self.assertGreater(JETSON_AGX_ORIN.gflops, SIEMENS_RS22X.gflops)
        self.assertGreater(CLOUD_DATACENTER.gflops, JETSON_AGX_ORIN.gflops)

    def test_get_hardware_spec_lookup(self):
        spec = get_hardware_spec("jetson_agx_orin")
        self.assertEqual(spec.name, "NVIDIA Jetson AGX Orin")

        spec_alias = get_hardware_spec("rsu_mid")
        self.assertEqual(spec_alias.name, "Siemens Sitraffic RS22X")

        spec_default = get_hardware_spec("unknown_device")
        self.assertEqual(spec_default.name, "Siemens Sitraffic RS22X")


if __name__ == "__main__":
    unittest.main()
