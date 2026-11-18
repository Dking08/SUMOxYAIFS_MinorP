"""Unit tests for wireless_channel module."""
import unittest

from modules.wireless_channel import (
    C_V2X_PC5,
    DSRC_80211P,
    NR_V2X_5G,
    calc_inter_rsu_latency_s,
    calc_path_loss_db,
    calc_shannon_rate_bps,
    calc_sinr_linear,
    calc_transmission_time_s,
    get_v2x_protocol,
)


class TestWirelessChannel(unittest.TestCase):
    def test_path_loss_growth_with_distance(self):
        pl_10m = calc_path_loss_db(10.0, freq_ghz=5.9, path_loss_exp=2.4)
        pl_100m = calc_path_loss_db(100.0, freq_ghz=5.9, path_loss_exp=2.4)
        self.assertGreater(pl_100m, pl_10m)
        # Expected difference for 10x distance with n=2.4 is 10 * 2.4 * log10(10) = 24 dB
        self.assertAlmostEqual(pl_100m - pl_10m, 24.0, delta=0.5)

    def test_sinr_and_shannon_rate(self):
        sinr_close = calc_sinr_linear(10.0, protocol=C_V2X_PC5)
        sinr_far = calc_sinr_linear(500.0, protocol=C_V2X_PC5)
        self.assertGreater(sinr_close, sinr_far)

        rate_close = calc_shannon_rate_bps(10.0, protocol=C_V2X_PC5)
        rate_far = calc_shannon_rate_bps(500.0, protocol=C_V2X_PC5)
        self.assertGreater(rate_close, rate_far)

    def test_transmission_time(self):
        # 1 Mbit payload at close distance vs far distance
        t_close = calc_transmission_time_s(10.0, payload_mbits=1.0, protocol=C_V2X_PC5)
        t_far = calc_transmission_time_s(500.0, payload_mbits=1.0, protocol=C_V2X_PC5)
        self.assertGreater(t_far, t_close)

    def test_inter_rsu_latency(self):
        t_f2f = calc_inter_rsu_latency_s(payload_mbits=1.0, hops=2, bandwidth_gbps=1.0)
        self.assertGreater(t_f2f, 0.002)  # Should be > 2ms due to propagation delay

    def test_protocol_lookup(self):
        proto = get_v2x_protocol("5g")
        self.assertEqual(proto.name, "5G_NR_V2X")


if __name__ == "__main__":
    unittest.main()
