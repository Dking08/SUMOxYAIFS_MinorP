"""Wireless channel physical layer modeling for V2I, V2V, Inter-RSU, and Cloud links."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class V2XProtocol:
    name: str
    freq_ghz: float = 5.9  # Carrier frequency in GHz
    tx_power_dbm: float = 23.0  # Transmit power in dBm (23 dBm = 0.2 W)
    bandwidth_hz: float = 20e6  # Channel bandwidth in Hz (20 MHz default)
    path_loss_exp: float = 2.4  # Urban micro path loss exponent
    antenna_gain_dbi: float = 3.0  # Combined Tx/Rx antenna gain (dBi)
    coding_efficiency: float = 0.75  # PHY/MAC FEC efficiency factor


# Pre-defined physical layer protocols
DSRC_80211P = V2XProtocol(
    name="DSRC_80211p",
    freq_ghz=5.9,
    tx_power_dbm=23.0,
    bandwidth_hz=10e6,  # 10 MHz
    path_loss_exp=2.7,
)

C_V2X_PC5 = V2XProtocol(
    name="C-V2X_PC5",
    freq_ghz=5.9,
    tx_power_dbm=23.0,
    bandwidth_hz=20e6,  # 20 MHz
    path_loss_exp=2.4,
)

NR_V2X_5G = V2XProtocol(
    name="5G_NR_V2X",
    freq_ghz=5.9,
    tx_power_dbm=30.0,  # 1 Watt
    bandwidth_hz=100e6,  # 100 MHz
    path_loss_exp=2.2,
)

V2V_P2P = V2XProtocol(
    name="V2V_P2P",
    freq_ghz=5.9,
    tx_power_dbm=20.0,  # 100 mW
    bandwidth_hz=10e6,
    path_loss_exp=2.8,
)

PROTOCOLS: dict[str, V2XProtocol] = {
    "dsrc": DSRC_80211P,
    "c-v2x": C_V2X_PC5,
    "c_v2x": C_V2X_PC5,
    "5g_nr_v2x": NR_V2X_5G,
    "5g": NR_V2X_5G,
    "v2v": V2V_P2P,
}


def get_v2x_protocol(key: str, default: V2XProtocol = C_V2X_PC5) -> V2XProtocol:
    name = str(key).strip().lower()
    return PROTOCOLS.get(name, default)


def calc_path_loss_db(
    distance_m: float,
    freq_ghz: float = 5.9,
    path_loss_exp: float = 2.4,
) -> float:
    """Log-distance path loss model: PL(d) = 32.4 + 20*log10(fc) + 10*n*log10(d)."""
    d = max(1.0, float(distance_m))
    fc = max(0.1, float(freq_ghz))
    n = max(1.0, float(path_loss_exp))
    return 32.4 + (20.0 * math.log10(fc)) + (10.0 * n * math.log10(d))


def calc_sinr_linear(
    distance_m: float,
    protocol: V2XProtocol = C_V2X_PC5,
    interference_w: float = 0.0,
) -> float:
    """Calculates linear SINR considering distance, transmit power, noise, and interference."""
    pl_db = calc_path_loss_db(distance_m, freq_ghz=protocol.freq_ghz, path_loss_exp=protocol.path_loss_exp)
    pl_linear = 10.0 ** (pl_db / 10.0)

    # Convert transmit power from dBm to Watts (e.g. 23 dBm -> 0.2 W)
    tx_gain_linear = 10.0 ** ((protocol.tx_power_dbm + (2.0 * protocol.antenna_gain_dbi)) / 10.0) / 1000.0
    rx_power_w = tx_gain_linear / max(1.0, pl_linear)

    # Thermal noise spectral density N0 = -174 dBm/Hz = 3.98e-21 W/Hz
    n0_w_per_hz = 10.0 ** ((-174.0 - 30.0) / 10.0)
    thermal_noise_w = n0_w_per_hz * protocol.bandwidth_hz
    total_noise_plus_interference = thermal_noise_w + max(0.0, float(interference_w))

    return rx_power_w / max(1e-25, total_noise_plus_interference)


def calc_shannon_rate_bps(
    distance_m: float,
    protocol: V2XProtocol = C_V2X_PC5,
    interference_w: float = 0.0,
) -> float:
    """Shannon capacity R = eta * B * log2(1 + SINR)."""
    sinr = calc_sinr_linear(distance_m, protocol=protocol, interference_w=interference_w)
    rate = protocol.coding_efficiency * protocol.bandwidth_hz * math.log2(1.0 + sinr)
    return max(100.0, rate)  # Minimum fallback rate of 100 bps


def calc_transmission_time_s(
    distance_m: float,
    payload_mbits: float,
    protocol: V2XProtocol = C_V2X_PC5,
    interference_w: float = 0.0,
) -> float:
    """Calculates physical transmission latency for a given payload (in Mbits)."""
    rate_bps = calc_shannon_rate_bps(distance_m, protocol=protocol, interference_w=interference_w)
    bits = max(0.0, float(payload_mbits)) * 1e6
    return bits / rate_bps


def calc_inter_rsu_latency_s(
    payload_mbits: float,
    hops: int = 1,
    bandwidth_gbps: float = 1.0,
    prop_delay_per_hop_s: float = 0.001,
) -> float:
    """Fiber / mmWave backhaul latency between RSUs."""
    bits = max(0.0, float(payload_mbits)) * 1e6
    bw_bps = max(1e6, float(bandwidth_gbps) * 1e9)
    trans_time = bits / bw_bps
    prop_time = max(1, int(hops)) * max(0.0, float(prop_delay_per_hop_s))
    return trans_time + prop_time


def calc_cloud_latency_s(
    payload_mbits: float,
    wan_rtt_s: float = 0.05,
    wan_bandwidth_mbps: float = 100.0,
) -> float:
    """WAN latency for sending tasks to Cloud Data Centers."""
    bits = max(0.0, float(payload_mbits)) * 1e6
    bw_bps = max(1e3, float(wan_bandwidth_mbps) * 1e6)
    trans_time = bits / bw_bps
    return (float(wan_rtt_s) / 2.0) + trans_time
