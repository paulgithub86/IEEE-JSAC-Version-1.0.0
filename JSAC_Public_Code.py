# -*- coding: utf-8 -*-
"""
----------------------------------

Author: Paul & Singh

IEEE-JSAC companion (manuscript-Sec.V.B, with QKD)
-------------------------------------------------------------------------------

Faithful simulated prototype implementation of the proposed framework:

(1) System Model (V2X clusterized ISAC network)
    - C clusters, each with multiple vehicles, UAVs, and 1 RSU.
    - Monostatic radar sensing per vehicle with ISAC transmit power split:
          α_i^s(t) + α_i^c(t) = 1.
    - Radar sensing quality Π_i(t) from radar SINR.
    - Finite-blocklength uplink (V->U) and front-haul (U->R) with interference.
    - Task offloading chain with:
          T_{i->U_i}^{UL}, T_{i,u}^{UAV}, T_{i,u}^{FH}, T_{i,u}^{RSU},
          T_i^tot(t) = UL + UAV + FH + RSU.

(2) Data generation:
    - S_i(t) ~ Poisson(5×10^3),
    - ζ = b_q c_c d_m = 8 × 0.5 × 1.0 = 4 bits/measurement,
    - Π_th = 1, Q_i(t) = min(1, Π_i(t)/Π_th),
    - D_i(t+1) = ζ S_i(t) Q_i(t).

(3) Hybrid actor (classical Transformer-style encoder + quantum VQC head).

(4) Local A2C-style agent updates with GAE and entropy regularization.

(5) Per-episode DP accountant for privacy budget ε(t), with Gaussian mechanism.

(6) QKD on V->U and U->R:
    - Decoy-state BB84 free-space model -> secure key rate R_sec(d),
    - Per-slot key generation,
    - Data-plane OTP: when sufficient key bits are available, the encrypted
      throughput is capped by the key buffer; otherwise, the simulator falls
      back to sending the remaining bits without OTP encryption (no key
      consumption for that portion).

(7) RSU-side FedAvg encoder sync across vehicle agents with AdaGrad-style
    preconditioning on the global encoder weights.

(8) Outputs:
    - convergence.csv (episode, mean_return, epsilon, mean_latency_ms),
    - privacy_budget.csv (episode, epsilon),
    - latency_traces.csv (per-step end-to-end latency per vehicle),
    - convergence_curve.png,
    - privacy_budget_curve.png,
    - tradeoff.xlsx (α_c vs sensing SINR & V->U SE),
    - tradeoff.png (ISAC sensing–throughput tradeoff curve),
    - ul_sinr_traces.csv (per-vehicle V->U SINR debug),
    - fh_sinr_traces.csv (per-UAV U->R SINR debug),
    - sensing_traces.csv (per-vehicle radar SINR Π_i(t) in linear & dB).  #

NOTE:
    This is a faithful system-level prototype: RF/optical-level details
    are mapped to tractable closed forms; remaining numerical constants can
    be tuned to our final JSAC manuscript values.
"""

import math
import random
from typing import Dict
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import sympy
import cirq
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 0) GLOBAL CONSTANTS (aligned with typical 6G V2X + ISAC setup)
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Topology
CLUSTERS = 4
VEH_PER_CL = 4
UAV_PER_CL = 3
RSU_PER_CL = 1
N_VEH = CLUSTERS * VEH_PER_CL
N_UAV = CLUSTERS * UAV_PER_CL
N_RSU = CLUSTERS * RSU_PER_CL   # = CLUSTERS (1 RSU/cluster)

# RF PHY (for RL simulator) – 20 MHz
B = 20e6                        # 20 MHz bandwidth
fc = 2.4e9                      # 2.4 GHz carrier
lambda_rf = 3e8 / fc            # RF wavelength
tau = 20e-3                     # slot 20 ms
n_blk = int(B * tau)            # finite blocklength

# Noise (for RL simulator)
kB = 1.38e-23
T0 = 290.0
NF_DB = 3.0  # reduced NF to slightly improve SNR
NF_LIN = 10 ** (NF_DB / 10.0)
sigma_t2 = kB * T0 * B * NF_LIN  # thermal noise power

# Power budgets (W)
P_VEH_MAX = 10.0                # vehicles: 15 W max (slightly higher power)
P_UAV_MAX = 10.0                # UAVs:    15 W max (slightly higher power)

# Computation budgets
kappa = 200.0                        # cycles per bit
F_R_MAX = 1e10                       # 10 Gcycles/s

# RL / model parameters
D_H = 128                            # encoder hidden dim
L_T = 4                              # "transformer" depth (stacked dense)
Q_QUBITS = 8                         # VQC qubits
D_V = 4                              # VQC depth
ROLL = 24                            # steps per episode
EPIS = 250                          # max training episodes (Make it 250 or  >)

LR_ACT = 2e-4
LR_CRT = 2e-4
LR_Q = 1e-3
L2_CLIP = 1.0
PARAM_SHIFT = math.pi / 2.0
PS_SUB = 12                          # parameter-shift subset for speed
ENTROPY_BETA = 1e-3                  # policy entropy weight

# A2C / GAE
GAMMA_DISCOUNT = 0.99
LAMBDA_GAE = 0.95

# DP parameters
DP_DELTA = 1e-5
EPSILON_MAX = 5.0
SIG_DP = math.sqrt(2 * EPIS * math.log(1 / DP_DELTA)) / EPSILON_MAX

# Data generation: bits per measurement
b_q = 8.0
c_c = 0.5
d_m = 50.0
zeta = b_q * c_c * d_m  # = 400 bits/measurement
PI_THRESH = 1.0

# Reward scaling (TIGHTENED AROUND DEADLINE)
REWARD_LAT_SCALE = 5e-3          #  stronger latency weight
lambda_energy = 0.2
lambda_deadline = 2.0            # extra penalty if T_tot_ms > DEADLINE_MS

FBL_EPS = 1e-3

# Shadowing (dB)
SHADOW_STD_DB_VU = 2.0  # slightly milder shadowing
SHADOW_STD_DB_UR = 2.0  # slightly milder shadowing

# VQC sampling parameters
VQC_REPS = 256
VQC_TEMP = 8.0

# QKD warm start
QKD_WARM_S = 0.2

# Radar / ISAC parameters
M_T_RADAR = 4
M_R_RADAR = 4
G_T_RADAR = 0.5
G_R_RADAR = 0.5
SIGMA_RCS = 10.0
ETA_VEH_SI = 0.02  # reduced vehicle SI factor
K_SCATTER = 1  # single dominant scatterer

# V->U large-scale fading model exponents
K0_VU = 1.0
D0_VU = 1.0
GAMMA_P_VU = 1.7  # pathloss exponent

# UAV self-interference factor
ETA_UAV_SI = 0.01  # reduced UAV SI factor

# U->R front-haul antenna gains
G_UAV_TX = 0.1
G_RSU_RX = 0.1

# Other PHY constants
G_VU_TX = 2.0   #  UL Tx gain
G_VU_RX = 2.0   #  UL Rx gain

# Deadlines (design target)
DEADLINE_MS = 250.0              # 250 ms target per vehicle
CLIP_LAT_MS = 250.0              # treat anything beyond as equally bad

# SNR outage thresholds (RELAXED to reduce "pure outage")
SNR_MIN_DB_UL = -30.0  # allow decoding down to -30 dB
SNR_MIN_DB_FH = -30.0  # allow decoding down to -30 dB
SNR_MIN_UL = 10 ** (SNR_MIN_DB_UL / 10.0)
SNR_MIN_FH = 10 ** (SNR_MIN_DB_FH / 10.0)

# Interference scaling (to avoid ultra-severe interference-limited regime)
INTERF_SCALE_UL = 0.1   # logical cross-vehicle interference scale in V->U
INTERF_SCALE_FH = 0.1   # logical cross-UAV interference scale in U->R

# Server-side AdaGrad for FedAvg (encoder)
SERVER_ENCODER_WEIGHTS = None
SERVER_ADAGRAD_ACCUM = None
SERVER_LR = 0.5
SERVER_ADAGRAD_EPS = 1e-8

# ──────────────────────────────────────────────────────────────────────────────
# 1) Privacy Accountant (Gaussian mechanism)
# ──────────────────────────────────────────────────────────────────────────────
class PrivacyAccountant:
    def __init__(self):
        self._moments = 0.0
        self.eps = 0.0

    def accumulate(self):
        # One effective DP round per episode
        self._moments += 1.0 / (SIG_DP ** 2)
        self.eps = math.sqrt(2 * self._moments * math.log(1 / DP_DELTA))

    def current(self):
        return self.eps

# ──────────────────────────────────────────────────────────────────────────────
# 2) PHY helpers (finite blocklength, pathloss, shadowing, etc.)
# ──────────────────────────────────────────────────────────────────────────────
def fbl_spectral_efficiency(snr: float, blk_len: int, eps: float) -> float:
    """Finite-blocklength spectral efficiency (bits/use)."""
    if snr <= 0:
        return 0.0
    v = (1.0 - (1.0 + snr) ** -2) * (math.log2(math.e) ** 2)
    return max(
        0.0,
        math.log2(1.0 + snr) - math.sqrt(v / blk_len) * stats.norm.isf(eps)
    )

def throughput_bits_per_slot(snr: float, B_: float, tau_: float,
                             nblk_: int, eps: float) -> float:
    """Bits in one slot using FBL normal approximation."""
    return B_ * tau_ * fbl_spectral_efficiency(snr, nblk_, eps)

def fspl(d: float, wavelength: float) -> float:
    """Free-space pathloss (linear)."""
    return (4.0 * math.pi * max(1e-3, d) / wavelength) ** 2

def lognorm_shadow(std_db: float) -> float:
    x_db = np.random.normal(0.0, std_db)
    return 10.0 ** (x_db / 10.0)

def rician_power(K_dB: float = 10.0) -> float:
    """ Rician fading power |h|^2 with K-factor in dB """
    K = 10.0 ** (K_dB / 10.0)
    # LoS amplitude
    s = math.sqrt(K / (K + 1.0))
    # Scatter std (I/Q)
    sigma = math.sqrt(1.0 / (2.0 * (K + 1.0)))
    x = np.random.normal(s, sigma)
    y = np.random.normal(0.0, sigma)
    return x * x + y * y


def H2(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def safe_db(x: float) -> float:
    return 10.0 * math.log10(max(1e-12, x))


def ula_steering(M: int, phi_rad: float) -> np.ndarray:
    n = np.arange(M)
    return np.exp(1j * np.pi * n * np.sin(phi_rad)) / np.sqrt(M)

def rand_unit_vec_c(M: int) -> np.ndarray:
    x = (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2.0)
    return x / np.linalg.norm(x)

def monostatic_beta(Gt: float, Gr: float, lam: float, d: float) -> float:
    return Gt * Gr * (lam / (4.0 * math.pi * max(d, 1e-3))) ** 4

def large_scale_vu(d: float) -> float:
    xi = lognorm_shadow(SHADOW_STD_DB_VU)
    return G_VU_TX * G_VU_RX * K0_VU * (D0_VU / max(d, 1.0)) ** GAMMA_P_VU * xi

def large_scale_ur(d: float) -> float:
    return G_UAV_TX * G_RSU_RX * (lambda_rf / (4.0 * math.pi * max(d, 1e-3))) ** 2

# ──────────────────────────────────────────────────────────────────────────────
# 3) QKD link (decoy-state BB84 FSO model)
# ──────────────────────────────────────────────────────────────────────────────
class QKDLink:
    def __init__(self,
                 wavelength_nm: float = 1550.0,
                 pulse_rate: float = 1e9,
                 mu: float = 0.5,
                 eta_det: float = 0.25,
                 p_dark: float = 5e-6,
                 f_ec: float = 1.16,
                 e_det: float = 0.015,
                 G_tx: float = 1e10,
                 G_rx: float = 1e10,
                 eta_sys: float = 1e-3):
        self.lambda_opt = wavelength_nm * 1e-9
        self.R_p = pulse_rate
        self.mu = mu
        self.eta_det = eta_det
        self.p_dark = p_dark
        self.f_ec = f_ec
        self.e_det = e_det
        self.q = 0.5
        self.G_tx = G_tx
        self.G_rx = G_rx
        self.eta_sys = eta_sys
        self.key_buffer_bits = 0.0

    def channel_eta(self, d_m: float) -> float:
        L_fs = fspl(d_m, self.lambda_opt)
        eta_ch = (self.G_tx * self.G_rx) / L_fs
        eta_tot = eta_ch * self.eta_sys * self.eta_det
        return float(np.clip(eta_tot, 0.0, 1.0))

    def secure_key_rate(self, d_m: float) -> float:
        eta = self.channel_eta(d_m)
        Y = self.mu * eta + self.p_dark
        if Y <= 0.0:
            return 0.0
        Q = (self.e_det * self.mu * eta + 0.5 * self.p_dark) / Y
        term = 1.0 - self.f_ec * H2(Q)
        R = self.q * self.R_p * Y * max(0.0, term)
        return max(0.0, R)

    def generate_for_slot(self, d_m: float, tau_s: float) -> float:
        r = self.secure_key_rate(d_m)
        gen = r * tau_s
        self.key_buffer_bits += gen
        return gen

    def consume_bits(self, need_bits: float) -> float:
        use = min(need_bits, self.key_buffer_bits)
        self.key_buffer_bits -= use
        return use

# ──────────────────────────────────────────────────────────────────────────────
# 4) Hybrid policy network: Transformer encoder + Cirq VQC head
# ──────────────────────────────────────────────────────────────────────────────
def build_ctx_encoder() -> keras.Model:
    inp = keras.Input(shape=(D_H,))
    x = inp
    for _ in range(L_T):
        x = layers.Dense(D_H, activation="gelu")(x)
    return keras.Model(inp, x, name="ctx_encoder")


# Build VQC circuit
PHI_SYM = []
VAR_SYM = []
qs = cirq.LineQubit.range(Q_QUBITS)
vqc_circ = cirq.Circuit()

# Angle encoding
for i, qubit in enumerate(qs):
    phi = sympy.Symbol(f"phi_enc_{i}")
    PHI_SYM.append(phi)
    vqc_circ.append(cirq.ry(phi)(qubit))

# Variational layers
for d in range(D_V):
    for i, qubit in enumerate(qs):
        th = sympy.Symbol(f"theta_{d}_{i}")
        zz = sympy.Symbol(f"zeta_{d}_{i}")
        VAR_SYM += [th, zz]
        vqc_circ.append([cirq.ry(th)(qubit), cirq.rz(zz)(qubit)])
    for i in range(Q_QUBITS):
        vqc_circ.append(cirq.CZ(qs[i], qs[(i + 1) % Q_QUBITS]))

vqc_circ.append(cirq.measure(*qs, key="m"))

N_VAR = len(VAR_SYM)
VQC_SIM = cirq.Simulator()

class HybridAgent:
    def __init__(self):
        self.enc = build_ctx_encoder()
        self.vpar = tf.Variable(tf.zeros(N_VAR, tf.float32), name="vqc_params")
        self.w_v = tf.Variable(
            tf.random.normal([D_H, 1], stddev=0.02), name="critic_head"
        )
        self.optE = keras.optimizers.Adam(LR_ACT)
        self.optQ = keras.optimizers.Adam(LR_Q)
        self.optV = keras.optimizers.Adam(LR_CRT)

    def _vqc_forward(self, ctx_vec, par_override=None) -> np.ndarray:
        par = self.vpar.numpy() if par_override is None else par_override
        binder = {}
        # encoder angles
        for i, sym in enumerate(PHI_SYM):
            binder[sym] = float(ctx_vec[i]) if i < len(ctx_vec) else 0.0
        # variational angles
        for j, sym in enumerate(VAR_SYM):
            binder[sym] = float(par[j])

        meas = VQC_SIM.run(
            vqc_circ,
            param_resolver=binder,
            repetitions=VQC_REPS
        ).measurements["m"]
        probs = meas.mean(axis=0)[:2]
        logits = tf.convert_to_tensor(probs * VQC_TEMP, dtype=tf.float32)
        return tf.nn.softmax(logits).numpy()

    def act(self, obs_vec: np.ndarray, epsilon: float = 0.1):
        """Return action (2-dim) and value estimate V(s)."""
        obs_tf = tf.convert_to_tensor(obs_vec[None, :], dtype=tf.float32)  # (1, D_H)
        ctx = self.enc(obs_tf, training=False)                             # (1, D_H)
        v_hat = tf.squeeze(ctx @ self.w_v, axis=1).numpy()[0]              # scalar

        ctx_np = ctx.numpy()[0]  # (D_H,) for VQC
        pi = self._vqc_forward(ctx_np)
        if random.random() < epsilon:
            pi = np.random.dirichlet([1.0, 1.0])
        return pi.astype(np.float32), float(v_hat)

    def update(self, obs_b: tf.Tensor, act_b: tf.Tensor, ret_b: tf.Tensor):
        """Batch A2C update with DP on encoder and param-shift on VQC."""
        with tf.GradientTape() as tape_v, tf.GradientTape() as tape_e:
            ctx_pred = self.enc(obs_b, training=True)
            v_hat = tf.squeeze(ctx_pred @ self.w_v, axis=1)
            loss_v = tf.reduce_mean((v_hat - ret_b) ** 2)

            logits = ctx_pred[:, :2]
            logp = tf.nn.log_softmax(logits)
            pi = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(pi * logp, axis=1)

            adv = tf.stop_gradient(ret_b - v_hat)
            pg = tf.reduce_sum(logp * act_b, axis=1)
            loss_e = -tf.reduce_mean(pg * adv + ENTROPY_BETA * entropy)

        # Critic
        gv = tape_v.gradient(loss_v, [self.w_v])
        self.optV.apply_gradients(zip(gv, [self.w_v]))

        # Encoder + DP
        ge = tape_e.gradient(loss_e, self.enc.trainable_variables)
        gn = tf.linalg.global_norm(ge)
        scale = tf.minimum(1.0, L2_CLIP / (gn + 1e-12))
        ge = [g * scale + tf.random.normal(tf.shape(g), stddev=SIG_DP) for g in ge]
        self.optE.apply_gradients(zip(ge, self.enc.trainable_variables))

        # Quantum params via parameter shift (subset)
        if N_VAR > 0:
            idxs = np.random.choice(N_VAR, min(PS_SUB, N_VAR), replace=False)
            grad_q = np.zeros(N_VAR, np.float32)
            ctx_np = self.enc(obs_b, training=False).numpy()
            act_np = act_b.numpy()
            v_hat_np = np.squeeze(ctx_np @ self.w_v.numpy(), axis=1)
            ret_np = ret_b.numpy()
            adv_np = ret_np - v_hat_np

            base_par = self.vpar.numpy()
            for k in idxs:
                acc = 0.0
                for sgn in (+1, -1):
                    psh = base_par.copy()
                    psh[k] += sgn * PARAM_SHIFT
                    out = np.array([self._vqc_forward(c, psh) for c in ctx_np])
                    loss_q = -np.mean(
                        np.sum(np.log(out + 1e-9) * act_np, axis=1) * adv_np
                    )
                    acc += sgn * loss_q
                grad_q[k] = 0.5 * acc

            grad_tensor = tf.convert_to_tensor(grad_q, dtype=tf.float32)
            self.optQ.apply_gradients([(grad_tensor, self.vpar)])

    # For FedAvg (vehicles)
    def get_encoder_weights(self):
        return [w.numpy() for w in self.enc.trainable_variables]

    def set_encoder_weights(self, weights):
        for var, val in zip(self.enc.trainable_variables, weights):
            var.assign(val)

class VehicleAgent(HybridAgent):
    def reward(self, T_tot_ms: float, e_joule: float) -> float:
        """
        Latency-centric reward with a hard 200 ms deadline:
            - scale latency by DEADLINE_MS
            - add extra penalty if T_tot_ms > DEADLINE_MS
        """
        T_clamped = min(T_tot_ms, CLIP_LAT_MS)
        violated = 1.0 if T_tot_ms > DEADLINE_MS else 0.0
        lat_norm = T_clamped / DEADLINE_MS

        return (
            -REWARD_LAT_SCALE * lat_norm
            - lambda_deadline * violated
            - lambda_energy * e_joule
        )

class UAVAgent(VehicleAgent):
    pass

class RSUAgent(HybridAgent):
    def reward(self, T_edge_ms: float) -> float:
        T_edge_ms = min(T_edge_ms, CLIP_LAT_MS)
        lat_norm = T_edge_ms / DEADLINE_MS
        return -(REWARD_LAT_SCALE * lat_norm)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Environment: clustered ISAC V2X with QKD + multi-hop offloading
# ──────────────────────────────────────────────────────────────────────────────
class VehicularEnv:
    def __init__(self):
        self.veh = list(range(N_VEH))
        self.uav = list(range(N_UAV))
        self.rsu = list(range(N_RSU))
        self.v2cl = {v: v // VEH_PER_CL for v in self.veh}
        self.u2cl = {u: u // UAV_PER_CL for u in self.uav}

        # Distances: tighter / closer geometry -> higher SINR
        self.d_vu = {
            (v, u): random.uniform(10.0, 50.0)  # slightly closer V->U geometry
            for v in self.veh
            for u in self.uav
            if self.v2cl[v] == self.u2cl[u]
        }
        self.d_ur = {
            (u, r): random.uniform(15.0, 60.0)  # slightly closer U->R geometry
            for u in self.uav
            for r in self.rsu
            if self.u2cl[u] == r
        }

        # Shadowing
        self.sh_vu = {(v, u): lognorm_shadow(SHADOW_STD_DB_VU)
                      for (v, u) in self.d_vu}
        self.sh_ur = {(u, r): lognorm_shadow(SHADOW_STD_DB_UR)
                      for (u, r) in self.d_ur}

        # QKD links
        self.qkd_vu = {(v, u): QKDLink() for (v, u) in self.d_vu}
        self.qkd_ur = {(u, r): QKDLink() for (u, r) in self.d_ur}

        self.rng = np.random.default_rng(SEED)

        self.last_ul_sinr = {v: 0.0 for v in self.veh}
        self.last_ul_rate = {v: 0.0 for v in self.veh}
        self.last_fh_sinr = {u: 0.0 for u in self.uav}
        self.last_Pi = {v: 1.0 for v in self.veh}

    def reset(self):
        self.t = 0
        self.S = {v: self.rng.poisson(5e3) for v in self.veh}
        self.D = {v: self.S[v] * zeta for v in self.veh}
        self.buf_u = {u: 0.0 for u in self.uav}
        self.edge_r = {r: 0.0 for r in self.rsu}

        # QKD warm start
        for (v, u), link in self.qkd_vu.items():
            link.key_buffer_bits = 0.0
            link.generate_for_slot(self.d_vu[(v, u)], QKD_WARM_S)
        for (u, r), link in self.qkd_ur.items():
            link.key_buffer_bits = 0.0
            link.generate_for_slot(self.d_ur[(u, r)], QKD_WARM_S)

        # reset SINR snapshots
        for v in self.veh:
            self.last_ul_sinr[v] = 0.0
        for u in self.uav:
            self.last_fh_sinr[u] = 0.0

        return self._obs()

    # Radar sensing quality Π_i(t)
    def _compute_sensing_quality(self, alpha_s: Dict[int, float]) -> Dict[int, float]:
        Pi = {}

        sigma_t2_sense = (10 ** (-108 / 10.0)) * 1e-3  # -108 dBm -> W
    
        si_factor = 0.2   # same as in sweep_alpha_tradeoff_jsac_strict
        j_factor  = 0.1   # same as in sweep_alpha_tradeoff_jsac_strict
    
        for v in self.veh:
            a_s = max(0.0, min(1.0, alpha_s.get(v, 0.5)))
    
            # --- target and channel as before ---
            d_i = random.uniform(10.0, 20.0)  # target range (m)
            beta_i = monostatic_beta(G_T_RADAR, G_R_RADAR, lambda_rf, d_i)
    
            phi_i = random.uniform(-math.pi / 3, math.pi / 3)
            a_tx_i = ula_steering(M_T_RADAR, phi_i)
            a_rx_i = ula_steering(M_R_RADAR, phi_i)
            w_s_i  = rand_unit_vec_c(M_T_RADAR)
            v_s_i  = rand_unit_vec_c(M_R_RADAR)
    
            theta_amp  = math.sqrt(max(SIGMA_RCS, 1e-9))
            theta_phase = 2 * math.pi * random.random()
            theta_i    = theta_amp * complex(math.cos(theta_phase),
                                             math.sin(theta_phase))
    
            g_rx = np.vdot(v_s_i, a_rx_i)
            g_tx = np.vdot(a_tx_i, w_s_i)
            h_eff = math.sqrt(beta_i) * theta_i * g_rx * g_tx
            num   = a_s * P_VEH_MAX * (abs(h_eff) ** 2)
    
            # --- interference from other vehicles (same as before, but only as interference) ---
            cl = self.v2cl[v]
            I_cross = 0.0
            for j in self.veh:
                if j == v or self.v2cl[j] != cl:
                    continue
                a_s_j = max(0.0, min(1.0, alpha_s.get(j, 0.5)))
                if a_s_j <= 0.0:
                    continue
                phi_ji = random.uniform(-math.pi / 3, math.pi / 3)
                a_tx_j = ula_steering(M_T_RADAR, phi_ji)
                w_s_j  = rand_unit_vec_c(M_T_RADAR)
                theta_amp_j  = math.sqrt(max(SIGMA_RCS, 1e-9))
                theta_phase_j = 2 * math.pi * random.random()
                theta_ji = theta_amp_j * complex(math.cos(theta_phase_j),
                                                 math.sin(theta_phase_j))
                g_tx_ji  = np.vdot(a_tx_j, w_s_j)
                h_ji_eff = math.sqrt(beta_i) * theta_ji * g_rx * g_tx_ji
                I_cross += INTERF_SCALE_UL * a_s_j * P_VEH_MAX * (abs(h_ji_eff) ** 2)
    
            # --- -:: radar-noise denominator (no huge 10.0 clutter term) ---
            sigma_SI = si_factor * sigma_t2_sense
            I_j      = j_factor * sigma_t2_sense
            denom = sigma_t2_sense + sigma_SI + I_j + I_cross
    
            Pi[v] = num / max(denom, 1e-15)
    
        return Pi

    def _obs(self):
        T_norm = EPIS * ROLL
        obs_v, obs_u, obs_r = {}, {}, {}

        for v in self.veh:
            cl = self.v2cl[v]
            vu_gains = []
            for u in self.uav:
                if self.u2cl[u] != cl:
                    continue
                d = self.d_vu[(v, u)]
                beta = large_scale_vu(d)
                vu_gains.append(beta)
            if not vu_gains:
                vu_gains = [1e-9]

            D_norm = min(1.0, self.D[v] / 1e7)
            Pi_val = max(self.last_Pi.get(v, 1e-3), 1e-6)
            Pi_db = 10.0 * math.log10(Pi_val)
            Pi_norm = np.clip(Pi_db, -10.0, 30.0) / 30.0

            sinr_ul_db = np.clip(safe_db(self.last_ul_sinr[v]),
                                 -10.0, 30.0) / 30.0
            rate_ul_mbps = (self.last_ul_rate[v] / 1e6) / 100.0

            feat = [
                D_norm,
                Pi_norm,
                float(np.max(vu_gains)),
                float(np.mean(vu_gains)),
                float(sinr_ul_db),
                float(rate_ul_mbps),
                min(1.0, self.t / max(1.0, T_norm)),
            ]
            obs_v[v] = np.pad(np.array(feat, dtype=np.float32),
                              (0, D_H - len(feat)))

        for u in self.uav:
            r = self.u2cl[u]
            d_ur = self.d_ur[(u, r)]
            beta_ur = large_scale_ur(d_ur) * self.sh_ur[(u, r)]
            sinr_fh_db = np.clip(safe_db(self.last_fh_sinr[u]),
                                 -10.0, 30.0) / 30.0
            feat = [
                min(1.0, self.buf_u[u] / 1e7),
                float(beta_ur),
                float(sinr_fh_db),
                min(1.0, self.t / max(1.0, T_norm)),
            ]
            obs_u[u] = np.pad(np.array(feat, dtype=np.float32),
                              (0, D_H - len(feat)))

        for r in self.rsu:
            feat = [
                min(1.0, self.edge_r[r] / 1e7),
                min(1.0, self.t / max(1.0, T_norm)),
            ]
            obs_r[r] = np.pad(np.array(feat, dtype=np.float32),
                              (0, D_H - len(feat)))

        return obs_v, obs_u, obs_r

    def step(self,
             veh_act: Dict[int, np.ndarray],
             uav_act: Dict[int, np.ndarray],
             rsu_act: Dict[int, np.ndarray]):
        # Normalize vehicle actions to α_s + α_c = 1
        norm_veh_act = {}
        for v, a in veh_act.items():
            arr = np.asarray(a, dtype=float)
            s = arr.sum()
            if s <= 0.0:
                arr = np.array([0.5, 0.5], dtype=float)
            else:
                arr = arr / s
            norm_veh_act[v] = arr
        veh_act = norm_veh_act

        rew_v, rew_u, rew_r = {}, {}, {}

        T_UL_ms = {}
        T_UAV_ms = {(v, u): 0.0 for v in self.veh for u in self.uav}
        T_FH_ms = {(v, u): 0.0 for v in self.veh for u in self.uav}
        T_RSU_ms = {(v, u): 0.0 for v in self.veh for u in self.uav}

        self.buf_u = {u: 0.0 for u in self.uav}
        self.edge_r = {r: 0.0 for r in self.rsu}

        alpha_s = {}
        alpha_c = {}
        u_best_map = {}

        uav_tx_power = {}
        for u, beta in uav_act.items():
            beta = np.asarray(beta, dtype=float)
            beta_f = max(0.3, min(1.0, float(beta[0])))  # ≥ 30% to compute
            beta_p = max(0.0, min(1.0, float(beta[1])))
            uav_tx_power[u] = beta_p * P_UAV_MAX

        # 1) UL with QKD + OTP fallback
        for v, alpha in veh_act.items():
            alpha_s_v, alpha_c_v = float(alpha[0]), float(alpha[1])
            alpha_s[v] = alpha_s_v
            alpha_c[v] = alpha_c_v
            D_bits = self.D[v]
            cl = self.v2cl[v]
            cand_u = [u for u in self.uav if self.u2cl[u] == cl]
            if not cand_u:
                continue

            # helper set U_i(t) reduced to single best helper for UL
            u_best = min(
                cand_u,
                key=lambda u: fspl(self.d_vu[(v, u)], lambda_rf) *
                              self.sh_vu[(v, u)]
            )
            u_best_map[v] = u_best

            d_vub = self.d_vu[(v, u_best)]
            beta_vub = large_scale_vu(d_vub) * self.sh_vu[(v, u_best)]
            g_small = rician_power(10.0)   # or 8.0 if you want slightly weaker LoS
            link_gain = beta_vub * g_small

            I_comm = 0.0
            I_sense = 0.0
            for j, a_j in veh_act.items():
                if j == v or self.v2cl[j] != cl:
                    continue
                alpha_s_j, alpha_c_j = float(a_j[0]), float(a_j[1])
                d_jub = self.d_vu[(j, u_best)]
                beta_jub = large_scale_vu(d_jub) * self.sh_vu[(j, u_best)]
                g_j = rician_power(10.0)
                gain_j = beta_jub * g_j
                # scale interference to avoid brutal SINR collapse
                I_comm += INTERF_SCALE_UL * alpha_c_j * P_VEH_MAX * gain_j
                I_sense += INTERF_SCALE_UL * alpha_s_j * P_VEH_MAX * gain_j

            p_u = uav_tx_power[u_best]
            I_SI_u = ETA_UAV_SI * p_u

            S_ul = alpha_c_v * P_VEH_MAX * link_gain
            gamma_ul = S_ul / (sigma_t2 + I_comm + I_sense + I_SI_u + 1e-15)

            # QKD generation is independent of RF outage
            qkd = self.qkd_vu[(v, u_best)]
            qkd.generate_for_slot(self.d_vu[(v, u_best)], tau)

            if gamma_ul < SNR_MIN_UL:
                # Treat as outage: no useful bits, large fixed latency
                phy_bits = 0.0
                svc_bits = 0.0
                T_ul_ms = CLIP_LAT_MS
            else:
                R_ul_bits_slot = throughput_bits_per_slot(
                    gamma_ul, B, tau, n_blk, FBL_EPS
                )
                phy_bits = R_ul_bits_slot
                available_keys = qkd.key_buffer_bits
                enc_bits = min(phy_bits, available_keys)
                if enc_bits > 0.0:
                    qkd.consume_bits(enc_bits)
                svc_bits = phy_bits
                T_ul_ms = (D_bits / max(1e-9, svc_bits)) * 1e3

            e_v = alpha_c_v * P_VEH_MAX * tau
            T_UL_ms[v] = T_ul_ms

            # Aggregate buffer at helper UAVs (equal split over cluster)
            share = D_bits / max(1, len(cand_u))
            for u in cand_u:
                self.buf_u[u] += share

            self.last_ul_sinr[v] = gamma_ul
            self.last_ul_rate[v] = svc_bits / tau if tau > 0 else 0.0

        # 2) UAV compute + FH with QKD (β_i,u=1 for u_best)
        FH_FORWARD_FRACTION = 0.2
        lat_u_total_ms = {u: 0.0 for u in self.uav}
        lat_r_total_ms = {r: 0.0 for r in self.rsu}

        fh_gain = {}
        I_fh = {r: 0.0 for r in self.rsu}
        for u in self.uav:
            r = self.u2cl[u]
            d_ur = self.d_ur[(u, r)]
            beta_ur = large_scale_ur(d_ur) * self.sh_ur[(u, r)]
            g_ur = rician_power(10.0)
            fh_gain[(u, r)] = beta_ur * g_ur

        for r in self.rsu:
            val = 0.0
            for u in self.uav:
                if self.u2cl[u] != r:
                    continue
                # scale cross-UAV interference contribution
                val += INTERF_SCALE_FH * uav_tx_power[u] * fh_gain[(u, r)]
            I_fh[r] = val

        for u, beta in uav_act.items():
            beta = np.asarray(beta, dtype=float)
            beta_f = max(0.3, min(1.0, float(beta[0])))  # ≥ 30% to compute
            beta_p = max(0.0, min(1.0, float(beta[1])))
            fu = beta_f * F_R_MAX
            pu = beta_p * P_UAV_MAX

            r = self.u2cl[u]
            gain_ur = fh_gain[(u, r)]
            # subtract own (unscaled) signal from interference bucket
            I_r_minus_u = max(I_fh[r] - INTERF_SCALE_FH * pu * gain_ur, 0.0)
            gamma_fh = pu * gain_ur / (sigma_t2 + I_r_minus_u + 1e-15)

            # QKD generation for U->R
            qkd = self.qkd_ur[(u, r)]
            qkd.generate_for_slot(self.d_ur[(u, r)], tau)

            if gamma_fh < SNR_MIN_FH:
                R_fh_bits_slot = 0.0
                phy_bits_fh = 0.0
                svc_fh_bits = 0.0
            else:
                R_fh_bits_slot = throughput_bits_per_slot(
                    gamma_fh, B, tau, n_blk, FBL_EPS
                )
                phy_bits_fh = R_fh_bits_slot
                avail_keys_fh = qkd.key_buffer_bits
                enc_bits_fh = min(phy_bits_fh, avail_keys_fh)
                if enc_bits_fh > 0.0:
                    qkd.consume_bits(enc_bits_fh)
                svc_fh_bits = phy_bits_fh

            for v in self.veh:
                if u_best_map.get(v, None) != u:
                    continue
                D_i = self.D[v]
                D_proc = D_i * (1 - FH_FORWARD_FRACTION)
                D_fwd = D_i * FH_FORWARD_FRACTION

                T_uav_ms = (D_proc * kappa) / (fu + 1e-9) * 1e3

                if svc_fh_bits <= 0.0:
                    T_fh_ms = CLIP_LAT_MS
                else:
                    T_fh_ms = (D_fwd / max(1e-9, svc_fh_bits)) * 1e3

                T_UAV_ms[(v, u)] = T_uav_ms
                T_FH_ms[(v, u)] = T_fh_ms
                lat_u_total_ms[u] += (T_uav_ms + T_fh_ms)
                self.edge_r[r] += D_fwd

            self.last_fh_sinr[u] = gamma_fh

        # 3) RSU compute
        for r, gamma in rsu_act.items():
            gamma = np.asarray(gamma, dtype=float)
            gamma_f = max(0.3, min(1.0, float(gamma[0])))  # ≥ 30% to compute
            fr = gamma_f * F_R_MAX
            if fr <= 0.0:
                fr = 1e3

            for v in self.veh:
                u = u_best_map.get(v, None)
                if u is None or self.u2cl[u] != r:
                    continue
                D_i = self.D[v]
                D_fwd_total = D_i * FH_FORWARD_FRACTION
                T_rsu_ms = (D_fwd_total * kappa) / (fr + 1e-9) * 1e3
                T_RSU_ms[(v, u)] = T_rsu_ms
                lat_r_total_ms[r] += T_rsu_ms

        # 4) End-to-end latency and rewards
        for v in self.veh:
            u = u_best_map.get(v, None)
            if u is None:
                T_tot_ms = CLIP_LAT_MS
            else:
                T_tot_ms = T_UL_ms.get(v, CLIP_LAT_MS) + (
                    T_UAV_ms[(v, u)] +
                    T_FH_ms[(v, u)] +
                    T_RSU_ms[(v, u)]
                )
                T_tot_ms = min(T_tot_ms, CLIP_LAT_MS)
            e_v = alpha_c.get(v, 0.0) * P_VEH_MAX * tau
            rew_v[v] = (T_tot_ms, e_v)

        for u in self.uav:
            e_u = uav_tx_power[u] * tau
            rew_u[u] = (lat_u_total_ms[u], e_u)

        for r in self.rsu:
            rew_r[r] = lat_r_total_ms[r]

        # 5) New data D_i(t+1) via sensing quality
        Pi_dict = self._compute_sensing_quality(alpha_s)
        self.last_Pi = Pi_dict
        for v in self.veh:
            S_new = self.rng.poisson(0.05e3)
            Q_i = min(1.0, Pi_dict[v] / PI_THRESH)
            self.S[v] = S_new
            self.D[v] = S_new * zeta * Q_i

        self.t += 1
        done = (self.t >= ROLL)
        return (*self._obs(), rew_v, rew_u, rew_r, done)

# ──────────────────────────────────────────────────────────────────────────────
# 6) FedAvg encoder aggregation for vehicle agents (AdaGrad-style)
# ──────────────────────────────────────────────────────────────────────────────
def federated_encoder_update(veh_agents: Dict[int, VehicleAgent]):
    global SERVER_ENCODER_WEIGHTS, SERVER_ADAGRAD_ACCUM
    weights = [ag.get_encoder_weights() for ag in veh_agents.values()]
    n_agents = len(weights)
    if n_agents == 0:
        return

    n_layers = len(weights[0])
    if SERVER_ENCODER_WEIGHTS is None:
        # Initialize server weights and AdaGrad accumulators
        SERVER_ENCODER_WEIGHTS = [
            np.mean([w[layer_idx] for w in weights], axis=0)
            for layer_idx in range(n_layers)
        ]
        SERVER_ADAGRAD_ACCUM = [
            np.zeros_like(w) for w in SERVER_ENCODER_WEIGHTS
        ]
    else:
        # Compute average local encoder weights
        avg_local = [
            np.mean([w[layer_idx] for w in weights], axis=0)
            for layer_idx in range(n_layers)
        ]
        # AdaGrad-style preconditioning on "pseudo-gradient"
        for layer_idx in range(n_layers):
            grad_l = avg_local[layer_idx] - SERVER_ENCODER_WEIGHTS[layer_idx]
            SERVER_ADAGRAD_ACCUM[layer_idx] += grad_l * grad_l
            precond = 1.0 / (np.sqrt(SERVER_ADAGRAD_ACCUM[layer_idx]) +
                             SERVER_ADAGRAD_EPS)
            SERVER_ENCODER_WEIGHTS[layer_idx] += SERVER_LR * precond * grad_l

    # Push global encoder weights back to all vehicle agents
    for ag in veh_agents.values():
        ag.set_encoder_weights(SERVER_ENCODER_WEIGHTS)

# ──────────────────────────────────────────────────────────────────────────────
# 7) GAE helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_returns_gae(rews, vals, last_val,
                        gamma=GAMMA_DISCOUNT,
                        lam=LAMBDA_GAE):
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = last_val if t == T - 1 else vals[t + 1]
        delta = rews[t] + gamma * next_val - vals[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    returns = adv + np.array(vals, dtype=np.float32)
    return returns.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# 8) Training loop
# ──────────────────────────────────────────────────────────────────────────────
def run_fdrl():
    env = VehicularEnv()
    va = {v: VehicleAgent() for v in env.veh}
    ua = {u: UAVAgent() for u in env.uav}
    ra = {r: RSUAgent() for r in env.rsu}
    acct = PrivacyAccountant()
    returns, epsilons = [], []
    epsilon = 0.5

    latency_records = []
    # #per-episode mean latency storage
    episode_mean_latencies = []

    #: per-episode mean sensing SINR (dB)
    episode_mean_sensing_db = []

    # SINR debug logs
    ul_sinr_records = []
    fh_sinr_records = []

    #: sensing debug logs
    sensing_records = []

    # Outer progress bar: episodes
    outer_pbar = tqdm(range(EPIS), desc="Episodes", ncols=90)

    for ep in outer_pbar:
        # DP accountant per episode (Gaussian mechanism upper bound)
        acct.accumulate()
        eps_now = acct.current()
        if eps_now > EPSILON_MAX:
            tqdm.write(
                f"[STOP] Episode {ep}: ε(t)={eps_now:.3f} "
                f"> EPSILON_MAX={EPSILON_MAX:.3f}"
            )
            break

        obs_v, obs_u, obs_r = env.reset()

        # Per-agent trajectory buffers
        traj_v = {
            v: {'obs': [], 'act': [], 'rew': [], 'val': []}
            for v in env.veh
        }
        traj_u = {
            u: {'obs': [], 'act': [], 'rew': [], 'val': []}
            for u in env.uav
        }
        traj_r = {
            r: {'obs': [], 'act': [], 'rew': [], 'val': []}
            for r in env.rsu
        }

        veh_ep_rewards = []

        #: collect all per-step per-vehicle latencies for this episode
        ep_latencies = []
        
        #: collect all per-step per-vehicle sensing SINR (dB) for this episode
        ep_sensing_db = []

        # Inner progress bar: steps within each episode
        inner_pbar = tqdm(
            range(ROLL),
            desc=f"  Ep {ep} steps",
            leave=False,
            ncols=90
        )

        for step in inner_pbar:
            # Actions & value estimates (vehicles)
            av, a_vals_v = {}, {}
            for v in env.veh:
                a_v, v_v = va[v].act(obs_v[v], epsilon)
                av[v] = a_v
                a_vals_v[v] = v_v
                traj_v[v]['obs'].append(obs_v[v].copy())
                traj_v[v]['act'].append(a_v.copy())
                traj_v[v]['val'].append(v_v)

            # UAVs
            au, a_vals_u = {}, {}
            for u in env.uav:
                a_u, v_u = ua[u].act(obs_u[u], epsilon)
                au[u] = a_u
                a_vals_u[u] = v_u
                traj_u[u]['obs'].append(obs_u[u].copy())
                traj_u[u]['act'].append(a_u.copy())
                traj_u[u]['val'].append(v_u)

            # RSUs
            ar, a_vals_r = {}, {}
            for r in env.rsu:
                a_r, v_r = ra[r].act(obs_r[r], epsilon)
                ar[r] = a_r
                a_vals_r[r] = v_r
                traj_r[r]['obs'].append(obs_r[r].copy())
                traj_r[r]['act'].append(a_r.copy())
                traj_r[r]['val'].append(v_r)

            # Environment step
            nxt_v, nxt_u, nxt_r, rv, ru, rr, done = env.step(av, au, ar)

            # DEBUG: log SINR after this step
            for v in env.veh:
                gamma_ul = env.last_ul_sinr.get(v, 0.0)
                ul_sinr_records.append({
                    'episode': ep,
                    'step': step,
                    'veh_id': v,
                    'sinr_linear': gamma_ul,
                    'sinr_db': safe_db(gamma_ul),
                })
            for u in env.uav:
                gamma_fh = env.last_fh_sinr.get(u, 0.0)
                fh_sinr_records.append({
                    'episode': ep,
                    'step': step,
                    'uav_id': u,
                    'sinr_linear': gamma_fh,
                    'sinr_db': safe_db(gamma_fh),
                })

            #: SENSING debug – log Π_i(t) for each vehicle
            for v in env.veh:
                Pi_val = env.last_Pi.get(v, 0.0)
                Pi_db_val = safe_db(Pi_val)
                sensing_records.append({
                    'episode': ep,
                    'step': step,
                    'veh_id': v,
                    'Pi_linear': Pi_val,
                    'Pi_db': Pi_db_val,
                })
                #: accumulate for per-episode mean
                ep_sensing_db.append(Pi_db_val)

            # Print compact SINR statistics for quick debugging
            ul_db_vals = [safe_db(env.last_ul_sinr[v]) for v in env.veh]
            fh_db_vals = [safe_db(env.last_fh_sinr[u]) for u in env.uav]
            tqdm.write(
                f"[Ep {ep} Step {step}] "
                f"UL SINR dB: mean={np.mean(ul_db_vals):.1f}, "
                f"min={np.min(ul_db_vals):.1f}, max={np.max(ul_db_vals):.1f}; "
                f"FH SINR dB: mean={np.mean(fh_db_vals):.1f}, "
                f"min={np.min(fh_db_vals):.1f}, max={np.max(fh_db_vals):.1f}"
            )

            # Rewards and per-step latency logging (vehicles)
            for v in env.veh:
                T_tot_ms, e_v = rv[v]
                r_v = va[v].reward(T_tot_ms, e_v)
                traj_v[v]['rew'].append(r_v)
                veh_ep_rewards.append(r_v)

                # log per-step latency
                latency_records.append({
                    'episode': ep,
                    'step': step,
                    'veh_id': v,
                    'T_tot_ms': T_tot_ms,
                    'deadline_ms': DEADLINE_MS,
                    'violation': int(T_tot_ms > DEADLINE_MS),
                })
                #: also add to per-episode accumulator
                ep_latencies.append(T_tot_ms)

            # UAV rewards
            for u in env.uav:
                lat_u_ms, e_u = ru[u]
                r_u = ua[u].reward(lat_u_ms, e_u)
                traj_u[u]['rew'].append(r_u)

            # RSU rewards
            for r in env.rsu:
                lat_r_ms = rr[r]
                r_r = ra[r].reward(lat_r_ms)
                traj_r[r]['rew'].append(r_r)

            obs_v, obs_u, obs_r = nxt_v, nxt_u, nxt_r

            # Show internal progress using per-vehicle average so far
            if veh_ep_rewards:
                inner_pbar.set_postfix(
                    mean_ret=f"{np.mean(veh_ep_rewards):+.3f}"
                )

            if done:
                break

        # Bootstrap values at final state for GAE
        last_vals_v = {}
        for v in env.veh:
            _, v_val = va[v].act(obs_v[v], epsilon=0.0)
            last_vals_v[v] = v_val

        last_vals_u = {}
        for u in env.uav:
            _, v_val = ua[u].act(obs_u[u], epsilon=0.0)
            last_vals_u[u] = v_val

        last_vals_r = {}
        for r in env.rsu:
            _, v_val = ra[r].act(obs_r[r], epsilon=0.0)
            last_vals_r[r] = v_val

        # Batch updates with GAE
        for v in env.veh:
            if not traj_v[v]['rew']:
                continue
            obs_arr = np.stack(traj_v[v]['obs'], axis=0)
            act_arr = np.stack(traj_v[v]['act'], axis=0)
            vals = traj_v[v]['val']
            rews = traj_v[v]['rew']
            ret_arr = compute_returns_gae(rews, vals, last_vals_v[v])
            va[v].update(
                tf.constant(obs_arr, dtype=tf.float32),
                tf.constant(act_arr, dtype=tf.float32),
                tf.constant(ret_arr, dtype=tf.float32),
            )

        for u in env.uav:
            if not traj_u[u]['rew']:
                continue
            obs_arr = np.stack(traj_u[u]['obs'], axis=0)
            act_arr = np.stack(traj_u[u]['act'], axis=0)
            vals = traj_u[u]['val']
            rews = traj_u[u]['rew']
            ret_arr = compute_returns_gae(rews, vals, last_vals_u[u])
            ua[u].update(
                tf.constant(obs_arr, dtype=tf.float32),
                tf.constant(act_arr, dtype=tf.float32),
                tf.constant(ret_arr, dtype=tf.float32),
            )

        for r in env.rsu:
            if not traj_r[r]['rew']:
                continue
            obs_arr = np.stack(traj_r[r]['obs'], axis=0)
            act_arr = np.stack(traj_r[r]['act'], axis=0)
            vals = traj_r[r]['val']
            rews = traj_r[r]['rew']
            ret_arr = compute_returns_gae(rews, vals, last_vals_r[r])
            ra[r].update(
                tf.constant(obs_arr, dtype=tf.float32),
                tf.constant(act_arr, dtype=tf.float32),
                tf.constant(ret_arr, dtype=tf.float32),
            )

        # Server-side FedAvg + AdaGrad on encoders (vehicles)
        federated_encoder_update(va)

        # Per-episode mean reward per vehicle
        mean_ret = float(np.mean(veh_ep_rewards)) if veh_ep_rewards else 0.0
        returns.append(mean_ret)
        epsilons.append(eps_now)

        #: per-episode mean latency over all vehicles & steps

        ep_mean_lat = float(np.mean(ep_latencies)) if ep_latencies else 0.0
        episode_mean_latencies.append(ep_mean_lat)
        
        #: per-episode mean sensing SINR (in dB) over all vehicles & steps
        ep_mean_sens_db = float(np.mean(ep_sensing_db)) if ep_sensing_db else 0.0
        episode_mean_sensing_db.append(ep_mean_sens_db)
        
        # Print / show in progress bar
        outer_pbar.set_postfix(
            ret=f"{mean_ret:+.3f}",
            eps=f"{eps_now:.3f}",
            lat=f"{ep_mean_lat:.1f} ms",
            sens=f"{ep_mean_sens_db:.1f} dB"
        )
        
        # OPTIONAL: explicit print line per episode
        print(f"[Episode {ep}] mean sensing SINR = {ep_mean_sens_db:.2f} dB")

        # Anneal exploration
        epsilon = max(0.01, epsilon * 0.95)

    outer_pbar.close()

    # Save convergence + privacy budget + mean latency
    df = pd.DataFrame({
    'episode': np.arange(len(returns)),
    'mean_return': returns,
    'epsilon': epsilons,
    'mean_latency_ms': episode_mean_latencies,   # COLUMN
    'mean_sensing_db': episode_mean_sensing_db,  # COLUMN
    })

    df.to_csv('convergence.csv', index=False)

    pd.DataFrame({
        'episode': df['episode'],
        'epsilon': df['epsilon'],
    }).to_csv('privacy_budget.csv', index=False)
    print("-> saved convergence.csv and privacy_budget.csv "
          "(now with mean_latency_ms)")

    # Save per-step latency traces
    if latency_records:
        df_lat = pd.DataFrame(latency_records)
        df_lat.to_csv('latency_traces.csv', index=False)
        print("-> saved latency_traces.csv")

    # Save SINR debug traces
    if ul_sinr_records:
        df_ul = pd.DataFrame(ul_sinr_records)
        df_ul.to_csv('ul_sinr_traces.csv', index=False)
        print("-> saved ul_sinr_traces.csv")
    if fh_sinr_records:
        df_fh = pd.DataFrame(fh_sinr_records)
        df_fh.to_csv('fh_sinr_traces.csv', index=False)
        print("-> saved fh_sinr_traces.csv")

    #: Save sensing debug traces
    if sensing_records:
        df_sens = pd.DataFrame(sensing_records)
        df_sens.to_csv('sensing_traces.csv', index=False)
        print("-> saved sensing_traces.csv")
    
    print("All files are saved, go for the next step, Please follow the README file")

# ──────────────────────────────────────────────────────────────────────────────
# 9) Main entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_fdrl()