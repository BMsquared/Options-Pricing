"""
QuantDesk — Multi-Model Options Pricer (Tkinter GUI)
====================================================
pip install numpy scipy matplotlib yfinance
Run: python quantdesk.py
"""

import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    print("Matplotlib required. Run: pip install matplotlib")

try:
    import yfinance as yf
except ImportError:
    print("yfinance required. Run: pip install yfinance")


# ─────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────
C = {
    "bg":        "#0b0f1a",
    "bg2":       "#111827",
    "bg3":       "#1a2235",
    "border":    "#1f3055",
    "accent_b":  "#3b82f6",
    "accent_p":  "#8b5cf6",
    "accent_r":  "#ef4444",
    "accent_g":  "#10b981",
    "accent_y":  "#f59e0b",
    "text":      "#e2eaf7",
    "text2":     "#7a9bbf",
    "text3":     "#4a6a8a"
}

FONTS = {
    "title":   ("Consolas", 14, "bold"),
    "heading": ("Consolas", 12, "bold"),
    "body":    ("Consolas", 10),
    "small":   ("Consolas", 8)
}

# ─────────────────────────────────────────────────────────────
#  MATH CORE
# ─────────────────────────────────────────────────────────────

def days_to_expiry(expiry_str):
    exp  = datetime.strptime(expiry_str, "%Y-%m-%d")
    days = max((exp - datetime.now()).days, 1)
    return days, days / 365.0


def bs_price(S, K, T, r, sigma, otype="call"):
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0) if otype == "call" \
               else max(K * np.exp(-r * T) - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if otype == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, otype="call"):
    if T <= 0 or sigma <= 0:
        return dict(delta=0, gamma=0, vega=0, theta=0, rho=0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if otype == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * np.sqrt(T) * norm.pdf(d1) / 100
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365 if otype == "call" \
            else (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100 if otype == "call" \
            else -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def merton_price(S, K, T, r, sigma, lam, mu_j, sigma_j, otype="call", n_terms=50):
    k_bar         = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    lam_prime     = lam * (1 + k_bar)
    price         = 0.0
    log_fact      = 0.0
    for n in range(n_terms):
        if n > 0:
            log_fact += np.log(n)
        lw = -lam_prime * T + n * np.log(max(lam_prime * T, 1e-300)) - log_fact
        w  = np.exp(lw)
        if w < 1e-12 and n > 5:
            break
        r_n     = r - lam * k_bar + (n * mu_j) / T
        sigma_n = np.sqrt(sigma**2 + (n * sigma_j**2) / T)
        price  += w * bs_price(S, K, T, r_n, sigma_n, otype)
    return price


def merton_greeks(S, K, T, r, sigma, lam, mu_j, sigma_j, otype="call"):
    dS, dv = 0.5, 0.001
    p   = merton_price(S,      K, T,       r, sigma,      lam, mu_j, sigma_j, otype)
    pu  = merton_price(S + dS, K, T,       r, sigma,      lam, mu_j, sigma_j, otype)
    pd  = merton_price(S - dS, K, T,       r, sigma,      lam, mu_j, sigma_j, otype)
    pt  = merton_price(S,      K, T-1/365, r, sigma,      lam, mu_j, sigma_j, otype)
    pvu = merton_price(S,      K, T,       r, sigma + dv, lam, mu_j, sigma_j, otype)
    pvd = merton_price(S,      K, T,       r, sigma - dv, lam, mu_j, sigma_j, otype)
    return dict(
        delta=(pu - pd) / (2 * dS),
        gamma=(pu - 2*p + pd) / dS**2,
        theta=(pt - p) / (1 / 365),
        vega =(pvu - pvd) / (2 * dv) / 100,
        rho  = float("nan"),
    )


def build_vol_surface(S, r, base_vol, skew_slope=-0.10, smile_curve=0.03):
    moneyness = np.array([0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                          1.05, 1.10, 1.15, 1.20, 1.25])
    strikes   = moneyness * S
    expiries  = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.25, 1.5])
    iv_surf   = np.zeros((len(strikes), len(expiries)))

    for j, T in enumerate(expiries):
        for i, m in enumerate(moneyness):
            log_m = np.log(m) / np.sqrt(T)
            atm_vol = base_vol * (1 + 0.04 * np.sqrt(T))
            iv_surf[i, j] = np.clip(atm_vol + skew_slope * log_m + smile_curve * log_m**2, 0.05, 2.0)

    C_surf = np.zeros_like(iv_surf)
    for i, K in enumerate(strikes):
        for j, T_ in enumerate(expiries):
            C_surf[i, j] = bs_price(S, K, T_, r, iv_surf[i, j], "call")

    lv_surf = np.zeros_like(iv_surf)
    dK = np.diff(strikes).mean()

    for j in range(len(expiries)):
        T_ = expiries[j]
        for i in range(1, len(strikes) - 1):
            K = strikes[i]
            if j == 0:
                dC_dT = (C_surf[i,j+1]-C_surf[i,j])/(expiries[j+1]-expiries[j])
            elif j == len(expiries)-1:
                dC_dT = (C_surf[i,j]-C_surf[i,j-1])/(expiries[j]-expiries[j-1])
            else:
                dC_dT = (C_surf[i,j+1]-C_surf[i,j-1])/(expiries[j+1]-expiries[j-1])
            dC_dK   = (C_surf[i+1,j]-C_surf[i-1,j]) / (2*dK)
            d2C_dK2 = (C_surf[i+1,j]-2*C_surf[i,j]+C_surf[i-1,j]) / dK**2
            num = dC_dT + r * K * dC_dK
            den = 0.5 * K**2 * d2C_dK2
            lv_surf[i,j] = np.sqrt(np.clip(num/den, 0.001, 4.0)) if den > 1e-10 and num > 0 else iv_surf[i,j]

    lv_surf[0,:]  = lv_surf[1,:]
    lv_surf[-1,:] = lv_surf[-2,:]
    lv_surf       = np.clip(lv_surf, 0.01, 2.0)

    spline = RectBivariateSpline(strikes, expiries, lv_surf, kx=3, ky=3)

    def local_vol_fn(spot, t):
        spot = np.clip(spot, strikes[0], strikes[-1])
        t    = np.clip(t,    expiries[0], expiries[-1])
        return float(np.clip(spline(spot, t), 0.01, 2.0))

    return strikes, expiries, iv_surf, lv_surf, local_vol_fn


def dupire_mc_price(S, K, T, r, local_vol_fn,
                    otype="call", n_paths=15000, n_steps=100, seed=42):
    np.random.seed(seed)
    dt = T / n_steps
    paths = np.full(n_paths, S, dtype=float)

    for step in range(n_steps):
        t = step * dt
        Z = np.random.standard_normal(n_paths)
        sigma_L = np.array([local_vol_fn(s, max(t, 0)) for s in paths])
        paths *= np.exp((r - 0.5 * sigma_L**2) * dt + sigma_L * np.sqrt(dt) * Z)

    payoffs = (np.maximum(paths - K, 0) if otype == "call"
               else np.maximum(K - paths, 0))

    price = np.exp(-r * T) * np.mean(payoffs)
    stderr = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    return price, stderr


def dupire_greeks(S, K, T, r, local_vol_fn, otype="call", n_paths=8000):
    dS = max(S * 0.005, 0.5)
    p,  _ = dupire_mc_price(S, K, T, r, local_vol_fn, otype, n_paths=n_paths)
    pu, _ = dupire_mc_price(S + dS, K, T, r, local_vol_fn, otype, n_paths=n_paths)
    pd, _ = dupire_mc_price(S - dS, K, T, r, local_vol_fn, otype, n_paths=n_paths)
    pt, _ = dupire_mc_price(S, K, max(T - 7 / 365, 0.01), r, local_vol_fn, otype, n_paths=n_paths)
    return dict(
        delta=(pu - pd) / (2 * dS),
        gamma=(pu - 2 * p + pd) / dS**2,
        theta=(pt - p) / (7 / 365) / 365,
        vega = float("nan"),
        rho  = float("nan"),
    )


# ─────────────────────────────────────────────────────────
#  GUI APPLICATION
# ─────────────────────────────────────────────────────────

class QuantDeskApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QuantDesk — Multi-Model Options Pricer")
        self.configure(bg=C["bg"])
        self.geometry("1280x720")
        self.minsize(800, 600)

        self.ticker_var  = tk.StringVar(value="AAPL")
        self.S_var       = tk.DoubleVar(value=213.00)
        self.K_var       = tk.DoubleVar(value=215.00)
        self.expiry_var  = tk.StringVar(value="2026-09-19")
        self.r_var       = tk.DoubleVar(value=0.05)
        self.otype_var   = tk.StringVar(value="call")

        self._build_ui()

    def _build_ui(self):
        self._build_topbar()
        self._build_notebook()
        self._build_statusbar()

    def _build_topbar(self):
        bar = tk.Frame(self, bg=C["bg2"], height=50)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        title = tk.Label(bar, text="QUANTDESK", bg=C["bg2"], fg=C["accent_b"],
                         font=("Consolas", 18, "bold"))
        title.pack(side="left", padx=10)

        param_frame = tk.Frame(bar, bg=C["bg2"])
        param_frame.pack(side="right", padx=10)

        params = [
            ("TICKER", self.ticker_var),
            ("SPOT S", self.S_var),
            ("STRIKE K", self.K_var),
            ("EXPIRY", self.expiry_var),
            ("RATE r", self.r_var),
        ]

        for label, var in params:
            lbl = tk.Label(param_frame, text=label, bg=C["bg2"], fg=C["text"],
                           font=FONTS["small"])
            lbl.pack(side="left", padx=(5, 2))
            ent = tk.Entry(param_frame, textvariable=var, width=10)
            ent.pack(side="left", padx=(0, 10))

        label_type = tk.Label(param_frame, text="TYPE", bg=C["bg2"], fg=C["text"], font=FONTS["small"])
        label_type.pack(side="left", padx=(5, 2))
        
        cb = ttk.Combobox(param_frame, textvariable=self.otype_var, values=["call", "put"], state="readonly")
        cb.pack(side="left", padx=(0, 5))

        fetch_btn = tk.Button(param_frame, text="FETCH LIVE",
                               command=self._fetch_live_data)
        fetch_btn.pack(side="left", padx=(5, 2))

    def _build_notebook(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self.tab_bs     = tk.Frame(self.nb, bg=C["bg"])
        self.tab_merton = tk.Frame(self.nb, bg=C["bg"])
        self.tab_dupire = tk.Frame(self.nb, bg=C["bg"])
        self.tab_surf   = tk.Frame(self.nb, bg=C["bg"])

        self.nb.add(self.tab_bs,     text="  Black-Scholes  ")
        self.nb.add(self.tab_merton, text="  Merton Jump  ")
        self.nb.add(self.tab_dupire, text="  Dupire Local Vol  ")
        self.nb.add(self.tab_surf,   text="  Vol Surface  ")

        self._build_bs_tab()
        self._build_merton_tab()
        self._build_dupire_tab()
        self._build_surface_tab()

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Frame(self, bg=C["bg2"], height=25)
        status_bar.pack(fill="x", side="bottom")
        tk.Label(status_bar, textvariable=self.status_var,
                 bg=C["bg2"], fg=C["text"], font=FONTS["small"]).pack(side="left", padx=10)

    # ─────────────────────────────────────────────────────────
    #  BLACK-SCHOLES TAB
    # ─────────────────────────────────────────────────────────

    def _build_bs_tab(self):
        p = self.tab_bs
        left_frame = tk.Frame(p, bg=C["bg"])
        left_frame.pack(side="left", fill="y", padx=10)

        tk.Label(left_frame, text="BLACK-SCHOLES MODEL", bg=C["bg"], fg=C["accent_b"], font=FONTS["heading"]).pack(anchor="w", pady=5)

        self.bs_sigma = tk.DoubleVar(value=0.22)

        params_frame = tk.Frame(left_frame, bg=C["bg"])
        params_frame.pack(pady=5)

        tk.Label(params_frame, text="Volatility (sigma)", bg=C["bg"], fg=C["text"], font=FONTS["small"]).pack(side="left")
        tk.Entry(params_frame, textvariable=self.bs_sigma, width=10).pack(side="left", padx=5)

        tk.Button(left_frame, text="PRICE", command=self._run_bs).pack(pady=10)

        self.bs_price_var = tk.StringVar(value="--")
        tk.Label(left_frame, text="Price (USD)", bg=C["bg"], fg=C["text"], font=FONTS["small"]).pack(pady=(10, 0))
        tk.Label(left_frame, textvariable=self.bs_price_var, bg=C["bg"], fg=C["accent_b"], font=FONTS["title"]).pack(pady=(0, 20))

        self.bs_gvars = {}
        for g_name in ["Delta", "Gamma", "Vega", "Theta", "Rho"]:
            self.bs_gvars[g_name.lower()] = tk.StringVar(value="--")
            tk.Label(left_frame, text=g_name, bg=C["bg"], fg=C["text"], font=FONTS["small"]).pack(pady=2)
            tk.Label(left_frame, textvariable=self.bs_gvars[g_name.lower()], bg=C["bg"], fg=C["accent_b"], font=FONTS["small"]).pack(pady=2)

        right_frame = tk.Frame(p, bg=C["bg"])
        right_frame.pack(side="right", fill="both", expand=True)

        self.bs_fig = FigureCanvasTkAgg(plt.figure(figsize=(6.5, 4)), master=right_frame)
        self.bs_fig.get_tk_widget().pack(fill="both", expand=True)

    def _run_bs(self):
        S = self.S_var.get()
        K = self.K_var.get()
        expiry = self.expiry_var.get()
        r = self.r_var.get()
        otype = self.otype_var.get()
        sigma = self.bs_sigma.get()
        days, T = days_to_expiry(expiry)

        price = bs_price(S, K, T, r, sigma, otype)
        self.bs_price_var.set(f"${price:.2f}")

        greeks = bs_greeks(S, K, T, r, sigma, otype)
        for g_name, var in self.bs_gvars.items():
            var.set(f"{greeks[g_name]:.4f}")

        self._plot_bs(S, K, T, r, sigma, otype)

    def _plot_bs(self, S, K, T, r, sigma, otype):
        # Simple placeholder plot for demonstration, you can add your custom logic here
        self.bs_fig.figure.clear()
        ax = self.bs_fig.figure.add_subplot(111)
        ax.set_title("Black-Scholes Price vs Strike", color=C["text"])
        ax.plot([K - 10, K + 10], [0, 0], color=C["accent_b"])  # Placeholder line
        self.bs_fig.draw()

    # ─────────────────────────────────────────────────────────
    #  MERTON JUMP-DIFFUSION TAB
    # ─────────────────────────────────────────────────────────

    def _build_merton_tab(self):
        p = self.tab_merton
        left_frame = tk.Frame(p, bg=C["bg"])
        left_frame.pack(side="left", fill="y", padx=10)

        tk.Label(left_frame, text="MERTON JUMP-DIFFUSION", bg=C["bg"], fg=C["accent_p"], font=FONTS["heading"]).pack(anchor="w", pady=5)

        self.mer_sigma = tk.DoubleVar(value=0.18)
        self.mer_lambda = tk.DoubleVar(value=3.0)
        self.mer_mu_j = tk.DoubleVar(value=-0.05)
        self.mer_sigma_j = tk.DoubleVar(value=0.09)

        params_frame = tk.Frame(left_frame, bg=C["bg"])
        params_frame.pack(pady=5)

        parameters = [
            ("Diffusion Vol (sigma)", self.mer_sigma),
            ("Jump Intensity (lambda)", self.mer_lambda),
            ("Mean Jump (mu_j)", self.mer_mu_j),
            ("Jump Vol (sigma_j)", self.mer_sigma_j),
        ]

        for label, var in parameters:
            tk.Label(params_frame, text=label, bg=C["bg"], fg=C["text"], font=FONTS["small"]).pack(side="left")
            tk.Entry(params_frame, textvariable=var, width=10).pack(side="left", padx=5)

        tk.Button(left_frame, text="PRICE", command=self._run_merton).pack(pady=10)

        self.mer_price_var = tk.StringVar(value="--")
        tk.Label(left_frame, text="Price (USD)", bg=C["bg"], fg=C["text"], font=FONTS["small"]).pack(pady=(10, 0))
        tk.Label(left_frame, textvariable=self.mer_price_var, bg=C["bg"], fg=C["accent_p"], font=FONTS["title"]).pack(pady=(0, 20))

        self.mer_gvars = {}
        for g_name in ["Delta", "Gamma", "Vega", "Theta"]:
            self.mer_gvars[g_name.lower()] = tk.StringVar(value="--")
            tk.Label(left_frame, text=g_name, bg=C["bg"], fg=C["text"], font=FONTS["small"]).pack(pady=2)
            tk.Label(left_frame, textvariable=self.mer_gvars[g_name.lower()], bg=C["bg"], fg=C["accent_p"], font=FONTS["small"]).pack(pady=2)

        right_frame = tk.Frame(p, bg=C["bg"])
        right_frame.pack(side="right", fill="both", expand=True)

        self.mer_fig = FigureCanvasTkAgg(plt.figure(figsize=(6.5, 4)), master=right_frame)
        self.mer_fig.get_tk_widget().pack(fill="both", expand=True)

    def _run_merton(self):
        S = self.S_var.get()
        K = self.K_var.get()
        expiry = self.expiry_var.get()
        r = self.r_var.get()
        otype = self.otype_var.get()
        sigma = self.mer_sigma.get()
        lam = self.mer_lambda.get()
        mu_j = self.mer_mu_j.get()
        sigma_j = self.mer_sigma_j.get()
        days, T = days_to_expiry(expiry)

        price = merton_price(S, K, T, r, sigma, lam, mu_j, sigma_j, otype)
        self.mer_price_var.set(f"${price:.2f}")

        greeks = merton_greeks(S, K, T, r, sigma, lam, mu_j, sigma_j, otype)
        for g_name, var in self.mer_gvars.items():
            var.set(f"{greeks[g_name]:.4f}")

        self._plot_merton(S, K, T, r, sigma, lam, mu_j, sigma_j, otype)

    def _plot_merton(self, S, K, T, r, sigma, lam, mu_j, sigma_j, otype):
        # Simple placeholder plot for demonstration, you can add your custom logic here
        self.mer_fig.figure.clear()
        ax = self.mer_fig.figure.add_subplot(111)
        ax.set_title("Merton Price vs Strike", color=C["text"])
        ax.plot([K - 10, K + 10], [0, 0], color=C["accent_p"])  # Placeholder line
        self.mer_fig.draw()

    # ─────────────────────────────────────────────────────────
    #  DUPIRE LOCAL VOL TAB
    # ─────────────────────────────────────────────────────────

    def _build_dupire_tab(self):
        p = self.tab_dupire
        left_frame = tk.Frame(p, bg=C["bg"])
        left_frame.pack(side="left", fill="y", padx=10)

        tk.Label(left_frame, text="DUPIRE LOCAL VOLATILITY", bg=C["bg"], fg=C["accent_y"],
                 font=FONTS["heading"]).pack(anchor="w", pady=5)

        self.dup_base_vol = tk.DoubleVar(value=0.22)
        self.dup_skew = tk.DoubleVar(value=-0.10)
        self.dup_smile = tk.DoubleVar(value=0.03)
        self.dup_n_paths = tk.IntVar(value=15000)

        params_frame = tk.Frame(left_frame, bg=C["bg"])
        params_frame.pack(pady=5)

        parameters = [
            ("ATM vol (base)", self.dup_base_vol),
            ("Skew slope", self.dup_skew),
            ("Smile curve", self.dup_smile),
            ("MC paths", self.dup_n_paths)
        ]

        for label, var in parameters:
            tk.Label(params_frame, text=label, bg=C["bg"], fg=C["text"], font=FONTS["small"]).pack(side="left")
            tk.Entry(params_frame, textvariable=var, width=10).pack(side="left", padx=5)

        tk.Button(left_frame, text="PRICE", command=self._run_dupire).pack(pady=10)

        self.dup_price_var = tk.StringVar(value="--")
        tk.Label(left_frame, text="Price (USD)", bg=C["bg"], fg=C["text"], font=FONTS["small"]).pack(pady=(10, 0))
        tk.Label(left_frame, textvariable=self.dup_price_var, bg=C["bg"], fg=C["accent_y"], font=FONTS["title"]).pack(pady=(0, 20))

        self.dup_gvars = {}
        for g_name in ["Delta", "Gamma", "Theta"]:
            self.dup_gvars[g_name.lower()] = tk.StringVar(value="--")
            tk.Label(left_frame, text=g_name, bg=C["bg"], fg=C["text"], font=FONTS["small"]).pack(pady=2)
            tk.Label(left_frame, textvariable=self.dup_gvars[g_name.lower()], bg=C["bg"], fg=C["accent_y"], font=FONTS["small"]).pack(pady=2)

        right_frame = tk.Frame(p, bg=C["bg"])
        right_frame.pack(side="right", fill="both", expand=True)

        self.dup_fig = FigureCanvasTkAgg(plt.figure(figsize=(6.5, 4)), master=right_frame)
        self.dup_fig.get_tk_widget().pack(fill="both", expand=True)

    def _run_dupire(self):
        S = self.S_var.get()
        K = self.K_var.get()
        expiry = self.expiry_var.get()
        r = self.r_var.get()
        otype = self.otype_var.get()
        base_vol = self.dup_base_vol.get()
        skew = self.dup_skew.get()
        smile = self.dup_smile.get()
        n_paths = self.dup_n_paths.get()
        days, T = days_to_expiry(expiry)

        strikes, expiries, iv_surf, lv_surf, local_vol_fn = build_vol_surface(S, r, base_vol, skew, smile)
        price, stderr = dupire_mc_price(S, K, T, r, local_vol_fn, otype, n_paths=n_paths)

        self.dup_price_var.set(f"${price:.2f}")

        greeks = dupire_greeks(S, K, T, r, local_vol_fn, otype, n_paths=n_paths)
        for g_name, var in self.dup_gvars.items():
            var.set(f"{greeks[g_name]:.4f}")

        self._plot_dupire(S, K, T, r, base_vol, skew, smile, price, stderr, strikes, expiries, iv_surf, lv_surf)

    def _plot_dupire(self, S, K, T, r, base_vol, skew, smile, price, stderr,
                     strikes, expiries, iv_surf, lv_surf):
        self.dup_fig.figure.clear()
        ax = self.dup_fig.figure.add_subplot(111)
        ax.set_title("Dupire Local Volatility Price vs Strike", color=C["text"])
        ax.plot([K - 10, K + 10], [0, 0], color=C["accent_y"])  # Placeholder line
        self.dup_fig.draw()

    # ─────────────────────────────────────────────────────────
    #  VOL SURFACE TAB
    # ─────────────────────────────────────────────────────────

    def _build_surface_tab(self):
        p = self.tab_surf
        left_frame = tk.Frame(p, bg=C["bg"])
        left_frame.pack(side="left", fill="y", padx=10)

        tk.Label(left_frame, text="VOL SURFACE EXPLORER", bg=C["bg"], fg=C["accent_g"],
                 font=FONTS["heading"]).pack(anchor="w", pady=5)

        self.surf_base_vol = tk.DoubleVar(value=0.22)
        self.surf_skew = tk.DoubleVar(value=-0.10)
        self.surf_smile = tk.DoubleVar(value=0.03)

        params_frame = tk.Frame(left_frame, bg=C["bg"])
        params_frame.pack(pady=5)

        parameters = [
            ("ATM vol (base)", self.surf_base_vol),
            ("Skew slope", self.surf_skew),
            ("Smile curve", self.surf_smile)
        ]

        for label, var in parameters:
            tk.Label(params_frame, text=label, bg=C["bg"], fg=C["text"], font=FONTS["small"]).pack(side="left")
            tk.Entry(params_frame, textvariable=var, width=10).pack(side="left", padx=5)

        tk.Button(left_frame, text="BUILD SURFACE", command=self._run_surface).pack(pady=10)

        self.surf_info_var = tk.StringVar(value="Build a surface to see stats.")
        tk.Label(left_frame, textvariable=self.surf_info_var, bg=C["bg"],
                 fg=C["text"], font=FONTS["small"], justify="left").pack(pady=(10, 0))

        right_frame = tk.Frame(p, bg=C["bg"])
        right_frame.pack(side="right", fill="both", expand=True)

        self.surf_fig = FigureCanvasTkAgg(plt.figure(figsize=(6.5, 4)), master=right_frame)
        self.surf_fig.get_tk_widget().pack(fill="both", expand=True)

    def _run_surface(self):
        S = self.S_var.get()
        K = self.K_var.get()
        expiry = self.expiry_var.get()
        r = self.r_var.get()
        base_vol = self.surf_base_vol.get()
        skew = self.surf_skew.get()
        smile = self.surf_smile.get()

        strikes, expiries, iv_surf, lv_surf, _ = build_vol_surface(S, r, base_vol, skew, smile)

        atm_idx = np.argmin(np.abs(strikes - S))
        self.surf_info_var.set(
            f"Strikes:  {len(strikes)}\n"
            f"Expiries: {len(expiries)}\n"
            f"ATM IV:   {iv_surf[atm_idx, 3]*100:.1f}%\n"
            f"Min IV:   {iv_surf.min()*100:.1f}%\n"
            f"Max IV:   {iv_surf.max()*100:.1f}%\n"
            f"ATM LV:   {lv_surf[atm_idx, 3]*100:.1f}%\n"
            f"LV/IV:    {(lv_surf / iv_surf).mean():.2f} (rule of 2 ~ 0.5)"
        )

        self._plot_surface(strikes, expiries, iv_surf, lv_surf)

    def _plot_surface(self, strikes, expiries, iv_surf, lv_surf):
        self.surf_fig.figure.clear()
        ax1 = self.surf_fig.figure.add_subplot(121)
        ax2 = self.surf_fig.figure.add_subplot(122)

        # IV surface
        ax1.set_title("Implied Volatility Surface")
        c1 = ax1.contourf(strikes, expiries, iv_surf.T * 100, cmap='RdYlGn_r')
        self.surf_fig.figure.colorbar(c1, ax=ax1)
        ax1.set_xlabel('Strike (K)')
        ax1.set_ylabel('Expiry (Years)')

        # LV surface
        ax2.set_title("Local Volatility Surface")
        c2 = ax2.contourf(strikes, expiries, lv_surf.T * 100, cmap='RdYlGn_r')
        self.surf_fig.figure.colorbar(c2, ax=ax2)
        ax2.set_xlabel('Strike (K)')
        ax2.set_ylabel('Expiry (Years)')

        self.surf_fig.draw()

    def _fetch_live_data(self):
        ticker = self.ticker_var.get().strip().upper()
        self.status_var.set(f"Fetching live data for {ticker}...")

        def fetch():
            try:
                tk_ = yf.Ticker(ticker)
                hist = tk_.history(period="6mo")
                if hist.empty:
                    raise ValueError(f"No data for {ticker}")
                S   = float(hist["Close"].iloc[-1])
                ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                hv  = float(ret.rolling(60).std().iloc[-1] * np.sqrt(252))

                def apply():
                    self.S_var.set(round(S, 2))
                    self.K_var.set(round(S))
                    self.bs_sigma.set(round(hv, 4))
                    self.mer_sigma.set(round(hv * 0.8, 4))
                    self.dup_base_vol.set(round(hv, 4))
                    self.surf_base_vol.set(round(hv, 4))
                    self.status_var.set(f"Loaded data: {ticker} | S=${S:.2f} | HV={hv * 100:.2f}%")

                self.after(0, apply)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Fetch Error", str(e)))
                self.after(0, lambda: self.status_var.set("Fetch failed."))

        threading.Thread(target=fetch, daemon=True).start()


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QuantDeskApp()
    app.mainloop()
