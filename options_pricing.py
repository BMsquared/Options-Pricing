"""
========================================================
  Multi-Model Options Pricing & Comparison
  Models: Black-Scholes | Merton Jump-Diffusion | Dupire Local Vol
========================================================
  pip install numpy scipy matplotlib yfinance
========================================================
"""

import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ── Try importing yfinance for live data ──────────────────────
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not installed — using manual inputs. Run: pip install yfinance")

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not installed — charts disabled. Run: pip install matplotlib")


# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 1 — FETCH LIVE DATA                            ║
# ╚══════════════════════════════════════════════════════════╝

def fetch_live_data(ticker="AAPL"):
    """
    Fetch live stock price and estimate historical volatility
    from Yahoo Finance. Falls back to manual input if unavailable.

    Returns
    -------
    S    : float - current stock price
    hv   : float - 60-day historical volatility (annualized)
    name : str   - company name
    """
    if not YFINANCE_AVAILABLE:
        S = float(input(f"Enter current price for {ticker}: $"))
        hv = float(input("Enter historical volatility (e.g. 0.25 for 25%): "))
        return S, hv, ticker

    print(f"\nFetching live data for {ticker}...")
    tk = yf.Ticker(ticker)
    hist = tk.history(period="6mo")

    if hist.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    S = float(hist['Close'].iloc[-1])
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    hv = float(returns.rolling(60).std().iloc[-1] * np.sqrt(252))
    name = tk.info.get('longName', ticker)

    print(f"  ✓ {name}")
    print(f"  ✓ Last price  : ${S:.2f}")
    print(f"  ✓ 60d HV      : {hv*100:.2f}%")
    return S, hv, name


def fetch_option_chain(ticker="AAPL", expiry_index=1):
    """
    Fetch real option chain from Yahoo Finance.
    Returns the mid prices for calls and puts at available strikes.

    Parameters
    ----------
    ticker       : str - stock ticker
    expiry_index : int - which expiry to use (0=nearest, 1=next, etc.)

    Returns
    -------
    calls_df, puts_df : DataFrames with strike, bid, ask, lastPrice, impliedVolatility
    expiry_date       : str
    """
    if not YFINANCE_AVAILABLE:
        print("yfinance not available — skipping option chain fetch.")
        return None, None, None

    tk = yf.Ticker(ticker)
    expiries = tk.options

    if not expiries:
        print(f"No options data found for {ticker}")
        return None, None, None

    expiry_index = min(expiry_index, len(expiries) - 1)
    expiry_date = expiries[expiry_index]
    print(f"\nFetching option chain for {ticker} expiry: {expiry_date}")

    chain = tk.option_chain(expiry_date)
    calls = chain.calls.copy()
    puts  = chain.puts.copy()

    # compute mid price
    calls['mid'] = (calls['bid'] + calls['ask']) / 2
    puts['mid']  = (puts['bid']  + puts['ask'])  / 2

    # filter out zero-volume / illiquid strikes
    calls = calls[calls['volume'].fillna(0) > 0].reset_index(drop=True)
    puts  = puts[puts['volume'].fillna(0)  > 0].reset_index(drop=True)

    print(f"  ✓ {len(calls)} liquid call strikes")
    print(f"  ✓ {len(puts)}  liquid put strikes")
    return calls, puts, expiry_date


# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 2 — MATH UTILITIES                             ║
# ╚══════════════════════════════════════════════════════════╝

def days_to_expiry(expiry_str):
    """Convert expiry date string (YYYY-MM-DD) to years."""
    from datetime import datetime
    exp = datetime.strptime(str(expiry_str)[:10], "%Y-%m-%d")
    now = datetime.now()
    days = max((exp - now).days, 1)
    return days, days / 365.0


# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 3 — MODEL 1: BLACK-SCHOLES (1973)              ║
# ╚══════════════════════════════════════════════════════════╝

def bs_price(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes closed-form option price.

    Assumes:
    - Constant volatility σ across all strikes and expiries
    - No jumps in the stock price
    - Log-normal stock price distribution

    Parameters
    ----------
    S           : float - current stock price
    K           : float - strike price
    T           : float - time to maturity (years)
    r           : float - risk-free rate
    sigma       : float - constant volatility
    option_type : str   - 'call' or 'put'
    """
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0) if option_type == 'call' \
               else max(K * np.exp(-r * T) - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes greeks."""
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * np.sqrt(T) * norm.pdf(d1) / 100          # per 1% vol
    if option_type == 'call':
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}


def implied_vol_bs(market_price, S, K, T, r, option_type='call', tol=1e-6):
    """Recover implied volatility from market price via Brent's method."""
    try:
        iv = brentq(
            lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price,
            1e-4, 5.0, xtol=tol
        )
        return iv
    except Exception:
        return np.nan


# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 4 — MODEL 2: MERTON JUMP-DIFFUSION (1976)      ║
# ╚══════════════════════════════════════════════════════════╝

def merton_price(S, K, T, r, sigma, lambda_, mu_j, sigma_j,
                 option_type='call', n_terms=50):
    """
    Merton Jump-Diffusion option price.

    Extends Black-Scholes by adding Poisson-distributed price jumps:
        dS/S = (r - λk̄)dt + σdW + JdN

    The price is a weighted sum of BS prices across n jump scenarios,
    weighted by Poisson probabilities.

    Parameters
    ----------
    S        : float - current stock price
    K        : float - strike price
    T        : float - time to maturity (years)
    r        : float - risk-free rate
    sigma    : float - continuous (diffusion) volatility
    lambda_  : float - jump intensity (expected jumps per year)
    mu_j     : float - mean log jump size (negative = downward jumps)
    sigma_j  : float - std dev of log jump size
    n_terms  : int   - Poisson series terms to sum
    """
    k_bar         = np.exp(mu_j + 0.5 * sigma_j**2) - 1   # expected jump size
    lambda_prime  = lambda_ * (1 + k_bar)                  # risk-neutral intensity
    price         = 0.0
    log_factorial = 0.0

    for n in range(n_terms):
        if n > 0:
            log_factorial += np.log(n)

        # Poisson weight for exactly n jumps
        log_weight = (-lambda_prime * T
                      + n * np.log(max(lambda_prime * T, 1e-300))
                      - log_factorial)
        weight = np.exp(log_weight)

        if weight < 1e-12 and n > 5:
            break

        # adjusted drift and vol conditional on n jumps
        r_n     = r - lambda_ * k_bar + (n * mu_j) / T
        sigma_n = np.sqrt(sigma**2 + (n * sigma_j**2) / T)

        price += weight * bs_price(S, K, T, r_n, sigma_n, option_type)

    return price


def merton_greeks(S, K, T, r, sigma, lambda_, mu_j, sigma_j,
                  option_type='call', dS=0.5, dv=0.001):
    """Numerical greeks for Merton model via finite differences."""
    p   = merton_price(S,      K, T,         r, sigma,      lambda_, mu_j, sigma_j, option_type)
    pu  = merton_price(S + dS, K, T,         r, sigma,      lambda_, mu_j, sigma_j, option_type)
    pd  = merton_price(S - dS, K, T,         r, sigma,      lambda_, mu_j, sigma_j, option_type)
    pt  = merton_price(S,      K, T-1/365,   r, sigma,      lambda_, mu_j, sigma_j, option_type)
    pvu = merton_price(S,      K, T,         r, sigma + dv, lambda_, mu_j, sigma_j, option_type)
    pvd = merton_price(S,      K, T,         r, sigma - dv, lambda_, mu_j, sigma_j, option_type)

    delta = (pu - pd) / (2 * dS)
    gamma = (pu - 2 * p + pd) / dS**2
    theta = (pt - p) / (1 / 365)
    vega  = (pvu - pvd) / (2 * dv) / 100   # per 1% vol move
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}


# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 5 — MODEL 3: DUPIRE LOCAL VOL (1994)           ║
# ╚══════════════════════════════════════════════════════════╝

def build_vol_surface(S, r, base_vol, skew_slope=-0.10, smile_curve=0.03):
    """
    Build a realistic implied volatility surface σ(K, T).

    The surface has:
    - Term structure: ATM vol rises with sqrt(T)
    - Skew: lower strikes have higher IV (left skew / crash fear)
    - Smile: symmetric curvature around ATM

    Formula (in log-moneyness m = log(K/S)/sqrt(T)):
        σ(K,T) = ATM(T) + skew_slope * m + smile_curve * m²

    Parameters
    ----------
    S            : float - current stock price
    r            : float - risk-free rate
    base_vol     : float - ATM volatility for reference expiry
    skew_slope   : float - slope of skew (negative = left skew, typical for equities)
    smile_curve  : float - curvature of smile (positive = U-shape)

    Returns
    -------
    strikes, expiries, iv_surface, local_vol_fn
    """
    moneyness = np.array([0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                          1.05, 1.10, 1.15, 1.20, 1.25])
    strikes   = moneyness * S
    expiries  = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.25, 1.5])
    iv_surf   = np.zeros((len(strikes), len(expiries)))

    for j, T in enumerate(expiries):
        for i, m in enumerate(moneyness):
            # log-moneyness scaled by sqrt(T) — standard parameterisation
            log_m   = np.log(m) / np.sqrt(T)
            atm_vol = base_vol * (1 + 0.04 * np.sqrt(T))      # upward term structure
            skew    = skew_slope * log_m
            smile   = smile_curve * log_m**2
            iv_surf[i, j] = np.clip(atm_vol + skew + smile, 0.05, 2.0)

    # ── Dupire formula: extract local vol from call price surface ──
    C_surf = np.zeros_like(iv_surf)
    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            C_surf[i, j] = bs_price(S, K, T, r, iv_surf[i, j], 'call')

    lv_surf = np.zeros_like(iv_surf)
    dK      = np.diff(strikes).mean()

    for j in range(len(expiries)):
        T = expiries[j]
        for i in range(1, len(strikes) - 1):
            K = strikes[i]
            # time derivative dC/dT
            if j == 0:
                dC_dT = (C_surf[i, j+1] - C_surf[i, j]) / (expiries[j+1] - expiries[j])
            elif j == len(expiries) - 1:
                dC_dT = (C_surf[i, j] - C_surf[i, j-1]) / (expiries[j] - expiries[j-1])
            else:
                dC_dT = (C_surf[i, j+1] - C_surf[i, j-1]) / (expiries[j+1] - expiries[j-1])

            dC_dK   = (C_surf[i+1, j] - C_surf[i-1, j]) / (2 * dK)
            d2C_dK2 = (C_surf[i+1, j] - 2*C_surf[i, j] + C_surf[i-1, j]) / dK**2

            num = dC_dT + r * K * dC_dK
            den = 0.5 * K**2 * d2C_dK2

            lv_surf[i, j] = np.sqrt(np.clip(num / den, 0.001, 4.0)) \
                             if den > 1e-10 and num > 0 else iv_surf[i, j]

    lv_surf[0, :]  = lv_surf[1, :]
    lv_surf[-1, :] = lv_surf[-2, :]
    lv_surf = np.clip(lv_surf, 0.01, 2.0)

    # 2D spline interpolator: local_vol_fn(spot, t)
    spline = RectBivariateSpline(strikes, expiries, lv_surf, kx=3, ky=3)

    def local_vol_fn(spot, t):
        spot = np.clip(spot, strikes[0], strikes[-1])
        t    = np.clip(t,    expiries[0], expiries[-1])
        return float(np.clip(spline(spot, t), 0.01, 2.0))

    return strikes, expiries, iv_surf, lv_surf, local_vol_fn


def dupire_mc_price(S, K, T, r, local_vol_fn,
                    option_type='call', n_paths=20000, n_steps=100, seed=42):
    """
    Price an option via Monte Carlo under the Dupire local vol model.

    At each time step, volatility is re-evaluated at the current
    stock price and time: σ_L = local_vol_fn(S_t, t)

    Parameters
    ----------
    S            : float    - current stock price
    K            : float    - strike price
    T            : float    - time to maturity (years)
    r            : float    - risk-free rate
    local_vol_fn : callable - σ_L(spot, t) from build_vol_surface()
    n_paths      : int      - Monte Carlo paths
    n_steps      : int      - time steps per path
    seed         : int      - random seed

    Returns
    -------
    price  : float - option price
    stderr : float - standard error (95% CI = ±1.96 * stderr)
    """
    np.random.seed(seed)
    dt    = T / n_steps
    paths = np.full(n_paths, S, dtype=float)

    for step in range(n_steps):
        t       = step * dt
        Z       = np.random.standard_normal(n_paths)
        sigma_L = np.array([local_vol_fn(s, max(t, expiries_global[0]))
                            for s in paths])
        paths  *= np.exp((r - 0.5 * sigma_L**2) * dt
                         + sigma_L * np.sqrt(dt) * Z)

    payoffs = (np.maximum(paths - K, 0) if option_type == 'call'
               else np.maximum(K - paths, 0))
    price   = np.exp(-r * T) * np.mean(payoffs)
    stderr  = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    return price, stderr


def dupire_greeks(S, K, T, r, local_vol_fn, option_type='call',
                  n_paths=10000, dS=1.0):
    """Numerical greeks for Dupire model."""
    p,  _ = dupire_mc_price(S,      K, T, r, local_vol_fn, option_type, n_paths, seed=1)
    pu, _ = dupire_mc_price(S + dS, K, T, r, local_vol_fn, option_type, n_paths, seed=2)
    pd, _ = dupire_mc_price(S - dS, K, T, r, local_vol_fn, option_type, n_paths, seed=3)
    pt, _ = dupire_mc_price(S, K, max(T - 7/365, 0.01), r, local_vol_fn,
                             option_type, n_paths, seed=4)
    delta = (pu - pd) / (2 * dS)
    gamma = (pu - 2*p + pd) / dS**2
    theta = (pt - p) / (7 / 365) / 365   # per day
    return {'delta': delta, 'gamma': gamma, 'theta': theta}


# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 6 — COMPARISON ENGINE                          ║
# ╚══════════════════════════════════════════════════════════╝

def run_comparison(S, K, T, r, market_price, option_type,
                   bs_sigma, merton_params, local_vol_fn,
                   iv_surf, lv_surf, strikes, expiries):
    """
    Run all three models and produce a full comparison report.

    Parameters
    ----------
    S             : float  - stock price
    K             : float  - strike price
    T             : float  - time to maturity (years)
    r             : float  - risk-free rate
    market_price  : float  - observed market mid price
    option_type   : str    - 'call' or 'put'
    bs_sigma      : float  - volatility for Black-Scholes (surface IV at K,T)
    merton_params : dict   - {sigma, lambda_, mu_j, sigma_j}
    local_vol_fn  : callable
    iv_surf       : 2D array
    lv_surf       : 2D array
    strikes       : array
    expiries      : array
    """
    days = int(T * 365)

    print("\n" + "=" * 65)
    print(f"  OPTION: {option_type.upper()}  |  K=${K:.2f}  |  S=${S:.2f}  "
          f"|  T={days}d  |  r={r*100:.1f}%")
    print("=" * 65)
    print(f"  Market mid price    : ${market_price:.4f}")
    print(f"  Surface IV at (K,T) : {bs_sigma*100:.2f}%")
    print("-" * 65)

    # ── BLACK-SCHOLES ─────────────────────────────────────────
    bs_p  = bs_price(S, K, T, r, bs_sigma, option_type)
    bs_iv = implied_vol_bs(market_price, S, K, T, r, option_type)
    bs_g  = bs_greeks(S, K, T, r, bs_sigma, option_type)

    # ── MERTON ───────────────────────────────────────────────
    mp = merton_params
    mer_p  = merton_price(S, K, T, r, mp['sigma'], mp['lambda_'],
                          mp['mu_j'], mp['sigma_j'], option_type)
    mer_g  = merton_greeks(S, K, T, r, mp['sigma'], mp['lambda_'],
                           mp['mu_j'], mp['sigma_j'], option_type)

    # ── DUPIRE ───────────────────────────────────────────────
    print("  Running Dupire MC (20,000 paths)...", end="", flush=True)
    dup_p, dup_se = dupire_mc_price(S, K, T, r, local_vol_fn,
                                    option_type, n_paths=20000)
    dup_g = dupire_greeks(S, K, T, r, local_vol_fn, option_type)
    print(" done.")

    # ── MARKET IMPLIED VOL ────────────────────────────────────
    mkt_iv = implied_vol_bs(market_price, S, K, T, r, option_type)

    # ── PRINT RESULTS ─────────────────────────────────────────
    print()
    print(f"  {'Model':<22} {'Price':>8}  {'vs Market':>10}  {'vs Mkt%':>8}")
    print(f"  {'-'*52}")
    for name, price in [("Black-Scholes (1973)", bs_p),
                         ("Merton Jump (1976)",   mer_p),
                         ("Dupire Local Vol (1994)", dup_p)]:
        diff    = price - market_price
        diff_pct= (diff / market_price) * 100
        flag    = "✓" if abs(diff_pct) < 2 else ("↑" if diff > 0 else "↓")
        print(f"  {name:<22} ${price:>7.4f}  {diff:>+10.4f}  {diff_pct:>+7.2f}%  {flag}")
    print(f"  {'Market Mid':<22} ${market_price:>7.4f}  {'—':>10}  {'—':>8}")
    print(f"\n  Market IV (from mid)    : {mkt_iv*100:.2f}%")
    print(f"  BS IV input (surface)   : {bs_sigma*100:.2f}%")

    print()
    print(f"  {'Greek':<12} {'BS':>10} {'Merton':>10} {'Dupire':>10}")
    print(f"  {'-'*44}")
    for g in ['delta', 'gamma', 'vega', 'theta']:
        dv = bs_g[g]
        mv = mer_g[g]
        dpv = dup_g.get(g, float('nan'))
        print(f"  {g.capitalize():<12} {dv:>10.5f} {mv:>10.5f} {dpv:>10.5f}")

    print()
    print("  INTERPRETATION")
    print("  " + "-" * 60)
    diff_bs  = bs_p  - market_price
    diff_mer = mer_p - market_price
    diff_dup = dup_p - market_price

    print(f"  BS prices {'above' if diff_bs>0 else 'below'} market by "
          f"${abs(diff_bs):.4f} — flat vol {'' if abs(diff_bs)<0.5 else 'mis'}pricing")
    print(f"  Merton {'above' if diff_mer>0 else 'below'} by ${abs(diff_mer):.4f} — "
          f"jump intensity λ={mp['lambda_']}, avg jump {mp['mu_j']*100:.1f}%")
    print(f"  Dupire {'above' if diff_dup>0 else 'below'} by ${abs(diff_dup):.4f} ± "
          f"${dup_se*1.96:.4f} — local vol captures skew")

    closest = min([("BS", abs(diff_bs)), ("Merton", abs(diff_mer)),
                   ("Dupire", abs(diff_dup))], key=lambda x: x[1])
    print(f"\n  Closest model to market: {closest[0]} (diff = ${closest[1]:.4f})")
    print("=" * 65)

    return {
        'bs':      {'price': bs_p,  'iv': bs_iv,  'greeks': bs_g},
        'merton':  {'price': mer_p, 'iv': np.nan,  'greeks': mer_g},
        'dupire':  {'price': dup_p, 'stderr': dup_se, 'greeks': dup_g},
        'market':  {'price': market_price, 'iv': mkt_iv},
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 7 — VOLATILITY SURFACE ANALYSIS                ║
# ╚══════════════════════════════════════════════════════════╝

def print_vol_surface(S, strikes, expiries, iv_surf, lv_surf, K_target, T_target):
    """Print IV and Local Vol surface as a table."""
    print("\n" + "=" * 65)
    print("  VOLATILITY SURFACE — IMPLIED VOL σ(K, T)")
    print("=" * 65)

    exp_labels = [f"{int(e*12)}m" if e < 1 else f"{e:.2f}y" for e in expiries]
    header = f"  {'Strike':>8} " + " ".join(f"{l:>7}" for l in exp_labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    atm = round(S / 5) * 5

    for i, K in enumerate(strikes):
        row = f"  {'→' if abs(K-K_target)<2.5 else ' '} ${K:>6.1f} "
        for j in range(len(expiries)):
            val = iv_surf[i, j] * 100
            row += f"  {val:>5.1f}%"
        tag = " ← ATM" if abs(K - atm) < 2.5 else ""
        print(row + tag)

    print("\n" + "=" * 65)
    print("  LOCAL VOLATILITY SURFACE — σ_L(K, T)  [Dupire]")
    print("=" * 65)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, K in enumerate(strikes):
        row = f"  {'→' if abs(K-K_target)<2.5 else ' '} ${K:>6.1f} "
        for j in range(len(expiries)):
            val = lv_surf[i, j] * 100
            row += f"  {val:>5.1f}%"
        tag = " ← ATM" if abs(K - atm) < 2.5 else ""
        print(row + tag)

    print("\n  Note: IV = what traders imply; Local Vol = what Dupire")
    print("  extracts. At short expiries LV < IV; at long expiries LV > IV.")
    print("  This is the 'rule of two' relationship.")


# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 8 — CHARTS                                     ║
# ╚══════════════════════════════════════════════════════════╝

def plot_comparison(S, K, T, r, results, iv_surf, lv_surf,
                    strikes, expiries, option_type, ticker):
    """
    4-panel chart:
    1. Model prices vs market
    2. Vol surface (IV)
    3. Local vol surface (Dupire)
    4. Price difference vs market across strikes
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available — skipping charts.")
        return

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0a0e17')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = {'BS': '#4da6ff', 'Merton': '#b07eff', 'Dupire': '#ffb830',
              'Market': '#00e5aa', 'bg': '#0a0e17', 'bg2': '#111827',
              'text': '#dce8f5', 'text2': '#7a9bbf', 'grid': '#1a3352'}

    def style_ax(ax, title):
        ax.set_facecolor(colors['bg2'])
        ax.set_title(title, color=colors['text'], fontsize=10, pad=8)
        ax.tick_params(colors=colors['text2'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(colors['grid'])
        ax.grid(True, color=colors['grid'], linewidth=0.5, alpha=0.6)

    # ── PANEL 1: Price bars ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, f"Model Prices vs Market\n{ticker} ${K} {option_type.upper()}, T={int(T*365)}d")
    models = ['BS', 'Merton', 'Dupire', 'Market']
    prices = [results['bs']['price'], results['merton']['price'],
              results['dupire']['price'], results['market']['price']]
    clrs   = [colors['BS'], colors['Merton'], colors['Dupire'], colors['Market']]
    bars   = ax1.bar(models, prices, color=clrs, alpha=0.85, width=0.6)
    ax1.axhline(y=results['market']['price'], color=colors['Market'],
                linestyle='--', linewidth=1, alpha=0.5, label='Market mid')
    for bar, val in zip(bars, prices):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'${val:.2f}', ha='center', va='bottom',
                 color=colors['text'], fontsize=8, fontfamily='monospace')
    ax1.set_ylabel('Price ($)', color=colors['text2'], fontsize=8)
    ax1.yaxis.label.set_color(colors['text2'])

    # ── PANEL 2: IV surface heatmap ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Implied Vol Surface σ(K,T)\n[Full skew + term structure]")
    exp_labels = [f"{int(e*12)}m" if e < 1 else f"{e:.1f}y" for e in expiries]
    im = ax2.imshow(iv_surf * 100, aspect='auto', cmap='RdYlGn_r',
                    origin='lower', vmin=15, vmax=80)
    ax2.set_xticks(range(len(expiries)))
    ax2.set_xticklabels(exp_labels, rotation=45, fontsize=7, color=colors['text2'])
    ax2.set_yticks(range(len(strikes)))
    ax2.set_yticklabels([f'${s:.0f}' for s in strikes], fontsize=7, color=colors['text2'])
    ax2.set_xlabel('Expiry', color=colors['text2'], fontsize=8)
    ax2.set_ylabel('Strike', color=colors['text2'], fontsize=8)
    plt.colorbar(im, ax=ax2, label='IV %').ax.yaxis.set_tick_params(color=colors['text2'])

    # ── PANEL 3: Local vol surface heatmap ───────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, "Dupire Local Vol Surface σ_L(K,T)\n[Extracted via Dupire formula]")
    im3 = ax3.imshow(lv_surf * 100, aspect='auto', cmap='RdYlGn_r',
                     origin='lower', vmin=15, vmax=80)
    ax3.set_xticks(range(len(expiries)))
    ax3.set_xticklabels(exp_labels, rotation=45, fontsize=7, color=colors['text2'])
    ax3.set_yticks(range(len(strikes)))
    ax3.set_yticklabels([f'${s:.0f}' for s in strikes], fontsize=7, color=colors['text2'])
    ax3.set_xlabel('Expiry', color=colors['text2'], fontsize=8)
    plt.colorbar(im3, ax=ax3, label='Local Vol %').ax.yaxis.set_tick_params(color=colors['text2'])

    # ── PANEL 4: BS price across strikes ─────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    style_ax(ax4, f"Model Prices Across Strikes — {option_type.upper()} — T={int(T*365)}d")
    sample_strikes = np.linspace(strikes[0], strikes[-1], 30)
    bs_prices, mer_prices, dup_prices = [], [], []
    mp = merton_params_global
    for Ks in sample_strikes:
        # surface IV for this strike
        iv_k = np.interp(Ks, strikes, iv_surf[:, 3])
        bs_prices.append(bs_price(S, Ks, T, r, iv_k, option_type))
        mer_prices.append(merton_price(S, Ks, T, r, mp['sigma'], mp['lambda_'],
                                       mp['mu_j'], mp['sigma_j'], option_type))
        lv_k = np.interp(Ks, strikes, lv_surf[:, 3])
        dup_prices.append(bs_price(S, Ks, T, r, lv_k, option_type))

    ax4.plot(sample_strikes, bs_prices,  color=colors['BS'],
             label='Black-Scholes', linewidth=1.5)
    ax4.plot(sample_strikes, mer_prices, color=colors['Merton'],
             label='Merton Jump',   linewidth=1.5, linestyle='--')
    ax4.plot(sample_strikes, dup_prices, color=colors['Dupire'],
             label='Dupire LV',     linewidth=1.5, linestyle=':')
    ax4.axvline(x=K, color=colors['Market'], linestyle='--',
                linewidth=1, alpha=0.6, label=f'Selected K=${K}')
    ax4.axvline(x=S, color=colors['text2'], linestyle=':',
                linewidth=1, alpha=0.4, label=f'Spot S=${S:.0f}')
    ax4.scatter([K], [results['market']['price']], color=colors['Market'],
                s=80, zorder=5, label=f'Market mid ${results["market"]["price"]:.2f}')
    ax4.set_xlabel('Strike', color=colors['text2'], fontsize=8)
    ax4.set_ylabel('Option Price ($)', color=colors['text2'], fontsize=8)
    legend = ax4.legend(fontsize=8, facecolor=colors['bg2'],
                        labelcolor=colors['text'], edgecolor=colors['grid'])

    # ── PANEL 5: Vol smile across strikes at selected T ───────
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, f"Vol Smile at T={int(T*365)}d\nIV vs Local Vol")
    iv_at_T  = np.interp(np.linspace(strikes[0], strikes[-1], 30),
                          strikes, iv_surf[:, min(3, len(expiries)-1)])
    lv_at_T  = np.interp(np.linspace(strikes[0], strikes[-1], 30),
                          strikes, lv_surf[:, min(3, len(expiries)-1)])
    xs = np.linspace(strikes[0], strikes[-1], 30)
    ax5.plot(xs, iv_at_T*100,  color=colors['BS'],    label='Implied Vol',   linewidth=1.5)
    ax5.plot(xs, lv_at_T*100,  color=colors['Dupire'],label='Local Vol (LV)',linewidth=1.5, linestyle='--')
    ax5.axvline(x=S, color=colors['Market'], linestyle=':', linewidth=1, alpha=0.5, label=f'ATM=${S:.0f}')
    ax5.set_xlabel('Strike', color=colors['text2'], fontsize=8)
    ax5.set_ylabel('Volatility (%)', color=colors['text2'], fontsize=8)
    ax5.legend(fontsize=8, facecolor=colors['bg2'],
               labelcolor=colors['text'], edgecolor=colors['grid'])

    fig.suptitle(f'QuantDesk — Multi-Model Analysis: {ticker}  ${K} {option_type.upper()}',
                 color=colors['text'], fontsize=13, fontweight='bold', y=0.98)

   # plt.savefig('/mnt/user-data/outputs/model_comparison_chart.png',
               # dpi=150, bbox_inches='tight', facecolor=colors['bg'])
    #print("\n  Chart saved: model_comparison_chart.png")
    plt.show()


# ╔══════════════════════════════════════════════════════════╗
# ║  SECTION 9 — MAIN: RUN EVERYTHING                       ║
# ╚══════════════════════════════════════════════════════════╝

# ── Global state (needed by MC functions) ────────────────────
expiries_global      = None
merton_params_global = None

if __name__ == "__main__":

    # ── 1. CONFIGURE YOUR OPTION HERE ────────────────────────
    TICKER      = "AAPL"       # stock ticker
    K           = 215.0        # strike price
    EXPIRY      = "2026-09-19" # expiry date (YYYY-MM-DD) — or use a real chain date
    r           = 0.05         # risk-free rate
    OPTION_TYPE = "call"       # 'call' or 'put'
    MARKET_PRICE = 54.20       # observed market mid price (replace with real data)

    # Merton parameters (calibrate to your stock)
    MERTON = {
        'sigma':   0.18,   # diffusion vol (continuous part)
        'lambda_': 3.0,    # jumps per year
        'mu_j':   -0.05,   # mean log jump (negative = crashes)
        'sigma_j': 0.09,   # jump vol
    }

    # Dupire surface parameters
    SKEW_SLOPE   = -0.10    # strength of left skew (typical for equities)
    SMILE_CURVE  =  0.03    # smile curvature

    # ── 2. FETCH LIVE DATA ────────────────────────────────────
    S, hv, name = fetch_live_data(TICKER)
    days, T     = days_to_expiry(EXPIRY)
    print(f"\n  Using T = {days} days = {T:.4f} years to {EXPIRY}")

    # ── 3. BUILD VOL SURFACE ──────────────────────────────────
    print("\nBuilding volatility surface...")
    atm_vol = hv  # use historical vol as ATM vol anchor
    strikes, expiries, iv_surf, lv_surf, local_vol_fn = build_vol_surface(
        S, r, atm_vol, skew_slope=SKEW_SLOPE, smile_curve=SMILE_CURVE
    )
    expiries_global = expiries
    merton_params_global = MERTON

    # surface IV at the target (K, T)
    T_clipped = np.clip(T, expiries[0], expiries[-1])
    K_clipped = np.clip(K, strikes[0], strikes[-1])
    # interpolate IV at our specific (K, T)
    from scipy.interpolate import RegularGridInterpolator
    iv_interp = RegularGridInterpolator((strikes, expiries), iv_surf,
                                        method='linear', bounds_error=False,
                                        fill_value=atm_vol)
    bs_sigma = float(iv_interp([[K_clipped, T_clipped]]))

    print(f"  Surface IV at K=${K}, T={days}d : {bs_sigma*100:.2f}%")
    print(f"  ATM vol (HV anchor)            : {atm_vol*100:.2f}%")

    # ── 4. PRINT VOL SURFACES ─────────────────────────────────
    print_vol_surface(S, strikes, expiries, iv_surf, lv_surf, K, T)

    # ── 5. RUN MODEL COMPARISON ───────────────────────────────
    results = run_comparison(
        S, K, T, r,
        market_price = MARKET_PRICE,
        option_type  = OPTION_TYPE,
        bs_sigma     = bs_sigma,
        merton_params= MERTON,
        local_vol_fn = local_vol_fn,
        iv_surf      = iv_surf,
        lv_surf      = lv_surf,
        strikes      = strikes,
        expiries     = expiries,
    )

    # ── 6. OPTIONALLY RUN ACROSS MULTIPLE STRIKES ────────────
    print("\n" + "=" * 65)
    print("  CROSS-STRIKE COMPARISON (same expiry, BS vs Merton vs Dupire)")
    print("=" * 65)
    atm = round(S / 5) * 5
    test_strikes = [atm - 15, atm - 10, atm - 5, atm, atm + 5, atm + 10, atm + 15]
    print(f"\n  {'Strike':>8}  {'Moneyness':>10}  {'BS':>8}  {'Merton':>8}  "
          f"{'Dupire':>8}  {'BS-Mer':>8}  {'BS-Dup':>8}")
    print(f"  {'-'*70}")
    for Ks in test_strikes:
        iv_k = float(iv_interp([[np.clip(Ks, strikes[0], strikes[-1]), T_clipped]]))
        bsp  = bs_price(S, Ks, T, r, iv_k, OPTION_TYPE)
        merp = merton_price(S, Ks, T, r, MERTON['sigma'], MERTON['lambda_'],
                            MERTON['mu_j'], MERTON['sigma_j'], OPTION_TYPE)
        lv_k_arr = np.interp(np.clip(Ks, strikes[0], strikes[-1]),
                             strikes, lv_surf[:, 3])
        dupp = bs_price(S, Ks, T, r, lv_k_arr, OPTION_TYPE)
        itm  = "ITM" if (OPTION_TYPE == 'call' and S > Ks) or \
                        (OPTION_TYPE == 'put'  and S < Ks) else \
               ("ATM" if abs(S - Ks) < 3 else "OTM")
        print(f"  ${Ks:>6.0f}   {itm:>10}   ${bsp:>6.2f}   ${merp:>6.2f}   "
              f"${dupp:>6.2f}   {bsp-merp:>+7.3f}   {bsp-dupp:>+7.3f}")

    # ── 7. CHARTS ─────────────────────────────────────────────
    plot_comparison(
        S, K, T, r, results,
        iv_surf, lv_surf, strikes, expiries,
        OPTION_TYPE, TICKER
    )