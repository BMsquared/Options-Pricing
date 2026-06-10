import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple

"""
Kelly's Criterion Portfolio Optimization
=========================================
Implementation based on Peterson (2017):
"Kelly's Criterion in Portfolio Optimization: A Decoupled Problem"

Models implemented:
  1. Mean-Variance (MV) model  — Markowitz 1952
  2. Decoupled Kelly model      — Peterson 2017

Optimizer: Differential Evolution (scipy)
Validator: Monte Carlo simulation
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  PAPER DATA  (Tables 1 & 2)
# ─────────────────────────────────────────────

# Average monthly returns for 10 stocks
AVG_RETURNS = np.array([
    0.1750, 0.0995, 0.3398, 0.2366, 0.1149,
    0.2799, 0.2158, 0.2593, 0.2686, 0.4405
])

# Covariance matrix
COV_MATRIX = np.array([
    [0.1817, 0.0978, 0.1403, 0.0962, 0.0481, 0.1745, 0.0752, 0.0574, 0.1326, 0.0040],
    [0.0978, 0.1370, 0.1246, 0.0696, 0.0288, 0.1823, 0.0924, 0.0438, 0.1121, 0.0402],
    [0.1403, 0.1246, 0.2778, 0.1352, 0.0582, 0.1756, 0.0914, 0.0953, 0.1621, 0.0703],
    [0.0962, 0.0696, 0.1352, 0.1121, 0.0443, 0.1440, 0.0617, 0.0572, 0.1198, 0.0321],
    [0.0481, 0.0288, 0.0582, 0.0443, 0.0619, 0.0970, 0.0338, 0.0537, 0.0690,-0.0141],
    [0.1745, 0.1823, 0.1756, 0.1440, 0.0970, 0.3495, 0.1543, 0.1055, 0.2195, 0.0291],
    [0.0752, 0.0924, 0.0914, 0.0617, 0.0338, 0.1543, 0.1161, 0.0581, 0.1160, 0.0484],
    [0.0574, 0.0438, 0.0953, 0.0572, 0.0537, 0.1055, 0.0581, 0.0763, 0.0844, 0.0216],
    [0.1326, 0.1121, 0.1621, 0.1198, 0.0690, 0.2195, 0.1160, 0.0844, 0.2068, 0.0466],
    [0.0040, 0.0402, 0.0703, 0.0321,-0.0141, 0.0291, 0.0484, 0.0216, 0.0466, 0.0839],
])

N_ASSETS = len(AVG_RETURNS)

# ─────────────────────────────────────────────
# 2.  DRIFT & VOLATILITY  (Eq. 20)
# ─────────────────────────────────────────────

def compute_drift_volatility(avg_returns: np.ndarray,
                              cov_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive log-normal drift (mu) and volatility (sigma) from
    sample mean and variance of monthly returns.

    mu_i    = ln(1 + Avg[R_i])
    sigma_i = sqrt( ln( Var[R_i] * exp(-2*mu_i) + 1 ) )
    """
    variances = np.diag(cov_matrix)
    mu    = np.log(1.0 + avg_returns)
    sigma = np.sqrt(np.log(variances * np.exp(-2.0 * mu) + 1.0))
    return mu, sigma


MU, SIGMA = compute_drift_volatility(AVG_RETURNS, COV_MATRIX)

# ─────────────────────────────────────────────
# 3.  KELLY EXPECTATION  (Eq. 21)
# ─────────────────────────────────────────────

def kelly_expectation(f_i: float, mu_i: float, sigma_i: float) -> float:
    """
    Numerically evaluate   E[ ln(1 + f_i * X_i) ]
    where X_i ~ log-normal with parameters mu_i, sigma_i.

    Integral over standard normal y:
      (1/sqrt(2π)) ∫ ln(1 + f_i*(exp(mu_i - 0.5*sigma_i² + sigma_i*y) - 1)) * exp(-y²/2) dy
    """
    adj_mu = mu_i - 0.5 * sigma_i ** 2

    def integrand(y: float) -> float:
        x_val = np.exp(adj_mu + sigma_i * y) - 1.0
        inner = 1.0 + f_i * x_val
        if inner <= 0:
            return -1e6 * norm.pdf(y)          # penalise ruin
        return np.log(inner) * norm.pdf(y)

    result, _ = quad(integrand, -8, 8, limit=100)
    return result


# ─────────────────────────────────────────────
# 4.  RETURN & RISK FUNCTIONS
# ─────────────────────────────────────────────

def mv_return(F: np.ndarray, avg_returns: np.ndarray) -> float:
    """MV linear return  (Eq. 2):  R = Σ F_i * E[X_i]"""
    return float(np.dot(F, avg_returns))


def mv_risk(F: np.ndarray, cov_matrix: np.ndarray) -> float:
    """MV quadratic risk  (Eq. 12):  Var = F' M F"""
    return float(F @ cov_matrix @ F)


def kelly_return(f: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Decoupled Kelly return  (Eq. 10):
      R = Σ f_i * ( exp(E[ln(1 + f_i*X_i)]) - 1 )
    Note: actual wealth fraction = f_i²
    """
    total = 0.0
    for i in range(len(f)):
        e_val = kelly_expectation(f[i], mu[i], sigma[i])
        total += f[i] * (np.exp(e_val) - 1.0)
    return total


def kelly_risk(f: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Decoupled Kelly risk  (Eq. 14):
      Var = Σ_i ( f_i^4 * M_ii  +  2 * Σ_{j>i} f_i² f_j² M_ij )
    """
    N = len(f)
    risk = 0.0
    for i in range(N):
        risk += f[i]**4 * cov_matrix[i, i]
        for j in range(i + 1, N):
            risk += 2.0 * f[i]**2 * f[j]**2 * cov_matrix[i, j]
    return risk


# ─────────────────────────────────────────────
# 5.  OBJECTIVE FUNCTIONS  (Eq. 16 & 17)
# ─────────────────────────────────────────────

def mv_objective(F: np.ndarray, P: float,
                 avg_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
    """MV single-objective  (Eq. 16) — negate for minimisation."""
    ret  = mv_return(F, avg_returns)
    risk = mv_risk(F, cov_matrix)
    return -(P * ret - (1.0 - P) * risk)


def kelly_objective(f: np.ndarray, P: float,
                    mu: np.ndarray, sigma: np.ndarray,
                    cov_matrix: np.ndarray) -> float:
    """Decoupled Kelly single-objective  (Eq. 17) — negate for minimisation."""
    ret  = kelly_return(f, mu, sigma)
    risk = kelly_risk(f, cov_matrix)
    return -(P * ret - (1.0 - P) * risk)


# ─────────────────────────────────────────────
# 6.  CONSTRAINT HELPERS
# ─────────────────────────────────────────────

K_MIN, K_MAX = 0.05, 0.95          # cardinality bounds

def project_mv(F: np.ndarray) -> np.ndarray:
    """Clip then re-normalise MV weights so Σ F_i = 1."""
    F = np.clip(F, K_MIN, K_MAX)
    return F / F.sum()


def project_kelly(f: np.ndarray) -> np.ndarray:
    """
    For Kelly the actual wealth fractions are f_i².
    Clip f so that f_i² ∈ [K_MIN, K_MAX], then
    normalise so Σ f_i² = 1.
    """
    f2 = np.clip(f**2, K_MIN, K_MAX)
    f2 /= f2.sum()
    return np.sqrt(f2)


# ─────────────────────────────────────────────
# 7.  DIFFERENTIAL EVOLUTION SOLVER
# ─────────────────────────────────────────────

def _make_mv_obj(P, avg_returns, cov_matrix):
    def obj(F):
        F = project_mv(F)
        return mv_objective(F, P, avg_returns, cov_matrix)
    return obj


def _make_kelly_obj(P, mu, sigma, cov_matrix):
    def obj(f):
        f = project_kelly(f)
        return kelly_objective(f, P, mu, sigma, cov_matrix)
    return obj


def solve_mv(P: float,
             avg_returns: np.ndarray = AVG_RETURNS,
             cov_matrix: np.ndarray  = COV_MATRIX,
             seed: int = 42) -> dict:
    """Solve the MV portfolio optimisation for a given risk parameter P."""
    bounds = [(K_MIN, K_MAX)] * N_ASSETS
    result = differential_evolution(
        _make_mv_obj(P, avg_returns, cov_matrix),
        bounds,
        seed=seed,
        maxiter=2000,
        tol=1e-8,
        popsize=20,
        mutation=(0.5, 1.0),
        recombination=0.75,
        polish=True,
    )
    F_opt = project_mv(result.x)
    return {
        "weights":  F_opt,
        "return":   mv_return(F_opt, avg_returns),
        "risk":     mv_risk(F_opt, cov_matrix),
        "P":        P,
        "model":    "MV",
        "success":  result.success,
    }


def solve_kelly(P: float,
                mu: np.ndarray         = MU,
                sigma: np.ndarray      = SIGMA,
                cov_matrix: np.ndarray = COV_MATRIX,
                seed: int = 42) -> dict:
    """Solve the decoupled Kelly portfolio optimisation for a given risk parameter P."""
    bounds = [(np.sqrt(K_MIN), np.sqrt(K_MAX))] * N_ASSETS
    result = differential_evolution(
        _make_kelly_obj(P, mu, sigma, cov_matrix),
        bounds,
        seed=seed,
        maxiter=2000,
        tol=1e-8,
        popsize=20,
        mutation=(0.5, 1.0),
        recombination=0.75,
        polish=True,
    )
    f_opt = project_kelly(result.x)
    F_opt = f_opt ** 2            # actual wealth fractions
    return {
        "f_raw":    f_opt,
        "weights":  F_opt,
        "return":   kelly_return(f_opt, mu, sigma),
        "risk":     kelly_risk(f_opt, cov_matrix),
        "P":        P,
        "model":    "Kelly",
        "success":  result.success,
    }


# ─────────────────────────────────────────────
# 8.  MONTE CARLO VALIDATOR
# ─────────────────────────────────────────────

def monte_carlo_portfolio(weights: np.ndarray,
                          mu: np.ndarray,
                          sigma: np.ndarray,
                          cov_matrix: np.ndarray,
                          n_samples: int = 10_000,
                          n_periods: int = 1,
                          seed: int = 0) -> dict:
    """
    Simulate portfolio returns using correlated log-normal draws.

    Returns dict with simulated mean return, std, and return-to-risk ratio.
    """
    rng = np.random.default_rng(seed)
    adj_mu = mu - 0.5 * sigma**2

    # Cholesky decomposition for correlated sampling
    corr = np.diag(1.0 / sigma) @ cov_matrix @ np.diag(1.0 / sigma)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr + 1e-8 * np.eye(N_ASSETS))

    portfolio_returns = []
    for _ in range(n_periods):
        z = rng.standard_normal((n_samples, N_ASSETS))
        z_corr = z @ L.T
        X = np.exp(adj_mu + sigma * z_corr) - 1.0   # log-normal returns
        port_ret = X @ weights
        portfolio_returns.append(port_ret)

    port_returns = np.mean(portfolio_returns, axis=0)
    avg_ret  = float(np.mean(port_returns))
    std_ret  = float(np.std(port_returns))
    rtr      = avg_ret / std_ret if std_ret > 0 else 0.0

    return {
        "mean_return": avg_ret,
        "std":         std_ret,
        "return_to_risk": rtr,
        "samples":     port_returns,
    }


# ─────────────────────────────────────────────
# 9.  EFFICIENT FRONTIER
# ─────────────────────────────────────────────

def compute_efficient_frontier(model: str = "mv",
                                n_points: int = 30) -> pd.DataFrame:
    """
    Sweep risk parameter P from 0.1 to 0.9 and collect (risk, return) pairs.
    model = 'mv' | 'kelly'
    """
    P_values = np.linspace(0.1, 0.9, n_points)
    records  = []
    for P in P_values:
        if model == "mv":
            res = solve_mv(P)
        else:
            res = solve_kelly(P)
        records.append({
            "P":      P,
            "return": res["return"],
            "risk":   res["risk"],
            "model":  res["model"],
        })
        print(f"  {model.upper()} P={P:.2f}  ret={res['return']:.4f}  risk={res['risk']:.4f}")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 10. PORTFOLIO SUMMARY HELPER
# ─────────────────────────────────────────────

def portfolio_summary(result: dict,
                       mc_result: dict | None = None) -> str:
    """Pretty-print a portfolio result."""
    lines = [
        f"\n{'='*55}",
        f"  Model : {result['model']}   |   Risk Param P = {result['P']}",
        f"{'='*55}",
        f"  Objective Return : {result['return']:.4f}",
        f"  Objective Risk   : {result['risk']:.4f}",
        f"  Return/Risk      : {result['return']/result['risk']:.4f}",
        "",
        "  Asset Weights:",
    ]
    for i, w in enumerate(result["weights"]):
        bar = "█" * int(w * 40)
        lines.append(f"    Stock {i+1:2d}: {w:.4f}  {bar}")
    if mc_result:
        lines += [
            "",
            "  Monte Carlo (10,000 simulations):",
            f"    Mean Return : {mc_result['mean_return']:.4f}",
            f"    Std Dev     : {mc_result['std']:.4f}",
            f"    Return/Risk : {mc_result['return_to_risk']:.4f}",
        ]
    lines.append("=" * 55)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 11. VISUALISATION
# ─────────────────────────────────────────────

def plot_results(mv_results: list[dict],
                 kelly_results: list[dict],
                 mc_mv: list[dict],
                 mc_kelly: list[dict],
                 save_path: str = "/mnt/user-data/outputs/kelly_portfolio_results.png"):
    """Generate a 4-panel summary figure."""

    P_vals = [r["P"] for r in mv_results]

    fig = plt.figure(figsize=(16, 12), facecolor="#0d1117")
    fig.suptitle("Kelly vs Mean-Variance Portfolio Optimisation",
                 fontsize=18, color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                           left=0.08, right=0.96, top=0.93, bottom=0.07)

    GOLD  = "#f4c430"
    BLUE  = "#4fc3f7"
    GRID  = "#2a2a3e"
    TEXT  = "#e0e0e0"

    def style_ax(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.set_title(title, color=TEXT, fontsize=11, pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)

    # ── Panel 1: Return vs P ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(P_vals, [r["return"] for r in mv_results],
             "o-", color=BLUE, lw=2, ms=6, label="Mean-Variance")
    ax1.plot(P_vals, [r["return"] for r in kelly_results],
             "s--", color=GOLD, lw=2, ms=6, label="Decoupled Kelly")
    ax1.set_xlabel("Risk Parameter P")
    ax1.set_ylabel("Portfolio Return")
    ax1.legend(facecolor="#1a1a2e", edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax1, "Return vs Risk Parameter")

    # ── Panel 2: Risk vs P ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(P_vals, [r["risk"] for r in mv_results],
             "o-", color=BLUE, lw=2, ms=6, label="Mean-Variance")
    ax2.plot(P_vals, [r["risk"] for r in kelly_results],
             "s--", color=GOLD, lw=2, ms=6, label="Decoupled Kelly")
    ax2.set_xlabel("Risk Parameter P")
    ax2.set_ylabel("Portfolio Risk (Variance)")
    ax2.legend(facecolor="#1a1a2e", edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax2, "Risk vs Risk Parameter")

    # ── Panel 3: Return-to-Risk ratio (MC) ────
    ax3 = fig.add_subplot(gs[1, 0])
    mc_mv_rtr    = [m["return_to_risk"] for m in mc_mv]
    mc_kelly_rtr = [m["return_to_risk"] for m in mc_kelly]
    ax3.plot(P_vals, mc_mv_rtr,    "o-",  color=BLUE, lw=2, ms=6, label="MV (Monte Carlo)")
    ax3.plot(P_vals, mc_kelly_rtr, "s--", color=GOLD, lw=2, ms=6, label="Kelly (Monte Carlo)")
    ax3.set_xlabel("Risk Parameter P")
    ax3.set_ylabel("Return / Risk Ratio")
    ax3.legend(facecolor="#1a1a2e", edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax3, "Return-to-Risk Ratio (Monte Carlo Validated)")

    # ── Panel 4: Asset weights at P=0.9 ───────
    ax4 = fig.add_subplot(gs[1, 1])
    idx9_mv    = next(i for i, r in enumerate(mv_results)    if abs(r["P"] - 0.9) < 0.01)
    idx9_kelly = next(i for i, r in enumerate(kelly_results) if abs(r["P"] - 0.9) < 0.01)
    x = np.arange(N_ASSETS)
    w = 0.35
    ax4.bar(x - w/2, mv_results[idx9_mv]["weights"],    w, color=BLUE, alpha=0.85, label="MV")
    ax4.bar(x + w/2, kelly_results[idx9_kelly]["weights"], w, color=GOLD, alpha=0.85, label="Kelly")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"S{i+1}" for i in range(N_ASSETS)], fontsize=8)
    ax4.set_xlabel("Stock")
    ax4.set_ylabel("Weight")
    ax4.legend(facecolor="#1a1a2e", edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    style_ax(ax4, "Portfolio Weights at P = 0.9")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Figure saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  Kelly Criterion Portfolio Optimisation")
    print("  Peterson (2017) — Full Implementation")
    print("="*55)

    P_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

    # ── Solve both models ─────────────────────
    print("\n[1/4] Solving Mean-Variance model …")
    mv_results = []
    for P in P_VALUES:
        print(f"  P = {P} …", end=" ", flush=True)
        res = solve_mv(P)
        mv_results.append(res)
        print(f"ret={res['return']:.4f}  risk={res['risk']:.5f}  ✓")

    print("\n[2/4] Solving Decoupled Kelly model …")
    kelly_results = []
    for P in P_VALUES:
        print(f"  P = {P} …", end=" ", flush=True)
        res = solve_kelly(P)
        kelly_results.append(res)
        print(f"ret={res['return']:.4f}  risk={res['risk']:.5f}  ✓")

    # ── Monte Carlo validation ────────────────
    print("\n[3/4] Running Monte Carlo validation (10k samples each) …")
    mc_mv, mc_kelly = [], []
    for i, P in enumerate(P_VALUES):
        mc_mv.append(monte_carlo_portfolio(
            mv_results[i]["weights"], MU, SIGMA, COV_MATRIX))
        mc_kelly.append(monte_carlo_portfolio(
            kelly_results[i]["weights"], MU, SIGMA, COV_MATRIX))
        print(f"  P={P}  MV rtr={mc_mv[-1]['return_to_risk']:.4f}"
              f"   Kelly rtr={mc_kelly[-1]['return_to_risk']:.4f}")

    # ── Print summaries ───────────────────────
    print("\n[4/4] Portfolio Summaries\n")
    for i, P in enumerate(P_VALUES):
        print(portfolio_summary(mv_results[i],    mc_mv[i]))
        print(portfolio_summary(kelly_results[i], mc_kelly[i]))

    # ── Comparison table ──────────────────────
    print("\n" + "="*55)
    print("  COMPARISON TABLE")
    print("="*55)
    header = f"{'P':>5} | {'MV Ret':>8} {'MV Risk':>9} | {'K Ret':>8} {'K Risk':>9} | {'MV MC RTR':>10} {'K MC RTR':>10}"
    print(header)
    print("-" * len(header))
    for i, P in enumerate(P_VALUES):
        print(
            f"{P:>5.1f} | "
            f"{mv_results[i]['return']:>8.4f} {mv_results[i]['risk']:>9.5f} | "
            f"{kelly_results[i]['return']:>8.4f} {kelly_results[i]['risk']:>9.5f} | "
            f"{mc_mv[i]['return_to_risk']:>10.4f} {mc_kelly[i]['return_to_risk']:>10.4f}"
        )

    # ── Plot ──────────────────────────────────
    plot_results(mv_results, kelly_results, mc_mv, mc_kelly)

    print("\n✅  Done. Results saved to /mnt/user-data/outputs/")


if __name__ == "__main__":
    main()
