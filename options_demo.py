"""
Options Market Demo — Portfolio Piece
======================================
Live US stock data (Yahoo Finance) + Black-Scholes pricing engine
Full paper trading: order book, positions, P&L tracker, Greeks dashboard

Author: Built with WQU MScFE knowledge
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import datetime
import numpy as np
from scipy.stats import norm
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional
import uuid


# ─────────────────────────────────────────────
# PRICING ENGINE
# ─────────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

def bs_put(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    if sigma <= 0 or T <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if option_type == "call":
        delta = norm.cdf(d1)
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * np.sqrt(T) * pdf_d1 / 100
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}

def implied_vol(C_market, S, K, T, r, option_type="call"):
    """Newton-Raphson implied vol solver."""
    sigma = 0.3
    for _ in range(100):
        price  = bs_call(S, K, T, r, sigma) if option_type == "call" else bs_put(S, K, T, r, sigma)
        diff   = price - C_market
        if abs(diff) < 1e-6:
            break
        d1     = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        v      = S * np.sqrt(T) * norm.pdf(d1)
        if abs(v) < 1e-10:
            break
        sigma -= diff / v
        sigma  = max(sigma, 1e-4)
    return sigma

def historical_vol(ticker_data, window=30):
    """Annualized historical volatility from price series."""
    closes  = ticker_data["Close"].squeeze()
    returns = np.log(closes / closes.shift(1)).dropna()
    return float(returns.rolling(window).std().iloc[-1] * np.sqrt(252))

def generate_strikes(S, n=7):
    """Generate realistic strike prices around current spot."""
    step = round(S * 0.025, 0)  # ~2.5% spacing
    step = max(step, 1.0)
    atm  = round(S / step) * step
    return [atm + (i - n // 2) * step for i in range(n)]

def generate_expiries():
    """Generate realistic expiry dates (monthly, next 4 months)."""
    today   = datetime.date.today()
    expiries = []
    for months_ahead in [1, 2, 3, 6]:
        d = today.replace(day=1) + datetime.timedelta(days=32 * months_ahead)
        # third Friday of that month
        first_day = d.replace(day=1)
        first_fri = first_day + datetime.timedelta(days=(4 - first_day.weekday()) % 7)
        third_fri = first_fri + datetime.timedelta(weeks=2)
        expiries.append(third_fri)
    return expiries


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class Option:
    symbol:      str
    option_type: str   # "call" or "put"
    strike:      float
    expiry:      datetime.date
    bid:         float = 0.0
    ask:         float = 0.0
    iv:          float = 0.0
    delta:       float = 0.0
    gamma:       float = 0.0
    vega:        float = 0.0
    theta:       float = 0.0

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    @property
    def T(self):
        days = (self.expiry - datetime.date.today()).days
        return max(days / 365, 1/365)

    @property
    def label(self):
        return f"{self.symbol} {self.expiry.strftime('%b%y')} {self.strike:.0f} {self.option_type.upper()}"


@dataclass
class Order:
    order_id:    str
    option:      Option
    direction:   str   # "buy" or "sell"
    quantity:    int
    price:       float
    timestamp:   datetime.datetime
    status:      str = "filled"  # paper trading: instant fill


@dataclass
class Position:
    option:    Option
    quantity:  int        # positive = long, negative = short
    avg_price: float
    orders:    List[Order] = field(default_factory=list)

    @property
    def market_value(self):
        return self.quantity * self.option.mid * 100  # 1 contract = 100 shares

    @property
    def cost_basis(self):
        return self.quantity * self.avg_price * 100

    @property
    def unrealized_pnl(self):
        return self.market_value - self.cost_basis

    @property
    def pnl_pct(self):
        if abs(self.cost_basis) < 0.01:
            return 0.0
        return (self.unrealized_pnl / abs(self.cost_basis)) * 100


# ─────────────────────────────────────────────
# MARKET DATA ENGINE
# ─────────────────────────────────────────────

class MarketData:
    def __init__(self):
        self.prices    = {}   # symbol -> float
        self.hist_data = {}   # symbol -> DataFrame
        self.hist_vols = {}   # symbol -> float
        self.r         = 0.05
        self._lock     = threading.Lock()

    def fetch(self, symbol: str) -> Optional[float]:
        try:
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(period="3mo", interval="1d")
            if hist.empty:
                return None
            price = float(hist["Close"].iloc[-1])
            vol   = historical_vol(hist, window=30)
            with self._lock:
                self.prices[symbol]    = price
                self.hist_data[symbol] = hist
                self.hist_vols[symbol] = vol
            return price
        except Exception:
            return None

    def price(self, symbol):
        with self._lock:
            return self.prices.get(symbol)

    def vol(self, symbol):
        with self._lock:
            return self.hist_vols.get(symbol, 0.3)

    def price_option(self, opt: Option, S: float, sigma: float) -> Option:
        """Price option and compute Greeks, add realistic bid/ask spread."""
        fn    = bs_call if opt.option_type == "call" else bs_put
        mid   = fn(S, opt.strike, opt.T, self.r, sigma)
        spread = max(0.05, mid * 0.02)   # 2% spread, min $0.05
        opt.bid = max(0.01, mid - spread / 2)
        opt.ask = mid + spread / 2
        opt.iv  = sigma
        g = bs_greeks(S, opt.strike, opt.T, self.r, sigma, opt.option_type)
        opt.delta = g["delta"]
        opt.gamma = g["gamma"]
        opt.vega  = g["vega"]
        opt.theta = g["theta"]
        return opt

    def build_chain(self, symbol: str) -> List[Option]:
        S     = self.price(symbol)
        sigma = self.vol(symbol)
        if S is None:
            return []
        chain   = []
        strikes  = generate_strikes(S)
        expiries = generate_expiries()
        for expiry in expiries:
            for strike in strikes:
                for otype in ["call", "put"]:
                    opt = Option(symbol=symbol, option_type=otype,
                                 strike=strike, expiry=expiry)
                    opt = self.price_option(opt, S, sigma)
                    chain.append(opt)
        return chain


# ─────────────────────────────────────────────
# PORTFOLIO
# ─────────────────────────────────────────────

class Portfolio:
    def __init__(self, starting_cash=100_000.0):
        self.cash      = starting_cash
        self.start     = starting_cash
        self.positions = {}   # label -> Position
        self.orders    = []

    def place_order(self, option: Option, direction: str, qty: int) -> tuple:
        price    = option.ask if direction == "buy" else option.bid
        cost     = price * qty * 100 * (1 if direction == "buy" else -1)
        if direction == "buy" and cost > self.cash:
            return False, f"Insufficient cash. Need ${cost:,.2f}, have ${self.cash:,.2f}"
        order = Order(
            order_id  = str(uuid.uuid4())[:8].upper(),
            option    = option,
            direction = direction,
            quantity  = qty,
            price     = price,
            timestamp = datetime.datetime.now()
        )
        self.orders.append(order)
        self.cash -= cost
        label = option.label
        signed_qty = qty if direction == "buy" else -qty
        if label in self.positions:
            pos = self.positions[label]
            new_qty = pos.quantity + signed_qty
            if new_qty == 0:
                del self.positions[label]
            else:
                total_cost = pos.avg_price * abs(pos.quantity) + price * qty
                pos.avg_price = total_cost / abs(new_qty)
                pos.quantity  = new_qty
                pos.orders.append(order)
        else:
            self.positions[label] = Position(
                option    = option,
                quantity  = signed_qty,
                avg_price = price,
                orders    = [order]
            )
        return True, f"Order {order.order_id} filled — {direction.upper()} {qty}x {option.label} @ ${price:.2f}"

    @property
    def total_value(self):
        pos_value = sum(p.market_value for p in self.positions.values())
        return self.cash + pos_value

    @property
    def total_pnl(self):
        return self.total_value - self.start

    @property
    def total_delta(self):
        return sum(p.quantity * p.option.delta * 100 for p in self.positions.values())

    @property
    def total_gamma(self):
        return sum(p.quantity * p.option.gamma * 100 for p in self.positions.values())

    @property
    def total_vega(self):
        return sum(p.quantity * p.option.vega * 100 for p in self.positions.values())

    @property
    def total_theta(self):
        return sum(p.quantity * p.option.theta * 100 for p in self.positions.values())


# ─────────────────────────────────────────────
# GUI APPLICATION
# ─────────────────────────────────────────────

POPULAR_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                   "NVDA", "META", "SPY",  "QQQ",  "JPM"]

BG       = "#0f1117"
BG2      = "#1a1d27"
BG3      = "#22263a"
ACCENT   = "#4f8ef7"
GREEN    = "#00c896"
RED      = "#ff4d6a"
AMBER    = "#f5a623"
WHITE    = "#e8eaf0"
MUTED    = "#6b7280"
BORDER   = "#2d3148"

FONT_H1  = ("Helvetica", 16, "bold")
FONT_H2  = ("Helvetica", 12, "bold")
FONT_SM  = ("Helvetica", 10)
FONT_XS  = ("Helvetica", 9)
FONT_MONO= ("Courier",   10)


class OptionsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Options Market Demo  |  WQU MScFE Portfolio")
        self.geometry("1400x860")
        self.configure(bg=BG)
        self.resizable(True, True)

        self.market    = MarketData()
        self.portfolio = Portfolio(100_000)
        self.chain     = []
        self.selected_opt = None
        self._refresh_after = None

        self._build_ui()
        self._load_ticker("AAPL")

    # ── UI CONSTRUCTION ──────────────────────

    def _build_ui(self):
        # ── top bar ──
        top = tk.Frame(self, bg=BG, pady=8)
        top.pack(fill="x", padx=16)

        tk.Label(top, text="⬡ Options Market Demo",
                 font=FONT_H1, bg=BG, fg=ACCENT).pack(side="left")

        tk.Label(top, text="WQU MScFE  |  Paper Trading  |  Live US Data",
                 font=FONT_XS, bg=BG, fg=MUTED).pack(side="left", padx=16)

        # ticker selector
        right = tk.Frame(top, bg=BG)
        right.pack(side="right")
        tk.Label(right, text="Ticker:", font=FONT_SM, bg=BG, fg=WHITE).pack(side="left")
        self.ticker_var = tk.StringVar(value="AAPL")
        combo = ttk.Combobox(right, textvariable=self.ticker_var,
                             values=POPULAR_TICKERS, width=8, font=FONT_SM)
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>", lambda e: self._load_ticker(self.ticker_var.get()))
        tk.Button(right, text="Load", font=FONT_SM, bg=ACCENT, fg=WHITE,
                  relief="flat", padx=8,
                  command=lambda: self._load_ticker(self.ticker_var.get())).pack(side="left", padx=4)

        self.status_lbl = tk.Label(right, text="", font=FONT_XS, bg=BG, fg=MUTED)
        self.status_lbl.pack(side="left", padx=8)

        # ── portfolio summary bar ──
        self.pf_bar = tk.Frame(self, bg=BG2, pady=6)
        self.pf_bar.pack(fill="x", padx=16, pady=(0, 8))
        self.pf_labels = {}
        for key in ["Cash", "Total Value", "P&L", "Delta", "Gamma", "Vega", "Theta"]:
            f = tk.Frame(self.pf_bar, bg=BG2, padx=20)
            f.pack(side="left")
            tk.Label(f, text=key, font=FONT_XS, bg=BG2, fg=MUTED).pack()
            lbl = tk.Label(f, text="—", font=("Helvetica", 11, "bold"), bg=BG2, fg=WHITE)
            lbl.pack()
            self.pf_labels[key] = lbl

        # ── main panes ──
        main = tk.PanedWindow(self, orient="horizontal", bg=BG, sashwidth=4)
        main.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        # LEFT: option chain + order entry
        left_frame = tk.Frame(main, bg=BG)
        main.add(left_frame, minsize=620)

        # RIGHT: positions + orders + greeks
        right_frame = tk.Frame(main, bg=BG)
        main.add(right_frame, minsize=400)

        self._build_chain_panel(left_frame)
        self._build_order_panel(left_frame)
        self._build_right_panel(right_frame)

    def _build_chain_panel(self, parent):
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill="x", pady=(0, 4))

        tk.Label(hdr, text="Option Chain", font=FONT_H2,
                 bg=BG, fg=WHITE).pack(side="left")

        # expiry filter
        tk.Label(hdr, text="Expiry:", font=FONT_XS, bg=BG, fg=MUTED).pack(side="left", padx=(16,4))
        self.expiry_var = tk.StringVar(value="All")
        self.expiry_combo = ttk.Combobox(hdr, textvariable=self.expiry_var,
                                          values=["All"], width=12, font=FONT_XS)
        self.expiry_combo.pack(side="left")
        self.expiry_combo.bind("<<ComboboxSelected>>", lambda e: self._filter_chain())

        # type filter
        self.type_var = tk.StringVar(value="All")
        ttk.Combobox(hdr, textvariable=self.type_var,
                     values=["All","call","put"], width=6,
                     font=FONT_XS).pack(side="left", padx=4)
        self.type_var.trace("w", lambda *_: self._filter_chain())

        # spot label
        self.spot_lbl = tk.Label(hdr, text="Spot: —", font=FONT_SM,
                                  bg=BG, fg=ACCENT)
        self.spot_lbl.pack(side="right")

        # chain treeview
        cols = ("Type","Strike","Expiry","Bid","Ask","Mid","IV","Delta","Gamma","Theta","Vega")
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Chain.Treeview",
                        background=BG2, foreground=WHITE,
                        fieldbackground=BG2, rowheight=22,
                        font=FONT_XS, borderwidth=0)
        style.configure("Chain.Treeview.Heading",
                        background=BG3, foreground=MUTED,
                        font=("Helvetica", 9, "bold"), relief="flat")
        style.map("Chain.Treeview", background=[("selected", BG3)],
                  foreground=[("selected", ACCENT)])

        self.chain_tree = ttk.Treeview(frame, columns=cols, show="headings",
                                        style="Chain.Treeview", height=16)
        widths = [45, 60, 75, 65, 65, 65, 60, 65, 65, 65, 60]
        for col, w in zip(cols, widths):
            self.chain_tree.heading(col, text=col)
            self.chain_tree.column(col, width=w, anchor="center")

        sb = ttk.Scrollbar(frame, orient="vertical", command=self.chain_tree.yview)
        self.chain_tree.configure(yscrollcommand=sb.set)
        self.chain_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.chain_tree.bind("<<TreeviewSelect>>", self._on_chain_select)
        self.chain_tree.tag_configure("call", foreground="#7dd3fc")
        self.chain_tree.tag_configure("put",  foreground="#fca5a5")
        self.chain_tree.tag_configure("itm",  background="#1a2e1a")

    def _build_order_panel(self, parent):
        f = tk.Frame(parent, bg=BG2, pady=10, padx=14)
        f.pack(fill="x", pady=8)

        tk.Label(f, text="Order Entry", font=FONT_H2, bg=BG2, fg=WHITE).grid(
            row=0, column=0, columnspan=6, sticky="w", pady=(0, 8))

        self.selected_lbl = tk.Label(f, text="No option selected",
                                      font=FONT_XS, bg=BG2, fg=MUTED)
        self.selected_lbl.grid(row=1, column=0, columnspan=6, sticky="w", pady=(0,6))

        labels = ["Direction", "Quantity", "Price"]
        for i, l in enumerate(labels):
            tk.Label(f, text=l, font=FONT_XS, bg=BG2, fg=MUTED).grid(
                row=2, column=i*2, sticky="w", padx=(0,4))

        self.dir_var = tk.StringVar(value="buy")
        ttk.Combobox(f, textvariable=self.dir_var, values=["buy","sell"],
                     width=7, font=FONT_SM).grid(row=2, column=1, padx=4)

        self.qty_var = tk.StringVar(value="1")
        tk.Entry(f, textvariable=self.qty_var, width=6,
                 bg=BG3, fg=WHITE, insertbackground=WHITE,
                 font=FONT_SM, relief="flat").grid(row=2, column=3, padx=4)

        self.price_lbl = tk.Label(f, text="—", font=FONT_SM, bg=BG2, fg=WHITE)
        self.price_lbl.grid(row=2, column=5, padx=4)

        self.submit_btn = tk.Button(f, text="Place Order",
                                     font=("Helvetica", 10, "bold"),
                                     bg=GREEN, fg="black", relief="flat",
                                     padx=16, pady=4,
                                     command=self._place_order)
        self.submit_btn.grid(row=2, column=6, padx=12)

        self.order_msg = tk.Label(f, text="", font=FONT_XS, bg=BG2, fg=GREEN)
        self.order_msg.grid(row=3, column=0, columnspan=7, sticky="w", pady=(6,0))

    def _build_right_panel(self, parent):
        nb = ttk.Notebook(parent)
        style = ttk.Style()
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG3, foreground=MUTED,
                        font=FONT_XS, padding=[10, 4])
        style.map("TNotebook.Tab", background=[("selected", BG2)],
                  foreground=[("selected", WHITE)])
        nb.pack(fill="both", expand=True)

        # Positions tab
        pos_frame = tk.Frame(nb, bg=BG)
        nb.add(pos_frame, text="Positions")
        self._build_positions_tab(pos_frame)

        # Orders tab
        ord_frame = tk.Frame(nb, bg=BG)
        nb.add(ord_frame, text="Order History")
        self._build_orders_tab(ord_frame)

        # Greeks tab
        grk_frame = tk.Frame(nb, bg=BG)
        nb.add(grk_frame, text="Greeks")
        self._build_greeks_tab(grk_frame)

    def _build_positions_tab(self, parent):
        cols = ("Option","Qty","Avg Price","Mkt Value","Unr. P&L","P&L %")
        style = ttk.Style()
        style.configure("Pos.Treeview",
                        background=BG2, foreground=WHITE,
                        fieldbackground=BG2, rowheight=22,
                        font=FONT_XS, borderwidth=0)
        style.configure("Pos.Treeview.Heading",
                        background=BG3, foreground=MUTED,
                        font=("Helvetica", 9, "bold"), relief="flat")

        self.pos_tree = ttk.Treeview(parent, columns=cols, show="headings",
                                      style="Pos.Treeview", height=10)
        widths = [200, 45, 75, 80, 80, 65]
        for col, w in zip(cols, widths):
            self.pos_tree.heading(col, text=col)
            self.pos_tree.column(col, width=w, anchor="center")
        self.pos_tree.tag_configure("profit", foreground=GREEN)
        self.pos_tree.tag_configure("loss",   foreground=RED)
        self.pos_tree.pack(fill="both", expand=True)

        # close position button
        tk.Button(parent, text="Close Selected Position",
                  font=FONT_XS, bg=RED, fg=WHITE, relief="flat", pady=4,
                  command=self._close_position).pack(pady=6)

    def _build_orders_tab(self, parent):
        cols = ("ID","Time","Option","Dir","Qty","Price","Total")
        style = ttk.Style()
        style.configure("Ord.Treeview",
                        background=BG2, foreground=WHITE,
                        fieldbackground=BG2, rowheight=20,
                        font=FONT_XS, borderwidth=0)
        style.configure("Ord.Treeview.Heading",
                        background=BG3, foreground=MUTED,
                        font=("Helvetica", 9, "bold"), relief="flat")

        self.ord_tree = ttk.Treeview(parent, columns=cols, show="headings",
                                      style="Ord.Treeview", height=20)
        widths = [65, 70, 160, 40, 35, 60, 80]
        for col, w in zip(cols, widths):
            self.ord_tree.heading(col, text=col)
            self.ord_tree.column(col, width=w, anchor="center")
        self.ord_tree.tag_configure("buy",  foreground=GREEN)
        self.ord_tree.tag_configure("sell", foreground=RED)
        sb = ttk.Scrollbar(parent, orient="vertical", command=self.ord_tree.yview)
        self.ord_tree.configure(yscrollcommand=sb.set)
        self.ord_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _build_greeks_tab(self, parent):
        tk.Label(parent, text="Portfolio Greeks (all positions)",
                 font=FONT_H2, bg=BG, fg=WHITE).pack(pady=8)

        self.greeks_frame = tk.Frame(parent, bg=BG)
        self.greeks_frame.pack(fill="both", expand=True, padx=16)
        self.greek_cards = {}

        greek_info = {
            "Delta":  ("Δ", "Price sensitivity\nper $1 move in S", ACCENT),
            "Gamma":  ("Γ", "Delta change\nper $1 move in S",  AMBER),
            "Vega":   ("ν", "Price change\nper 1% vol move",   GREEN),
            "Theta":  ("Θ", "Daily time decay\n(per calendar day)", RED),
        }
        for i, (name, (sym, desc, color)) in enumerate(greek_info.items()):
            card = tk.Frame(self.greeks_frame, bg=BG2, padx=20, pady=16)
            card.grid(row=i//2, column=i%2, padx=8, pady=8, sticky="nsew")
            self.greeks_frame.columnconfigure(i%2, weight=1)
            tk.Label(card, text=f"{sym}  {name}", font=FONT_H2,
                     bg=BG2, fg=color).pack()
            val_lbl = tk.Label(card, text="0.0000", font=("Helvetica", 20, "bold"),
                               bg=BG2, fg=WHITE)
            val_lbl.pack(pady=4)
            tk.Label(card, text=desc, font=FONT_XS, bg=BG2,
                     fg=MUTED, justify="center").pack()
            self.greek_cards[name] = val_lbl

        # selected option greeks
        tk.Label(parent, text="Selected Option Greeks",
                 font=FONT_H2, bg=BG, fg=WHITE).pack(pady=(16, 4))
        self.sel_greeks_lbl = tk.Label(parent, text="Select an option from the chain",
                                        font=FONT_XS, bg=BG, fg=MUTED, justify="left")
        self.sel_greeks_lbl.pack(padx=16, anchor="w")

    # ── DATA LOADING ─────────────────────────

    def _load_ticker(self, symbol: str):
        symbol = symbol.upper().strip()
        self.status_lbl.config(text=f"Loading {symbol}...", fg=AMBER)
        self.update()

        def fetch():
            price = self.market.fetch(symbol)
            if price is None:
                self.after(0, lambda: self.status_lbl.config(
                    text=f"Failed to load {symbol}", fg=RED))
                return
            chain = self.market.build_chain(symbol)
            self.after(0, lambda: self._on_data_loaded(symbol, price, chain))

        threading.Thread(target=fetch, daemon=True).start()

    def _on_data_loaded(self, symbol, price, chain):
        self.chain = chain
        expiries = sorted(set(o.expiry.strftime("%Y-%m-%d") for o in chain))
        self.expiry_combo["values"] = ["All"] + expiries
        self.expiry_var.set("All")
        self.type_var.set("All")
        vol = self.market.vol(symbol) * 100
        self.spot_lbl.config(
            text=f"Spot: ${price:.2f}  |  HV30: {vol:.1f}%")
        self.status_lbl.config(
            text=f"{symbol} loaded — {len(chain)} contracts", fg=GREEN)
        self._filter_chain()
        self._refresh_portfolio()
        self._schedule_refresh(symbol)

    def _schedule_refresh(self, symbol, interval_ms=60_000):
        if self._refresh_after:
            self.after_cancel(self._refresh_after)
        def refresh():
            self._load_ticker(symbol)
        self._refresh_after = self.after(interval_ms, refresh)

    def _filter_chain(self):
        self.chain_tree.delete(*self.chain_tree.get_children())
        exp_filter  = self.expiry_var.get()
        type_filter = self.type_var.get()
        S = self.market.price(self.ticker_var.get().upper())

        for opt in self.chain:
            if exp_filter != "All" and opt.expiry.strftime("%Y-%m-%d") != exp_filter:
                continue
            if type_filter != "All" and opt.option_type != type_filter:
                continue
            itm = (opt.option_type == "call" and S and opt.strike < S) or \
                  (opt.option_type == "put"  and S and opt.strike > S)
            tags = (opt.option_type, "itm") if itm else (opt.option_type,)
            self.chain_tree.insert("", "end", iid=opt.label, tags=tags, values=(
                opt.option_type.upper(),
                f"{opt.strike:.0f}",
                opt.expiry.strftime("%Y-%m-%d"),
                f"{opt.bid:.2f}",
                f"{opt.ask:.2f}",
                f"{opt.mid:.2f}",
                f"{opt.iv*100:.1f}%",
                f"{opt.delta:.3f}",
                f"{opt.gamma:.4f}",
                f"{opt.theta:.4f}",
                f"{opt.vega:.4f}",
            ))

    # ── ORDER HANDLING ───────────────────────

    def _on_chain_select(self, event):
        sel = self.chain_tree.selection()
        if not sel:
            return
        label = sel[0]
        opt   = next((o for o in self.chain if o.label == label), None)
        if opt is None:
            return
        self.selected_opt = opt
        self.selected_lbl.config(text=opt.label, fg=WHITE)
        direction = self.dir_var.get()
        price = opt.ask if direction == "buy" else opt.bid
        self.price_lbl.config(text=f"${price:.2f}  (x100 per contract)")

        # update selected greeks
        txt = (f"  Delta: {opt.delta:+.4f}    Gamma: {opt.gamma:.5f}\n"
               f"  Vega:  {opt.vega:+.4f}    Theta: {opt.theta:+.4f}\n"
               f"  IV:    {opt.iv*100:.2f}%       T: {opt.T*365:.0f} days")
        self.sel_greeks_lbl.config(text=txt, fg=WHITE)

    def _place_order(self):
        if self.selected_opt is None:
            self.order_msg.config(text="Select an option first.", fg=RED)
            return
        try:
            qty = int(self.qty_var.get())
            assert qty > 0
        except Exception:
            self.order_msg.config(text="Quantity must be a positive integer.", fg=RED)
            return

        direction = self.dir_var.get()
        ok, msg   = self.portfolio.place_order(self.selected_opt, direction, qty)
        color     = GREEN if ok else RED
        self.order_msg.config(text=msg, fg=color)
        if ok:
            self._refresh_portfolio()

    def _close_position(self):
        sel = self.pos_tree.selection()
        if not sel:
            return
        label = sel[0]
        pos   = self.portfolio.positions.get(label)
        if pos is None:
            return
        direction = "sell" if pos.quantity > 0 else "buy"
        ok, msg   = self.portfolio.place_order(pos.option, direction, abs(pos.quantity))
        self.order_msg.config(text=msg, fg=GREEN if ok else RED)
        if ok:
            self._refresh_portfolio()

    # ── UI REFRESH ───────────────────────────

    def _refresh_portfolio(self):
        pf = self.portfolio

        # portfolio bar
        pnl_color = GREEN if pf.total_pnl >= 0 else RED
        self.pf_labels["Cash"].config(text=f"${pf.cash:>10,.2f}")
        self.pf_labels["Total Value"].config(text=f"${pf.total_value:>10,.2f}")
        self.pf_labels["P&L"].config(
            text=f"${pf.total_pnl:>+10,.2f}", fg=pnl_color)
        self.pf_labels["Delta"].config(text=f"{pf.total_delta:>+.2f}")
        self.pf_labels["Gamma"].config(text=f"{pf.total_gamma:>+.4f}")
        self.pf_labels["Vega"].config(text=f"{pf.total_vega:>+.4f}")
        self.pf_labels["Theta"].config(text=f"{pf.total_theta:>+.4f}")

        # positions
        self.pos_tree.delete(*self.pos_tree.get_children())
        for label, pos in pf.positions.items():
            tag   = "profit" if pos.unrealized_pnl >= 0 else "loss"
            short = label if len(label) <= 30 else label[:28] + "…"
            self.pos_tree.insert("", "end", iid=label, tags=(tag,), values=(
                short,
                pos.quantity,
                f"${pos.avg_price:.2f}",
                f"${pos.market_value:,.2f}",
                f"${pos.unrealized_pnl:+,.2f}",
                f"{pos.pnl_pct:+.1f}%",
            ))

        # orders
        self.ord_tree.delete(*self.ord_tree.get_children())
        for o in reversed(pf.orders):
            total = o.price * o.quantity * 100
            self.ord_tree.insert("", "end", tags=(o.direction,), values=(
                o.order_id,
                o.timestamp.strftime("%H:%M:%S"),
                o.option.label[:25],
                o.direction.upper(),
                o.quantity,
                f"${o.price:.2f}",
                f"${total:,.2f}",
            ))

        # greeks cards
        self.greek_cards["Delta"].config(text=f"{pf.total_delta:+.4f}")
        self.greek_cards["Gamma"].config(text=f"{pf.total_gamma:+.5f}")
        self.greek_cards["Vega"].config( text=f"{pf.total_vega:+.4f}")
        self.greek_cards["Theta"].config(text=f"{pf.total_theta:+.4f}")


if __name__ == "__main__":
    app = OptionsApp()
    app.mainloop()