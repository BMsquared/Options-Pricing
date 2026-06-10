import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import threading
import time

class PortfolioTracker(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("MSCFE Quantitative Portfolio Dashboard")
        self.geometry("850x500")
        self.configure(bg="#1e1e2e")  # Sleek dark theme

        # Define your specific risk-parity optimal weights & share quantities
        self.portfolio_data = {
            "GOOGL": {"shares": 11.58, "target_w": "8.43%"},
            "XOM":   {"shares": 25.81, "target_w": "7.81%"},
            "JPM":   {"shares": 14.95, "target_w": "9.33%"},
            "NVDA":  {"shares": 24.86, "target_w": "10.37%"},
            "LLY":   {"shares": 4.52,  "target_w": "10.51%"},
            "AAPL":  {"shares": 19.00, "target_w": "11.71%"},
            "AMZN":  {"shares": 12.53, "target_w": "6.14%"},
            "MSFT":  {"shares": 5.46,  "target_w": "4.48%"},
            "TSLA":  {"shares": 16.29, "target_w": "13.37%"},
            "V":     {"shares": 27.91, "target_w": "17.84%"}
        }
        
        self.initial_capital = 50000.00
        self.is_running = True

        self.setup_styles()
        self.create_header_widgets()
        self.create_matrix_grid()
        
        # Deploy background thread for API streaming
        self.data_thread = threading.Thread(target=self.live_stream_handler, daemon=True)
        self.data_thread.start()
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Treeview", background="#252538", foreground="#ffffff", fieldbackground="#252538", rowheight=28)
        self.style.configure("Treeview.Heading", background="#3b3b4f", foreground="#ffffff", font=("Arial", 10, "bold"))
        self.style.map("Treeview", background=[('selected', '#585b70')])

    def create_header_widgets(self):
        # FIX: Removed 'padding' and fixed the layout manager keyword constraints
        header_frame = tk.Frame(self, bg="#11111b")
        header_frame.pack(fill="x", padx=10, pady=15, side="top")

        # Total Net Asset Value metric
        self.lbl_nav = tk.Label(header_frame, text="Net Asset Value: Calculating...", font=("Arial", 18, "bold"), fg="#a6e3a1", bg="#11111b")
        self.lbl_nav.pack(side="left", padx=10)

        # Total Return percentage metric
        self.lbl_return = tk.Label(header_frame, text="Total Return: 0.00%", font=("Arial", 12, "bold"), fg="#ffffff", bg="#11111b")
        self.lbl_return.pack(side="right", padx=10)

    def create_matrix_grid(self):
        # Setup asset table view 
        cols = ("Ticker", "Target Risk Weight", "Shares Owned", "Live Price", "Position Value", "Daily Delta")
        self.tree = ttk.Treeview(self, columns=cols, show="headings")
        
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=125)
            
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        # Populate baseline structural keys into table rows
        for ticker, info in self.portfolio_data.items():
            self.tree.insert("", "end", iid=ticker, values=(ticker, info["target_w"], info["shares"], "Loading...", "Loading...", "Loading..."))

    def live_stream_handler(self):
        ticker_string = " ".join(self.portfolio_data.keys())
        
        while self.is_running:
            try:
                # Optimized single-request batch pull to decrease network latency
                tickers_group = yf.Tickers(ticker_string)
                total_nav = 0.0
                
                updates = {}
                for ticker in self.portfolio_data.keys():
                    info = tickers_group.tickers[ticker].fast_info
                    live_price = info.last_price
                    prev_close = info.previous_close if info.previous_close else live_price
                    
                    if live_price is None:
                        continue
                        
                    shares = self.portfolio_data[ticker]["shares"]
                    position_value = shares * live_price
                    total_nav += position_value
                    
                    # Compute fast intra-day momentum movement
                    price_delta = ((live_price - prev_close) / prev_close) * 100
                    
                    updates[ticker] = {
                        "price": f"${live_price:.2f}",
                        "value": f"${position_value:.2f}",
                        "delta": f"{price_delta:+.2f}%",
                        "raw_delta": price_delta
                    }
                
                # Push gathered metrics onto main loop stack for safe UI alteration
                self.after(0, self.update_display_interface, updates, total_nav)
                
            except Exception as e:
                print(f"Network error parsing real-time metrics: {e}")
                
            time.sleep(10)  # Stream refresh cadence set to 10 seconds

    def update_display_interface(self, updates, total_nav):
        # Refresh header metric components
        self.lbl_nav.config(text=f"Net Asset Value: ${total_nav:,.2f}")
        
        net_pct = ((total_nav - self.initial_capital) / self.initial_capital) * 100
        self.lbl_return.config(text=f"Total Return: {net_pct:+.2f}%")
        
        # Color profile change depending on positive or negative equity variance
        self.lbl_return.config(fg="#a6e3a1" if net_pct >= 0 else "#f38ba8")

        # Refresh cell text arrays across matrix grid rows
        for ticker, data in updates.items():
            if self.tree.exists(ticker):
                current_vals = self.tree.item(ticker, "values")
                
                self.tree.item(ticker, values=(
                    current_vals[0],  # Ticker
                    current_vals[1],  # Target Risk Weight
                    current_vals[2],  # Shares Owned
                    data["price"],    # Live Price
                    data["value"],    # Position Value
                    data["delta"]     # Daily Delta
                ))

    def on_close(self):
        self.is_running = False
        self.destroy()

if __name__ == "__main__":
    app = PortfolioTracker()
    app.mainloop()
