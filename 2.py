import time
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from pybit.unified_trading import HTTP
import logging
import MetaTrader5 as mt5

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log MT5 package version
logging.info(f"MetaTrader5 version: {mt5.__version__}")
logging.info(f"Has account_balance: {hasattr(mt5, 'account_balance')}")
logging.info(f"Has account_info: {hasattr(mt5, 'account_info')}")

from config import API_KEY, API_SECRET  # Ensure config.py exists

# Initialize Bybit API
client = HTTP(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
symbol = "BTCUSDT"

# Initialize MT5 connection
if not mt5.initialize():
    logging.error(f"Failed to initialize MT5: {mt5.last_error()}")
    exit()

mt5_login = 3263156
mt5_password = "Bl1SrRhFb0JP@E4"
mt5_server = "Bybit-Demo"
if not mt5.login(mt5_login, mt5_password, mt5_server):
    logging.error(f"Failed to login to MT5: {mt5.last_error()}")
    exit()
logging.info("Connected to MT5 demo account")

account_info = mt5.account_info()
if account_info is None:
    logging.error("Failed to fetch MT5 account info")
    exit()
initial_balance = account_info.balance
logging.info(f"MT5 Account Balance: {initial_balance} USDT")

# Symbol setup
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    logging.warning(f"{symbol} not found. Searching for BTCUSDT variant...")
    all_symbols = [s.name for s in mt5.symbols_get() if "BTCUSDT" in s.name]
    if not all_symbols:
        logging.error("No BTCUSDT variant found in MT5. Check Market Watch.")
        exit()
    symbol = all_symbols[0]
    logging.info(f"Using symbol: {symbol}")
    symbol_info = mt5.symbol_info(symbol)

raw_point = symbol_info.point
min_volume = symbol_info.volume_min
max_volume = symbol_info.volume_max
volume_step = symbol_info.volume_step
tick_size = raw_point
min_stop_level = symbol_info.trade_stops_level

# Wallet class
class Wallet:
    def __init__(self, max_risk_per_trade=0.05, max_drawdown=0.1):
        account_info = mt5.account_info()
        self.balance = account_info.balance if account_info else 0
        self.initial_balance = self.balance
        self.positions = {}
        self.trade_history = []
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = max_drawdown
        self.paused = False

    def sync_balance(self):
        account_info = mt5.account_info()
        if account_info is not None:
            self.balance = account_info.balance
            drawdown = (self.initial_balance - self.balance) / self.initial_balance
            if drawdown >= self.max_drawdown:
                self.paused = True
                logging.warning(f"Max drawdown ({self.max_drawdown*100}%) reached. Trading paused.")
        else:
            logging.warning("Failed to sync balance with MT5")

    def calculate_position_size(self, price, stop_loss_distance, fixed_qty=None, volume_multiplier=1.0):
        if fixed_qty is not None:
            return adjust_volume(fixed_qty, min_volume, max_volume, volume_step)
        risk_amount = self.balance * self.max_risk_per_trade
        qty = (risk_amount / stop_loss_distance) * volume_multiplier
        return adjust_volume(qty, min_volume, max_volume, volume_step)

    def open_position(self, symbol, side, qty, price, stop_loss, take_profit):
        if self.paused:
            logging.warning("Trading paused due to drawdown limit.")
            return False
        cost = qty * price
        self.sync_balance()
        if self.balance >= cost:
            self.balance -= cost
            self.positions[symbol] = {
                'qty': qty, 'entry_price': price, 'side': side,
                'stop_loss': stop_loss, 'take_profit': take_profit
            }
            logging.info(f"Opened {side} position: {qty} {symbol} @ {price}, SL: {stop_loss}, TP: {take_profit}")
            return True
        else:
            logging.warning(f"Insufficient funds: {cost} > {self.balance}")
            return False

    def close_position(self, symbol, price):
        if symbol in self.positions and self.positions[symbol]['qty'] > 0:
            pos = self.positions[symbol]
            qty = pos['qty']
            entry_price = pos['entry_price']
            side = pos['side']
            exit_value = qty * price
            self.balance += exit_value
            profit = exit_value - (qty * entry_price) if side == "Buy" else (qty * entry_price) - exit_value
            trade = {
                'symbol': symbol, 'side': side, 'qty': qty, 'entry_price': entry_price,
                'exit_price': price, 'profit': profit, 'timestamp': datetime.now()
            }
            self.trade_history.append(trade)
            logging.info(f"Closed {side} position: {qty} {symbol} @ {price}, Profit: {profit}, New balance: {self.balance}")
            del self.positions[symbol]
        self.sync_balance()

    def get_performance_summary(self, trades=None):
        if trades is None:
            trades = self.trade_history
        self.sync_balance()
        total_profit = sum(trade['profit'] for trade in trades)
        total_trades = len(trades)
        final_value = self.initial_balance + total_profit
        total_return = (final_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        profit_factor = float('inf') if sum(1 for t in trades if t['profit'] <= 0) == 0 else sum(t['profit'] for t in trades if t['profit'] > 0) / abs(sum(t['profit'] for t in trades if t['profit'] <= 0))
        avg_trade_return = np.mean([t['profit'] / (t['qty'] * t['entry_price']) * 100 for t in trades]) if total_trades > 0 else 0
        win_rate = len([t for t in trades if t['profit'] > 0]) / total_trades * 100 if total_trades > 0 else 0
        return {
            'start_value': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return * 100 if total_return != float('inf') else float('inf'),
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'current_balance': self.balance
        }

    def get_periodic_performance(self):
        now = datetime.now()
        periods = {
            'Hourly': timedelta(hours=1),
            '4-Hourly': timedelta(hours=4),
            '12-Hourly': timedelta(hours=12),
            'Daily': timedelta(days=1),
            'Weekly': timedelta(weeks=1),
            'Monthly': timedelta(days=30)
        }
        performance = {}
        for period_name, delta in periods.items():
            start_time = now - delta
            period_trades = [trade for trade in self.trade_history if trade['timestamp'] >= start_time]
            if period_trades:
                summary = self.get_performance_summary(period_trades)
                performance[period_name] = summary
            else:
                performance[period_name] = {
                    'start_value': self.initial_balance,
                    'final_value': self.initial_balance,
                    'total_return': 0.0,
                    'profit_factor': 0.0,
                    'avg_trade_return': 0.0,
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'current_balance': self.balance
                }
        return performance

wallet = Wallet()

# Markov Chain
class MarkovChain:
    def __init__(self, states, transition_matrix):
        self.states = states
        self.transition_matrix = transition_matrix
        self.current_state = random.choice(states)
        self.stationary_dist = self.compute_stationary_distribution()

    def compute_stationary_distribution(self):
        if np.allclose(self.transition_matrix, [[0, 1], [0, 1]]):
            return [0, 1]
        return [0.5, 0.5]

    def next_state(self):
        try:
            probabilities = self.transition_matrix[self.states.index(self.current_state)]
            probabilities = np.clip(probabilities, 0, None)
            prob_sum = np.sum(probabilities)
            if prob_sum == 0:
                probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                probabilities = probabilities / prob_sum
            self.current_state = np.random.choice(self.states, p=probabilities)
            return self.current_state, self.stationary_dist
        except Exception as e:
            logging.error(f"Error in Markov Chain transition: {e}")
            self.current_state = random.choice(self.states)
            return self.current_state, [0.5, 0.5]

    def entropy(self):
        dist = self.stationary_dist
        return -np.sum([p * np.log2(p + 1e-10) for p in dist if p > 0])

states = ["Loss", "Win"]
transition_matrix = np.array([[0, 1], [0, 1]], dtype=float)
markov_chain = MarkovChain(states, transition_matrix)

# Updated fetch_data with sorting
def fetch_data(symbol, timeframe, limit=200):
    try:
        response = client.get_kline(category="linear", symbol=symbol, interval=timeframe, limit=limit)
        if "result" not in response or "list" not in response["result"]:
            logging.error(f"Invalid API response structure for {timeframe} timeframe")
            return pd.DataFrame()
        
        data = response["result"]["list"]
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
        df[["open", "high", "low", "close", "volume", "turnover"]] = df[["open", "high", "low", "close", "volume", "turnover"]].astype(float)
        df["returns"] = df["close"].pct_change()
        df["high_low"] = df["high"] - df["low"]
        df["high_close_prev"] = abs(df["high"] - df["close"].shift(1))
        df["low_close_prev"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["high_low", "high_close_prev", "low_close_prev"]].max(axis=1)
        df["atr"] = df["tr"].rolling(window=14).mean()
        df = df.drop(columns=["high_low", "high_close_prev", "low_close_prev", "tr"])
        df = df.dropna()
        df.columns = [f"{col}_{timeframe}m" if col not in ["timestamp"] else col for col in df.columns]
        # Sort by timestamp to ensure merge_asof compatibility
        df = df.sort_values("timestamp")
        return df
    except Exception as e:
        logging.error(f"Error fetching {timeframe}min data: {e}")
        return pd.DataFrame()

# Updated fetch_combined_data with sorting and validation
def fetch_combined_data(symbol, timeframes=["1", "3", "5"], limit=200):
    dfs = {}
    for tf in timeframes:
        df = fetch_data(symbol, tf, limit)
        if not df.empty:
            dfs[tf] = df
        else:
            logging.warning(f"Failed to fetch {tf}min data")
    
    if not dfs:
        logging.error("No data fetched for any timeframe. Exiting.")
        return pd.DataFrame()
    
    # Use the shortest timeframe (1m) as base and ensure it's sorted
    base_tf = min(timeframes, key=int)
    combined_df = dfs[base_tf]
    if not combined_df["timestamp"].is_monotonic_increasing:
        logging.warning(f"Base timeframe {base_tf}m timestamps not sorted. Sorting now.")
        combined_df = combined_df.sort_values("timestamp")
    
    # Merge other timeframes
    for tf in timeframes:
        if tf != base_tf:
            if not dfs[tf]["timestamp"].is_monotonic_increasing:
                logging.warning(f"{tf}m timestamps not sorted. Sorting now.")
                dfs[tf] = dfs[tf].sort_values("timestamp")
            try:
                combined_df = pd.merge_asof(
                    combined_df,
                    dfs[tf],
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta(minutes=int(tf)),
                    suffixes=("", f"_{tf}m")
                )
            except ValueError as e:
                logging.error(f"Failed to merge {tf}m data: {e}")
                return pd.DataFrame()
    
    combined_df = combined_df.dropna()
    logging.info(f"Combined data shape: {combined_df.shape}, Timeframes: {timeframes}")
    return combined_df

# Other utility functions
def fetch_order_book(symbol, df):
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.warning(f"Failed to fetch tick data for {symbol}. Using fallback.")
            return [100.0], [10], [100.1], [10]
        bid_price = tick.bid
        ask_price = tick.ask
        current_price = (bid_price + ask_price) / 2
        last_close = df["close_5m"].iloc[-1] if not df.empty else None
        if last_close and abs(current_price - last_close) / last_close > 0.5:
            logging.warning(f"Tick price {current_price} deviates significantly from close {last_close}. Using close price.")
            current_price = last_close
        return [bid_price], [10], [ask_price], [10]
    except Exception as e:
        logging.error(f"Error fetching order book: {e}")
        return [100.0], [10], [100.1], [10]

def calculate_bid_ask_imbalance(bid_volumes, ask_volumes):
    total_bid_volume = sum(bid_volumes)
    total_ask_volume = sum(ask_volumes)
    return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0

def hurst_exponent(time_series):
    return 0.5  # Placeholder; implement if needed

def adjust_volume(volume, min_vol, max_vol, step):
    volume = max(min_vol, min(max_vol, volume))
    volume = round(volume / step) * step
    if volume == max_vol:
        logging.warning(f"Volume capped at max: {max_vol}")
    return round(volume, 6)

# Train models
def train_models(df):
    if df.empty or not any(col.endswith("_5m") for col in df.columns):
        logging.error("DataFrame is empty or missing required 5m columns")
        return None, None, None
    
    feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover")]
    X = df[feature_cols].iloc[:-1]
    y = (df["close_5m"].shift(-1) > df["close_5m"]).astype(int).iloc[:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_accuracy = rf_model.score(X_test_scaled, y_test)

    dt_model = DecisionTreeClassifier(max_depth=1, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    dt_accuracy = dt_model.score(X_test_scaled, y_test)

    logging.info(f"RF Model Accuracy: {rf_accuracy:.2f}")
    logging.info(f"DT Model Accuracy: {dt_accuracy:.2f}, Max Depth: {dt_model.get_depth()}, Leaves: {dt_model.get_n_leaves()}")
    return rf_model, dt_model, scaler

# Train models at startup
df_initial = fetch_combined_data(symbol, timeframes=["1", "3", "5"], limit=500)
if df_initial.empty:
    logging.error("Initial data fetch failed. Exiting.")
    exit()
rf_model, dt_model, scaler = train_models(df_initial)
if rf_model is None or dt_model is None or scaler is None:
    logging.error("Model training failed. Exiting.")
    exit()

# Execute trade
def execute_trade(prediction, symbol, df, confidence_threshold=0.9, markov_win_prob=0.0):
    if wallet.paused:
        logging.warning("Trading paused.")
        return
    
    side = "Buy" if prediction == 1 else "Sell"
    try:
        bid_prices, bid_volumes, ask_prices, ask_volumes = fetch_order_book(symbol, df)
        current_price = (min(ask_prices) + max(bid_prices)) / 2
        if current_price == 100.05:
            current_price = df["close_5m"].iloc[-1]
            logging.warning(f"Using fallback current price: {current_price} from DataFrame")
        atr_5m = df["atr_5m"].iloc[-1]
        atr_1m = df["atr_1m"].iloc[-1] if "atr_1m" in df.columns else atr_5m
        
        max_sl_distance = current_price * 0.01
        max_tp_distance = current_price * 0.015
        stop_loss_distance = min(max(atr_5m * 5, min_stop_level * tick_size), max_sl_distance)
        take_profit_distance = min(max(atr_5m * 7.5, min_stop_level * tick_size * 1.5), max_tp_distance)
        stop_loss = round(current_price - stop_loss_distance if side == "Buy" else current_price + stop_loss_distance, 6)
        take_profit = round(current_price + take_profit_distance if side == "Buy" else current_price - take_profit_distance, 6)
        
        sl_distance = abs(current_price - stop_loss)
        tp_distance = abs(current_price - take_profit)
        min_distance = min_stop_level * tick_size
        if sl_distance < min_distance or tp_distance < min_distance * 1.5:
            logging.warning(f"Invalid stops: SL={stop_loss}, TP={take_profit}, Min Stop={min_distance}")
            return

        feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover")]
        X_latest = pd.DataFrame(df[feature_cols].iloc[-1]).T
        X_latest_scaled = scaler.transform(X_latest)
        rf_confidence = rf_model.predict_proba(X_latest_scaled)[0][1] if prediction == 1 else 1 - rf_model.predict_proba(X_latest_scaled)[0][1]
        dt_confidence = dt_model.predict_proba(X_latest_scaled)[0][1] if prediction == 1 else 1 - dt_model.predict_proba(X_latest_scaled)[0][1]
        hurst_value = hurst_exponent(df["close_5m"].values)
        imbalance = calculate_bid_ask_imbalance(bid_volumes, ask_volumes)
        volatility_1m = df["atr_1m"].iloc[-1] / df["close_1m"].iloc[-1] if "atr_1m" in df.columns else 0

        rf_score = max(0, (rf_confidence - confidence_threshold) / (1 - confidence_threshold))
        dt_score = max(0, (dt_confidence - confidence_threshold) / (1 - confidence_threshold))
        markov_score = max(0, (markov_win_prob / 100 - 0.9) / 0.1)
        hurst_score = 1 - abs(hurst_value - 0.5) * 2
        imbalance_score = abs(imbalance)
        volatility_score = min(volatility_1m / 0.01, 1.0)

        total_score = (0.35 * rf_score + 0.15 * dt_score + 0.2 * markov_score + 
                       0.1 * hurst_score + 0.1 * imbalance_score + 0.1 * volatility_score)
        total_score = np.clip(total_score, 0, 1)

        volume_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        volume_index = min(int(total_score * (len(volume_range) - 1)), len(volume_range) - 1)
        trade_qty = volume_range[volume_index]

        logging.info(f"Trade Factors - RF: {rf_confidence:.2f}, DT: {dt_confidence:.2f}, Markov: {markov_win_prob:.2f}%, "
                     f"Hurst: {hurst_value:.2f}, Imbalance: {imbalance:.2f}, Volatility 1m: {volatility_1m:.4f}")
        logging.info(f"Scores - RF: {rf_score:.2f}, DT: {dt_score:.2f}, Markov: {markov_score:.2f}, "
                     f"Hurst: {hurst_score:.2f}, Imbalance: {imbalance_score:.2f}, Volatility: {volatility_score:.2f}, Total: {total_score:.2f}")
        logging.info(f"Selected trade size: {trade_qty} BTC from range {volume_range}")

        adjusted_qty = wallet.calculate_position_size(current_price, stop_loss_distance, fixed_qty=trade_qty)
        if adjusted_qty < min_volume:
            logging.warning(f"Trade qty {adjusted_qty} below minimum {min_volume}. Skipping.")
            return

        cost = adjusted_qty * current_price
        wallet.sync_balance()
        if cost > wallet.balance:
            logging.warning(f"Trade cost {cost:.2f} USDT exceeds balance {wallet.balance:.2f} USDT. Skipping.")
            return

        if (side == "Buy" and (hurst_value < 0.5 or imbalance < 0)) or (side == "Sell" and (hurst_value > 0.5 or imbalance > 0)):
            logging.warning(f"Unfavorable conditions for {side}: Hurst={hurst_value}, Imbalance={imbalance}")
            return

        order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": adjusted_qty,
            "type": order_type,
            "price": current_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 123456,
            "comment": f"{side} via Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed: {result.comment} (retcode: {result.retcode})")
            return

        logging.info(f"Placed {side} order: {adjusted_qty} {symbol} @ {current_price}, SL: {stop_loss}, TP: {take_profit}")
        if side == "Buy":
            if wallet.open_position(symbol, side, adjusted_qty, current_price, stop_loss, take_profit):
                logging.info(f"Trade recorded in wallet for {symbol}")
        elif side == "Sell" and symbol in wallet.positions:
            wallet.close_position(symbol, current_price)

    except Exception as e:
        logging.error(f"Error executing trade: {e}")

# Main loop
def main_loop():
    iteration = 0
    start_time = datetime.now()
    last_performance_log = datetime.now()
    performance_interval = timedelta(minutes=60)

    while True:
        try:
            df = fetch_combined_data(symbol, timeframes=["1", "3", "5"], limit=200)
            if df.empty:
                logging.warning("No data fetched, skipping iteration.")
                time.sleep(5)
                continue

            current_state, stationary_dist = markov_chain.next_state()
            entropy = markov_chain.entropy()
            markov_win_prob = stationary_dist[1] * 100
            logging.info(f"Current State: {current_state}, Entropy: {entropy:.2f} bits, Markov Win Prob: {markov_win_prob:.2f}%")

            if current_state == "Win" and markov_win_prob >= 90:
                feature_cols = [col for col in df.columns if col not in ["timestamp"] and not col.startswith("turnover")]
                X_latest = pd.DataFrame(df[feature_cols].iloc[-1]).T
                X_latest_scaled = scaler.transform(X_latest)
                rf_prediction = rf_model.predict_proba(X_latest_scaled)[0][1]
                dt_prediction = dt_model.predict_proba(X_latest_scaled)[0][1]
                prediction = 1 if rf_prediction > 0.9 or dt_prediction > 0.9 else 0
                logging.info(f"RF Win Prob: {rf_prediction*100:.2f}%, DT Win Prob: {dt_prediction*100:.2f}%, Prediction: {prediction}")
                execute_trade(prediction, symbol, df, confidence_threshold=0.9, markov_win_prob=markov_win_prob)

            iteration += 1
            if iteration % 10 == 0:
                summary = wallet.get_performance_summary()
                logging.info(f"Overall Performance Summary (Since {start_time.strftime('%Y-%m-%d %H:%M:%S')}): "
                             f"Start Value: {summary['start_value']:.2f} USDT, Final Value: {summary['final_value']:.2f} USDT, "
                             f"Total Return: {summary['total_return']:.2f}%, Profit Factor: {summary['profit_factor']:.2f}, "
                             f"Avg Trade Return: {summary['avg_trade_return']:.2f}%, Trades: {summary['total_trades']}, "
                             f"Win Rate: {summary['win_rate']:.2f}%, Balance: {summary['current_balance']:.2f} USDT")

                current_time = datetime.now()
                if current_time - last_performance_log >= performance_interval:
                    periodic_performance = wallet.get_periodic_performance()
                    for period, metrics in periodic_performance.items():
                        logging.info(f"{period} Performance: "
                                     f"Start Value: {metrics['start_value']:.2f} USDT, Final Value: {metrics['final_value']:.2f} USDT, "
                                     f"Total Return: {metrics['total_return']:.2f}%, Profit Factor: {metrics['profit_factor']:.2f}, "
                                     f"Avg Trade Return: {metrics['avg_trade_return']:.2f}%, Trades: {metrics['total_trades']}, "
                                     f"Win Rate: {metrics['win_rate']:.2f}%, Balance: {metrics['current_balance']:.2f} USDT")
                    last_performance_log = current_time

                if summary["win_rate"] < 30 and summary["total_trades"] > 10:
                    wallet.max_risk_per_trade = max(0.005, wallet.max_risk_per_trade * 0.8)
                    logging.info(f"Reduced risk to {wallet.max_risk_per_trade*100}% due to low win rate.")
                elif summary["win_rate"] > 70:
                    wallet.max_risk_per_trade = min(0.03, wallet.max_risk_per_trade * 1.2)
                    logging.info(f"Increased risk to {wallet.max_risk_per_trade*100}% due to high win rate.")

            time.sleep(5)
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main_loop()
