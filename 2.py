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
symbol = "BTCUSDT"  # May need to be BTCUSDT.P

# Initialize MT5 connection
if not mt5.initialize():
    logging.error(f"Failed to initialize MT5: {mt5.last_error()}")
    exit()

# Replace with your MT5 demo credentials
mt5_login = 3263156
mt5_password = "Bl1SrRhFb0JP@E4"
mt5_server = "Bybit-Demo"
if not mt5.login(mt5_login, mt5_password, mt5_server):
    logging.error(f"Failed to login to MT5: {mt5.last_error()}")
    exit()
logging.info("Connected to MT5 demo account")

# Fetch initial MT5 account balance via account_info
account_info = mt5.account_info()
if account_info is None:
    logging.error("Failed to fetch MT5 account info")
    exit()
initial_balance = account_info.balance
logging.info(f"MT5 Account Balance: {initial_balance} USDT")

# Find the correct symbol and get specs
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
    if symbol_info is None:
        logging.error(f"Still failed to fetch info for {symbol}")
        exit()

logging.info(f"SymbolInfo attributes: {dir(symbol_info)}")
raw_point = symbol_info.point
logging.info(f"Raw Point Value: {raw_point}")
min_volume = symbol_info.volume_min
max_volume = symbol_info.volume_max
volume_step = symbol_info.volume_step
tick_size = raw_point
if tick_size != 0.00001:
    logging.warning(f"Unexpected tick size {tick_size} for BTCUSDT. Expected 0.00001.")
min_stop_level = symbol_info.trade_stops_level
logging.info(f"Symbol specs for {symbol}: Min Volume: {min_volume}, Max Volume: {max_volume}, Step: {volume_step}, Tick Size: {tick_size}, Min Stop Level: {min_stop_level}")

# Wallet class with risk management and periodic summaries
class Wallet:
    def __init__(self, max_risk_per_trade=0.05, max_drawdown=0.1):  # 5% risk per trade
        account_info = mt5.account_info()
        self.balance = account_info.balance if account_info else 0
        self.initial_balance = self.balance
        self.positions = {}
        self.trade_history = []  # List of dicts with trade details
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

    def get_performance_summary(self):
        self.sync_balance()
        total_profit = sum(trade['profit'] for trade in self.trade_history)
        total_trades = len(self.trade_history)
        final_value = self.balance
        total_return = (final_value - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        profit_factor = float('inf') if sum(1 for t in self.trade_history if t['profit'] <= 0) == 0 else sum(t['profit'] for t in self.trade_history if t['profit'] > 0) / abs(sum(t['profit'] for t in self.trade_history if t['profit'] <= 0))
        avg_trade_return = np.mean([t['profit'] / (t['qty'] * t['entry_price']) * 100 for t in self.trade_history]) if total_trades > 0 else 0
        win_rate = len([t for t in self.trade_history if t['profit'] > 0]) / total_trades * 100 if total_trades > 0 else 0
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

    def get_periodic_summary(self, period='daily'):
        """Generate summary for daily, weekly, or monthly periods."""
        now = datetime.now()
        if period == 'daily':
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'weekly':
            start_time = now - timedelta(days=now.weekday())  # Start of the week (Monday)
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == 'monthly':
            start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError("Period must be 'daily', 'weekly', or 'monthly'")

        period_trades = [t for t in self.trade_history if t['timestamp'] >= start_time]
        total_profit = sum(t['profit'] for t in period_trades)
        total_trades = len(period_trades)
        win_trades = len([t for t in period_trades if t['profit'] > 0])
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
        profit_factor = float('inf') if sum(1 for t in period_trades if t['profit'] <= 0) == 0 else sum(t['profit'] for t in period_trades if t['profit'] > 0) / abs(sum(t['profit'] for t in period_trades if t['profit'] <= 0))
        avg_trade_profit = total_profit / total_trades if total_trades > 0 else 0

        summary = {
            'period': period.capitalize(),
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_profit': total_profit,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_profit': avg_trade_profit
        }
        return summary

    def log_periodic_summaries(self):
        """Log summaries for daily, weekly, and monthly periods."""
        for period in ['daily', 'weekly', 'monthly']:
            summary = self.get_periodic_summary(period)
            logging.info(f"{summary['period']} Summary (Since {summary['start_time']}): "
                         f"Total Profit: {summary['total_profit']:.2f} USDT, "
                         f"Total Trades: {summary['total_trades']}, "
                         f"Win Rate: {summary['win_rate']:.2f}%, "
                         f"Profit Factor: {summary['profit_factor']:.2f}, "
                         f"Avg Trade Profit: {summary['avg_trade_profit']:.2f} USDT")

wallet = Wallet()

# Markov Chain with updated transition matrix
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

# Fetch multi-timeframe data
def fetch_multi_timeframe_data(symbol, timeframes=["5", "15", "60"], limit=200):
    data = {}
    for tf in timeframes:
        try:
            response = client.get_kline(category="linear", symbol=symbol, interval=tf, limit=limit)
            if "result" not in response or "list" not in response["result"]:
                logging.error(f"Invalid API response for timeframe {tf}")
                continue
            
            df = pd.DataFrame(response["result"]["list"], columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
            df[["open", "high", "low", "close", "volume", "turnover"]] = df[["open", "high", "low", "close", "volume", "turnover"]].astype(float)
            df["returns"] = df["close"].pct_change()
            df["high_low"] = df["high"] - df["low"]
            df["high_close_prev"] = abs(df["high"] - df["close"].shift(1))
            df["low_close_prev"] = abs(df["low"] - df["close"].shift(1))
            df["tr"] = df[["high_low", "high_close_prev", "low_close_prev"]].max(axis=1)
            df["atr"] = df["tr"].rolling(window=14).mean()
            df = df.drop(columns=["high_low", "high_close_prev", "low_close_prev", "tr"]).dropna()
            data[tf] = df
            logging.info(f"Fetched {tf}-minute data with {len(df)} rows")
        except Exception as e:
            logging.error(f"Error fetching {tf}-minute data: {e}")
    return data

# Fetch real-time order book data from MT5
def fetch_order_book(symbol, df):
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.warning(f"Failed to fetch tick data for {symbol}. Using fallback.")
            return [100.0], [10], [100.1], [10]
        bid_price = tick.bid
        ask_price = tick.ask
        current_price = (bid_price + ask_price) / 2
        last_close = df["close"].iloc[-1] if not df.empty else None
        if last_close and abs(current_price - last_close) / last_close > 0.5:
            logging.warning(f"Tick price {current_price} deviates significantly from close {last_close}. Using close price.")
            current_price = last_close
        return [bid_price], [10], [ask_price], [10]
    except Exception as e:
        logging.error(f"Error fetching order book: {e}")
        return [100.0], [10], [100.1], [10]

def fetch_l2_order_book(symbol):
    return [{"price": 100.0, "size": 5}], [{"price": 100.1, "size": 5}]

def calculate_microstructure_noise(bid_prices, bid_volumes, ask_prices, ask_volumes):
    return 0.001

def calculate_fractal_dimension(df):
    return 1.5

def hurst_exponent(time_series):
    return 0.5

def calculate_bid_ask_imbalance(bid_volumes, ask_volumes):
    total_bid_volume = sum(bid_volumes)
    total_ask_volume = sum(ask_volumes)
    return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0

# Adjusted volume function with warning for max cap
def adjust_volume(volume, min_vol, max_vol, step):
    volume = max(min_vol, min(max_vol, volume))
    volume = round(volume / step) * step
    if volume == max_vol:
        logging.warning(f"Volume capped at max: {max_vol}")
    return round(volume, 6)

# Train models with multi-timeframe data
def train_multi_tf_models(multi_tf_data):
    if not multi_tf_data or any(tf not in multi_tf_data for tf in ["5", "15", "60"]):
        logging.error("Missing timeframe data for training")
        return None, None, None
    
    # Use 5-minute data as the base timeframe
    df_5m = multi_tf_data["5"]
    df_15m = multi_tf_data["15"]
    df_60m = multi_tf_data["60"]
    
    # Resample 15m and 60m to align with 5m timestamps
    df_15m_resampled = df_15m.set_index("timestamp").resample("5min").ffill().reindex(df_5m["timestamp"]).reset_index()
    df_60m_resampled = df_60m.set_index("timestamp").resample("5min").ffill().reindex(df_5m["timestamp"]).reset_index()
    
    # Combine features into a single DataFrame
    combined_df = pd.DataFrame(index=df_5m.index)
    for tf, df in [("5", df_5m), ("15", df_15m_resampled), ("60", df_60m_resampled)]:
        combined_df[f"{tf}_open"] = df["open"]
        combined_df[f"{tf}_high"] = df["high"]
        combined_df[f"{tf}_low"] = df["low"]
        combined_df[f"{tf}_close"] = df["close"]
        combined_df[f"{tf}_volume"] = df["volume"]
        combined_df[f"{tf}_returns"] = df["returns"]
        combined_df[f"{tf}_atr"] = df["atr"]
    
    # Calculate target based on 5-minute data
    y = (df_5m["close"].shift(-1) > df_5m["close"]).astype(int)
    
    # Align X and y by dropping NaNs from both
    combined_df["target"] = y
    aligned_df = combined_df.dropna()
    
    if len(aligned_df) < 2:  # Need at least 2 samples for train-test split
        logging.error(f"Insufficient aligned data for training: {len(aligned_df)} samples")
        return None, None, None
    
    X = aligned_df.drop(columns=["target"])
    y = aligned_df["target"]
    
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
multi_tf_data_initial = fetch_multi_timeframe_data(symbol, timeframes=["5", "15", "60"], limit=500)
rf_model, dt_model, scaler = train_multi_tf_models(multi_tf_data_initial)
if rf_model is None or dt_model is None or scaler is None:
    logging.error("Multi-timeframe model training failed. Exiting.")
    exit()

# Execute Trade with multi-timeframe data and additional logging
def execute_trade(prediction, symbol, multi_tf_data, confidence_threshold=0.9):
    if wallet.paused:
        logging.warning("Trading paused.")
        return
    
    side = "Buy" if prediction == 1 else "Sell"
    try:
        df_5m = multi_tf_data["5"]
        df_15m = multi_tf_data["15"]
        df_60m = multi_tf_data["60"]
        logging.info(f"Data rows - 5m: {len(df_5m)}, 15m: {len(df_15m)}, 60m: {len(df_60m)}")
        
        bid_prices, bid_volumes, ask_prices, ask_volumes = fetch_order_book(symbol, df_5m)
        current_price = (min(ask_prices) + max(bid_prices)) / 2
        if current_price == 100.05:
            current_price = df_5m["close"].iloc[-1]
            logging.warning(f"Using fallback current price: {current_price} from DataFrame")
        
        # Use 60-minute ATR for stop-loss/take-profit
        atr_60m = df_60m["atr"].iloc[-1]
        logging.info(f"60m ATR: {atr_60m}")
        max_sl_distance = current_price * 0.01
        max_tp_distance = current_price * 0.015
        stop_loss_distance = min(max(atr_60m * 5, min_stop_level * tick_size), max_sl_distance)
        take_profit_distance = min(max(atr_60m * 7.5, min_stop_level * tick_size * 1.5), max_tp_distance)
        stop_loss = round(current_price - stop_loss_distance if side == "Buy" else current_price + stop_loss_distance, 6)
        take_profit = round(current_price + take_profit_distance if side == "Buy" else current_price - take_profit_distance, 6)
        
        sl_distance = abs(current_price - stop_loss)
        tp_distance = abs(current_price - take_profit)
        min_distance = min_stop_level * tick_size
        logging.info(f"SL Distance: {sl_distance}, TP Distance: {tp_distance}, Min Distance: {min_distance}, Current Price: {current_price}")
        if sl_distance < min_distance or tp_distance < min_distance * 1.5:
            logging.warning(f"Invalid stops: SL={stop_loss}, TP={take_profit}, Min Stop={min_distance}")
            return

        # Combine features from all timeframes
        latest_time = min(df["timestamp"].iloc[-1] for df in multi_tf_data.values())
        logging.info(f"Latest time: {latest_time}")
        combined_features = pd.DataFrame()
        for tf, df in multi_tf_data.items():
            logging.info(f"{tf} latest timestamp: {df['timestamp'].iloc[-1]}")
            filtered_df = df[df["timestamp"] <= latest_time]
            if filtered_df.empty:
                logging.warning(f"No rows in {tf} with timestamp <= {latest_time}. Using latest row.")
                latest_row = df.iloc[-1]
            else:
                latest_row = filtered_df.iloc[-1]
            logging.info(f"{tf} rows after filter: {len(filtered_df)}")
            combined_features[f"{tf}_open"] = [latest_row["open"]]
            combined_features[f"{tf}_high"] = [latest_row["high"]]
            combined_features[f"{tf}_low"] = [latest_row["low"]]
            combined_features[f"{tf}_close"] = [latest_row["close"]]
            combined_features[f"{tf}_volume"] = [latest_row["volume"]]
            combined_features[f"{tf}_returns"] = [latest_row["returns"]]
            combined_features[f"{tf}_atr"] = [latest_row["atr"]]
        
        logging.info("Features combined for prediction")
        X_latest_scaled = scaler.transform(combined_features)
        rf_prediction = rf_model.predict_proba(X_latest_scaled)[0][1]
        
        # Trade size range based on confidence
        trade_sizes = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        confidence_range = 1.0 - confidence_threshold
        normalized_confidence = (rf_prediction - confidence_threshold) / confidence_range if rf_prediction >= confidence_threshold else 0
        index = min(int(normalized_confidence * (len(trade_sizes) - 1)), len(trade_sizes) - 1)
        fixed_qty = trade_sizes[index]
        logging.info(f"Confidence: {rf_prediction*100:.2f}%, Normalized: {normalized_confidence:.2f}, Selected trade size: {fixed_qty} BTC")
        
        trade_qty = wallet.calculate_position_size(current_price, stop_loss_distance, fixed_qty=fixed_qty)
        logging.info(f"Trade qty calculated: {trade_qty}")
        if trade_qty < min_volume:
            logging.warning(f"Trade qty {trade_qty} below minimum {min_volume}. Skipping.")
            return
        
        cost = trade_qty * current_price
        wallet.sync_balance()
        if cost > wallet.balance:
            logging.warning(f"Trade cost {cost:.2f} USDT exceeds balance {wallet.balance:.2f} USDT. Skipping.")
            return

        # Trend alignment check with 15-minute timeframe
        logging.info(f"Checking 15m returns: {df_15m['returns'].iloc[-1]}")
        if (side == "Buy" and df_15m["returns"].iloc[-1] < 0) or (side == "Sell" and df_15m["returns"].iloc[-1] > 0):
            logging.warning(f"15-minute trend opposes {side} signal. Skipping.")
            return

        logging.info("Passed 15m trend check")
        hurst_value = hurst_exponent(df_5m["close"].values)
        logging.info(f"Hurst value: {hurst_value}")
        imbalance = calculate_bid_ask_imbalance(bid_volumes, ask_volumes)
        logging.info(f"Bid-ask imbalance: {imbalance}")
        if (side == "Buy" and (hurst_value < 0.5 or imbalance < 0)) or (side == "Sell" and (hurst_value > 0.5 or imbalance > 0)):
            logging.warning(f"Unfavorable conditions for {side}: Hurst={hurst_value}, Imbalance={imbalance}")
            return

        logging.info("Preparing MT5 order")
        order_type = mt5.ORDER_TYPE_BUY if side == "Buy" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": trade_qty,
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

        logging.info(f"Placed {side} order: {trade_qty} {symbol} @ {current_price}, SL: {stop_loss}, TP: {take_profit}")
        if side == "Buy":
            if wallet.open_position(symbol, side, trade_qty, current_price, stop_loss, take_profit):
                logging.info(f"Trade recorded in wallet for {symbol}")
        elif side == "Sell" and symbol in wallet.positions:
            wallet.close_position(symbol, current_price)

    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        raise  # Re-raise to see full stack trace in logs

# Main Loop with multi-timeframe data and periodic summaries
def main_loop():
    iteration = 0
    start_time = datetime.now()
    last_summary_time = start_time  # Track last summary log time
    summary_interval = timedelta(minutes=60)  # Log summaries every hour

    while True:
        try:
            multi_tf_data = fetch_multi_timeframe_data(symbol, timeframes=["5", "15", "60"], limit=200)
            if not multi_tf_data or "5" not in multi_tf_data:
                logging.warning("No 5-minute data fetched, skipping iteration.")
                time.sleep(5)
                continue

            current_state, stationary_dist = markov_chain.next_state()
            entropy = markov_chain.entropy()
            markov_win_prob = stationary_dist[1] * 100
            logging.info(f"Current State: {current_state}, Entropy: {entropy:.2f} bits, Markov Win Prob: {markov_win_prob:.2f}%")

            if current_state == "Win" and markov_win_prob >= 90:
                # Combine features for prediction
                combined_features = pd.DataFrame()
                for tf, df in multi_tf_data.items():
                    latest_row = df.iloc[-1]
                    combined_features[f"{tf}_open"] = [latest_row["open"]]
                    combined_features[f"{tf}_high"] = [latest_row["high"]]
                    combined_features[f"{tf}_low"] = [latest_row["low"]]
                    combined_features[f"{tf}_close"] = [latest_row["close"]]
                    combined_features[f"{tf}_volume"] = [latest_row["volume"]]
                    combined_features[f"{tf}_returns"] = [latest_row["returns"]]
                    combined_features[f"{tf}_atr"] = [latest_row["atr"]]
                
                X_latest_scaled = scaler.transform(combined_features)
                rf_prediction = rf_model.predict_proba(X_latest_scaled)[0][1]
                dt_prediction = dt_model.predict_proba(X_latest_scaled)[0][1]
                prediction = 1 if rf_prediction > 0.9 or dt_prediction > 0.9 else 0
                logging.info(f"RF Win Prob: {rf_prediction*100:.2f}%, DT Win Prob: {dt_prediction*100:.2f}%, Prediction: {prediction}")
                execute_trade(prediction, symbol, multi_tf_data, confidence_threshold=0.9)

            iteration += 1
            current_time = datetime.now()

            # Log overall performance summary every 10 iterations
            if iteration % 10 == 0:
                summary = wallet.get_performance_summary()
                logging.info(f"Performance Summary (Since {start_time.strftime('%Y-%m-%d %H:%M:%S')}): "
                             f"Start Value: {summary['start_value']:.2f} USDT, Final Value: {summary['final_value']:.2f} USDT, "
                             f"Total Return: {summary['total_return']:.2f}%, Profit Factor: {summary['profit_factor']:.2f}, "
                             f"Avg Trade Return: {summary['avg_trade_return']:.2f}%, Trades: {summary['total_trades']}, "
                             f"Win Rate: {summary['win_rate']:.2f}%, Balance: {summary['current_balance']:.2f} USDT")
                if summary["win_rate"] < 30 and summary["total_trades"] > 10:
                    wallet.max_risk_per_trade = max(0.005, wallet.max_risk_per_trade * 0.8)
                    logging.info(f"Reduced risk to {wallet.max_risk_per_trade*100}% due to low win rate.")
                elif summary["win_rate"] > 70:
                    wallet.max_risk_per_trade = min(0.03, wallet.max_risk_per_trade * 1.2)
                    logging.info(f"Increased risk to {wallet.max_risk_per_trade*100}% due to high win rate.")

            # Log periodic summaries every hour
            if current_time - last_summary_time >= summary_interval:
                wallet.log_periodic_summaries()
                last_summary_time = current_time

            time.sleep(5)
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main_loop()