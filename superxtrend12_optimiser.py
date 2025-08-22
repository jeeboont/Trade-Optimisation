"""
Super XTrend v1.2 Optimizer - Streamlit Application
Version: 1.2.0
Last Updated: 2025-08-22
Author: Super XTrend Trading Systems

FEATURES:
- Complete Super XTrend v1.2 strategy implementation
- Advanced parameter optimization with staged approach
- Multiple timeframe support with HTF X-Trend
- Comprehensive filter testing (ADX, EMA, Pivot Supertrend)
- Interactive results visualization
- CSV upload and Yahoo Finance data integration
- cBot-ready parameter export
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import io
import zipfile
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Version display
__version__ = "1.2.0"
__last_updated__ = "2025-08-22"

# Initialize session state
if 'downloaded_data' not in st.session_state:
    st.session_state.downloaded_data = {}
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = {}

# ==================== SUPER XTREND v1.2 INDICATOR CALCULATIONS ====================

def calculate_pivot_points(df, period=5):
    """Calculate pivot highs and lows for Pivot Supertrend"""
    try:
        if len(df) < period * 2 + 1:
            return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)
        
        pivot_highs = pd.Series(index=df.index, dtype=float)
        pivot_lows = pd.Series(index=df.index, dtype=float)
        
        # Calculate pivots with period lookback/lookahead
        for i in range(period, len(df) - period):
            # Pivot High check
            is_pivot_high = True
            high_val = df['high'].iloc[i]
            
            for j in range(i - period, i + period + 1):
                if j != i and j < len(df):
                    if df['high'].iloc[j] >= high_val:
                        is_pivot_high = False
                        break
            
            if is_pivot_high:
                pivot_highs.iloc[i + period] = high_val
            
            # Pivot Low check
            is_pivot_low = True
            low_val = df['low'].iloc[i]
            
            for j in range(i - period, i + period + 1):
                if j != i and j < len(df):
                    if df['low'].iloc[j] <= low_val:
                        is_pivot_low = False
                        break
            
            if is_pivot_low:
                pivot_lows.iloc[i + period] = low_val
        
        return pivot_highs, pivot_lows
        
    except Exception as e:
        print(f"Pivot calculation error: {e}")
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)

def calculate_atr_wilder(df, period=15):
    """Calculate ATR using Wilder's smoothing method"""
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Wilder's smoothing (exponential with alpha = 1/period)
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        return atr.fillna(true_range)
    except Exception as e:
        print(f"ATR calculation error: {e}")
        return pd.Series(df['high'] - df['low'], index=df.index)

def calculate_pivot_supertrend(df, pivot_period=5, atr_factor=1.25, atr_period=15):
    """Calculate Pivot Supertrend - Super XTrend v1.2 implementation"""
    try:
        df = df.copy()
        
        # Calculate pivot points
        pivot_highs, pivot_lows = calculate_pivot_points(df, pivot_period)
        
        # Calculate ATR with Wilder's method
        atr = calculate_atr_wilder(df, atr_period)
        
        # Initialize center (pivot center calculation)
        center = pd.Series(index=df.index, dtype=float)
        current_center = None
        
        for i in range(len(df)):
            if not pd.isna(pivot_highs.iloc[i]):
                if current_center is None:
                    current_center = pivot_highs.iloc[i]
                else:
                    # TradingView formula: (center * 2 + lastpp) / 3
                    current_center = (current_center * 2 + pivot_highs.iloc[i]) / 3
            elif not pd.isna(pivot_lows.iloc[i]):
                if current_center is None:
                    current_center = pivot_lows.iloc[i]
                else:
                    current_center = (current_center * 2 + pivot_lows.iloc[i]) / 3
            
            center.iloc[i] = current_center if current_center is not None else df['close'].iloc[i]
        
        # Calculate bands
        up_band = center - (atr_factor * atr)
        down_band = center + (atr_factor * atr)
        
        # Initialize Supertrend
        tup = pd.Series(index=df.index, dtype=float)
        tdown = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        
        for i in range(len(df)):
            if i == 0:
                tup.iloc[i] = up_band.iloc[i]
                tdown.iloc[i] = down_band.iloc[i]
                trend.iloc[i] = 1
            else:
                # TUp calculation
                if df['close'].iloc[i-1] > tup.iloc[i-1]:
                    tup.iloc[i] = max(up_band.iloc[i], tup.iloc[i-1])
                else:
                    tup.iloc[i] = up_band.iloc[i]
                
                # TDown calculation
                if df['close'].iloc[i-1] < tdown.iloc[i-1]:
                    tdown.iloc[i] = min(down_band.iloc[i], tdown.iloc[i-1])
                else:
                    tdown.iloc[i] = down_band.iloc[i]
                
                # Trend determination
                if df['close'].iloc[i] > tdown.iloc[i-1]:
                    trend.iloc[i] = 1  # Bullish
                elif df['close'].iloc[i] < tup.iloc[i-1]:
                    trend.iloc[i] = -1  # Bearish
                else:
                    trend.iloc[i] = trend.iloc[i-1]
        
        df['pvt_trend'] = trend
        df['pvt_tup'] = tup
        df['pvt_tdown'] = tdown
        df['pvt_trailing_sl'] = np.where(trend == 1, tup, tdown)
        df['pvt_signal'] = trend.diff().fillna(0)
        
        return df
        
    except Exception as e:
        print(f"Error in Pivot Supertrend: {e}")
        df['pvt_trend'] = 1
        df['pvt_signal'] = 0
        return df

def calculate_x_trend(df):
    """Calculate X Trend - Core Super XTrend signal"""
    try:
        df = df.copy()
        
        # X Trend components
        lowest_low = df['low'].rolling(window=3, min_periods=1).min()
        highest_high = df['high'].rolling(window=2, min_periods=1).max()
        ma_low = df['low'].ewm(span=3, adjust=False).mean()
        ma_high = df['high'].rolling(window=2).mean()
        
        # Initialize X Trend variables
        next_trend = pd.Series(1.0, index=df.index)  # 1 or 0
        x_trend = pd.Series(0.0, index=df.index)     # 0 = bullish, 1 = bearish
        low_max = pd.Series(index=df.index, dtype=float)
        high_min = pd.Series(index=df.index, dtype=float)
        line_ht = pd.Series(index=df.index, dtype=float)
        
        # Initialize first values
        if len(df) > 0:
            low_max.iloc[0] = df['low'].iloc[0]
            high_min.iloc[0] = df['high'].iloc[0]
            line_ht.iloc[0] = df['close'].iloc[0]
        
        for i in range(1, len(df)):
            # Copy previous values
            next_trend.iloc[i] = next_trend.iloc[i-1]
            x_trend.iloc[i] = x_trend.iloc[i-1]
            low_max.iloc[i] = low_max.iloc[i-1]
            high_min.iloc[i] = high_min.iloc[i-1]
            
            # X Trend logic (matching cTrader v1.2)
            if next_trend.iloc[i] == 1:
                low_max.iloc[i] = max(low_max.iloc[i], lowest_low.iloc[i])
                if (ma_high.iloc[i] < low_max.iloc[i] and 
                    df['close'].iloc[i] < df['low'].iloc[i-1]):
                    x_trend.iloc[i] = 1
                    next_trend.iloc[i] = 0
                    high_min.iloc[i] = highest_high.iloc[i]
            
            if next_trend.iloc[i] == 0:
                high_min.iloc[i] = min(high_min.iloc[i], highest_high.iloc[i])
                if (ma_low.iloc[i] > high_min.iloc[i] and 
                    df['close'].iloc[i] > df['high'].iloc[i-1]):
                    x_trend.iloc[i] = 0
                    next_trend.iloc[i] = 1
                    low_max.iloc[i] = lowest_low.iloc[i]
            
            # Line_HT calculation
            if x_trend.iloc[i] == 0 and x_trend.iloc[i-1] == 0:
                line_ht.iloc[i] = max(low_max.iloc[i], line_ht.iloc[i-1])
            elif x_trend.iloc[i] == 1 and x_trend.iloc[i-1] == 1:
                line_ht.iloc[i] = min(high_min.iloc[i], line_ht.iloc[i-1])
            else:
                line_ht.iloc[i] = line_ht.iloc[i-1]
        
        df['x_trend'] = x_trend
        df['x_trend_signal'] = x_trend.diff().fillna(0)
        df['x_trend_line'] = line_ht
        
        return df
        
    except Exception as e:
        print(f"Error in X Trend calculation: {e}")
        df['x_trend'] = 0
        df['x_trend_signal'] = 0
        df['x_trend_line'] = df['close']
        return df

def calculate_htf_x_trend(df, htf_multiplier):
    """Calculate Higher Timeframe X Trend"""
    try:
        if htf_multiplier <= 1:
            df['htf_x_trend'] = df['x_trend']
            return df
        
        # Resample to higher timeframe
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index, unit='s')
        
        # Create HTF bars
        htf_minutes = htf_multiplier
        htf_data = df_copy.resample(f'{htf_minutes}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calculate X Trend on HTF
        htf_data = calculate_x_trend(htf_data)
        
        # Map back to original timeframe
        df['htf_x_trend'] = df['x_trend']  # Default
        
        for i in range(len(df)):
            try:
                bar_time = pd.to_datetime(df.index[i], unit='s')
                # Find corresponding HTF bar
                htf_bar = htf_data[htf_data.index <= bar_time].iloc[-1]
                df.loc[i, 'htf_x_trend'] = htf_bar['x_trend']
            except:
                pass
        
        return df
        
    except Exception as e:
        print(f"Error in HTF X Trend: {e}")
        df['htf_x_trend'] = df['x_trend']
        return df

def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index)"""
    try:
        # Calculate directional movement
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
        minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff
        
        # Calculate True Range and smooth
        atr = calculate_atr_wilder(df, period)
        
        # Calculate DI+ and DI-
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        df['adx'] = adx.fillna(25)
        df['di_plus'] = plus_di.fillna(0)
        df['di_minus'] = minus_di.fillna(0)
        
        return df
        
    except Exception as e:
        print(f"Error in ADX calculation: {e}")
        df['adx'] = 25
        return df

def calculate_ema(df, period=200):
    """Calculate Exponential Moving Average"""
    try:
        df['ema'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    except Exception as e:
        print(f"Error in EMA calculation: {e}")
        df['ema'] = df['close']
        return df

# ==================== SUPER XTREND v1.2 STRATEGY LOGIC ====================

class SuperXTrendTradeManager:
    """Trade manager for Super XTrend v1.2 strategy"""
    def __init__(self):
        # Trade state
        self.in_trade = False
        self.current_direction = 0  # 1 = long, -1 = short
        self.entry_price = 0
        self.entry_time = 0
        
        # Pending system (v1.2 feature)
        self.buy_signal_pending = False
        self.sell_signal_pending = False
        
        # Trade history
        self.trades = []
        
    def reset_pending_on_xtrend_flip(self, x_trend_flip_buy, x_trend_flip_sell):
        """Reset pending signals on X Trend direction change"""
        if x_trend_flip_sell:
            self.buy_signal_pending = False
        if x_trend_flip_buy:
            self.sell_signal_pending = False

def run_super_xtrend_backtest(df, params, asset_name=""):
    """Run Super XTrend v1.2 backtest"""
    try:
        # Calculate all indicators
        df = calculate_pivot_supertrend(df, 
                                       pivot_period=params['pivot_period'],
                                       atr_factor=params['atr_factor'],
                                       atr_period=params['atr_period'])
        df = calculate_x_trend(df)
        df = calculate_adx(df, period=14)
        df = calculate_ema(df, period=params.get('ema_period', 200))
        
        # HTF X Trend if enabled
        if params.get('use_htf_xtrend', False):
            df = calculate_htf_x_trend(df, params.get('htf_multiplier', 3))
            use_htf = True
        else:
            df['htf_x_trend'] = df['x_trend']
            use_htf = False
        
        # Initialize trade manager
        tm = SuperXTrendTradeManager()
        
        # Strategy mode
        strategy_mode = params.get('strategy_mode', 'TwoBarConfirmation')
        
        for i in range(2, len(df)):  # Start from bar 2 for 2-bar confirmation
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            prev_prev_row = df.iloc[i-2]
            
            # Determine final X Trend (local or HTF)
            if use_htf and params.get('htf_agreement_only', True):
                # Both must agree
                final_x_trend_bullish = (row['x_trend'] == 0) and (row['htf_x_trend'] == 0)
                final_x_trend_bearish = (row['x_trend'] == 1) and (row['htf_x_trend'] == 1)
            elif use_htf:
                # Use HTF only
                final_x_trend_bullish = row['htf_x_trend'] == 0
                final_x_trend_bearish = row['htf_x_trend'] == 1
            else:
                # Use local X Trend
                final_x_trend_bullish = row['x_trend'] == 0
                final_x_trend_bearish = row['x_trend'] == 1
            
            # Detect X Trend flips
            x_trend_flip_to_buy = (row['x_trend'] == 0) and (prev_row['x_trend'] == 1)
            x_trend_flip_to_sell = (row['x_trend'] == 1) and (prev_row['x_trend'] == 0)
            
            # Reset pending signals on X Trend flip
            tm.reset_pending_on_xtrend_flip(x_trend_flip_to_buy, x_trend_flip_to_sell)
            
            # Pivot Supertrend flips
            pvt_flip_to_buy = (row['pvt_trend'] == 1) and (prev_row['pvt_trend'] == -1)
            pvt_flip_to_sell = (row['pvt_trend'] == -1) and (prev_row['pvt_trend'] == 1)
            
            # 2-Bar confirmation signals
            x_2bar_confirm_bull = (final_x_trend_bullish and 
                                  (prev_row['x_trend'] == 0 if not use_htf else prev_row['htf_x_trend'] == 0) and
                                  (prev_prev_row['x_trend'] == 1 if not use_htf else prev_prev_row['htf_x_trend'] == 1))
            
            x_2bar_confirm_bear = (final_x_trend_bearish and 
                                  (prev_row['x_trend'] == 1 if not use_htf else prev_row['htf_x_trend'] == 1) and
                                  (prev_prev_row['x_trend'] == 0 if not use_htf else prev_prev_row['htf_x_trend'] == 0))
            
            # Apply filter settings
            adx_filter_passed = not params.get('use_adx', False) or row['adx'] >= params.get('adx_threshold', 25)
            ema_filter_bullish = not params.get('use_ema', False) or row['close'] > row['ema']
            ema_filter_bearish = not params.get('use_ema', False) or row['close'] < row['ema']
            
            # Generate signals based on strategy mode
            buy_signal = False
            sell_signal = False
            
            if strategy_mode == 'ClassicXTrend':
                # Classic X Trend mode
                if x_trend_flip_to_buy:
                    if not params.get('use_pivot_filter', True) or row['pvt_trend'] == 1:
                        buy_signal = True
                    else:
                        tm.buy_signal_pending = True
                        tm.sell_signal_pending = False
                elif tm.buy_signal_pending and pvt_flip_to_buy:
                    buy_signal = True
                    tm.buy_signal_pending = False
                
                if x_trend_flip_to_sell:
                    if not params.get('use_pivot_filter', True) or row['pvt_trend'] == -1:
                        sell_signal = True
                    else:
                        tm.sell_signal_pending = True
                        tm.buy_signal_pending = False
                elif tm.sell_signal_pending and pvt_flip_to_sell:
                    sell_signal = True
                    tm.sell_signal_pending = False
            
            else:  # TwoBarConfirmation or HighWinRate
                # 2-Bar confirmation mode
                if x_2bar_confirm_bull:
                    if not params.get('use_pivot_filter', True) or row['pvt_trend'] == 1:
                        buy_signal = True
                    else:
                        tm.buy_signal_pending = True
                        tm.sell_signal_pending = False
                elif tm.buy_signal_pending and pvt_flip_to_buy:
                    buy_signal = True
                    tm.buy_signal_pending = False
                
                if x_2bar_confirm_bear:
                    if not params.get('use_pivot_filter', True) or row['pvt_trend'] == -1:
                        sell_signal = True
                    else:
                        tm.sell_signal_pending = True
                        tm.buy_signal_pending = False
                elif tm.sell_signal_pending and pvt_flip_to_sell:
                    sell_signal = True
                    tm.sell_signal_pending = False
            
            # Apply additional filters
            buy_signal = buy_signal and adx_filter_passed and ema_filter_bullish
            sell_signal = sell_signal and adx_filter_passed and ema_filter_bearish
            
            # Process entry signals
            if not tm.in_trade:
                if buy_signal:
                    tm.in_trade = True
                    tm.current_direction = 1
                    tm.entry_price = row['close']
                    tm.entry_time = i
                elif sell_signal:
                    tm.in_trade = True
                    tm.current_direction = -1
                    tm.entry_price = row['close']
                    tm.entry_time = i
            
            # Process exit signals
            elif tm.in_trade:
                exit_signal = False
                exit_reason = ""
                
                exit_mode = params.get('exit_mode', 'XTrendFlip')
                
                if exit_mode == 'XTrendFlip':
                    # Exit on X Trend flip
                    if tm.current_direction == 1 and x_trend_flip_to_sell:
                        exit_signal = True
                        exit_reason = "X Trend Flip"
                    elif tm.current_direction == -1 and x_trend_flip_to_buy:
                        exit_signal = True
                        exit_reason = "X Trend Flip"
                else:
                    # Exit on Pivot Supertrend flip
                    if tm.current_direction == 1 and pvt_flip_to_sell:
                        exit_signal = True
                        exit_reason = "Pivot Supertrend Flip"
                    elif tm.current_direction == -1 and pvt_flip_to_buy:
                        exit_signal = True
                        exit_reason = "Pivot Supertrend Flip"
                
                if exit_signal:
                    # Calculate profit
                    if 'BTC' in asset_name.upper() or 'ETH' in asset_name.upper():
                        # For crypto, use points
                        if tm.current_direction == 1:
                            profit_pips = row['close'] - tm.entry_price
                        else:
                            profit_pips = tm.entry_price - row['close']
                    else:
                        # For forex, use pips
                        if tm.current_direction == 1:
                            profit_pips = (row['close'] - tm.entry_price) / 0.0001
                        else:
                            profit_pips = (tm.entry_price - row['close']) / 0.0001
                    
                    tm.trades.append({
                        'entry_time': tm.entry_time,
                        'exit_time': i,
                        'direction': 'long' if tm.current_direction == 1 else 'short',
                        'entry_price': tm.entry_price,
                        'exit_price': row['close'],
                        'profit_pips': profit_pips,
                        'exit_reason': exit_reason
                    })
                    
                    # Reset trade state
                    tm.in_trade = False
                    tm.current_direction = 0
        
        # Calculate performance metrics
        if len(tm.trades) > 0:
            trades_df = pd.DataFrame(tm.trades)
            winning_trades = trades_df[trades_df['profit_pips'] > 0]
            losing_trades = trades_df[trades_df['profit_pips'] < 0]
            
            metrics = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades_df) * 100,
                'total_pips': trades_df['profit_pips'].sum(),
                'avg_win': winning_trades['profit_pips'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': abs(losing_trades['profit_pips'].mean()) if len(losing_trades) > 0 else 0,
                'profit_factor': (winning_trades['profit_pips'].sum() / abs(losing_trades['profit_pips'].sum())) 
                                if len(losing_trades) > 0 and losing_trades['profit_pips'].sum() != 0 else 999,
                'max_drawdown': calculate_max_drawdown(trades_df),
                'largest_win': winning_trades['profit_pips'].max() if len(winning_trades) > 0 else 0,
                'largest_loss': abs(losing_trades['profit_pips'].min()) if len(losing_trades) > 0 else 0
            }
        else:
            metrics = {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'total_pips': 0, 'avg_win': 0, 'avg_loss': 0,
                'profit_factor': 0, 'max_drawdown': 0, 'largest_win': 0, 'largest_loss': 0
            }
        
        return metrics, tm.trades
        
    except Exception as e:
        print(f"Error in Super XTrend backtest: {e}")
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'total_pips': 0, 'avg_win': 0, 'avg_loss': 0,
            'profit_factor': 0, 'max_drawdown': 0, 'largest_win': 0, 'largest_loss': 0
        }, []

def calculate_max_drawdown(trades_df):
    """Calculate maximum drawdown from trades"""
    try:
        cumulative = trades_df['profit_pips'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        return abs(drawdown.min())
    except:
        return 0

# ==================== OPTIMIZATION ENGINE ====================

def run_super_xtrend_optimization(df, asset_name, optimization_settings):
    """Run Super XTrend v1.2 parameter optimization"""
    try:
        # Limit data for performance
        max_bars = optimization_settings.get('max_bars', 500)
        if len(df) > max_bars:
            df = df.tail(max_bars)
            st.info(f"Using last {max_bars} bars for optimization")
        
        # Define parameter ranges
        mode = optimization_settings.get('mode', 'Quick')
        
        if mode == 'Quick':
            pivot_periods = [3, 5, 7]
            atr_factors = [1.0, 1.25, 1.5, 2.0]
            atr_periods = [10, 15, 20]
            htf_multipliers = [1, 2, 3] if optimization_settings.get('use_htf', False) else [1]
        elif mode == 'Standard':
            pivot_periods = [3, 5, 7, 10]
            atr_factors = [1.0, 1.25, 1.5, 2.0, 2.5]
            atr_periods = [10, 14, 15, 20]
            htf_multipliers = [1, 2, 3, 4, 6] if optimization_settings.get('use_htf', False) else [1]
        else:  # Comprehensive
            pivot_periods = [3, 5, 7, 10, 15]
            atr_factors = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
            atr_periods = [10, 12, 14, 15, 18, 20]
            htf_multipliers = [1, 2, 3, 4, 6, 8] if optimization_settings.get('use_htf', False) else [1]
        
        # Additional parameters
        strategy_modes = optimization_settings.get('strategy_modes', ['TwoBarConfirmation'])
        adx_thresholds = optimization_settings.get('adx_thresholds', [25])
        ema_periods = optimization_settings.get('ema_periods', [200])
        
        total_combinations = (len(pivot_periods) * len(atr_factors) * len(atr_periods) * 
                            len(htf_multipliers) * len(strategy_modes) * len(adx_thresholds) * len(ema_periods))
        
        st.info(f"Testing {total_combinations:,} parameter combinations...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        combination_count = 0
        
        for pivot_period in pivot_periods:
            for atr_factor in atr_factors:
                for atr_period in atr_periods:
                    for htf_mult in htf_multipliers:
                        for strategy_mode in strategy_modes:
                            for adx_threshold in adx_thresholds:
                                for ema_period in ema_periods:
                                    combination_count += 1
                                    progress = combination_count / total_combinations
                                    progress_bar.progress(progress)
                                    status_text.text(f"Testing {combination_count}/{total_combinations}")
                                    
                                    params = {
                                        'pivot_period': pivot_period,
                                        'atr_factor': atr_factor,
                                        'atr_period': atr_period,
                                        'strategy_mode': strategy_mode,
                                        'use_pivot_filter': optimization_settings.get('use_pivot_filter', True),
                                        'exit_mode': optimization_settings.get('exit_mode', 'XTrendFlip'),
                                        'use_adx': optimization_settings.get('use_adx', False),
                                        'adx_threshold': adx_threshold,
                                        'use_ema': optimization_settings.get('use_ema', False),
                                        'ema_period': ema_period,
                                        'use_htf_xtrend': htf_mult > 1,
                                        'htf_multiplier': htf_mult,
                                        'htf_agreement_only': optimization_settings.get('htf_agreement_only', True)
                                    }
                                    
                                    metrics, trades = run_super_xtrend_backtest(df, params, asset_name)
                                    
                                    # Skip if too few trades
                                    if optimization_settings.get('skip_low_volume', True) and metrics['total_trades'] < 5:
                                        continue
                                    
                                    # Calculate optimization score
                                    if metrics['total_trades'] > 0:
                                        score = (
                                            metrics['win_rate'] * 0.3 +
                                            min(metrics['profit_factor'], 3) * 20 +
                                            (metrics['total_pips'] / metrics['total_trades']) * 0.5 -
                                            metrics['max_drawdown'] * 0.1
                                        )
                                    else:
                                        score = 0
                                    
                                    results.append({
                                        'pivot_period': pivot_period,
                                        'atr_factor': atr_factor,
                                        'atr_period': atr_period,
                                        'strategy_mode': strategy_mode,
                                        'htf_multiplier': htf_mult if htf_mult > 1 else None,
                                        'adx_threshold': adx_threshold if optimization_settings.get('use_adx', False) else None,
                                        'ema_period': ema_period if optimization_settings.get('use_ema', False) else None,
                                        'total_trades': metrics['total_trades'],
                                        'win_rate': round(metrics['win_rate'], 2),
                                        'total_pips': round(metrics['total_pips'], 2),
                                        'profit_factor': round(metrics['profit_factor'], 2),
                                        'avg_win': round(metrics['avg_win'], 2),
                                        'avg_loss': round(metrics['avg_loss'], 2),
                                        'max_drawdown': round(metrics['max_drawdown'], 2),
                                        'largest_win': round(metrics['largest_win'], 2),
                                        'largest_loss': round(metrics['largest_loss'], 2),
                                        'score': round(score, 2)
                                    })
        
        progress_bar.empty()
        status_text.empty()
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        return results_df
        
    except Exception as e:
        st.error(f"Optimization error: {e}")
        return pd.DataFrame()

# ==================== DATA FUNCTIONS ====================

def download_yahoo_data(symbol, period='7d', interval='5m'):
    """Download data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            st.error(f"No data found for {symbol}")
            return None
        
        # Format data
        data.reset_index(inplace=True)
        data.columns = [col.lower() for col in data.columns]
        data = data.rename(columns={'datetime': 'time'})
        
        # Convert to timestamp
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time']).astype(int) // 10**9
        
        # Limit data
        if len(data) > 1000:
            data = data.tail(1000)
            st.info(f"Using last 1000 bars for {symbol}")
        
        return data[['time', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        st.error(f"Error downloading {symbol}: {e}")
        return None

def process_uploaded_csv(df, filename):
    """Process uploaded CSV file"""
    try:
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]
        
        required = ['time', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            missing = [col for col in required if col not in df.columns]
            st.error(f"CSV must have columns: {required}. Missing: {missing}")
            return None
        
        # Convert to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['time'] = pd.to_numeric(df['time'], errors='coerce').astype('int64')
        df = df.dropna()
        
        if len(df) > 1000:
            df = df.tail(1000)
            st.info(f"Using last 1000 bars from {filename}")
        
        st.success(f"Successfully processed {filename}: {len(df)} valid bars")
        return df
        
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None

# ==================== VISUALIZATION FUNCTIONS ====================

def create_performance_chart(results_df, metric='total_pips'):
    """Create performance visualization chart"""
    if results_df.empty:
        return None
    
    fig = px.scatter(
        results_df.head(20),
        x='win_rate',
        y='total_pips',
        size='total_trades',
        color='profit_factor',
        hover_data=['pivot_period', 'atr_factor', 'strategy_mode'],
        title="Performance Analysis - Top 20 Results",
        labels={'win_rate': 'Win Rate (%)', 'total_pips': 'Total Pips'}
    )
    
    fig.update_layout(height=500)
    return fig

def create_parameter_heatmap(results_df):
    """Create parameter performance heatmap"""
    if results_df.empty or len(results_df) < 5:
        return None
    
    # Create pivot table for heatmap
    pivot_data = results_df.pivot_table(
        values='score',
        index='pivot_period',
        columns='atr_factor',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        pivot_data,
        title="Parameter Performance Heatmap (Average Score)",
        labels=dict(x="ATR Factor", y="Pivot Period", color="Score"),
        aspect="auto"
    )
    
    fig.update_layout(height=400)
    return fig

# ==================== MAIN APPLICATION ====================

def main():
    st.set_page_config(
        page_title="Super XTrend v1.2 Optimizer",
        page_icon="🚀",
        layout="wide"
    )
    
    # Header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
        <h1 style="color: white; margin: 0;">🚀 Super XTrend v{__version__} Optimizer</h1>
        <p style="color: #e8f4f8; margin: 5px 0 0 0;">Advanced X Trend Strategy Parameter Optimization</p>
        <p style="color: #d0e8f0; margin: 3px 0 0 0; font-size: 0.9em;">Last Updated: {__last_updated__}</p>
        <p style="color: #ffd700; margin: 8px 0 0 0; font-size: 0.95em; font-weight: bold;">
            ✨ Features: Pivot Supertrend Filter | X-Trend HTF | 2-Bar Confirmation | Advanced Risk Management
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Asset configuration
    assets = {
        'BTCUSD': {'yf': 'BTC-USD', 'name': 'Bitcoin/USD'},
        'ETHUSD': {'yf': 'ETH-USD', 'name': 'Ethereum/USD'},
        'EURUSD': {'yf': 'EURUSD=X', 'name': 'Euro/USD'},
        'GBPUSD': {'yf': 'GBPUSD=X', 'name': 'GBP/USD'},
        'USDJPY': {'yf': 'USDJPY=X', 'name': 'USD/JPY'},
        'XAUUSD': {'yf': 'GC=F', 'name': 'Gold/USD'},
        'AUDUSD': {'yf': 'AUDUSD=X', 'name': 'AUD/USD'},
        'USDCAD': {'yf': 'USDCAD=X', 'name': 'USD/CAD'}
    }
    
    # Sidebar configuration
    st.sidebar.header("📊 Configuration")
    
    # Data Source
    data_source = st.sidebar.radio(
        "Data Source",
        options=["Upload CSV", "Yahoo Finance"],
        index=0,
        help="Upload CSV files or download from Yahoo Finance"
    )
    
    # Initialize variables
    selected_assets = []
    uploaded_files = None
    
    if data_source == "Upload CSV":
        st.sidebar.subheader("📁 Upload CSV Files")
        uploaded_files = st.sidebar.file_uploader(
            "Choose CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="CSV must have columns: time, open, high, low, close, volume"
        )
        
        if uploaded_files:
            st.sidebar.success(f"✅ {len(uploaded_files)} file(s) uploaded")
    
    else:  # Yahoo Finance
        selected_assets = st.sidebar.multiselect(
            "Select Assets",
            options=list(assets.keys()),
            default=['EURUSD', 'BTCUSD'],
            format_func=lambda x: f"{x} ({assets[x]['name']})"
        )
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            timeframe = st.selectbox(
                "Timeframe",
                options=['1m', '5m', '15m', '30m', '1h'],
                index=1
            )
        
        with col2:
            period = st.selectbox(
                "Period",
                options=['7d', '1mo', '3mo'],
                index=0
            )
    
    # Optimization Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Optimization Settings")
    
    optimization_mode = st.sidebar.radio(
        "Optimization Mode",
        options=["Quick", "Standard", "Comprehensive"],
        index=0,
        help="Quick: Fast basic optimization | Standard: Balanced | Comprehensive: Thorough (slower)"
    )
    
    # Strategy Settings
    st.sidebar.markdown("### 🎯 Strategy Settings")
    
    strategy_modes = st.sidebar.multiselect(
        "Strategy Modes",
        options=['ClassicXTrend', 'TwoBarConfirmation', 'HighWinRate'],
        default=['TwoBarConfirmation'],
        help="Different signal generation methods"
    )
    
    use_pivot_filter = st.sidebar.checkbox(
        "Use Pivot Supertrend Filter",
        value=True,
        help="Filter signals with Pivot Supertrend trend"
    )
    
    exit_mode = st.sidebar.radio(
        "Exit Mode",
        options=["XTrendFlip", "PivotSupertrendFlip"],
        index=0,
        help="XTrendFlip: Exit on X Trend direction change | PivotSupertrendFlip: Exit on Supertrend change"
    )
    
    # Filter Settings
    st.sidebar.markdown("### 🔧 Filter Settings")
    
    use_adx = st.sidebar.checkbox("Use ADX Filter", value=False)
    if use_adx:
        adx_thresholds = st.sidebar.multiselect(
            "ADX Thresholds",
            options=[15, 20, 25, 30, 35, 40],
            default=[25],
            help="ADX values to test"
        )
    else:
        adx_thresholds = [25]
    
    use_ema = st.sidebar.checkbox("Use EMA Filter", value=False)
    if use_ema:
        ema_periods = st.sidebar.multiselect(
            "EMA Periods",
            options=[50, 100, 150, 200, 250],
            default=[200],
            help="EMA periods to test"
        )
    else:
        ema_periods = [200]
    
    # HTF Settings
    st.sidebar.markdown("### 📈 Higher Timeframe X-Trend")
    
    use_htf = st.sidebar.checkbox(
        "Test HTF X-Trend Variations",
        value=False,
        help="Test different HTF multipliers"
    )
    
    if use_htf:
        htf_agreement_only = st.sidebar.checkbox(
            "Require HTF Agreement",
            value=True,
            help="Both local and HTF X-Trend must agree"
        )
    else:
        htf_agreement_only = True
    
    # Advanced Settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        max_bars = st.slider(
            "Max Bars to Process",
            min_value=200,
            max_value=2000,
            value=500,
            step=100,
            help="Fewer bars = faster processing"
        )
        
        skip_low_volume = st.checkbox(
            "Skip Low Volume Results",
            value=True,
            help="Skip combinations with < 5 trades"
        )
    
    # Main content area
    st.markdown("### 📊 Data Management")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if data_source == "Upload CSV":
            if uploaded_files:
                if st.button("📤 Process CSV Files", type="primary", use_container_width=True):
                    st.session_state.downloaded_data.clear()
                    
                    for file in uploaded_files:
                        df = pd.read_csv(file)
                        processed = process_uploaded_csv(df, file.name)
                        if processed is not None:
                            asset_name = file.name.replace('.csv', '').split('_')[0].upper()
                            st.session_state.downloaded_data[asset_name] = processed
            else:
                st.info("👆 Please upload CSV files to proceed")
        
        else:  # Yahoo Finance
            if st.button("📥 Download Data", type="primary", use_container_width=True):
                st.session_state.downloaded_data.clear()
                
                for asset in selected_assets:
                    with st.spinner(f"Downloading {asset}..."):
                        data = download_yahoo_data(
                            assets[asset]['yf'],
                            period=period,
                            interval=timeframe
                        )
                        if data is not None:
                            st.session_state.downloaded_data[asset] = data
                            st.success(f"✅ {asset}: {len(data)} bars")
    
    with col2:
        if st.session_state.downloaded_data:
            if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
                st.session_state.optimization_results.clear()
                
                # Prepare optimization settings
                optimization_settings = {
                    'mode': optimization_mode,
                    'strategy_modes': strategy_modes,
                    'use_pivot_filter': use_pivot_filter,
                    'exit_mode': exit_mode,
                    'use_adx': use_adx,
                    'adx_thresholds': adx_thresholds,
                    'use_ema': use_ema,
                    'ema_periods': ema_periods,
                    'use_htf': use_htf,
                    'htf_agreement_only': htf_agreement_only,
                    'max_bars': max_bars,
                    'skip_low_volume': skip_low_volume
                }
                
                for asset, data in st.session_state.downloaded_data.items():
                    with st.container():
                        st.write(f"**Optimizing {asset}...**")
                        
                        results = run_super_xtrend_optimization(data, asset, optimization_settings)
                        
                        if not results.empty:
                            st.session_state.optimization_results[asset] = results
                            
                            # Show brief summary
                            best = results.iloc[0]
                            summary = f"Win Rate: {best['win_rate']}%, Profit: {best['total_pips']:.1f} pips"
                            st.success(f"Best: {summary}")
    
    with col3:
        if st.session_state.optimization_results:
            if st.button("📊 Clear Results", type="secondary", use_container_width=True):
                st.session_state.optimization_results.clear()
                st.session_state.downloaded_data.clear()
                st.rerun()
    
    # Display current data status
    if st.session_state.downloaded_data:
        st.markdown("---")
        st.markdown("### 📈 Loaded Data")
        
        data_cols = st.columns(len(st.session_state.downloaded_data))
        for idx, (asset, data) in enumerate(st.session_state.downloaded_data.items()):
            with data_cols[idx]:
                # Determine timeframe
                if len(data) > 1:
                    time_diff = data['time'].iloc[1] - data['time'].iloc[0]
                    tf_minutes = time_diff / 60
                    tf_str = f"{int(tf_minutes)}m" if tf_minutes < 60 else f"{int(tf_minutes/60)}h"
                else:
                    tf_str = "N/A"
                
                st.metric(
                    label=asset,
                    value=f"{len(data)} bars",
                    delta=f"{tf_str} timeframe"
                )
    
    # Results section
    if st.session_state.optimization_results:
        st.markdown("---")
        st.markdown("### 🏆 Optimization Results")
        
        # Summary table
        summary_data = []
        for asset, results in st.session_state.optimization_results.items():
            if not results.empty:
                best = results.iloc[0]
                summary_data.append({
                    'Asset': asset,
                    'Win Rate': f"{best['win_rate']}%",
                    'Total Pips': f"{best['total_pips']:.1f}",
                    'Profit Factor': f"{best['profit_factor']:.2f}",
                    'Trades': best['total_trades'],
                    'Max DD': f"{best['max_drawdown']:.1f}",
                    'Score': f"{best['score']:.1f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
        
        # Detailed results tabs
        tabs = st.tabs(list(st.session_state.optimization_results.keys()))
        
        for tab, asset in zip(tabs, st.session_state.optimization_results.keys()):
            with tab:
                results = st.session_state.optimization_results[asset]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**🥇 Top 10 Results:**")
                    
                    # Display columns
                    display_cols = ['pivot_period', 'atr_factor', 'atr_period', 'strategy_mode']
                    
                    if any(results['htf_multiplier'].notna()):
                        display_cols.append('htf_multiplier')
                    if any(results['adx_threshold'].notna()):
                        display_cols.append('adx_threshold')
                    if any(results['ema_period'].notna()):
                        display_cols.append('ema_period')
                    
                    display_cols.extend(['total_trades', 'win_rate', 'total_pips', 'profit_factor', 'score'])
                    
                    st.dataframe(
                        results[display_cols].head(10),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    # Best configuration
                    best = results.iloc[0]
                    
                    st.write("**📍 Optimal Settings:**")
                    
                    # Format for cBot
                    st.code(f"""// Super XTrend v1.2 Settings for {asset}
// === CORE STRATEGY ===
Pivot Period: {best['pivot_period']}
ATR Factor: {best['atr_factor']}
ATR Period: {best['atr_period']}
Strategy Mode: {best['strategy_mode']}
Exit Mode: {exit_mode}

// === FILTERS ===
Use Pivot Filter: {use_pivot_filter}
Use ADX Filter: {use_adx}
ADX Threshold: {best['adx_threshold'] or 25}
Use EMA Filter: {use_ema}
EMA Period: {best['ema_period'] or 200}

// === HTF X-TREND ===
Use HTF X-Trend: {best['htf_multiplier'] is not None}
HTF Multiplier: {best['htf_multiplier'] or 1}
Require HTF Agreement: {htf_agreement_only}

// === PERFORMANCE ===
Win Rate: {best['win_rate']}%
Total Trades: {best['total_trades']}
Total Pips: {best['total_pips']:.1f}
Profit Factor: {best['profit_factor']:.2f}
Max Drawdown: {best['max_drawdown']:.1f}
Largest Win: {best['largest_win']:.1f}
Largest Loss: {best['largest_loss']:.1f}
                    """)
                
                # Performance analysis
                st.write("**📊 Performance Analysis:**")
                
                col_perf1, col_perf2 = st.columns(2)
                
                with col_perf1:
                    # Create performance chart
                    perf_chart = create_performance_chart(results)
                    if perf_chart:
                        st.plotly_chart(perf_chart, use_container_width=True)
                
                with col_perf2:
                    # Create parameter heatmap
                    heatmap = create_parameter_heatmap(results)
                    if heatmap:
                        st.plotly_chart(heatmap, use_container_width=True)
                
                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {asset} Results CSV",
                    data=csv,
                    file_name=f"super_xtrend_v12_optimization_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        if not st.session_state.downloaded_data:
            st.info("👈 Upload CSV files or select assets to begin optimization")
        else:
            st.info("✨ Data ready! Click 'Run Optimization' to find optimal settings")
    
    # Information sections
    st.markdown("---")
    with st.expander("🚀 Super XTrend v1.2 Features"):
        st.markdown("""
        ### 🎯 Core Strategy Components:
        
        **1. Pivot Supertrend Filter**
        - Uses pivot highs/lows to create dynamic center line
        - Advanced ATR-based volatility bands
        - Wilder's smoothing for consistent signals
        
        **2. X Trend Primary Signal**
        - Proprietary trend detection algorithm
        - 3-period low EMA and 2-period high SMA
        - Dynamic support/resistance line calculation
        
        **3. Strategy Modes**
        - **Classic X Trend**: Direct X Trend flips with pending system
        - **2-Bar Confirmation**: Enhanced signal validation
        - **High Win Rate**: Same as 2-Bar with stricter filters
        
        **4. Pending Signal System**
        - Signals can wait for Pivot Supertrend confirmation
        - Reduces false entries in choppy markets
        - Maintains signal integrity across timeframes
        
        **5. Higher Timeframe X Trend**
        - Optional HTF trend confirmation
        - Configurable multipliers (2x, 3x, 4x, etc.)
        - Agreement mode for reduced noise
        
        **6. Advanced Filters**
        - **ADX Filter**: Minimum trend strength requirement
        - **EMA Filter**: Long-term trend alignment
        - **Exit Modes**: X Trend flip or Supertrend change
        
        ### 📊 Risk Management Features:
        - Comprehensive drawdown analysis
        - Win/loss ratio optimization
        - Trade frequency control
        - Performance scoring system
        """)
    
    with st.expander("📋 How to Apply Settings in cTrader"):
        st.markdown("""
        ### 📋 Applying Optimized Settings to cTrader cBot:
        
        **Step 1: Core Strategy Settings**
        ```
        Pivot Point Period: [from optimization]
        ATR Factor: [from optimization]
        ATR Period: [from optimization]
        Strategy Mode: [from optimization - ClassicXTrend/TwoBarConfirmation/HighWinRate]
        ```
        
        **Step 2: Filter Settings**
        ```
        Use Pivot Supertrend Filter: [Yes/No from optimization]
        Exit Mode: [XTrendFlip/PivotSupertrendFlip from optimization]
        Use ADX Filter: [Yes/No from optimization]
        ADX Threshold: [from optimization if ADX enabled]
        Use EMA Filter: [Yes/No from optimization]
        EMA Period: [from optimization if EMA enabled]
        ```
        
        **Step 3: HTF X Trend Settings**
        ```
        Use HTF X Trend: [Yes if HTF Multiplier > 1]
        HTF Multiplier: [from optimization]
        Require HTF Agreement: [Yes/No from optimization]
        ```
        
        **Step 4: Risk Management (cBot specific)**
        ```
        USD Amount per Trade: [your preference, e.g., $10]
        Max Concurrent Trades: 1
        Daily Loss Limit: [your preference, e.g., $200]
        Circuit Breaker Loss: [your preference, e.g., $400]
        Max Slippage (pips): [your preference, e.g., 50]
        ```
        
        **Step 5: Display Settings**
        ```
        Show Statistics Table: Yes
        Max Trades to Display: 30
        Show Buy/Sell Labels: Yes
        Show Exit Labels: Yes
        Show Entry/Exit Markers: Yes
        Show Trade Connection Lines: Yes
        ```
        
        ### ⚠️ Important Notes:
        - Use exact parameter values from optimization results
        - Test on demo account before live trading
        - Monitor performance and adjust risk limits as needed
        - Consider market conditions when applying settings
        """)
    
    with st.expander("📈 Understanding the Results"):
        st.markdown("""
        ### 📊 Key Performance Metrics Explained:
        
        **Win Rate**: Percentage of profitable trades
        - Higher is generally better, but balance with other metrics
        - 60%+ is excellent, 50%+ is good for most strategies
        
        **Total Pips**: Sum of all trade profits/losses
        - Measures absolute performance
        - Consider in context of number of trades
        
        **Profit Factor**: Gross profit ÷ Gross loss
        - Values > 1.0 are profitable
        - 1.5+ is good, 2.0+ is excellent
        
        **Average Win/Loss**: Mean profit/loss per winning/losing trade
        - Important for risk-reward assessment
        - Aim for Avg Win > Avg Loss
        
        **Maximum Drawdown**: Largest peak-to-trough decline
        - Lower is better for capital preservation
        - Should be manageable relative to account size
        
        **Score**: Composite optimization score
        - Balances multiple performance factors
        - Higher scores indicate better overall performance
        
        ### 🎯 Optimization Tips:
        - Focus on consistency over maximum profit
        - Consider trade frequency vs. profit per trade
        - Test multiple timeframes and market conditions
        - Validate results with out-of-sample data
        - Monitor live performance vs. backtest results
        """)
    
# Footer
    st.markdown(
        f"""
        <div style="text-align: center; color: #666; margin-top: 40px;">
            <small>
            Super XTrend v{__version__} Optimizer | Advanced X Trend Strategy Optimization<br>
            Pivot Supertrend Filter | HTF X-Trend | 2-Bar Confirmation | Risk Management<br>
            Last Updated: {__last_updated__} | Compatible with cTrader Super XTrend v1.2
            </small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
