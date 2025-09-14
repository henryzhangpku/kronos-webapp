#!/usr/bin/env python3
"""
Kronos AI Web Application
A modern web interface for Kronos AI market prediction and backtesting

This Flask application provides a web UI for:
- Generating market predictions for stocks and cryptocurrencies
- Running backtests on historical data
- Managing custom symbols and asset lists
- Visualizing results with interactive charts
"""

import os
import sys
import json
import logging
from datetime import datetime
from io import BytesIO
import base64

from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web deployment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KronosWebAPI:
    """
    Web API wrapper for Kronos AI functionality
    Handles predictions, backtesting, and data visualization
    """
    
    def __init__(self):
        self.popular_stocks = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
            'NVDA', 'META', 'NFLX', 'ADBE', 'CRM',
            'SPY', 'QQQ', 'IWM', 'GLD', 'TLT'
        ]
        
        self.popular_crypto = [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA',
            'SOL', 'DOGE', 'DOT', 'AVAX', 'MATIC',
            'LINK', 'UNI', 'LTC', 'BCH', 'ETC'
        ]
        
        self.custom_symbols = {
            'stock': set(),
            'crypto': set()
        }
        
        # Set to True since we have self-contained modules
        self.kronos_available = True
        
    def get_currency_symbol(self, asset_type):
        """Get currency symbol for price axis"""
        return 'USD' if asset_type == 'crypto' else 'USD'
        
    def get_available_symbols(self, asset_type):
        """Get list of available symbols for given asset type"""
        base_symbols = self.popular_stocks if asset_type == 'stock' else self.popular_crypto
        custom_symbols = list(self.custom_symbols[asset_type])
        return sorted(list(set(base_symbols + custom_symbols)))
    
    def add_symbol(self, symbol, asset_type):
        """Add a custom symbol to the available list"""
        try:
            symbol = symbol.upper().strip()
            if not symbol:
                return False, "Empty symbol provided"
            
            if asset_type not in ['stock', 'crypto']:
                return False, "Invalid asset type"
            
            self.custom_symbols[asset_type].add(symbol)
            logger.info(f"Added custom {asset_type} symbol: {symbol}")
            return True, f"Symbol {symbol} added successfully"
            
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
            return False, str(e)
    
    def generate_prediction_plot(self, symbol, asset_type, pred_length=120):
        """Generate prediction plot for given symbol"""
        try:
            # Import from local backtest module
            from backtest.kronos_pred import run_kronos_future_prediction
            
            logger.info(f"Generating prediction for {symbol} ({asset_type})")
            
            # Use our local prediction function
            result = run_kronos_future_prediction(
                symbol=symbol.upper(),
                data_type=asset_type,
                pred_len=pred_length
            )
            
            if not result['success']:
                logger.error(f"Prediction failed: {result['error']}")
                return False, None, {'error': result['error']}
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plt.style.use('dark_background')
            
            # Plot historical data
            hist_data = result['historical_data'].tail(200)  # Show last 200 points
            hist_dates = hist_data.index
            
            # Create date range for the entire plot
            pred_data = result['predictions']
            pred_dates = pred_data.index
            
            # Plot historical data with actual dates
            plt.plot(hist_dates, hist_data['close'].values, 
                    label='Historical', color='#4285f4', linewidth=2)
            
            # Plot prediction with actual dates
            plt.plot(pred_dates, pred_data['close'].values, 
                    label='Prediction', color='#34a853', linewidth=2, linestyle='--')
            
            plt.title(f'{symbol} - {pred_length} Period Prediction', 
                     fontsize=16, color='white', pad=20)
            plt.xlabel('Date', fontsize=12, color='white')
            plt.ylabel(f'Price ({self.get_currency_symbol(asset_type)})', fontsize=12, color='white')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().tick_params(axis='x', rotation=45)
            
            # Format y-axis prices
            from matplotlib.ticker import FuncFormatter
            def price_formatter(x, pos):
                if x >= 1000:
                    return f'${x:,.0f}'
                else:
                    return f'${x:.2f}'
            plt.gca().yaxis.set_major_formatter(FuncFormatter(price_formatter))
            
            # Style the plot
            plt.gca().set_facecolor('#1a1a1a')
            plt.gcf().patch.set_facecolor('#1a1a1a')
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight',
                       facecolor='#1a1a1a', edgecolor='none')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            # Return plot and stats
            current_price = hist_data['close'].iloc[-1]
            predicted_price = pred_data['close'].iloc[-1]
            predicted_change = (predicted_price - current_price) / current_price
            
            stats = {
                'current_price': round(current_price, 2),
                'predicted_change': round(predicted_change, 4)
            }
            
            return True, img_base64, stats
            
        except Exception as e:
            logger.error(f"Error generating prediction plot: {e}")
            return False, None, {'error': str(e)}
    
    def generate_backtest_plot(self, symbol, asset_type, lookback=512):
        """Generate backtest plot for given symbol"""
        try:
            # Import from local backtest module
            from backtest.kronos_bt import run_kronos_prediction
            
            logger.info(f"Generating backtest for {symbol} ({asset_type})")
            
            # Run backtest
            result = run_kronos_prediction(
                symbol=symbol.upper(),
                data_type=asset_type,
                lookback=lookback
            )
            
            if not result['success']:
                logger.error(f"Backtest failed: {result['error']}")
                return False, None, {'error': result['error']}
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plt.style.use('dark_background')
            
            # Plot actual vs predicted
            plt.subplot(2, 1, 1)
            hist_data = result['historical_data']
            pred_data = result['predictions']
            ground_truth = result.get('ground_truth', pd.DataFrame())
            
            plt.plot(range(len(hist_data)), hist_data['close'].values, 
                    label='Historical', color='#4285f4', linewidth=2)
            
            if not ground_truth.empty:
                gt_start = len(hist_data)
                gt_range = range(gt_start, gt_start + len(ground_truth))
                plt.plot(gt_range, ground_truth['close'].values,
                        label='Actual', color='#00ff00', linewidth=2)
            
            pred_start = len(hist_data)
            pred_range = range(pred_start, pred_start + len(pred_data))
            plt.plot(pred_range, pred_data['close'].values,
                    label='Predicted', color='#ff6b35', linewidth=2)
            
            plt.title(f'{symbol} - Backtest Analysis', 
                     fontsize=16, color='white', pad=20)
            plt.ylabel('Price', fontsize=12, color='white')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.gca().set_facecolor('#1a1a1a')
            
            # Plot performance metrics
            plt.subplot(2, 1, 2)
            if not ground_truth.empty and len(pred_data) > 0:
                min_len = min(len(pred_data), len(ground_truth))
                if min_len > 0:
                    pred_values = pred_data['close'].values[:min_len]
                    actual_values = ground_truth['close'].values[:min_len]
                    error_pct = np.abs(pred_values - actual_values) / actual_values * 100
                    
                    plt.plot(range(min_len), error_pct, 
                            color='#fbbc05', linewidth=2, label='Error %')
                    plt.title('Prediction Error', fontsize=14, color='white')
                    plt.ylabel('Error %', fontsize=12, color='white')
                    plt.xlabel('Time Period', fontsize=12, color='white')
                    plt.legend(fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.gca().set_facecolor('#1a1a1a')
            
            plt.tight_layout()
            plt.gcf().patch.set_facecolor('#1a1a1a')
            
            # Convert to base64
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight',
                       facecolor='#1a1a1a', edgecolor='none')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            # Calculate stats
            if not ground_truth.empty and len(pred_data) > 0:
                min_len = min(len(pred_data), len(ground_truth))
                if min_len > 0:
                    pred_values = pred_data['close'].values[:min_len]
                    actual_values = ground_truth['close'].values[:min_len]
                    
                    # Calculate accuracy (percentage of correct directional predictions)
                    if min_len > 1:
                        pred_changes = np.diff(pred_values)
                        actual_changes = np.diff(actual_values)
                        correct_directions = np.sum(np.sign(pred_changes) == np.sign(actual_changes))
                        accuracy = correct_directions / len(pred_changes)
                    else:
                        accuracy = 0.5
                    
                    # Calculate returns
                    total_return = (actual_values[-1] - actual_values[0]) / actual_values[0]
                    
                    # Simple Sharpe ratio approximation
                    returns = np.diff(actual_values) / actual_values[:-1]
                    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                else:
                    accuracy = 0.5
                    total_return = 0
                    sharpe_ratio = 0
            else:
                accuracy = 0.5
                total_return = 0
                sharpe_ratio = 0
            
            stats = {
                'accuracy': accuracy,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio
            }
            
            return True, img_base64, stats
            
        except Exception as e:
            logger.error(f"Error generating backtest plot: {e}")
            return False, None, {'error': str(e)}


# Initialize Flask app and Kronos API
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'kronos-ai-web-app-secret-key')
kronos_api = KronosWebAPI()

@app.route('/')
def index():
    """Main web interface"""
    return render_template('index.html',
                         popular_stocks=kronos_api.popular_stocks,
                         popular_crypto=kronos_api.popular_crypto,
                         kronos_available=kronos_api.kronos_available)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for generating predictions"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        asset_type = data.get('asset_type', 'stock')
        pred_length = int(data.get('pred_length', 120))
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'})
        
        if pred_length < 30 or pred_length > 500:
            return jsonify({'success': False, 'error': 'Prediction length must be between 30 and 500'})
        
        success, plot_data, stats = kronos_api.generate_prediction_plot(symbol, asset_type, pred_length)
        
        if success:
            return jsonify({
                'success': True,
                'plot': plot_data,
                'stats': stats
            })
        else:
            return jsonify({'success': False, 'error': stats.get('error', 'Unknown error')})
            
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """API endpoint for running backtests"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        asset_type = data.get('asset_type', 'stock')
        lookback = int(data.get('lookback', 512))
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'})
        
        if lookback < 100 or lookback > 1000:
            return jsonify({'success': False, 'error': 'Lookback period must be between 100 and 1000'})
        
        success, plot_data, stats = kronos_api.generate_backtest_plot(symbol, asset_type, lookback)
        
        if success:
            return jsonify({
                'success': True,
                'plot': plot_data,
                'stats': stats
            })
        else:
            return jsonify({'success': False, 'error': stats.get('error', 'Unknown error')})
            
    except Exception as e:
        logger.error(f"Error in backtest endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/add_symbol', methods=['POST'])
def api_add_symbol():
    """API endpoint for adding custom symbols"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        asset_type = data.get('asset_type', 'stock')
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'})
        
        success, message = kronos_api.add_symbol(symbol, asset_type)
        
        return jsonify({
            'success': success,
            'message': message
        })
        
    except Exception as e:
        logger.error(f"Error in add_symbol endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'kronos_available': kronos_api.kronos_available
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Kronos AI Web Application on port {port}")
    logger.info(f"Kronos AI available: {kronos_api.kronos_available}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)