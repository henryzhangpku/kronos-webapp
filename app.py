#!/usr/bin/env python3
"""
Kronos AI Web Application
========================
Flask web app for Kronos market prediction and backtesting.
Supports both stocks and cryptocurrencies with interactive web interface.

Features:
- Web UI for asset selection (stocks/crypto)
- Real-time prediction generation 
- Backtesting visualization
- Search and add new symbols
- Responsive design for various devices
- Optimized for vast.ai deployment

Author: QuantS
"""

import os
import sys
import logging
import json
from datetime import datetime
import io
import base64
from typing import Dict, List, Optional

# Add parent directory to path for imports
webapp_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(webapp_dir)
sys.path.append(parent_dir)

# Flask imports
from flask import Flask, render_template, request, jsonify, send_file
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import Kronos functionality
try:
    # Add kronos directory to path
    kronos_path = os.path.join(parent_dir, 'kronos')
    if kronos_path not in sys.path:
        sys.path.append(kronos_path)
    
    # Import the core Kronos functions
    from ai.kronos import get_enhanced_market_data
    
    # Import plotting functions from existing scripts
    sys.path.append(os.path.join(parent_dir, 'backtest'))
    
    KRONOS_AVAILABLE = True
    logging.info("✅ Kronos model imported successfully")
except ImportError as e:
    KRONOS_AVAILABLE = False
    logging.warning(f"❌ Kronos model not available: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'kronos-web-app-secret-key'

# Configuration
POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 
    'AMD', 'INTC', 'CRM', 'ADBE', 'ORCL', 'AVGO', 'QCOM', 'TXN',
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'ARKK', 'SOXL'
]

POPULAR_CRYPTO = [
    'BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'LTC', 'BCH',
    'XRP', 'DOGE', 'SHIB', 'AVAX', 'MATIC', 'ALGO', 'ATOM', 'NEAR',
    'SOL', 'LUNA', 'FTT', 'BNB', 'CRO', 'MANA', 'SAND', 'AXS'
]

# Default parameters
DEFAULT_PRED_LEN = 120
DEFAULT_LOOKBACK = 512
DEFAULT_DEVICE = 'cpu'
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9

class KronosWebAPI:
    """Web API wrapper for Kronos functionality"""
    
    def __init__(self):
        self.plot_cache = {}
    
    def generate_prediction_plot(self, symbol: str, asset_type: str, pred_len: int = DEFAULT_PRED_LEN) -> Optional[str]:
        """Generate prediction plot and return base64 encoded image"""
        try:
            # Import and run the prediction function from the parent directory
            sys.path.append(os.path.join(parent_dir, 'backtest'))
            from kronos_pred import run_kronos_future_prediction
            
            logger.info(f"Generating prediction for {symbol} ({asset_type})")
            
            # Run prediction
            result = run_kronos_future_prediction(
                symbol=symbol.upper(),
                data_type=asset_type,
                pred_len=pred_len,
                device=DEFAULT_DEVICE,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P
            )
            
            if not result['success']:
                logger.error(f"Prediction failed: {result['error']}")
                return None
            
            # Generate plot
            fig, ax = plt.subplots(2, 1, figsize=(16, 12))
            
            # Apply dark theme
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#0B0E14')
            
            # Plot historical and predicted data (simplified version)
            hist_data = result['historical_data']
            pred_data = result['predictions']
            
            # Plot recent historical data
            context_periods = min(200, len(hist_data))
            recent_hist = hist_data.tail(context_periods)
            
            ax[0].plot(range(len(recent_hist)), recent_hist['close'].values, 
                      color='#00D2FF', linewidth=2.5, label='Historical Data', alpha=0.9)
            
            # Plot predictions
            pred_start = len(recent_hist)
            pred_range = range(pred_start, pred_start + len(pred_data))
            ax[0].plot(pred_range, pred_data['close'].values,
                      color='#FF6B35', linewidth=2.5, label='Predictions', alpha=0.9)
            
            ax[0].set_title(f'{symbol} - Kronos AI Prediction', color='white', fontsize=16, fontweight='bold')
            ax[0].set_ylabel('Price', color='white', fontsize=12)
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)
            
            # Volume plot (if available)
            if 'volume' in hist_data.columns:
                ax[1].bar(range(len(recent_hist)), recent_hist['volume'].values, 
                         color='#FFD700', alpha=0.6, label='Volume')
                ax[1].set_title('Volume', color='white', fontsize=14)
                ax[1].set_ylabel('Volume', color='white', fontsize=12)
                ax[1].legend()
                ax[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', 
                       facecolor='#0B0E14', edgecolor='none', dpi=100)
            img_buffer.seek(0)
            
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error generating prediction plot: {e}")
            return None
    
    def generate_backtest_plot(self, symbol: str, asset_type: str, lookback: int = DEFAULT_LOOKBACK, 
                              pred_len: int = DEFAULT_PRED_LEN) -> Optional[str]:
        """Generate backtest plot and return base64 encoded image"""
        try:
            # Import and run the backtest function from the parent directory
            sys.path.append(os.path.join(parent_dir, 'backtest'))
            from kronos_bt import run_kronos_prediction
            
            logger.info(f"Generating backtest for {symbol} ({asset_type})")
            
            # Run backtest
            result = run_kronos_prediction(
                symbol=symbol.upper(),
                data_type=asset_type,
                lookback=lookback,
                pred_len=pred_len,
                device=DEFAULT_DEVICE,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P
            )
            
            if not result['success']:
                logger.error(f"Backtest failed: {result['error']}")
                return None
            
            # Generate plot (simplified version)
            fig, ax = plt.subplots(2, 1, figsize=(16, 12))
            
            # Apply dark theme
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#0B0E14')
            
            # Plot historical, ground truth, and predicted data
            hist_data = result['historical_data']
            pred_data = result['predictions']
            ground_truth = result.get('ground_truth', pd.DataFrame())
            
            # Plot data
            ax[0].plot(range(len(hist_data)), hist_data['close'].values, 
                      color='#00D2FF', linewidth=2, label='Historical Data', alpha=0.8)
            
            if not ground_truth.empty and len(ground_truth) > 0:
                gt_start = len(hist_data)
                gt_range = range(gt_start, gt_start + len(ground_truth))
                ax[0].plot(gt_range, ground_truth['close'].values,
                          color='#00FF00', linewidth=2.5, label='Ground Truth', alpha=0.9)
            
            pred_start = len(hist_data)
            pred_range = range(pred_start, pred_start + len(pred_data))
            ax[0].plot(pred_range, pred_data['close'].values,
                      color='#FF6B35', linewidth=2.5, label='Predictions', alpha=0.9)
            
            ax[0].set_title(f'{symbol} - Kronos AI Backtest', color='white', fontsize=16, fontweight='bold')
            ax[0].set_ylabel('Price', color='white', fontsize=12)
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)
            
            # Accuracy metrics plot (if available)
            if not ground_truth.empty and len(ground_truth) > 0:
                # Calculate and plot error
                min_len = min(len(pred_data), len(ground_truth))
                if min_len > 0:
                    pred_values = pred_data['close'].values[:min_len]
                    gt_values = ground_truth['close'].values[:min_len]
                    error = np.abs(pred_values - gt_values) / gt_values * 100
                    
                    ax[1].plot(range(min_len), error, color='#FF6B35', linewidth=2, label='Error %')
                    ax[1].set_title('Prediction Error %', color='white', fontsize=14)
                    ax[1].set_ylabel('Error %', color='white', fontsize=12)
                    ax[1].legend()
                    ax[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', 
                       facecolor='#0B0E14', edgecolor='none', dpi=100)
            img_buffer.seek(0)
            
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close(fig)
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error generating backtest plot: {e}")
            return None

# Initialize API
kronos_api = KronosWebAPI()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         popular_stocks=POPULAR_STOCKS,
                         popular_crypto=POPULAR_CRYPTO,
                         kronos_available=KRONOS_AVAILABLE)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Generate prediction plot"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        asset_type = data.get('type', 'stock')
        pred_len = int(data.get('pred_len', DEFAULT_PRED_LEN))
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'})
        
        if not KRONOS_AVAILABLE:
            return jsonify({'success': False, 'error': 'Kronos model not available'})
        
        # Generate plot
        plot_data = kronos_api.generate_prediction_plot(symbol, asset_type, pred_len)
        
        if plot_data:
            return jsonify({
                'success': True,
                'plot': plot_data,
                'symbol': symbol,
                'type': asset_type
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to generate prediction'})
            
    except Exception as e:
        logger.error(f"API predict error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Generate backtest plot"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        asset_type = data.get('type', 'stock')
        lookback = int(data.get('lookback', DEFAULT_LOOKBACK))
        pred_len = int(data.get('pred_len', DEFAULT_PRED_LEN))
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'})
        
        if not KRONOS_AVAILABLE:
            return jsonify({'success': False, 'error': 'Kronos model not available'})
        
        # Generate plot
        plot_data = kronos_api.generate_backtest_plot(symbol, asset_type, lookback, pred_len)
        
        if plot_data:
            return jsonify({
                'success': True,
                'plot': plot_data,
                'symbol': symbol,
                'type': asset_type
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to generate backtest'})
            
    except Exception as e:
        logger.error(f"API backtest error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/add_symbol', methods=['POST'])
def api_add_symbol():
    """Add new symbol to the list"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        asset_type = data.get('type', 'stock')
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'})
        
        # Add to appropriate list
        if asset_type == 'stock' and symbol not in POPULAR_STOCKS:
            POPULAR_STOCKS.append(symbol)
        elif asset_type == 'crypto' and symbol not in POPULAR_CRYPTO:
            POPULAR_CRYPTO.append(symbol)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'type': asset_type,
            'stocks': POPULAR_STOCKS,
            'crypto': POPULAR_CRYPTO
        })
        
    except Exception as e:
        logger.error(f"API add symbol error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({
        'status': 'healthy',
        'kronos_available': KRONOS_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Get port from environment variable for vast.ai compatibility
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)