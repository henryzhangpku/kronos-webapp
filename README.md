# Kronos AI Web Application

🚀 **Advanced AI-Powered Market Prediction & Backtesting Platform**

A modern web application that brings the power of Kronos AI to your browser, featuring real-time market predictions and comprehensive backtesting capabilities for both stocks and cryptocurrencies.

![Kronos AI](https://img.shields.io/badge/Kronos-AI%20Powered-blue?style=for-the-badge&logo=chart-line)
![Python](https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3.3-red?style=for-the-badge&logo=flask)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)

## 🌟 Features

### 📈 Market Prediction
- **AI-Powered Forecasting**: Generate future price predictions using advanced machine learning models
- **Multi-Asset Support**: Supports both stocks and cryptocurrencies
- **Customizable Timeframes**: Adjust prediction length from 30 to 500 periods
- **Real-time Visualization**: Interactive charts with professional trading theme

### 🕰️ Backtesting Engine
- **Historical Analysis**: Test prediction accuracy on historical data
- **Performance Metrics**: Comprehensive statistics including accuracy, returns, and Sharpe ratio
- **Configurable Lookback**: Analyze performance over 100-1000 historical periods
- **Visual Reports**: Clear charts showing prediction vs actual performance

### 🎨 Modern Web Interface
- **Dark Trading Theme**: Professional interface designed for traders
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Real-time Updates**: AJAX-powered interface with loading indicators
- **Symbol Management**: Add custom symbols beyond the popular lists

### 🚀 Deployment Ready
- **Docker Support**: Containerized for easy deployment
- **vast.ai Compatible**: Optimized for GPU cloud deployment
- **Production Ready**: Gunicorn WSGI server with multi-worker support
- **Auto-deployment**: One-click deployment script included

## 📷 Screenshots

*Dark trading interface with prediction and backtesting capabilities*

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/henryzhangpku/kronos-webapp.git
cd kronos-webapp

# Run the automated deployment script
bash deploy.sh
```

The application will be available at `http://localhost:5000`

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/henryzhangpku/kronos-webapp.git
cd kronos-webapp

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Option 3: vast.ai Deployment

1. Create a new instance on vast.ai
2. Upload this repository to your instance
3. Run the deployment script:
   ```bash
   bash deploy.sh
   ```
4. Access via your instance's public IP on port 5000

## 📚 API Documentation

### Endpoints

#### `POST /api/predict`
Generate market predictions for a given symbol.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "asset_type": "stock",
  "pred_length": 120
}
```

**Response:**
```json
{
  "success": true,
  "plot": "base64_encoded_image",
  "stats": {
    "current_price": 175.50,
    "predicted_change": 0.025
  }
}
```

#### `POST /api/backtest`
Run backtesting analysis for a given symbol.

**Request Body:**
```json
{
  "symbol": "BTC",
  "asset_type": "crypto",
  "lookback": 512
}
```

**Response:**
```json
{
  "success": true,
  "plot": "base64_encoded_image",
  "stats": {
    "accuracy": 0.73,
    "total_return": 0.15,
    "sharpe_ratio": 1.45
  }
}
```

#### `POST /api/add_symbol`
Add a custom symbol to the available list.

**Request Body:**
```json
{
  "symbol": "NVDA",
  "asset_type": "stock"
}
```

## 📝 Project Structure

```
kronos-webapp/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── deploy.sh             # Automated deployment script
├── README.md             # This file
├── templates/
│   └── index.html        # Main web interface
└── static/
    ├── css/
    │   └── styles.css    # Dark trading theme styles
    └── js/
        └── app.js        # Frontend JavaScript logic
```

## 🔧 Technical Details

### Backend
- **Framework**: Flask 2.3.3 with RESTful API design
- **AI Integration**: Seamless integration with Kronos prediction models
- **Data Visualization**: Matplotlib with base64 encoding for web delivery
- **Production Server**: Gunicorn WSGI server with multiple workers

### Frontend
- **Framework**: Bootstrap 5 for responsive design
- **JavaScript**: ES6+ with modern async/await patterns
- **Styling**: Custom CSS with CSS variables for theming
- **Icons**: Font Awesome for professional iconography

### Deployment
- **Containerization**: Docker with multi-stage builds
- **Cloud Ready**: Optimized for vast.ai and other cloud platforms
- **Scalability**: Configurable worker processes and timeout settings
- **Monitoring**: Built-in health checks and logging

## 📊 Supported Assets

### Popular Stocks
- AAPL, GOOGL, MSFT, AMZN, TSLA
- NVDA, META, NFLX, ADBE, CRM
- SPY, QQQ, IWM, GLD, TLT

### Popular Cryptocurrencies
- BTC, ETH, BNB, XRP, ADA
- SOL, DOGE, DOT, AVAX, MATIC
- LINK, UNI, LTC, BCH, ETC

*Custom symbols can be added through the web interface*

## 🔍 Configuration

### Environment Variables

```bash
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=production
PYTHONPATH=/app

# Application Settings (optional)
MAX_PREDICTION_LENGTH=500
MAX_LOOKBACK_PERIOD=1000
DEFAULT_PREDICTION_LENGTH=120
DEFAULT_LOOKBACK_PERIOD=512
```

### Docker Configuration

The included `Dockerfile` uses:
- Python 3.11 slim base image
- Multi-worker Gunicorn server
- 120-second timeout for ML operations
- Automatic port exposure on 5000

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill process on port 5000
   sudo lsof -t -i tcp:5000 | xargs kill -9
   ```

2. **Docker Build Fails**
   ```bash
   # Clean Docker cache
   docker system prune -a
   ```

3. **AI Models Not Loading**
   - Ensure the original Kronos AI codebase is available
   - Check the import paths in `app.py`
   - Verify all dependencies are installed

4. **Memory Issues**
   - Reduce the number of Gunicorn workers
   - Increase Docker memory allocation
   - Use smaller prediction/lookback periods

### Logs and Debugging

```bash
# View container logs
docker logs kronos-app

# Check container status
docker ps -a

# Enter container for debugging
docker exec -it kronos-app bash
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Support

For support, please:
1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed information

## 🔮 Future Enhancements

- [ ] Real-time market data streaming
- [ ] Portfolio management features
- [ ] Multi-timeframe analysis
- [ ] Alert system for predictions
- [ ] Export functionality for reports
- [ ] User authentication and sessions
- [ ] Historical prediction tracking
- [ ] Advanced charting with TradingView

---

**Built with ❤️ by the Kronos AI Team**

*Empowering traders with artificial intelligence since 2024*