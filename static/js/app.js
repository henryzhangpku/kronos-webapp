/**
 * Kronos AI Web Application
 * Main JavaScript file for handling user interactions and API calls
 */

class KronosApp {
    constructor() {
        this.currentSymbol = '';
        this.currentAssetType = 'stock';
        this.predictionLength = 120;
        this.lookback = 512;
        this.isLoading = false;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.populateSymbols();
        this.updateUI();
    }

    setupEventListeners() {
        // Asset type change
        document.querySelectorAll('input[name="assetType"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentAssetType = e.target.value;
                this.populateSymbols();
                this.updateUI();
            });
        });

        // Symbol selection
        document.getElementById('symbolSelect').addEventListener('change', (e) => {
            this.currentSymbol = e.target.value;
            this.updateUI();
        });

        // Add custom symbol
        document.getElementById('addSymbolBtn').addEventListener('click', () => {
            this.addCustomSymbol();
        });

        // Enter key for custom symbol
        document.getElementById('customSymbol').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.addCustomSymbol();
            }
        });

        // Prediction length
        document.getElementById('predLength').addEventListener('change', (e) => {
            this.predictionLength = parseInt(e.target.value);
        });

        // Lookback period
        document.getElementById('lookback').addEventListener('change', (e) => {
            this.lookback = parseInt(e.target.value);
        });

        // Generate buttons
        document.getElementById('generatePredictionBtn').addEventListener('click', () => {
            this.generatePrediction();
        });

        document.getElementById('generateBacktestBtn').addEventListener('click', () => {
            this.generateBacktest();
        });

        // Refresh buttons
        document.getElementById('refreshPredictionBtn').addEventListener('click', () => {
            this.generatePrediction();
        });

        document.getElementById('refreshBacktestBtn').addEventListener('click', () => {
            this.generateBacktest();
        });
    }

    populateSymbols() {
        const symbolSelect = document.getElementById('symbolSelect');
        symbolSelect.innerHTML = '<option value="">Select a symbol...</option>';
        
        const symbols = this.currentAssetType === 'stock' ? 
            window.appData.popularStocks : window.appData.popularCrypto;
        
        symbols.forEach(symbol => {
            const option = document.createElement('option');
            option.value = symbol;
            option.textContent = symbol;
            symbolSelect.appendChild(option);
        });

        // Reset current symbol
        this.currentSymbol = '';
    }

    updateUI() {
        const hasSymbol = this.currentSymbol !== '';
        const kronosAvailable = window.appData.kronosAvailable;
        
        // Enable/disable buttons
        document.getElementById('generatePredictionBtn').disabled = !hasSymbol || !kronosAvailable || this.isLoading;
        document.getElementById('generateBacktestBtn').disabled = !hasSymbol || !kronosAvailable || this.isLoading;
        
        // Update button text based on loading state
        if (this.isLoading) {
            document.getElementById('generatePredictionBtn').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            document.getElementById('generateBacktestBtn').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        } else {
            document.getElementById('generatePredictionBtn').innerHTML = '<i class="fas fa-crystal-ball"></i> Generate Prediction';
            document.getElementById('generateBacktestBtn').innerHTML = '<i class="fas fa-history"></i> Generate Backtest';
        }
    }

    addCustomSymbol() {
        const customSymbolInput = document.getElementById('customSymbol');
        const symbol = customSymbolInput.value.trim().toUpperCase();
        
        if (!symbol) {
            this.showAlert('Please enter a valid symbol', 'danger');
            return;
        }

        // Add to the select dropdown
        const symbolSelect = document.getElementById('symbolSelect');
        const existingOption = Array.from(symbolSelect.options).find(option => option.value === symbol);
        
        if (existingOption) {
            symbolSelect.value = symbol;
            this.currentSymbol = symbol;
            customSymbolInput.value = '';
            this.showAlert(`Symbol ${symbol} already exists and has been selected`, 'info');
        } else {
            // Call API to add symbol
            this.callAPI('/api/add_symbol', {
                symbol: symbol,
                asset_type: this.currentAssetType
            }).then(response => {
                if (response.success) {
                    const option = document.createElement('option');
                    option.value = symbol;
                    option.textContent = symbol;
                    symbolSelect.appendChild(option);
                    
                    symbolSelect.value = symbol;
                    this.currentSymbol = symbol;
                    customSymbolInput.value = '';
                    
                    this.showAlert(`Symbol ${symbol} added successfully`, 'success');
                    this.updateUI();
                } else {
                    this.showAlert(response.error || 'Failed to add symbol', 'danger');
                }
            }).catch(error => {
                this.showAlert('Error adding symbol: ' + error.message, 'danger');
            });
        }
        
        this.updateUI();
    }

    async generatePrediction() {
        if (!this.currentSymbol) {
            this.showAlert('Please select a symbol first', 'warning');
            return;
        }

        this.setLoading(true, 'Generating prediction...', 'Kronos AI is analyzing market data');
        
        try {
            const response = await this.callAPI('/api/predict', {
                symbol: this.currentSymbol,
                asset_type: this.currentAssetType,
                pred_length: this.predictionLength
            });

            if (response.success) {
                this.displayPrediction(response.plot, response.stats);
                this.showAlert('Prediction generated successfully', 'success');
                
                // Switch to prediction tab
                document.getElementById('prediction-tab').click();
            } else {
                this.showAlert(response.error || 'Failed to generate prediction', 'danger');
            }
        } catch (error) {
            this.showAlert('Error generating prediction: ' + error.message, 'danger');
        } finally {
            this.setLoading(false);
        }
    }

    async generateBacktest() {
        if (!this.currentSymbol) {
            this.showAlert('Please select a symbol first', 'warning');
            return;
        }

        this.setLoading(true, 'Running backtest...', 'Analyzing historical performance');
        
        try {
            const response = await this.callAPI('/api/backtest', {
                symbol: this.currentSymbol,
                asset_type: this.currentAssetType,
                lookback: this.lookback
            });

            if (response.success) {
                this.displayBacktest(response.plot, response.stats);
                this.showAlert('Backtest completed successfully', 'success');
                
                // Switch to backtest tab
                document.getElementById('backtest-tab').click();
            } else {
                this.showAlert(response.error || 'Failed to generate backtest', 'danger');
            }
        } catch (error) {
            this.showAlert('Error generating backtest: ' + error.message, 'danger');
        } finally {
            this.setLoading(false);
        }
    }

    displayPrediction(plotData, stats) {
        const content = document.getElementById('predictionContent');
        
        let html = `
            <div class="chart-container">
                <img src="data:image/png;base64,${plotData}" alt="Prediction Chart" class="img-fluid">
            </div>
        `;
        
        if (stats) {
            html += `
                <div class="row mt-3">
                    <div class="col-md-6">
                        <div class="card bg-dark">
                            <div class="card-body text-center">
                                <h6 class="card-title">Current Price</h6>
                                <h4 class="text-primary">${stats.current_price || 'N/A'}</h4>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card bg-dark">
                            <div class="card-body text-center">
                                <h6 class="card-title">Predicted Change</h6>
                                <h4 class="${stats.predicted_change >= 0 ? 'text-success' : 'text-danger'}">
                                    ${stats.predicted_change >= 0 ? '+' : ''}${(stats.predicted_change * 100).toFixed(2)}%
                                </h4>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        content.innerHTML = html;
    }

    displayBacktest(plotData, stats) {
        const content = document.getElementById('backtestContent');
        
        let html = `
            <div class="chart-container">
                <img src="data:image/png;base64,${plotData}" alt="Backtest Chart" class="img-fluid">
            </div>
        `;
        
        if (stats) {
            html += `
                <div class="row mt-3">
                    <div class="col-md-4">
                        <div class="card bg-dark">
                            <div class="card-body text-center">
                                <h6 class="card-title">Accuracy</h6>
                                <h4 class="text-info">${(stats.accuracy * 100).toFixed(1)}%</h4>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-dark">
                            <div class="card-body text-center">
                                <h6 class="card-title">Total Return</h6>
                                <h4 class="${stats.total_return >= 0 ? 'text-success' : 'text-danger'}">
                                    ${stats.total_return >= 0 ? '+' : ''}${(stats.total_return * 100).toFixed(2)}%
                                </h4>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-dark">
                            <div class="card-body text-center">
                                <h6 class="card-title">Sharpe Ratio</h6>
                                <h4 class="text-warning">${stats.sharpe_ratio ? stats.sharpe_ratio.toFixed(2) : 'N/A'}</h4>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        content.innerHTML = html;
    }

    setLoading(loading, text = 'Processing...', subtext = 'Please wait') {
        this.isLoading = loading;
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        
        if (loading) {
            document.getElementById('loadingText').textContent = text;
            document.getElementById('loadingSubtext').textContent = subtext;
            modal.show();
        } else {
            modal.hide();
        }
        
        this.updateUI();
    }

    async callAPI(endpoint, data) {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }

    showAlert(message, type = 'info') {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.kronosApp = new KronosApp();
});

// Utility functions
function formatNumber(num, decimals = 2) {
    if (typeof num !== 'number') return 'N/A';
    return num.toFixed(decimals);
}

function formatPercent(num, decimals = 2) {
    if (typeof num !== 'number') return 'N/A';
    return (num * 100).toFixed(decimals) + '%';
}

function formatCurrency(num, symbol = '$') {
    if (typeof num !== 'number') return 'N/A';
    return symbol + num.toFixed(2);
}