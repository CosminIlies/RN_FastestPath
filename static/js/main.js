// Main JavaScript file for Urban Agglomeration GNN Analysis
class App {
    constructor() {
        this.selectedCityId = null;
        this.originalCityData = null;
        this.currentData = null;
        this.dataChecksum = null;
        this.statusUpdateInterval = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupRangeSliders();
        this.startStatusUpdates();
        this.updateStatus();
    }

    setupEventListeners() {
        // Control buttons
        document.getElementById('btn-predict').addEventListener('click', () => this.controlAction('predict'));
        
        // Parameter controls
        document.getElementById('btn-apply-params').addEventListener('click', () => this.applyParameters());
        document.getElementById('btn-reset-params').addEventListener('click', () => this.resetParameters());
    }

    setupRangeSliders() {
        // Range sliders for score parameters
        const scoreSliders = ['education-range', 'infrastructure-range', 'location-range'];
        
        scoreSliders.forEach(sliderId => {
            const slider = document.getElementById(sliderId);
            const valueDisplay = document.getElementById(sliderId.replace('-range', '-value'));
            
            slider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                valueDisplay.textContent = value.toFixed(2);
            });
        });
    }

    async controlAction(action) {
        try {
            this.showLoading();
            console.log(`Executing action: ${action}`);
            
            const response = await fetch(`/api/control/${action}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (response.ok) {
                const result = await response.json();
                this.updateStatusDisplay(result);
                
                console.log(`Action '${action}' result:`, result);
                
                // Force reload graph data after predict action
                if (action === 'predict') {
                    setTimeout(() => {
                        this.loadGraphData();
                        this.showMessage('Predictions updated successfully', 'success');
                    }, 500);
                }
                
                this.showMessage(`Action '${action}' completed successfully`, 'success');
            } else {
                const error = await response.json();
                console.error(`Action '${action}' failed:`, error);
                this.showMessage(`Error: ${error.error}`, 'error');
            }
        } catch (error) {
            console.error(`Network error for action '${action}':`, error);
            this.showMessage(`Network error: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            if (response.ok) {
                const status = await response.json();
                this.updateStatusDisplay(status);
                
                console.log('Current status:', status);
                
                // Load graph data if available and not already loaded
                if (status.cities_loaded && status.edges_loaded && !this.currentData) {
                    console.log('Loading initial graph data...');
                    this.loadGraphData();
                }
                
                // If we have predictions available, make sure to load them
                if (status.predictions_available) {
                    console.log('Predictions available, refreshing graph data...');
                    this.loadGraphData();
                }
            }
        } catch (error) {
            console.error('Failed to update status:', error);
        }
    }

    updateStatusDisplay(status) {
        const statusIndicator = document.getElementById('status-indicator');
        const systemStatus = document.getElementById('system-status');
        
        // Update status indicator (only if element exists)
        if (statusIndicator) {
            statusIndicator.textContent = status.current_state.replace('_', ' ').toUpperCase();
            statusIndicator.className = `badge status-${status.current_state.replace('_', '-')}`;
        }
        
        // Update system status card (only if element exists)
        if (systemStatus) {
            const stateEmoji = {
                'initialization': '🔄',
                'generating_data': '⚙️',
                'saving_data': '💾',
                'creating_model': '🧠',
                'train_model': '🏋️',
                'prediction': '🔮',
                'idle': '✅',
                'complete': '🎉',
                'error': '❌'
            };
            
            systemStatus.innerHTML = `
                <div class="d-flex align-items-center">
                    <span class="me-2" style="font-size: 1.2em;">${stateEmoji[status.current_state] || '⚪'}</span>
                    <div>
                        <strong>${status.current_state.replace('_', ' ')}</strong>
                        ${status.error ? `<br><small class="text-danger">${status.error}</small>` : ''}
                    </div>
                </div>
            `;
            
            systemStatus.className = `alert alert-${status.error ? 'danger' : 'info'}`;
        }
        
        // Update model metrics (only if elements exist)
        if (status.predictions_summary) {
            const mseValue = document.getElementById('mse-value');
            const maeValue = document.getElementById('mae-value');
            const predictionsCount = document.getElementById('predictions-count');
            
            if (mseValue) {
                mseValue.textContent = status.predictions_summary.mse_loss ? 
                    status.predictions_summary.mse_loss.toFixed(4) : '-';
            }
            if (maeValue) {
                maeValue.textContent = status.predictions_summary.mae_loss ? 
                    status.predictions_summary.mae_loss.toFixed(4) : '-';
            }
            if (predictionsCount) {
                predictionsCount.textContent = status.predictions_summary.num_predictions || '-';
            }
        }
        
        // Update model status (only if element exists)
        const modelStatus = document.getElementById('model-status');
        if (modelStatus) {
            modelStatus.textContent = status.model_created ? 'Trained' : 'Not Trained';
            modelStatus.className = `text-${status.model_created ? 'success' : 'warning'}`;
        }
    }

    async loadGraphData() {
        try {
            const response = await fetch('/api/model');
            if (response.ok) {
                const data = await response.json();
                
                // Create a simple checksum to detect data changes
                const newChecksum = JSON.stringify({
                    nodeCount: data.nodes ? data.nodes.length : 0,
                    edgeCount: data.edges ? data.edges.length : 0,
                    firstNodeData: data.nodes && data.nodes[0] ? 
                        JSON.stringify(data.nodes[0]) : null
                });
                
                // Only update if data has actually changed
                if (this.dataChecksum !== newChecksum) {
                    this.currentData = data;
                    this.dataChecksum = newChecksum;
                    
                    // Update graph visualization
                    if (window.graphVisualization) {
                        window.graphVisualization.updateGraph(data);
                    }
                    
                    // Update charts
                    if (window.chartsManager) {
                        window.chartsManager.updateCharts(data);
                    }
                    
                    // Update city editor if a city is selected
                    if (this.selectedCityId !== null) {
                        this.updateCityEditor();
                    }
                    this.updateSelectionInfo();
                }
            }
        } catch (error) {
            console.error('Failed to load graph data:', error);
        }
    }

    async applyParameters() {
        if (!this.currentData || !this.currentData.metadata.model_trained) {
            this.showMessage('Model must be trained before applying parameter changes', 'warning');
            return;
        }

        if (this.selectedCityId === null) {
            this.showMessage('Please select a city to modify its parameters', 'warning');
            return;
        }

        // Use the helper method
        await this.applyParametersFromData(null, false);
    }

    selectCity(cityIndex) {
        // Only allow selecting one city at a time for individual editing
        if (this.selectedCityId === cityIndex) {
            this.selectedCityId = null;
            this.originalCityData = null;
        } else {
            this.selectedCityId = cityIndex;
            if (this.currentData && this.currentData.nodes[cityIndex]) {
                this.originalCityData = {...this.currentData.nodes[cityIndex]};
            }
        }
        
        this.updateCityEditor();
        this.updateSelectionInfo();
    }

    updateSelectionInfo() {
        const selectionInfo = document.getElementById('selection-info');
        
        if (this.selectedCityId === null) {
            selectionInfo.innerHTML = 'No cities selected. Click on a city in the graph to view and edit its parameters.';
            selectionInfo.className = 'alert alert-light';
        } else {
            const cityName = this.currentData && this.currentData.nodes[this.selectedCityId] ?
                (this.currentData.nodes[this.selectedCityId].city_name || `City ${this.selectedCityId}`) :
                `City ${this.selectedCityId}`;
            
            selectionInfo.innerHTML = `
                <strong>Selected:</strong> ${cityName}<br>
                <small>Edit parameters below and click Apply to see predicted changes.</small>
            `;
            selectionInfo.className = 'alert alert-primary';
        }
    }

    updateCityEditor() {
        const noSelection = document.getElementById('no-city-selected');
        const cityParameters = document.getElementById('city-parameters');
        const cityNameEl = document.getElementById('selected-city-name');
        
        if (this.selectedCityId === null || !this.currentData) {
            noSelection.classList.remove('d-none');
            cityParameters.classList.add('d-none');
            return;
        }
        
        const cityData = this.currentData.nodes[this.selectedCityId];
        if (!cityData) {
            noSelection.classList.remove('d-none');
            cityParameters.classList.add('d-none');
            return;
        }
        
        // Show the parameters editor
        noSelection.classList.add('d-none');
        cityParameters.classList.remove('d-none');
        
        // Update city name
        cityNameEl.textContent = cityData.city_name || `City ${this.selectedCityId}`;
        
        // Populate form fields with current city data
        document.getElementById('population-input').value = Math.round(cityData.population || 0);
        document.getElementById('gdp-input').value = Math.round(cityData.gdp_per_capita || 0);
        document.getElementById('education-range').value = (cityData.education_score || 0.5).toFixed(2);
        document.getElementById('infrastructure-range').value = (cityData.infrastructure_score || 0.5).toFixed(2);
        document.getElementById('location-range').value = (cityData.location_score || 0.5).toFixed(2);
        
        // Update display values
        document.getElementById('education-value').textContent = (cityData.education_score || 0.5).toFixed(2);
        document.getElementById('infrastructure-value').textContent = (cityData.infrastructure_score || 0.5).toFixed(2);
        document.getElementById('location-value').textContent = (cityData.location_score || 0.5).toFixed(2);
    }

    resetParameters() {
        if (this.selectedCityId === null) {
            this.showMessage('No city selected', 'warning');
            return;
        }
        
        if (!this.originalCityData) {
            this.showMessage('Original data unavailable - try selecting the city again', 'warning');
            return;
        }
        
        // Send reset request to backend
        this.applyParametersFromData(this.originalCityData, true);
    }

    async applyParametersFromData(cityData, isReset = false) {
        if (this.selectedCityId === null) {
            return;
        }

        try {
            this.showLoading();
            
            const parameters = isReset ? {
                reset_to_original: true,
                affected_cities: [this.selectedCityId]
            } : {
                population: parseInt(document.getElementById('population-input').value),
                gdp_per_capita: parseFloat(document.getElementById('gdp-input').value),
                education_score: parseFloat(document.getElementById('education-range').value),
                infrastructure_score: parseFloat(document.getElementById('infrastructure-range').value),
                location_score: parseFloat(document.getElementById('location-range').value),
                affected_cities: [this.selectedCityId],
                individual_values: true
            };

            const response = await fetch('/api/model', {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(parameters)
            });

            if (response.ok) {
                const result = await response.json();
                this.currentData = result;
                
                // Update visualizations with new predictions
                if (window.graphVisualization) {
                    window.graphVisualization.updateGraph(result);
                }
                
                if (window.chartsManager) {
                    window.chartsManager.updateCharts(result);
                }
                
                // Update the city info display
                this.updateCityInfoDisplay();
                
                this.showMessage(isReset ? 'Parameters reset to original values' : 'Parameter changes applied successfully', 'success');
            } else {
                const error = await response.json();
                this.showMessage(`Error: ${error.error}`, 'error');
            }
        } catch (error) {
            this.showMessage(`Network error: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    updateCityInfoDisplay() {
        // Update city info if a city is still selected
        if (this.selectedCityId !== null && this.currentData && this.currentData.nodes[this.selectedCityId]) {
            this.updateCityEditor();
        }
    }

    startStatusUpdates() {
        // Update status every 5 seconds
        this.statusUpdateInterval = setInterval(() => {
            this.updateStatus();
        }, 5000);
    }

    showLoading() {
        document.body.classList.add('loading');
    }

    hideLoading() {
        document.body.classList.remove('loading');
    }

    showMessage(message, type = 'info') {
        // Create or update message toast
        const toastContainer = document.getElementById('toast-container') || this.createToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // Remove toast element after it hides
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '1060';
        document.body.appendChild(container);
        return container;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});