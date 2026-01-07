// Dashboard-specific functionality
document.addEventListener('DOMContentLoaded', () => {
    // Initialize dashboard components
    initializeDashboard();
});

function initializeDashboard() {
    // Add any dashboard-specific initialization here
    
    // Set up keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // Set up auto-refresh toggle
    setupAutoRefresh();
    
    // Initialize tooltips
    initializeTooltips();
}

function handleKeyboardShortcuts(event) {
    // Ctrl+P: Predict
    if (event.ctrlKey && event.key === 'p') {
        event.preventDefault();
        document.getElementById('btn-predict').click();
    }
    
    // Escape: Clear selection
    if (event.key === 'Escape') {
        if (window.app) {
            window.app.selectedCityId = null;
            window.app.originalCityData = null;
            window.app.updateCityEditor();
            window.app.updateSelectionInfo();
            if (window.graphVisualization) {
                window.graphVisualization.updateNodeSelection();
            }
        }
    }
}

function setupAutoRefresh() {
    // This could be extended to add auto-refresh controls
    console.log('Auto-refresh setup completed');
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Utility functions
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function getStatusColor(state) {
    const colors = {
        'initialization': 'secondary',
        'generating_data': 'warning',
        'saving_data': 'info',
        'creating_model': 'primary',
        'train_model': 'warning',
        'prediction': 'info',
        'idle': 'success',
        'complete': 'success',
        'error': 'danger'
    };
    return colors[state] || 'secondary';
}

// Export utility functions to global scope
window.dashboardUtils = {
    formatNumber,
    getStatusColor
};