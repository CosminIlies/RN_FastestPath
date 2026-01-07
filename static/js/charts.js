// Charts visualization using Chart.js
class ChartsManager {
    constructor() {
        this.charts = {};
        this.initializeCharts();
    }

    initializeCharts() {
        this.createChart('population-chart', 'Population vs Agglomeration', 'rgba(54, 162, 235, 0.6)');
        this.createChart('gdp-chart', 'GDP per Capita vs Agglomeration', 'rgba(75, 192, 192, 0.6)');
        this.createChart('education-chart', 'Education vs Agglomeration', 'rgba(255, 99, 132, 0.6)');
        this.createChart('infrastructure-chart', 'Infrastructure vs Agglomeration', 'rgba(153, 102, 255, 0.6)');
    }

    createChart(canvasId, title, color) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        this.charts[canvasId] = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: title,
                    data: [],
                    backgroundColor: color,
                    borderColor: color.replace('0.6', '1'),
                    borderWidth: 1,
                    pointRadius: 5,
                    pointHoverRadius: 7
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: false
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const point = context.parsed;
                                const dataIndex = context.dataIndex;
                                const dataset = context.dataset.data[dataIndex];
                                
                                return [
                                    `City: ${dataset.cityName || 'Unknown'}`,
                                    `${context.dataset.label.split(' vs ')[0]}: ${dataset.yLabel || point.y}`,
                                    `Agglomeration: ${point.x.toFixed(3)}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Agglomeration'
                        },
                        min: 0,
                        max: 1
                    },
                    y: {
                        title: {
                            display: true,
                            text: this.getYAxisLabel(canvasId)
                        },
                        beginAtZero: false
                    }
                },
                onHover: (event, activeElements) => {
                    event.native.target.style.cursor = activeElements.length > 0 ? 'pointer' : 'default';
                },
                onClick: (event, activeElements) => {
                    if (activeElements.length > 0) {
                        const dataIndex = activeElements[0].index;
                        const cityIndex = this.charts[canvasId].data.datasets[0].data[dataIndex].cityIndex;
                        
                        if (window.app && cityIndex !== undefined) {
                            window.app.selectCity(cityIndex);
                            if (window.graphVisualization) {
                                window.graphVisualization.updateNodeSelection();
                            }
                        }
                    }
                }
            }
        });
    }

    getYAxisLabel(canvasId) {
        const labels = {
            'population-chart': 'Population',
            'gdp-chart': 'GDP per Capita ($)',
            'education-chart': 'Education Score',
            'infrastructure-chart': 'Infrastructure Score'
        };
        return labels[canvasId] || 'Value';
    }

    updateCharts(data) {
        if (!data || !data.nodes) {
            this.clearAllCharts();
            return;
        }

        const nodes = data.nodes;
        
        // Update each chart
        this.updateChart('population-chart', nodes, (node) => node.population, (node) => node.population?.toLocaleString());
        this.updateChart('gdp-chart', nodes, (node) => node.gdp_per_capita, (node) => `$${node.gdp_per_capita?.toLocaleString()}`);
        this.updateChart('education-chart', nodes, (node) => node.education_score, (node) => `${(node.education_score * 100).toFixed(0)}%`);
        this.updateChart('infrastructure-chart', nodes, (node) => node.infrastructure_score, (node) => `${(node.infrastructure_score * 100).toFixed(0)}%`);
    }

    updateChart(chartId, nodes, valueExtractor, labelFormatter) {
        const chart = this.charts[chartId];
        if (!chart) return;

        const chartData = nodes.map((node, index) => {
            // Use predicted agglomeration if available, otherwise use actual
            const agglomeration = node.predicted_agglomeration !== undefined ? 
                                node.predicted_agglomeration : node.agglomeration;
            
            const yValue = valueExtractor(node);
            
            return {
                x: agglomeration,
                y: yValue,
                cityName: node.city_name || `City ${index}`,
                cityIndex: index,
                yLabel: labelFormatter(node)
            };
        }).filter(point => point.y !== undefined && point.y !== null);

        chart.data.datasets[0].data = chartData;
        
        // Update y-axis range
        if (chartData.length > 0) {
            const yValues = chartData.map(point => point.y);
            const minY = Math.min(...yValues);
            const maxY = Math.max(...yValues);
            const padding = (maxY - minY) * 0.1;
            
            chart.options.scales.y.min = Math.max(0, minY - padding);
            chart.options.scales.y.max = maxY + padding;
        }

        chart.update('none'); // No animation for better performance
    }

    clearAllCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.data.datasets[0].data = [];
                chart.update('none');
            }
        });
    }

    highlightCities(cityIndices) {
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.data.datasets[0].data) {
                chart.data.datasets[0].data.forEach((point, index) => {
                    const isSelected = cityIndices.includes(point.cityIndex);
                    // You could modify the point appearance here for selected cities
                });
                chart.update('none');
            }
        });
    }

    destroy() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
        this.charts = {};
    }
}

// Initialize charts manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chartsManager = new ChartsManager();
});