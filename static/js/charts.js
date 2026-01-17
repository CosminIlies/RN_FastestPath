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
        this.createComparisonChart();
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
        
        // Update comparison chart if predictions are available
        if (data.predictions && this.hasPredictions(nodes)) {
            this.updateComparisonChart(nodes, data.predictions);
        }
    }

    hasPredictions(nodes) {
        return nodes.some(node => node.predicted_agglomeration !== undefined);
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

    createComparisonChart() {
        const ctx = document.getElementById('comparison-chart');
        if (!ctx) return;

        this.charts['comparison-chart'] = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Perfect Prediction',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    showLine: true,
                    tension: 0
                }, {
                    label: 'Actual vs Predicted',
                    data: [],
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Actual vs Predicted Agglomeration Values'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                if (context.datasetIndex === 0) return 'Perfect Prediction Line';
                                const point = context.parsed;
                                const cityData = context.dataset.data[context.dataIndex];
                                const difference = Math.abs(point.y - point.x);
                                return [
                                    `City: ${cityData.cityName || 'Unknown'}`,
                                    `Actual: ${point.x.toFixed(3)}`,
                                    `Predicted: ${point.y.toFixed(3)}`,
                                    `Difference: ${difference.toFixed(3)}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Actual Agglomeration'
                        },
                        min: 0.1,
                        max: 1.0
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Predicted Agglomeration'
                        },
                        min: 0.1,
                        max: 1.0
                    }
                },
                interaction: {
                    intersect: false
                },
                animation: {
                    duration: 500
                }
            }
        });
    }

    updateComparisonChart(cities, predictions) {
        const chart = this.charts['comparison-chart'];
        if (!chart || !predictions) return;

        // Create perfect prediction line (diagonal)
        const perfectLine = [
            { x: 0.1, y: 0.1 },
            { x: 1.0, y: 1.0 }
        ];

        // Create actual vs predicted points
        const comparisonData = [];
        cities.forEach((city, index) => {
            if (index < predictions.predicted_values.length) {
                comparisonData.push({
                    x: city.agglomeration,
                    y: predictions.predicted_values[index],
                    cityName: city.city_name,
                    cityIndex: index
                });
            }
        });

        chart.data.datasets[0].data = perfectLine;
        chart.data.datasets[1].data = comparisonData;
        chart.update();

        // Update metrics
        this.updateMetrics(predictions);
    }

    updateMetrics(predictions) {
        if (!predictions) return;

        document.getElementById('r2-score').textContent = 
            predictions.r2_score ? predictions.r2_score.toFixed(4) : '--';
        document.getElementById('mae-loss').textContent = 
            predictions.mae_loss ? predictions.mae_loss.toFixed(4) : '--';
        document.getElementById('num-predictions').textContent = 
            predictions.num_predictions || '--';

        // Add color coding based on R² score
        const r2Element = document.getElementById('r2-score');
        if (predictions.r2_score !== undefined) {
            r2Element.className = 'metric-value';
            if (predictions.r2_score > 0.7) {
                r2Element.classList.add('positive');
            } else if (predictions.r2_score < 0.3) {
                r2Element.classList.add('negative');
            }
        }
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