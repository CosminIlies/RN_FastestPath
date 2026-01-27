// Graph visualization using D3.js
class GraphVisualization {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = d3.select(`#${containerId}`);
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        this.nodeElements = null;
        this.linkElements = null;
        this.labelElements = null;
        this.nodePositions = new Map(); // Store node positions
        
        this.width = 0;
        this.height = 0;
        
        this.initializeVisualization();
    }

    initializeVisualization() {
        // Clear container
        this.container.selectAll('*').remove();
        
        // Get container dimensions
        const containerRect = document.getElementById(this.containerId).getBoundingClientRect();
        this.width = containerRect.width - 20;
        this.height = containerRect.height - 20;
        
        // Create SVG
        this.svg = this.container.append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .style('background', '#f8f9fa')
            .style('border-radius', '8px');
        
        // Add zoom and pan
        const zoom = d3.zoom()
            .scaleExtent([0.5, 3])
            .on('zoom', (event) => {
                this.svg.select('.graph-group').attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        // Create group for graph elements
        this.graphGroup = this.svg.append('g')
            .attr('class', 'graph-group');
        
        // Add empty state message
        this.showEmptyState();
    }

    showEmptyState() {
        this.graphGroup.selectAll('*').remove();
        
        const emptyGroup = this.graphGroup.append('g')
            .attr('class', 'empty-state')
            .attr('transform', `translate(${this.width/2}, ${this.height/2})`);
        
        // City icon
        emptyGroup.append('text')
            .text('🏙️')
            .attr('text-anchor', 'middle')
            .attr('y', -20)
            .style('font-size', '48px');
        
        // Message
        emptyGroup.append('text')
            .text('Generate data to view the city network graph')
            .attr('text-anchor', 'middle')
            .attr('y', 20)
            .style('font-size', '14px')
            .style('fill', '#6c757d');
    }

    updateGraph(data) {
        if (!data || !data.nodes || !data.edges) {
            this.showEmptyState();
            return;
        }

        // Clear empty state
        this.graphGroup.selectAll('.empty-state').remove();

        // Store current positions before updating
        if (this.nodes.length > 0) {
            this.nodes.forEach(node => {
                this.nodePositions.set(node.id, {x: node.x, y: node.y});
            });
        }

        // Debug: Check if nodes have predicted agglomeration values
        const nodesWithPredictions = data.nodes.filter(node => node.predicted_agglomeration !== undefined);
        console.log(`Graph update: ${nodesWithPredictions.length}/${data.nodes.length} nodes have predicted agglomeration values`);

        // Prepare data
        this.nodes = data.nodes.map((node, i) => {
            const existingPos = this.nodePositions.get(i);
            return {
                id: i,
                ...node,
                x: existingPos ? existingPos.x : this.width * 0.2 + Math.random() * (this.width * 0.6),
                y: existingPos ? existingPos.y : this.height * 0.2 + Math.random() * (this.height * 0.6)
            };
        });

        this.links = data.edges.map(edge => ({
            source: edge[0],
            target: edge[1],
            distance: edge[2] || 50
        }));

        // Only re-render if this is the first time or structure changed significantly
        const shouldRerender = !this.nodeElements || 
                              this.nodeElements.size() !== this.nodes.length;
        
        if (shouldRerender) {
            this.renderGraph();
            this.setupSimulation();
        } else {
            // Just update node properties without disrupting the simulation
            this.updateNodePropertiesGently();
        }
    }

    updateNodePropertiesGently() {
        if (!this.nodeElements) return;
        
        // Update node colors and sizes based on new data without disrupting simulation
        this.nodeElements
            .each((d, i) => {
                // Update the data properties in place without changing object reference
                const newData = this.nodes[i];
                if (newData) {
                    Object.assign(d, {
                        city_name: newData.city_name,
                        population: newData.population,
                        gdp_per_capita: newData.gdp_per_capita,
                        education_score: newData.education_score,
                        infrastructure_score: newData.infrastructure_score,
                        location_score: newData.location_score,
                        agglomeration: newData.agglomeration,
                        predicted_agglomeration: newData.predicted_agglomeration
                    });
                }
            })
            .style('fill', d => this.getNodeColor(d))
            .attr('r', d => Math.max(5, Math.min(20, Math.sqrt(d.population / 1000))));
        
        // Update labels in case city names changed
        this.labelElements
            .text(d => d.city_name || `City ${d.id}`);
        
        console.log('Node properties updated gently without disrupting simulation');
    }

    updateNodeProperties() {
        if (!this.nodeElements) return;
        
        // Update the data bound to existing elements with new data
        this.nodeElements = this.nodeElements.data(this.nodes, d => d.id);
        this.labelElements = this.labelElements.data(this.nodes, d => d.id);
        
        // Update node colors and sizes based on new data
        this.nodeElements
            .style('fill', d => this.getNodeColor(d))
            .attr('r', d => Math.max(5, Math.min(20, Math.sqrt(d.population / 1000))));
        
        // Update labels in case city names changed
        this.labelElements
            .text(d => d.city_name || `City ${d.id}`);
        
        // Update simulation nodes reference to new data
        if (this.simulation) {
            this.simulation.nodes(this.nodes);
            // Update the link force with new node references
            this.simulation.force('link').links(this.links);
            // Gently restart simulation with low alpha to settle any changes
            this.simulation.alpha(0.1).restart();
        }
        
        console.log('Node properties and data updated for tooltip consistency');
    }

    renderGraph() {
        // Remove existing elements
        this.graphGroup.selectAll('.link').remove();
        this.graphGroup.selectAll('.node').remove();
        this.graphGroup.selectAll('.label').remove();

        // Create links
        this.linkElements = this.graphGroup.selectAll('.link')
            .data(this.links)
            .enter().append('line')
            .attr('class', 'link')
            .style('stroke', '#999')
            .style('stroke-opacity', 0.6)
            .style('stroke-width', 1);

        // Create nodes
        this.nodeElements = this.graphGroup.selectAll('.node')
            .data(this.nodes)
            .enter().append('circle')
            .attr('class', 'node')
            .attr('r', d => Math.max(5, Math.min(20, Math.sqrt(d.population / 1000))))
            .style('fill', d => this.getNodeColor(d))
            .style('stroke', '#fff')
            .style('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('click', (event, d) => this.onNodeClick(event, d))
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip())
            .call(d3.drag()
                .on('start', (event, d) => this.dragstarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragended(event, d)));

        // Create labels
        this.labelElements = this.graphGroup.selectAll('.label')
            .data(this.nodes)
            .enter().append('text')
            .attr('class', 'label node-label')
            .text(d => d.city_name || `City ${d.id}`)
            .style('font-size', '10px')
            .style('font-weight', '500')
            .style('fill', '#333')
            .style('text-anchor', 'middle')
            .style('pointer-events', 'none');
    }

    getNodeColor(node) {
        // Use predicted agglomeration if available, otherwise use actual
        const agglomeration = node.predicted_agglomeration !== undefined ? 
                            node.predicted_agglomeration : node.agglomeration;
        
        // Color scale from blue (low) to red (high)
        const scale = d3.scaleLinear()
            .domain([0, 1])
            .range(['#4e79a7', '#f28e2c']);
        
        return scale(agglomeration);
    }

    setupSimulation() {
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(d => Math.max(5, Math.min(20, Math.sqrt(d.population / 1000))) + 5))
            .on('tick', () => this.ticked());

        // Use gentler restart to avoid jarring movement
        this.simulation.alpha(0.5).restart();
    }

    ticked() {
        if (this.linkElements) {
            this.linkElements
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
        }

        if (this.nodeElements) {
            this.nodeElements
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
        }

        if (this.labelElements) {
            this.labelElements
                .attr('x', d => d.x)
                .attr('y', d => d.y + 25);
        }
    }

    onNodeClick(event, node) {
        // Toggle selection
        if (window.app) {
            window.app.selectCity(node.id);
            
            // Update visual selection
            this.updateNodeSelection();
        }
    }

    updateNodeSelection() {
        if (this.nodeElements && window.app) {
            this.nodeElements
                .classed('selected', d => window.app.selectedCityId === d.id)
                .style('stroke', d => window.app.selectedCityId === d.id ? '#ff6b6b' : '#fff')
                .style('stroke-width', d => window.app.selectedCityId === d.id ? 4 : 2);
        }
    }

    showTooltip(event, node) {
        // Debug log to verify we have the latest data
        console.log('Tooltip data for node', node.id, ':', {
            agglomeration: node.agglomeration,
            predicted_agglomeration: node.predicted_agglomeration,
            population: node.population
        });
        
        const tooltip = d3.select('body').selectAll('.graph-tooltip')
            .data([0]);
        
        const tooltipEnter = tooltip.enter().append('div')
            .attr('class', 'graph-tooltip tooltip');
            
        const tooltipMerge = tooltipEnter.merge(tooltip);
        
        const agglomeration = node.predicted_agglomeration !== undefined ? 
                            node.predicted_agglomeration : node.agglomeration;
        
        // Check if we have a prediction
        const hasPrediction = node.predicted_agglomeration !== undefined;
        const originalAgglomeration = node.agglomeration;
        
        // Check if this node has been modified (if app has selected city and original data)
        let isModified = false;
        let originalValues = null;
        if (window.app && window.app.selectedCityId === node.id && window.app.originalCityData) {
            originalValues = window.app.originalCityData;
            isModified = (
                originalValues.population !== node.population ||
                originalValues.gdp_per_capita !== node.gdp_per_capita ||
                originalValues.education_score !== node.education_score ||
                originalValues.infrastructure_score !== node.infrastructure_score ||
                originalValues.location_score !== node.location_score
            );
        }
        
        // Build agglomeration display
        let agglomerationDisplay = '';
        if (hasPrediction) {
            const change = agglomeration - originalAgglomeration;
            const changePercent = originalAgglomeration !== 0 ? ((change / originalAgglomeration) * 100) : 0;
            
            agglomerationDisplay = `
                <div style="background: #e3f2fd; padding: 8px; border-radius: 6px; margin: 6px 0; border-left: 4px solid #1976d2;">
                    <div style="display: flex; align-items: center; margin-bottom: 4px;">
                        <span style="font-size: 16px; margin-right: 6px;">🔮</span>
                        <strong style="color: #1976d2; font-size: 13px;">PREDICTED AGGLOMERATION</strong>
                    </div>
                    <div style="font-size: 14px; font-weight: bold; color: #1976d2; margin-bottom: 4px;">
                        ${agglomeration.toFixed(3)}
                    </div>
                    <div style="font-size: 11px; color: #666; line-height: 1.3;">
                        <div>Original: <strong>${originalAgglomeration?.toFixed(3)}</strong></div>
                        ${Math.abs(change) > 0.001 ? 
                            `<div style="color: ${change > 0 ? '#28a745' : '#dc3545'}; font-weight: 500;">
                                Change: ${change > 0 ? '+' : ''}${change.toFixed(3)} (${changePercent > 0 ? '+' : ''}${changePercent.toFixed(1)}%)
                            </div>` : 
                            '<div style="color: #666;">No change from original</div>'
                        }
                    </div>
                </div>
            `;
        } else {
            agglomerationDisplay = `
                <div style="background: #f8f9fa; padding: 8px; border-radius: 6px; margin: 6px 0; border-left: 4px solid #6c757d;">
                    <div style="display: flex; align-items: center; margin-bottom: 4px;">
                        <span style="font-size: 16px; margin-right: 6px;">📊</span>
                        <strong style="color: #495057; font-size: 13px;">ORIGINAL AGGLOMERATION</strong>
                    </div>
                    <div style="font-size: 14px; font-weight: bold; color: #495057;">
                        ${agglomeration?.toFixed(3)}
                    </div>
                    <div style="font-size: 11px; color: #666;">
                        (No predictions available - run "Predict" action)
                    </div>
                </div>
            `;
        }
        
        // Build parameter comparison if modified
        let parameterComparison = '';
        if (isModified && originalValues) {
            parameterComparison = `
                <div style="background: #fff3cd; padding: 6px; border-radius: 4px; margin: 4px 0; border-left: 3px solid #ffc107;">
                    <strong style="color: #856404;">📝 Modified Parameters:</strong>
                    <div style="margin-top: 4px; font-size: 10px; line-height: 1.3;">
                        ${originalValues.population !== node.population ? 
                            `Population: <span style="color: #666;">${originalValues.population?.toLocaleString()}</span> → <strong>${node.population?.toLocaleString()}</strong><br>` : ''
                        }
                        ${originalValues.gdp_per_capita !== node.gdp_per_capita ? 
                            `GDP: <span style="color: #666;">$${originalValues.gdp_per_capita?.toLocaleString()}</span> → <strong>$${node.gdp_per_capita?.toLocaleString()}</strong><br>` : ''
                        }
                        ${originalValues.education_score !== node.education_score ? 
                            `Education: <span style="color: #666;">${(originalValues.education_score * 100).toFixed(0)}%</span> → <strong>${(node.education_score * 100).toFixed(0)}%</strong><br>` : ''
                        }
                        ${originalValues.infrastructure_score !== node.infrastructure_score ? 
                            `Infrastructure: <span style="color: #666;">${(originalValues.infrastructure_score * 100).toFixed(0)}%</span> → <strong>${(node.infrastructure_score * 100).toFixed(0)}%</strong><br>` : ''
                        }
                        ${originalValues.location_score !== node.location_score ? 
                            `Location: <span style="color: #666;">${(originalValues.location_score * 100).toFixed(0)}%</span> → <strong>${(node.location_score * 100).toFixed(0)}%</strong><br>` : ''
                        }
                    </div>
                </div>
            `;
        }
        
        // Current parameter values
        const currentParameters = `
            <div style="margin-top: 8px; font-size: 11px; color: #666;">
                <strong style="color: #333;">Current Parameters:</strong><br>
                Population: ${node.population?.toLocaleString()}<br>
                GDP per Capita: $${node.gdp_per_capita?.toLocaleString()}<br>
                Education: ${(node.education_score * 100).toFixed(0)}%<br>
                Infrastructure: ${(node.infrastructure_score * 100).toFixed(0)}%<br>
                Location Score: ${(node.location_score * 100).toFixed(0)}%
            </div>
        `;
        
        tooltipMerge
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .style('max-width', '300px')
            .html(`
                <div style="font-size: 12px; line-height: 1.4;">
                    <strong style="color: #333; font-size: 13px;">${node.city_name || `City ${node.id}`}</strong>
                    <hr style="margin: 6px 0; border: 0; border-top: 1px solid #eee;">
                    ${agglomerationDisplay}
                    ${parameterComparison}
                    ${currentParameters}
                </div>
            `);
    }

    hideTooltip() {
        d3.select('body').selectAll('.graph-tooltip')
            .style('opacity', 0)
            .remove();
    }

    dragstarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.1).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    resize() {
        const containerRect = document.getElementById(this.containerId).getBoundingClientRect();
        this.width = containerRect.width - 20;
        this.height = containerRect.height - 20;
        
        this.svg
            .attr('width', this.width)
            .attr('height', this.height);
        
        if (this.simulation) {
            this.simulation
                .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                .alpha(0.3)
                .restart();
        }
    }
}

// Initialize graph visualization when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.graphVisualization = new GraphVisualization('graph-container');
    
    // Handle window resize
    window.addEventListener('resize', () => {
        if (window.graphVisualization) {
            window.graphVisualization.resize();
        }
    });
});