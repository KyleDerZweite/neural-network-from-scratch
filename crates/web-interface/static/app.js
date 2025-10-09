const trainButton = document.getElementById('train_button');
const stopTrainingButton = document.getElementById('stop_training_button');
const predictButton = document.getElementById('predict_button');
const backendSelect = document.getElementById('backend');
const optimizerSelect = document.getElementById('optimizer');
const useGpuCheckbox = document.getElementById('use_gpu');
const gpuThresholdInput = document.getElementById('gpu_threshold');
const targetTypeSelect = document.getElementById('target_type');
const functionInputDiv = document.getElementById('function_input');
const logicalOperationInputDiv = document.getElementById('logical_operation_input');
const libraryOptionsDiv = document.getElementById('library_options');

const jobIdSpan = document.getElementById('job_id');
const statusSpan = document.getElementById('status');
const progressSpan = document.getElementById('progress');
const currentLossSpan = document.getElementById('current_loss');
const predictionOutputSpan = document.getElementById('prediction_output');
const actualResultSpan = document.getElementById('actual_result');
const warningSpan = document.getElementById('warning');
const messageSpan = document.getElementById('status_message');
const backendNameSpan = document.getElementById('backend_name');
const learningRateInput = document.getElementById('learning_rate');
const learningRateHint = document.getElementById('learning_rate_hint');
const layersInput = document.getElementById('layers');
const layersHint = document.getElementById('layers_hint');

const targetChartCanvas = document.getElementById('target_chart');
const lossChartCanvas = document.getElementById('loss_chart');
const comparisonChartCanvas = document.getElementById('comparison_chart');
const logicalPreviewTable = document.getElementById('logical_preview_table');
const logicalPreviewTableBody = document.getElementById('logical_preview_table_body');
const plot3dDiv = document.getElementById('plot_3d');
const weightsBiasesDisplay = document.getElementById('weights_biases_display');
const weightsBiasesContent = document.getElementById('weights_biases_content');

let trainingJobId = null;
let trainingInterval = null;
let current3DPlot = null;

predictButton.disabled = true;

const MAX_LOSS_POINTS = 500;

// Tab functionality
const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');

tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        const tabName = button.getAttribute('data-tab');
        
        // Remove active class from all buttons and contents
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));
        
        // Add active class to clicked button and corresponding content
        button.classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        // Trigger chart updates when switching to graphs tab
        if (tabName === 'graphs') {
            lossChart.update();
            targetChart.update();
            comparisonChart.update();
            if (current3DPlot) {
                Plotly.Plots.resize('plot_3d');
            }
        }
        
        // Trigger network visualization update when switching to network-viz tab
        if (tabName === 'network-viz' && trainingJobId) {
            drawNetworkVisualization();
        }
    });
});

// Network Visualization Canvas
const networkCanvas = document.getElementById('network_canvas');
const networkCtx = networkCanvas.getContext('2d');
let currentNetworkConfig = null;
let networkVisualizationData = null;

// Loss chart
const lossChart = new Chart(lossChartCanvas.getContext('2d'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Loss',
            data: [],
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1,
            pointRadius: 0,
        }],
    },
    options: {
        responsive: true,
        animation: false,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'MSE',
                },
            },
            x: {
                title: {
                    display: true,
                    text: 'Checkpoint',
                },
            },
        },
    },
});

// Target-only chart
const targetChart = new Chart(targetChartCanvas.getContext('2d'), {
    type: 'line',
    data: {
        datasets: [
            {
                label: 'Target',
                data: [],
                borderColor: '#81C784',
                backgroundColor: 'rgba(129, 199, 132, 0.15)',
                fill: false,
                pointRadius: 0,
                tension: 0.2,
            },
        ],
    },
    options: {
        responsive: true,
        animation: false,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: 'linear',
                title: {
                    display: true,
                    text: 'Input',
                },
            },
            y: {
                title: {
                    display: true,
                    text: 'Target',
                },
            },
        },
        plugins: {
            legend: {
                labels: {
                    color: '#e0e0e0',
                },
            },
        },
    },
});

// Overlay chart (target vs prediction)
const comparisonChart = new Chart(comparisonChartCanvas.getContext('2d'), {
    type: 'line',
    data: {
        datasets: [
            {
                label: 'Target',
                data: [],
                borderColor: 'rgba(129, 199, 132, 0.4)',
                backgroundColor: 'rgba(129, 199, 132, 0.1)',
                fill: false,
                pointRadius: 0,
                tension: 0.2,
            },
            {
                label: 'Prediction',
                data: [],
                borderColor: '#FFB74D',
                backgroundColor: 'rgba(255, 183, 77, 0.15)',
                fill: false,
                pointRadius: 0,
                borderDash: [6, 4],
                tension: 0.2,
            },
        ],
    },
    options: {
        responsive: true,
        animation: false,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: 'linear',
                title: {
                    display: true,
                    text: 'Input',
                },
            },
            y: {
                title: {
                    display: true,
                    text: 'Output',
                },
            },
        },
        plugins: {
            legend: {
                labels: {
                    color: '#e0e0e0',
                },
            },
        },
    },
});

function toggleTargetInputs() {
    const isLogicalOperation = targetTypeSelect.value === 'logical_operation';
    
    if (targetTypeSelect.value === 'function') {
        functionInputDiv.style.display = 'block';
        logicalOperationInputDiv.style.display = 'none';
        
        // Suggest lower learning rate for functions (unless user has customized it)
        const currentLR = parseFloat(learningRateInput.value);
        if (currentLR >= 0.4) {
            learningRateInput.value = '0.1';
        }
        learningRateHint.textContent = 'Recommended: 0.01-0.2 for functions';
        
        // Suggest appropriate layers for functions
        const currentLayers = layersInput.value;
        if (currentLayers === '2,2,1' || currentLayers === '2,4,1') {
            layersInput.value = '1,32,1';
        }
        layersHint.textContent = 'For sin(x): 1,32,1 or 1,64,1';
    } else {
        functionInputDiv.style.display = 'none';
        logicalOperationInputDiv.style.display = 'block';
        
        // Suggest higher learning rate for logical operations (unless user has customized it)
        const currentLR = parseFloat(learningRateInput.value);
        if (currentLR < 0.3) {
            learningRateInput.value = '0.5';
        }
        learningRateHint.textContent = 'Recommended: 0.5 for XOR/logical operations';
        
        // Suggest appropriate layers for logical operations
        const currentLayers = layersInput.value;
        if (currentLayers === '1,32,1' || currentLayers === '1,64,1') {
            layersInput.value = '2,2,1';
        }
        layersHint.textContent = 'For XOR: 2,2,1 or 2,4,1';
    }
}

function toggleBackendOptions() {
    const isLibrary = backendSelect.value === 'nn-core-library';
    libraryOptionsDiv.style.display = isLibrary ? 'block' : 'none';
}

function formatBackendLabel(value) {
    switch (value) {
        case 'nn-core-library':
            return 'nn-core-library (ndarray)';
        case 'nn-core':
        default:
            return 'nn-core (baseline CPU)';
    }
}

function resetVisualisations() {
    const chartsGrid = document.querySelector('.charts-grid');
    
    lossChart.data.labels.length = 0;
    lossChart.data.datasets[0].data.length = 0;
    lossChart.update();
    lossChartCanvas.style.display = 'block';

    targetChart.data.datasets[0].data = [];
    targetChart.update();
    targetChartCanvas.style.display = 'none';

    comparisonChart.data.datasets.forEach(dataset => {
        dataset.data = [];
    });
    comparisonChart.update();
    comparisonChartCanvas.style.display = 'none';

    logicalPreviewTableBody.innerHTML = '';
    logicalPreviewTable.style.display = 'none';
    
    plot3dDiv.style.display = 'none';
    plot3dDiv.innerHTML = '';
    current3DPlot = null;
    
    weightsBiasesDisplay.style.display = 'none';
    weightsBiasesContent.innerHTML = '';
    
    // Reset grid to show only loss chart
    if (chartsGrid) chartsGrid.style.gridTemplateRows = '300px';
}

function updateLossHistory(lossValue) {
    lossChart.data.labels.push(lossChart.data.labels.length + 1);
    lossChart.data.datasets[0].data.push(lossValue);
    if (lossChart.data.labels.length > MAX_LOSS_POINTS) {
        lossChart.data.labels.shift();
        lossChart.data.datasets[0].data.shift();
    }
    
    // Update chart options to show point markers if we have few data points
    if (lossChart.data.datasets[0].data.length <= 5) {
        lossChart.data.datasets[0].pointRadius = 4;
        lossChart.data.datasets[0].pointHoverRadius = 6;
    } else {
        lossChart.data.datasets[0].pointRadius = 0;
        lossChart.data.datasets[0].pointHoverRadius = 0;
    }
    
    lossChart.update();
}

function updateLossHistoryComplete(lossHistoryArray) {
    // Clear existing data
    lossChart.data.labels.length = 0;
    lossChart.data.datasets[0].data.length = 0;
    
    // Add all history data points
    lossHistoryArray.forEach((loss, index) => {
        lossChart.data.labels.push(index + 1);
        lossChart.data.datasets[0].data.push(loss);
    });
    
    // Trim if exceeds max points (keep evenly distributed samples)
    if (lossChart.data.labels.length > MAX_LOSS_POINTS) {
        const step = Math.ceil(lossChart.data.labels.length / MAX_LOSS_POINTS);
        const sampledLabels = [];
        const sampledData = [];
        
        for (let i = 0; i < lossChart.data.labels.length; i += step) {
            sampledLabels.push(lossChart.data.labels[i]);
            sampledData.push(lossChart.data.datasets[0].data[i]);
        }
        
        // Always include the last point
        if (sampledLabels[sampledLabels.length - 1] !== lossChart.data.labels[lossChart.data.labels.length - 1]) {
            sampledLabels.push(lossChart.data.labels[lossChart.data.labels.length - 1]);
            sampledData.push(lossChart.data.datasets[0].data[lossChart.data.datasets[0].data.length - 1]);
        }
        
        lossChart.data.labels = sampledLabels;
        lossChart.data.datasets[0].data = sampledData;
    }
    
    // Update chart options to show point markers if we have few data points
    if (lossChart.data.datasets[0].data.length <= 5) {
        lossChart.data.datasets[0].pointRadius = 4;
        lossChart.data.datasets[0].pointHoverRadius = 6;
    } else {
        lossChart.data.datasets[0].pointRadius = 0;
        lossChart.data.datasets[0].pointHoverRadius = 0;
    }
    
    lossChart.update();
}

function updatePredictionVisuals(progress) {
    const preview = progress.prediction_preview ?? [];
    const chartsGrid = document.querySelector('.charts-grid');
    
    if (!preview.length) {
        targetChartCanvas.style.display = 'none';
        comparisonChartCanvas.style.display = 'none';
        logicalPreviewTable.style.display = 'none';
        plot3dDiv.style.display = 'none';
        lossChartCanvas.style.display = 'block';
        if (chartsGrid) chartsGrid.style.gridTemplateRows = '300px';
        return;
    }

    const canPlotContinuously =
        progress.target_type === 'function' &&
        preview.every(
            item => item.input.length === 1 && item.prediction.length === 1 && item.target.length === 1,
        );
    
    const canPlot3D = 
        preview.every(
            item => item.input.length === 2 && item.prediction.length === 1 && item.target.length === 1
        );

    if (canPlotContinuously) {
        // Show 2D charts for 1D continuous functions (like sin(x))
        const sorted = [...preview].sort((a, b) => a.input[0] - b.input[0]);
        const targetPoints = sorted.map(item => ({ x: item.input[0], y: item.target[0] }));
        const predictionPoints = sorted.map(item => ({ x: item.input[0], y: item.prediction[0] }));
        targetChart.data.datasets[0].data = targetPoints;
        comparisonChart.data.datasets[0].data = targetPoints;
        comparisonChart.data.datasets[1].data = predictionPoints;
        targetChartCanvas.style.display = 'block';
        comparisonChartCanvas.style.display = 'block';
        lossChartCanvas.style.display = 'block';
        logicalPreviewTable.style.display = 'none';
        plot3dDiv.style.display = 'none';
        if (chartsGrid) chartsGrid.style.gridTemplateRows = '300px 450px';
        targetChart.update();
        comparisonChart.update();
    } else if (canPlot3D) {
        // Show 3D visualization for 2D inputs (like XOR)
        targetChartCanvas.style.display = 'none';
        comparisonChartCanvas.style.display = 'none';
        lossChartCanvas.style.display = 'block';
        logicalPreviewTable.style.display = 'table';
        plot3dDiv.style.display = 'block';
        // Adjust grid to show only loss chart (single row)
        if (chartsGrid) chartsGrid.style.gridTemplateRows = '300px';
        populateLogicalPreview(preview);
        create3DPlot(preview);
    } else {
        // Fallback: hide charts, show table only
        targetChartCanvas.style.display = 'none';
        comparisonChartCanvas.style.display = 'none';
        lossChartCanvas.style.display = 'block';
        plot3dDiv.style.display = 'none';
        if (chartsGrid) chartsGrid.style.gridTemplateRows = '300px';
        populateLogicalPreview(preview);
    }
    
    // Fetch and display weights/biases if training is complete or in progress
    if (trainingJobId && (progress.status === 'Completed' || progress.status === 'Running')) {
        fetchAndDisplayWeightsBiases();
    }
}

function populateLogicalPreview(preview) {
    logicalPreviewTableBody.innerHTML = '';
    preview.slice(0, 20).forEach(item => {
        const row = document.createElement('tr');

        const inputCell = document.createElement('td');
        inputCell.textContent = item.input.map(value => value.toFixed(2)).join(', ');
        row.appendChild(inputCell);

        const predictionCell = document.createElement('td');
        predictionCell.textContent = item.prediction.map(value => value.toFixed(4)).join(', ');
        row.appendChild(predictionCell);

        const targetCell = document.createElement('td');
        targetCell.textContent = item.target.map(value => value.toFixed(4)).join(', ');
        row.appendChild(targetCell);

        const errorCell = document.createElement('td');
        const denominator = item.prediction.length || 1;
        const error = item.prediction
            .map((value, index) => Math.abs(value - (item.target[index] ?? 0)))
            .reduce((acc, value) => acc + value, 0) / denominator;
        errorCell.textContent = Number.isFinite(error) ? error.toFixed(4) : '—';
        row.appendChild(errorCell);

        logicalPreviewTableBody.appendChild(row);
    });

    logicalPreviewTable.style.display = 'table';
}

function create3DPlot(preview) {
    // Bilinear interpolation function
    function bilinearInterpolate(x, y, points) {
        // If we have the exact point, return it
        const exact = points.find(p => Math.abs(p.input[0] - x) < 0.001 && Math.abs(p.input[1] - y) < 0.001);
        if (exact) {
            return { target: exact.target[0], prediction: exact.prediction[0] };
        }
        
        // Find the 4 nearest points (corners of the cell containing (x, y))
        const xVals = [...new Set(points.map(p => p.input[0]))].sort((a, b) => a - b);
        const yVals = [...new Set(points.map(p => p.input[1]))].sort((a, b) => a - b);
        
        // Find surrounding x values
        let x1 = xVals[0], x2 = xVals[xVals.length - 1];
        for (let i = 0; i < xVals.length - 1; i++) {
            if (xVals[i] <= x && x <= xVals[i + 1]) {
                x1 = xVals[i];
                x2 = xVals[i + 1];
                break;
            }
        }
        
        // Find surrounding y values
        let y1 = yVals[0], y2 = yVals[yVals.length - 1];
        for (let i = 0; i < yVals.length - 1; i++) {
            if (yVals[i] <= y && y <= yVals[i + 1]) {
                y1 = yVals[i];
                y2 = yVals[i + 1];
                break;
            }
        }
        
        // Get the 4 corner points
        const p11 = points.find(p => Math.abs(p.input[0] - x1) < 0.001 && Math.abs(p.input[1] - y1) < 0.001);
        const p12 = points.find(p => Math.abs(p.input[0] - x1) < 0.001 && Math.abs(p.input[1] - y2) < 0.001);
        const p21 = points.find(p => Math.abs(p.input[0] - x2) < 0.001 && Math.abs(p.input[1] - y1) < 0.001);
        const p22 = points.find(p => Math.abs(p.input[0] - x2) < 0.001 && Math.abs(p.input[1] - y2) < 0.001);
        
        // If any corner is missing, fall back to nearest neighbor
        if (!p11 || !p12 || !p21 || !p22) {
            const nearest = points.reduce((prev, curr) => {
                const prevDist = Math.sqrt(Math.pow(prev.input[0] - x, 2) + Math.pow(prev.input[1] - y, 2));
                const currDist = Math.sqrt(Math.pow(curr.input[0] - x, 2) + Math.pow(curr.input[1] - y, 2));
                return currDist < prevDist ? curr : prev;
            });
            return { target: nearest.target[0], prediction: nearest.prediction[0] };
        }
        
        // Bilinear interpolation
        const dx = (x2 - x1) || 1;
        const dy = (y2 - y1) || 1;
        const tx = (x - x1) / dx;
        const ty = (y - y1) / dy;
        
        // Interpolate target values
        const targetTop = p11.target[0] * (1 - tx) + p21.target[0] * tx;
        const targetBottom = p12.target[0] * (1 - tx) + p22.target[0] * tx;
        const targetValue = targetTop * (1 - ty) + targetBottom * ty;
        
        // Interpolate prediction values
        const predTop = p11.prediction[0] * (1 - tx) + p21.prediction[0] * tx;
        const predBottom = p12.prediction[0] * (1 - tx) + p22.prediction[0] * tx;
        const predValue = predTop * (1 - ty) + predBottom * ty;
        
        return { target: targetValue, prediction: predValue };
    }
    
    // Create a finer grid for smoother surface (increased from 50 to 100)
    const gridSize = 100;
    const xRange = [0, 1];
    const yRange = [0, 1];
    
    // Generate grid points
    const xGrid = [];
    const yGrid = [];
    const zTarget = [];
    const zPrediction = [];
    
    for (let i = 0; i < gridSize; i++) {
        const xRow = [];
        const yRow = [];
        const zTargetRow = [];
        const zPredictionRow = [];
        
        for (let j = 0; j < gridSize; j++) {
            const x = xRange[0] + (xRange[1] - xRange[0]) * i / (gridSize - 1);
            const y = yRange[0] + (yRange[1] - yRange[0]) * j / (gridSize - 1);
            
            xRow.push(x);
            yRow.push(y);
            
            // Use bilinear interpolation for smooth values
            const interpolated = bilinearInterpolate(x, y, preview);
            
            zTargetRow.push(interpolated.target);
            zPredictionRow.push(interpolated.prediction);
        }
        
        xGrid.push(xRow);
        yGrid.push(yRow);
        zTarget.push(zTargetRow);
        zPrediction.push(zPredictionRow);
    }
    
    // Extract actual data points for scatter
    const xPoints = preview.map(p => p.input[0]);
    const yPoints = preview.map(p => p.input[1]);
    const zTargetPoints = preview.map(p => p.target[0]);
    const zPredictionPoints = preview.map(p => p.prediction[0]);
    
    const data = [
        {
            type: 'surface',
            x: xGrid,
            y: yGrid,
            z: zTarget,
            name: 'Target Surface',
            colorscale: 'Blues',
            opacity: 0.7,
            showscale: false,
            contours: {
                z: {
                    show: true,
                    usecolormap: true,
                    highlightcolor: "#42a5f5",
                    project: { z: false }
                }
            }
        },
        {
            type: 'surface',
            x: xGrid,
            y: yGrid,
            z: zPrediction,
            name: 'Prediction Surface',
            colorscale: 'Reds',
            opacity: 0.7,
            showscale: false,
            contours: {
                z: {
                    show: true,
                    usecolormap: true,
                    highlightcolor: "#ef5350",
                    project: { z: false }
                }
            }
        },
        {
            type: 'scatter3d',
            x: xPoints,
            y: yPoints,
            z: zTargetPoints,
            mode: 'markers',
            name: 'Target Points',
            marker: {
                size: 8,
                color: 'blue',
                symbol: 'circle'
            }
        },
        {
            type: 'scatter3d',
            x: xPoints,
            y: yPoints,
            z: zPredictionPoints,
            mode: 'markers',
            name: 'Prediction Points',
            marker: {
                size: 8,
                color: 'red',
                symbol: 'diamond'
            }
        }
    ];
    
    const layout = {
        title: {
            text: '3D Visualization: Neural Network Output',
            font: { color: '#e0e0e0' }
        },
        scene: {
            xaxis: { title: 'Input X1', color: '#e0e0e0', gridcolor: '#444' },
            yaxis: { title: 'Input X2', color: '#e0e0e0', gridcolor: '#444' },
            zaxis: { title: 'Output', color: '#e0e0e0', gridcolor: '#444' },
            bgcolor: '#2a2a2a'
        },
        paper_bgcolor: '#2a2a2a',
        plot_bgcolor: '#2a2a2a',
        font: { color: '#e0e0e0' },
        margin: { l: 0, r: 0, b: 0, t: 40 },
        showlegend: true,
        legend: {
            font: { color: '#e0e0e0' },
            bgcolor: '#333'
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    Plotly.newPlot('plot_3d', data, layout, config);
    current3DPlot = true;
}

function fetchAndDisplayWeightsBiases() {
    if (!trainingJobId) return;
    
    fetch(`/network-internals/${trainingJobId}`)
        .then(response => response.json())
        .then(data => {
            if (data.layers) {
                displayWeightsBiases(data.layers);
            }
        })
        .catch(error => {
            console.error('Error fetching weights/biases:', error);
        });
}

function displayWeightsBiases(layers) {
    weightsBiasesContent.innerHTML = '';
    
    layers.forEach((layer, idx) => {
        const layerDiv = document.createElement('div');
        layerDiv.className = 'layer-weights';
        
        const layerTitle = document.createElement('h4');
        layerTitle.textContent = `Layer ${layer.layer_index + 1}: ${layer.weights[0].length} → ${layer.weights.length} neurons`;
        layerDiv.appendChild(layerTitle);
        
        // Display weight statistics
        const statsDiv = document.createElement('div');
        statsDiv.style.marginBottom = '10px';
        statsDiv.innerHTML = `
            <span style="color: #bbbbbb;">Weights:</span> 
            <span style="color: #FFD54F;">Min: ${layer.weight_stats.min.toFixed(4)}</span>, 
            <span style="color: #FFD54F;">Max: ${layer.weight_stats.max.toFixed(4)}</span>, 
            <span style="color: #FFD54F;">Mean: ${layer.weight_stats.mean.toFixed(4)}</span>, 
            <span style="color: #FFD54F;">StdDev: ${layer.weight_stats.std_dev.toFixed(4)}</span>
        `;
        layerDiv.appendChild(statsDiv);
        
        // Display actual weights (limited to avoid overwhelming display)
        const maxWeightsToShow = 64;
        const totalWeights = layer.weights.length * layer.weights[0].length;
        
        if (totalWeights <= maxWeightsToShow) {
            const weightsTitle = document.createElement('h5');
            weightsTitle.textContent = 'Weights Matrix:';
            weightsTitle.style.color = '#bbbbbb';
            weightsTitle.style.marginTop = '10px';
            weightsTitle.style.marginBottom = '5px';
            layerDiv.appendChild(weightsTitle);
            
            const weightsGrid = document.createElement('div');
            weightsGrid.className = 'weights-grid';
            
            layer.weights.forEach((neuronWeights, neuronIdx) => {
                neuronWeights.forEach((weight, weightIdx) => {
                    const weightCell = document.createElement('div');
                    weightCell.className = 'weight-cell';
                    weightCell.textContent = weight.toFixed(3);
                    weightCell.title = `Neuron ${neuronIdx}, Weight ${weightIdx}: ${weight}`;
                    weightsGrid.appendChild(weightCell);
                });
            });
            
            layerDiv.appendChild(weightsGrid);
        } else {
            const note = document.createElement('p');
            note.style.color = '#bbbbbb';
            note.style.fontSize = '0.85em';
            note.textContent = `(${totalWeights} weights - showing statistics only due to size)`;
            layerDiv.appendChild(note);
        }
        
        // Display biases
        const biasesContainer = document.createElement('div');
        biasesContainer.className = 'biases-container';
        
        const biasesTitle = document.createElement('h5');
        biasesTitle.textContent = 'Biases:';
        biasesContainer.appendChild(biasesTitle);
        
        const biasesGrid = document.createElement('div');
        biasesGrid.className = 'weights-grid';
        
        layer.biases.forEach((bias, idx) => {
            const biasCell = document.createElement('div');
            biasCell.className = 'bias-cell';
            biasCell.textContent = bias.toFixed(3);
            biasCell.title = `Neuron ${idx} Bias: ${bias}`;
            biasesGrid.appendChild(biasCell);
        });
        
        biasesContainer.appendChild(biasesGrid);
        layerDiv.appendChild(biasesContainer);
        
        weightsBiasesContent.appendChild(layerDiv);
    });
    
    weightsBiasesDisplay.style.display = 'block';
}

targetTypeSelect.addEventListener('change', toggleTargetInputs);
backendSelect.addEventListener('change', toggleBackendOptions);

toggleTargetInputs();
toggleBackendOptions();

// Forward pass visualization button
const vizForwardButton = document.getElementById('viz_forward_button');
const vizInput = document.getElementById('viz_input');

vizForwardButton.addEventListener('click', () => {
    const inputValue = vizInput.value.trim();
    const input = inputValue.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
    
    if (input.length === 0) {
        alert('Please provide valid input values');
        return;
    }
    
    if (!trainingJobId) {
        alert('Please train a network first');
        return;
    }
    
    performForwardPassVisualization(input);
});

trainButton.addEventListener('click', async () => {
    const layers = document
        .getElementById('layers')
        .value.split(',')
        .map(value => Number(value.trim()))
        .filter(value => !Number.isNaN(value));
    const activation_function = document.getElementById('activation_function').value;
    const learning_rate = Number.parseFloat(document.getElementById('learning_rate').value);
    const epochs = Number.parseInt(document.getElementById('epochs').value, 10);
    const target_type = targetTypeSelect.value;
    const backend = backendSelect.value;

    if (!layers.length || layers.some(value => value <= 0)) {
        alert('Please provide a valid layer configuration.');
        return;
    }
    if (!Number.isFinite(learning_rate) || learning_rate <= 0) {
        alert('Please provide a valid learning rate.');
        return;
    }
    if (!Number.isInteger(epochs) || epochs <= 0) {
        alert('Epochs must be a positive integer.');
        return;
    }

    const target_expression = target_type === 'function'
        ? document.getElementById('target_expression').value
        : document.getElementById('logical_operation').value;

    const network_config = {
        backend,
        layers,
        activation_function,
        learning_rate,
        epochs,
    };

    if (backend === 'nn-core-library') {
        const options = {
            optimizer: optimizerSelect.value,
            use_gpu: useGpuCheckbox.checked,
        };
        const threshold = Number.parseInt(gpuThresholdInput.value, 10);
        if (Number.isFinite(threshold) && threshold > 0) {
            options.gpu_workload_threshold = threshold;
        }
        network_config.core_library_options = options;
    }

    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                network_config,
                target_type,
                target_expression,
            }),
        });

        const data = await response.json();
        if (response.ok) {
            trainingJobId = data.job_id;
            jobIdSpan.textContent = trainingJobId;
            statusSpan.textContent = data.status;
            warningSpan.textContent = data.warning ?? '';
            messageSpan.textContent = data.message ?? '';
             backendNameSpan.textContent = formatBackendLabel(backend);
            stopTrainingButton.style.display = 'inline-block';
            trainButton.disabled = true;
            predictButton.disabled = true;
            resetVisualisations();
            storeNetworkConfig(network_config);
            startTrainingProgressPolling();
        } else {
            alert(`Error: ${data.message ?? data.status}`);
            messageSpan.textContent = data.message ?? '';
        }
    } catch (error) {
        console.error('Failed to submit training job', error);
        alert('Failed to submit training job. See console for details.');
    }
});

stopTrainingButton.addEventListener('click', async () => {
    if (!trainingJobId) {
        return;
    }

    try {
        const response = await fetch(`/stop-training/${trainingJobId}`, {
            method: 'POST',
        });
        const data = await response.json();
        if (response.ok) {
            statusSpan.textContent = data.status;
            stopTrainingButton.style.display = 'none';
            trainButton.disabled = false;
            predictButton.disabled = false;
            messageSpan.textContent = data.message ?? '';
            stopTrainingProgressPolling();
        } else {
            alert(`Error: ${data.message ?? data.status}`);
        }
    } catch (error) {
        console.error('Failed to stop training job', error);
        alert('Failed to stop training job. See console for details.');
    }
});

predictButton.addEventListener('click', async () => {
    if (!trainingJobId) {
        alert('Please train a network first.');
        return;
    }

    const input = document
        .getElementById('prediction_input')
        .value.split(',')
        .map(value => Number(value.trim()))
        .filter(value => !Number.isNaN(value));

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                job_id: trainingJobId,
                input,
            }),
        });

        const data = await response.json();
        if (response.ok) {
            predictionOutputSpan.textContent = data.prediction.map(value => value.toFixed(6)).join(', ');
            actualResultSpan.textContent = data.actual_result !== undefined ? data.actual_result : 'N/A';
        } else {
            alert(`Error: ${data.message ?? data.status}`);
        }
    } catch (error) {
        console.error('Failed to fetch prediction', error);
        alert('Failed to fetch prediction. See console for details.');
    }
});

function startTrainingProgressPolling() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
    }
    trainingInterval = setInterval(async () => {
        if (!trainingJobId) {
            return;
        }

        try {
            const response = await fetch(`/training-progress/${trainingJobId}`);
            const data = await response.json();

            if (response.ok) {
                statusSpan.textContent = data.status;
                progressSpan.textContent = data.progress.toFixed(2);
                currentLossSpan.textContent =
                    typeof data.current_loss === 'number' ? data.current_loss.toFixed(6) : 'N/A';
                warningSpan.textContent = data.warning ?? '';
                messageSpan.textContent = data.message ?? '';
                backendNameSpan.textContent = formatBackendLabel(data.backend ?? backendSelect.value);

                if (typeof data.current_loss === 'number') {
                    updateLossHistory(data.current_loss);
                }

                updatePredictionVisuals(data);

                if (['Completed', 'Stopped', 'Failed'].includes(data.status)) {
                    stopTrainingProgressPolling();
                    trainButton.disabled = false;
                    stopTrainingButton.style.display = 'none';
                    predictButton.disabled = !['Completed', 'Stopped'].includes(data.status);
                    
                    // Update with complete loss history when training finishes
                    if (data.loss_history && data.loss_history.length > 0) {
                        updateLossHistoryComplete(data.loss_history);
                    }
                    
                    // Fetch network internals when training completes
                    if (['Completed', 'Stopped'].includes(data.status)) {
                        fetchNetworkInternals();
                        
                        // Perform a demo forward pass visualization
                        if (data.prediction_preview && data.prediction_preview.length > 0) {
                            const sampleInput = data.prediction_preview[0].input;
                            performForwardPassVisualization(sampleInput);
                        }
                    }
                }
            } else {
                console.error('Failed to fetch training progress:', data);
                stopTrainingProgressPolling();
                trainButton.disabled = false;
                stopTrainingButton.style.display = 'none';
            }
        } catch (error) {
            console.error('Training progress polling failed', error);
            stopTrainingProgressPolling();
            trainButton.disabled = false;
            stopTrainingButton.style.display = 'none';
        }
    }, 1000);
}

function stopTrainingProgressPolling() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
}

// Network Visualization Functions
function drawNetworkVisualization() {
    if (!currentNetworkConfig || !currentNetworkConfig.layers) {
        return;
    }
    
    // Set canvas size
    networkCanvas.width = networkCanvas.offsetWidth;
    networkCanvas.height = networkCanvas.offsetHeight;
    
    const ctx = networkCtx;
    const layers = currentNetworkConfig.layers;
    const padding = 60;
    const width = networkCanvas.width - padding * 2;
    const height = networkCanvas.height - padding * 2;
    
    // Clear canvas
    ctx.fillStyle = '#2a2a2a';
    ctx.fillRect(0, 0, networkCanvas.width, networkCanvas.height);
    
    // Calculate positions
    const maxNeurons = Math.max(...layers);
    const layerSpacing = width / (layers.length - 1 || 1);
    
    // Draw connections first (so they appear behind neurons)
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 0.5;
    
    for (let i = 0; i < layers.length - 1; i++) {
        const currentLayer = layers[i];
        const nextLayer = layers[i + 1];
        const x1 = padding + i * layerSpacing;
        const x2 = padding + (i + 1) * layerSpacing;
        
        for (let j = 0; j < currentLayer; j++) {
            const y1 = getNodeY(j, currentLayer, height, padding);
            
            for (let k = 0; k < nextLayer; k++) {
                const y2 = getNodeY(k, nextLayer, height, padding);
                
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
            }
        }
    }
    
    // Draw neurons
    layers.forEach((neuronCount, layerIndex) => {
        const x = padding + layerIndex * layerSpacing;
        const isInputLayer = layerIndex === 0;
        const isOutputLayer = layerIndex === layers.length - 1;
        
        for (let i = 0; i < neuronCount; i++) {
            const y = getNodeY(i, neuronCount, height, padding);
            
            // Neuron circle
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            
            if (isInputLayer) {
                ctx.fillStyle = '#90CAF9';
            } else if (isOutputLayer) {
                ctx.fillStyle = '#FFB74D';
            } else {
                ctx.fillStyle = '#81C784';
            }
            
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
        
        // Layer label
        ctx.fillStyle = '#e0e0e0';
        ctx.font = '12px "Segoe UI"';
        ctx.textAlign = 'center';
        const layerName = isInputLayer ? 'Input' : isOutputLayer ? 'Output' : `Hidden ${layerIndex}`;
        ctx.fillText(layerName, x, padding - 25);
        ctx.fillText(`(${neuronCount})`, x, padding - 10);
    });
    
    // Update layer details
    updateLayerDetails();
}

function getNodeY(index, totalNodes, height, padding) {
    if (totalNodes === 1) {
        return padding + height / 2;
    }
    const spacing = height / (totalNodes - 1 || 1);
    return padding + index * spacing;
}

function updateLayerDetails() {
    if (!currentNetworkConfig) return;
    
    const layerDetailsDiv = document.getElementById('layer_details');
    const layers = currentNetworkConfig.layers;
    const activation = currentNetworkConfig.activation_function || 'sigmoid';
    const learningRate = currentNetworkConfig.learning_rate || 0.01;
    
    let html = '<h4 style="margin-top: 0; color: #81C784;">Architecture</h4>';
    html += `<p><strong>Layers:</strong> ${layers.join(' → ')}</p>`;
    html += `<p><strong>Activation:</strong> ${activation.charAt(0).toUpperCase() + activation.slice(1)}</p>`;
    html += `<p><strong>Learning Rate:</strong> ${learningRate}</p>`;
    
    // Calculate total parameters
    let totalParams = 0;
    for (let i = 0; i < layers.length - 1; i++) {
        const weights = layers[i] * layers[i + 1];
        const biases = layers[i + 1];
        totalParams += weights + biases;
    }
    
    html += `<p><strong>Total Parameters:</strong> ${totalParams.toLocaleString()}</p>`;
    html += '<h4 style="color: #81C784; margin-top: 15px;">Layer Breakdown</h4>';
    
    for (let i = 0; i < layers.length - 1; i++) {
        const weights = layers[i] * layers[i + 1];
        const biases = layers[i + 1];
        const layerParams = weights + biases;
        html += `<p><strong>Layer ${i} → ${i + 1}:</strong> ${layerParams.toLocaleString()} params (${weights} weights + ${biases} biases)</p>`;
    }
    
    layerDetailsDiv.innerHTML = html;
}

function storeNetworkConfig(config) {
    currentNetworkConfig = config;
    if (document.querySelector('.tab-button[data-tab="network-viz"]').classList.contains('active')) {
        drawNetworkVisualization();
    }
}

// Resize canvas when window resizes
window.addEventListener('resize', () => {
    if (document.getElementById('network-viz-tab').classList.contains('active')) {
        drawNetworkVisualization();
    }
});

// Fetch and display network internals
async function fetchNetworkInternals() {
    if (!trainingJobId) return;
    
    try {
        const response = await fetch(`/network-internals/${trainingJobId}`);
        const data = await response.json();
        
        if (response.ok && data.layers) {
            displayWeightTable(data.layers);
        }
    } catch (error) {
        console.error('Failed to fetch network internals:', error);
    }
}

function displayWeightTable(layers) {
    const weightStatsDiv = document.getElementById('weight_stats');
    if (!weightStatsDiv) return;
    
    let html = '<h4 style="color: #81C784; margin-top: 15px;">Weight Statistics</h4>';
    html += '<table>';
    html += '<thead><tr><th>Layer</th><th>Min</th><th>Max</th><th>Mean</th><th>Std Dev</th></tr></thead>';
    html += '<tbody>';
    
    layers.forEach((layer, idx) => {
        const stats = layer.weight_stats;
        html += `<tr>`;
        html += `<td>Layer ${idx}</td>`;
        html += `<td>${stats.min.toFixed(4)}</td>`;
        html += `<td>${stats.max.toFixed(4)}</td>`;
        html += `<td>${stats.mean.toFixed(4)}</td>`;
        html += `<td>${stats.std_dev.toFixed(4)}</td>`;
        html += `</tr>`;
    });
    
    html += '</tbody></table>';
    weightStatsDiv.innerHTML = html;
}

// Forward pass visualization
let forwardPassAnimationFrame = null;
let currentForwardPassData = null;

async function performForwardPassVisualization(input) {
    if (!trainingJobId) return;
    
    try {
        const response = await fetch('/forward-pass', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                job_id: trainingJobId,
                input: input,
            }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentForwardPassData = data;
            animateForwardPass();
        }
    } catch (error) {
        console.error('Failed to perform forward pass visualization:', error);
    }
}

function animateForwardPass() {
    if (!currentForwardPassData || !currentNetworkConfig) return;
    
    // Cancel any ongoing animation
    if (forwardPassAnimationFrame) {
        cancelAnimationFrame(forwardPassAnimationFrame);
    }
    
    let step = 0;
    const maxSteps = currentForwardPassData.activations.length - 1;
    
    function animate() {
        if (step <= maxSteps) {
            drawNetworkVisualizationWithActivations(step);
            step++;
            forwardPassAnimationFrame = requestAnimationFrame(animate);
        } else {
            forwardPassAnimationFrame = null;
        }
    }
    
    animate();
}

function drawNetworkVisualizationWithActivations(activeStep) {
    if (!currentNetworkConfig || !currentForwardPassData) {
        drawNetworkVisualization();
        return;
    }
    
    // Set canvas size
    networkCanvas.width = networkCanvas.offsetWidth;
    networkCanvas.height = networkCanvas.offsetHeight;
    
    const ctx = networkCtx;
    const layers = currentNetworkConfig.layers;
    const padding = 60;
    const width = networkCanvas.width - padding * 2;
    const height = networkCanvas.height - padding * 2;
    
    // Clear canvas
    ctx.fillStyle = '#2a2a2a';
    ctx.fillRect(0, 0, networkCanvas.width, networkCanvas.height);
    
    // Calculate positions
    const layerSpacing = width / (layers.length - 1 || 1);
    
    // Draw connections with activation highlighting
    for (let i = 0; i < layers.length - 1; i++) {
        const currentLayer = layers[i];
        const nextLayer = layers[i + 1];
        const x1 = padding + i * layerSpacing;
        const x2 = padding + (i + 1) * layerSpacing;
        
        // Check if this connection layer is active
        const isActive = activeStep > i;
        
        for (let j = 0; j < currentLayer; j++) {
            const y1 = getNodeY(j, currentLayer, height, padding);
            
            for (let k = 0; k < nextLayer; k++) {
                const y2 = getNodeY(k, nextLayer, height, padding);
                
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                
                if (isActive) {
                    ctx.strokeStyle = 'rgba(76, 175, 80, 0.3)';
                    ctx.lineWidth = 1;
                } else {
                    ctx.strokeStyle = '#555';
                    ctx.lineWidth = 0.5;
                }
                
                ctx.stroke();
            }
        }
    }
    
    // Draw neurons with activation values
    layers.forEach((neuronCount, layerIndex) => {
        const x = padding + layerIndex * layerSpacing;
        const isInputLayer = layerIndex === 0;
        const isOutputLayer = layerIndex === layers.length - 1;
        const isActive = activeStep >= layerIndex;
        
        const activations = isActive && currentForwardPassData.activations[layerIndex] 
            ? currentForwardPassData.activations[layerIndex] 
            : null;
        
        for (let i = 0; i < neuronCount; i++) {
            const y = getNodeY(i, neuronCount, height, padding);
            
            // Determine neuron color and size based on activation
            let neuronColor, neuronSize = 8;
            
            if (isActive && activations) {
                const activation = activations[i];
                const intensity = Math.min(1, Math.abs(activation));
                
                if (isInputLayer) {
                    neuronColor = `rgba(144, 202, 249, ${0.5 + intensity * 0.5})`;
                } else if (isOutputLayer) {
                    neuronColor = `rgba(255, 183, 77, ${0.5 + intensity * 0.5})`;
                } else {
                    neuronColor = `rgba(129, 199, 132, ${0.5 + intensity * 0.5})`;
                }
                neuronSize = 8 + intensity * 4;
            } else {
                if (isInputLayer) {
                    neuronColor = 'rgba(144, 202, 249, 0.3)';
                } else if (isOutputLayer) {
                    neuronColor = 'rgba(255, 183, 77, 0.3)';
                } else {
                    neuronColor = 'rgba(129, 199, 132, 0.3)';
                }
            }
            
            // Draw neuron circle
            ctx.beginPath();
            ctx.arc(x, y, neuronSize, 0, Math.PI * 2);
            ctx.fillStyle = neuronColor;
            ctx.fill();
            ctx.strokeStyle = isActive ? '#fff' : '#666';
            ctx.lineWidth = isActive ? 2 : 1;
            ctx.stroke();
            
            // Draw activation value if active
            if (isActive && activations && neuronCount <= 10) {
                ctx.fillStyle = '#fff';
                ctx.font = '10px "Segoe UI"';
                ctx.textAlign = 'left';
                ctx.fillText(activations[i].toFixed(2), x + neuronSize + 5, y + 4);
            }
        }
        
        // Layer label
        ctx.fillStyle = isActive ? '#e0e0e0' : '#888';
        ctx.font = '12px "Segoe UI"';
        ctx.textAlign = 'center';
        const layerName = isInputLayer ? 'Input' : isOutputLayer ? 'Output' : `Hidden ${layerIndex}`;
        ctx.fillText(layerName, x, padding - 25);
        ctx.fillText(`(${neuronCount})`, x, padding - 10);
    });
}
