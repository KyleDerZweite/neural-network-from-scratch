const trainButton = document.getElementById('train_button');
const stopTrainingButton = document.getElementById('stop_training_button');
const predictButton = document.getElementById('predict_button');
const targetTypeSelect = document.getElementById('target_type');
const functionInputDiv = document.getElementById('function_input');
const logicalOperationInputDiv = document.getElementById('logical_operation_input');

const jobIdSpan = document.getElementById('job_id');
const statusSpan = document.getElementById('status');
const progressSpan = document.getElementById('progress');
const currentLossSpan = document.getElementById('current_loss');
const predictionOutputSpan = document.getElementById('prediction_output');
const actualResultSpan = document.getElementById('actual_result');
const warningSpan = document.getElementById('warning');
const messageSpan = document.getElementById('status_message');

let trainingJobId = null;
let trainingInterval = null;

predictButton.disabled = true;

// Chart.js setup
const ctx = document.getElementById('loss_chart').getContext('2d');
const lossChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Loss',
            data: [],
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

targetTypeSelect.addEventListener('change', (event) => {
    if (event.target.value === 'function') {
        functionInputDiv.style.display = 'block';
        logicalOperationInputDiv.style.display = 'none';
    } else {
        functionInputDiv.style.display = 'none';
        logicalOperationInputDiv.style.display = 'block';
    }
});

trainButton.addEventListener('click', async () => {
    const layers = document.getElementById('layers').value.split(',').map(Number);
    const activation_function = document.getElementById('activation_function').value;
    const learning_rate = parseFloat(document.getElementById('learning_rate').value);
    const epochs = parseInt(document.getElementById('epochs').value);
    const target_type = document.getElementById('target_type').value;
    let target_expression = '';

    if (target_type === 'function') {
        target_expression = document.getElementById('target_expression').value;
    } else {
        target_expression = document.getElementById('logical_operation').value; // e.g., 'xor'
    }

    const network_config = {
        layers,
        activation_function,
        learning_rate,
        epochs,
    };

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
        stopTrainingButton.style.display = 'inline-block';
        trainButton.disabled = true;
        predictButton.disabled = true; // Disable predict button during training
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.update();
        startTrainingProgressPolling();
    } else {
        alert(`Error: ${data.message ?? data.status}`);
        messageSpan.textContent = data.message ?? '';
    }
});

stopTrainingButton.addEventListener('click', async () => {
    if (trainingJobId) {
        const response = await fetch(`/stop-training/${trainingJobId}`, {
            method: 'POST',
        });
        const data = await response.json();
        if (response.ok) {
            statusSpan.textContent = data.status;
            stopTrainingButton.style.display = 'none';
            trainButton.disabled = false;
            predictButton.disabled = false; // Enable predict button after stopping
            messageSpan.textContent = data.message ?? '';
            stopTrainingProgressPolling();
        } else {
            alert(`Error: ${data.message ?? data.status}`);
        }
    }
});

predictButton.addEventListener('click', async () => {
    if (!trainingJobId) {
        alert('Please train a network first.');
        return;
    }

    const input = document.getElementById('prediction_input').value.split(',').map(Number);

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
        predictionOutputSpan.textContent = data.prediction.join(', ');
        actualResultSpan.textContent = data.actual_result !== undefined ? data.actual_result : 'N/A';
    } else {
        alert(`Error: ${data.message ?? data.status}`);
    }
});

function startTrainingProgressPolling() {
    if (trainingInterval) {
        clearInterval(trainingInterval);
    }
    trainingInterval = setInterval(async () => {
        if (!trainingJobId) return;

        const response = await fetch(`/training-progress/${trainingJobId}`);
        const data = await response.json();

        if (response.ok) {
            statusSpan.textContent = data.status;
            progressSpan.textContent = data.progress.toFixed(2);
            currentLossSpan.textContent = data.current_loss !== undefined ? data.current_loss.toFixed(6) : 'N/A';
            warningSpan.textContent = data.warning ?? '';
            messageSpan.textContent = data.message ?? '';

            if (typeof data.current_loss === 'number') {
                lossChart.data.labels.push(lossChart.data.labels.length + 1);
                lossChart.data.datasets[0].data.push(data.current_loss);
                lossChart.update();
            }

            if (data.status === 'Completed' || data.status === 'Stopped' || data.status === 'Failed') {
                stopTrainingProgressPolling();
                trainButton.disabled = false;
                stopTrainingButton.style.display = 'none';
                predictButton.disabled = data.status !== 'Completed';
            }
        } else {
            console.error('Failed to fetch training progress:', data);
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
