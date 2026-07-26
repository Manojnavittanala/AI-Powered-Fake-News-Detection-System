document.addEventListener('DOMContentLoaded', () => {
    const predictForm = document.getElementById('predict-form');
    const resultCard = document.getElementById('prediction-result');
    const chartCanvas = document.getElementById('probability-chart');
    const loadingIndicator = document.getElementById('loading-indicator');
    let probabilityChart;

    const showError = (messageNode, message) => {
        messageNode.textContent = message;
        messageNode.style.display = 'block';
    };

    if (predictForm) {
        predictForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const text = document.getElementById('news-text').value.trim();
            const messageNode = document.getElementById('predict-alert');

            if (!text) {
                messageNode.textContent = 'Paste your news article before predicting.';
                messageNode.style.display = 'block';
                return;
            }
            if (text.length < 100) {
                messageNode.textContent = 'Please provide at least 100 characters of article text.';
                messageNode.style.display = 'block';
                return;
            }

            messageNode.style.display = 'none';
            loadingIndicator.style.display = 'block';

            try {
                let response;
                try {
                    response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text }),
                    });
                } catch (error) {
                    console.error('Prediction request failed:', error);
                    showError(messageNode, 'Unable to connect to the prediction API. Check that the Flask server is running.');
                    return;
                }

                let data;
                try {
                    data = await response.json();
                } catch (error) {
                    console.error('Prediction API returned invalid JSON:', error);
                    showError(messageNode, 'The prediction service returned an invalid response.');
                    return;
                }

                if (!response.ok) {
                    showError(messageNode, data.error || 'Prediction failed.');
                    return;
                }

                const verdict = data.label;
                const confidence = Number(data.confidence) || 0;
                const keywords = data.keywords || [];
                const probabilities = data.probabilities || {
                    Real: verdict === 'Real' ? Number(data.probability) || 0 : 0,
                    Fake: verdict === 'Fake' ? Number(data.probability) || 0 : 0,
                };

                document.getElementById('result-label').textContent = verdict;
                document.getElementById('result-confidence').textContent = `${confidence}% confidence`;
                document.getElementById('result-probability').textContent =
                    `Real: ${probabilities.Real}% | Fake: ${probabilities.Fake}%`;
                document.getElementById('result-model').textContent = `Model: ${data.model_used || 'Classical ML'}`;
                if (data.message) {
                    messageNode.textContent = data.message;
                    messageNode.style.display = 'block';
                }
                document.getElementById('keywords-list').innerHTML = keywords
                    .map((term) => `<span class="keyword-pill">${term}</span>`)
                    .join('');

                if (chartCanvas && typeof window.Chart === 'function') {
                    try {
                        const ctx = chartCanvas.getContext('2d');
                        if (probabilityChart) {
                            probabilityChart.destroy();
                        }
                        probabilityChart = new window.Chart(ctx, {
                            type: 'doughnut',
                            data: {
                                labels: ['Confidence', 'Remaining'],
                                datasets: [{
                                    data: [confidence, 100 - confidence],
                                    backgroundColor: ['#2563eb', '#334155'],
                                    borderWidth: 0,
                                }],
                            },
                            options: {
                                plugins: { legend: { display: false } },
                                cutout: '70%',
                            },
                        });
                    } catch (error) {
                        console.error('Probability chart could not render:', error);
                    }
                }

                resultCard.style.display = 'grid';
            } finally {
                loadingIndicator.style.display = 'none';
            }
        });
    }
});
