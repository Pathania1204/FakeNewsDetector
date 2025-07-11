// Fake News Detector JavaScript functionality

document.addEventListener('DOMContentLoaded', function() {
    // Check model status on page load
    checkModelStatus();
    
    // Initialize form validation
    initializeFormValidation();
    
    // Initialize UI enhancements
    initializeUIEnhancements();
});

/**
 * Check if the model is trained and ready
 */
function checkModelStatus() {
    const statusCard = document.getElementById('model-status-card');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    if (!statusCard) return;
    
    fetch('/model_status')
        .then(response => response.json())
        .then(data => {
            updateModelStatusCard(statusCard, data);
            updateAnalyzeButton(analyzeBtn, data);
        })
        .catch(error => {
            console.error('Error checking model status:', error);
            updateModelStatusCard(statusCard, { error: 'Failed to check model status' });
            updateAnalyzeButton(analyzeBtn, { model_trained: false });
        });
}

/**
 * Update the model status card
 */
function updateModelStatusCard(statusCard, data) {
    if (data.error) {
        statusCard.innerHTML = `
            <div class="card-body">
                <div class="d-flex align-items-center text-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <span>Error: ${data.error}</span>
                </div>
            </div>
        `;
        return;
    }
    
    if (data.model_trained) {
        statusCard.innerHTML = `
            <div class="card-body">
                <div class="d-flex align-items-center text-success">
                    <i class="fas fa-check-circle me-2"></i>
                    <span>Model is ready for predictions</span>
                </div>
            </div>
        `;
        statusCard.classList.add('border-success');
    } else {
        statusCard.innerHTML = `
            <div class="card-body">
                <div class="d-flex align-items-center text-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <span>Model not trained. Please <a href="/train" class="text-warning">train the model</a> first.</span>
                </div>
            </div>
        `;
        statusCard.classList.add('border-warning');
    }
}

/**
 * Update the analyze button state
 */
function updateAnalyzeButton(analyzeBtn, data) {
    if (!analyzeBtn) return;
    
    if (data.model_trained) {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search me-2"></i>Analyze News';
        analyzeBtn.classList.remove('btn-secondary');
        analyzeBtn.classList.add('btn-primary');
    } else {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>Model Not Ready';
        analyzeBtn.classList.remove('btn-primary');
        analyzeBtn.classList.add('btn-secondary');
    }
}

/**
 * Initialize form validation
 */
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            const newsText = form.querySelector('#news_text');
            
            if (newsText) {
                const text = newsText.value.trim();
                
                if (text.length < 10) {
                    event.preventDefault();
                    showAlert('Please enter at least 10 characters for better accuracy.', 'warning');
                    return;
                }
                
                // Show loading state
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
                }
            }
        });
    });
}

/**
 * Initialize UI enhancements
 */
function initializeUIEnhancements() {
    // Add character counter to textarea
    const textArea = document.getElementById('news_text');
    if (textArea) {
        addCharacterCounter(textArea);
    }
    
    // Add copy functionality to results
    addCopyToClipboard();
    
    // Add smooth scrolling to results
    scrollToResults();
    
    // Initialize tooltips
    initializeTooltips();
}

/**
 * Add character counter to textarea
 */
function addCharacterCounter(textArea) {
    const counter = document.createElement('div');
    counter.className = 'form-text text-end';
    counter.innerHTML = '<span id="char-count">0</span> characters';
    
    textArea.parentNode.appendChild(counter);
    
    textArea.addEventListener('input', function() {
        const charCount = document.getElementById('char-count');
        charCount.textContent = this.value.length;
        
        // Color coding based on length
        if (this.value.length < 10) {
            charCount.className = 'text-danger';
        } else if (this.value.length < 50) {
            charCount.className = 'text-warning';
        } else {
            charCount.className = 'text-success';
        }
    });
}

/**
 * Add copy to clipboard functionality
 */
function addCopyToClipboard() {
    const results = document.querySelectorAll('[data-copy]');
    
    results.forEach(element => {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'btn btn-sm btn-outline-secondary ms-2';
        copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
        copyBtn.onclick = () => copyToClipboard(element.textContent);
        
        element.appendChild(copyBtn);
    });
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showAlert('Copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Failed to copy: ', err);
        showAlert('Failed to copy to clipboard', 'error');
    });
}

/**
 * Scroll to results section
 */
function scrollToResults() {
    const resultsCard = document.querySelector('.card.mt-4.shadow-sm');
    if (resultsCard && window.location.search.includes('result')) {
        setTimeout(() => {
            resultsCard.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    const alertContainer = document.querySelector('.container.my-4');
    const alert = document.createElement('div');
    alert.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    alert.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : type === 'success' ? 'check-circle' : 'info-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    alertContainer.insertBefore(alert, alertContainer.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.classList.remove('show');
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.remove();
                }
            }, 150);
        }
    }, 5000);
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Validate dataset file before upload
 */
function validateDatasetFile(fileInput) {
    const file = fileInput.files[0];
    
    if (!file) return false;
    
    const allowedTypes = ['text/csv', 'text/plain'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!allowedTypes.includes(file.type)) {
        showAlert('Please select a CSV or TXT file.', 'error');
        fileInput.value = '';
        return false;
    }
    
    if (file.size > maxSize) {
        showAlert('File size must be less than 16MB.', 'error');
        fileInput.value = '';
        return false;
    }
    
    return true;
}

// Add file validation to upload forms
document.addEventListener('DOMContentLoaded', function() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            validateDatasetFile(this);
        });
    });
});

// Add progressive enhancement for better UX
window.addEventListener('load', function() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('fade-in');
        }, index * 100);
    });
});
