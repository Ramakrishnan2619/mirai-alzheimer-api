/**
 * MirAI Core Functions
 * Shared utilities and initialization
 */

// Initialize authentication state on page load
document.addEventListener('DOMContentLoaded', function () {
    // Check if API is available
    if (typeof window.API !== 'undefined') {
        console.log('MirAI API loaded successfully');

        // Auto-validate session if token exists
        if (window.API.session.isValid()) {
            window.API.auth.validate().then(valid => {
                if (!valid) {
                    console.log('Session expired, clearing...');
                    window.API.session.clear();
                }
            });
        }
    }
});

// Utility function to show loading state
function showLoading(element, text = 'Loading...') {
    if (element) {
        element.disabled = true;
        element.dataset.originalText = element.textContent;
        element.textContent = text;
    }
}

// Utility function to hide loading state
function hideLoading(element) {
    if (element && element.dataset.originalText) {
        element.disabled = false;
        element.textContent = element.dataset.originalText;
    }
}

// Utility function to show error message
function showError(message, containerId = 'error-container') {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `<div class="alert alert-danger">${message}</div>`;
        container.style.display = 'block';
    } else {
        alert(message);
    }
}

// Utility function to show success message
function showSuccess(message, containerId = 'success-container') {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `<div class="alert alert-success">${message}</div>`;
        container.style.display = 'block';
    }
}

// Format risk category with color
function formatRiskCategory(category) {
    const colors = {
        'low': 'success',
        'moderate': 'warning',
        'high': 'danger',
        'elevated': 'danger'
    };
    const color = colors[category.toLowerCase()] || 'secondary';
    return `<span class="badge bg-${color}">${category.toUpperCase()}</span>`;
}

// Format percentage
function formatPercentage(value) {
    return `${(value * 100).toFixed(1)}%`;
}

// Export utilities
window.MirAICore = {
    showLoading,
    hideLoading,
    showError,
    showSuccess,
    formatRiskCategory,
    formatPercentage
};
