// Auto-dismiss alerts after a few seconds
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.alert').forEach((alertEl) => {
        setTimeout(() => {
            const alert = bootstrap.Alert.getOrCreateInstance(alertEl);
            if (alert) alert.close();
        }, 5000);
    });
});
