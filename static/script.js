// Dance Grading System - JavaScript

// DOM Elements
const teacherUpload = document.getElementById('teacherUpload');
const teacherFile = document.getElementById('teacherFile');
const teacherBtn = document.getElementById('teacherBtn');
const teacherResult = document.getElementById('teacherResult');

const studentUpload = document.getElementById('studentUpload');
const studentFile = document.getElementById('studentFile');
const studentBtn = document.getElementById('studentBtn');
const studentName = document.getElementById('studentName');

const statusBar = document.getElementById('statusBar');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');

const gradeAnotherBtn = document.getElementById('gradeAnotherBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    loadStatus();
    loadHistory();
});

// ============================================
// Event Listeners
// ============================================

function setupEventListeners() {
    // Teacher upload
    teacherBtn.addEventListener('click', () => teacherFile.click());
    teacherFile.addEventListener('change', handleTeacherFileSelect);
    
    // Drag and drop for teacher
    setupDragAndDrop(teacherUpload, teacherFile);
    
    // Student upload
    studentBtn.addEventListener('click', () => studentFile.click());
    studentFile.addEventListener('change', handleStudentFileSelect);
    
    // Drag and drop for student
    setupDragAndDrop(studentUpload, studentFile);
    
    // Other
    gradeAnotherBtn.addEventListener('click', resetStudentSection);
    clearHistoryBtn.addEventListener('click', clearHistory);
}

// ============================================
// Drag and Drop
// ============================================

function setupDragAndDrop(uploadArea, fileInput) {
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            if (fileInput.id === 'teacherFile') {
                handleTeacherFileSelect();
            } else {
                handleStudentFileSelect();
            }
        }
    });
}

// ============================================
// File Handlers
// ============================================

async function handleTeacherFileSelect() {
    const file = teacherFile.files[0];
    if (!file) return;
    
    if (!isValidFile(file)) {
        showError('Invalid file. Please use MP4, AVI, MOV, MKV, or WebM');
        return;
    }
    
    await uploadTeacher(file);
}

async function handleStudentFileSelect() {
    const file = studentFile.files[0];
    if (!file) return;
    
    if (!isValidFile(file)) {
        showError('Invalid file. Please use MP4, AVI, MOV, MKV, or WebM');
        return;
    }
    
    await uploadStudent(file);
}

function isValidFile(file) {
    const validTypes = [
        'video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm',
        'application/x-msvideo', 'video/x-m4v'
    ];
    
    const validExtensions = ['mp4', 'avi', 'mov', 'mkv', 'webm'];
    const fileName = file.name.toLowerCase();
    const ext = fileName.split('.').pop();
    
    return validExtensions.includes(ext) && file.size <= 500 * 1024 * 1024;
}

// ============================================
// API Calls
// ============================================

async function uploadTeacher(file) {
    showLoading(true);
    
    try {
        const formData = new FormData();
        formData.append('video', file);
        
        const response = await fetch('/api/upload-teacher', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            showError(data.error || 'Failed to upload teacher video');
            showLoading(false);
            return;
        }
        
        // Show success
        teacherResult.classList.remove('hidden');
        document.getElementById('teacherFrames').textContent = `Frames extracted: ${data.frames}`;
        
        showSuccess(`Teacher reference created with ${data.frames} frames!`);
        
        // Update status
        loadStatus();
        
        // Enable student section
        enableStudentSection();
        
    } catch (error) {
        showError(`Error: ${error.message}`);
    }
    
    showLoading(false);
}

async function uploadStudent(file) {
    const name = studentName.value || 'Student';
    const movementType = document.getElementById('movementType').value;
    
    showLoading(true);
    
    try {
        const formData = new FormData();
        formData.append('video', file);
        formData.append('name', name);
        formData.append('movement_type', movementType);
        
        const response = await fetch('/api/upload-student', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            showError(data.error || 'Failed to grade student video');
            showLoading(false);
            return;
        }
        
        // Show results
        displayResults(data);
        
        // Update status and history
        loadStatus();
        loadHistory();
        
    } catch (error) {
        showError(`Error: ${error.message}`);
    }
    
    showLoading(false);
}

// ============================================
// Results Display
// ============================================

function displayResults(results) {
    resultsSection.classList.remove('hidden');
    
    // Student info
    document.getElementById('resultName').textContent = results.student_name;
    document.getElementById('resultFrames').textContent = `Frames: ${results.frames}`;
    
    // Scores
    document.getElementById('overallScore').textContent = results.overall_score;
    document.getElementById('armScore').textContent = results.arm_score;
    document.getElementById('legScore').textContent = results.leg_score;
    
    // Fill bars with animation
    setTimeout(() => {
        document.getElementById('overallFill').style.width = results.overall_score + '%';
        document.getElementById('armFill').style.width = results.arm_score + '%';
        document.getElementById('legFill').style.width = results.leg_score + '%';
    }, 100);
    
    // Feedback
    document.getElementById('feedbackText').textContent = results.feedback;
    document.getElementById('starRating').textContent = '⭐'.repeat(results.star_rating);
    
    // Timestamp
    const date = new Date(results.timestamp);
    const timeStr = date.toLocaleString();
    document.getElementById('resultTime').textContent = `Graded on ${timeStr}`;
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// ============================================
// Status & History
// ============================================

async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        
        // Update teacher status
        const teacherStatus = document.getElementById('teacherStatus');
        if (status.teacher_exists) {
            teacherStatus.innerHTML = `
                <span class="status-indicator ready"></span>
                Teacher Reference: ${status.teacher_frames} frames
            `;
            // Enable student button
            enableStudentSection();
        }
        
        // Update student status
        const studentStatus = document.getElementById('studentStatus');
        studentStatus.innerHTML = `
            <span class="status-indicator ${status.students_graded > 0 ? 'ready' : 'waiting'}"></span>
            Students Graded: ${status.students_graded}
        `;
        
    } catch (error) {
        console.error('Error loading status:', error);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const history = await response.json();
        
        const historyList = document.getElementById('historyList');
        
        if (history.length === 0) {
            historyList.innerHTML = '<p class="empty-message">No grading history yet</p>';
            return;
        }
        
        // Sort by timestamp (newest first)
        history.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        
        historyList.innerHTML = history.map(item => {
            const date = new Date(item.timestamp);
            const timeStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit'
            });
            
            const scoreColor = item.overall_score >= 80 ? '#10b981' : 
                              item.overall_score >= 70 ? '#3b82f6' :
                              item.overall_score >= 60 ? '#f59e0b' : '#ef4444';
            
            return `
                <div class="history-item">
                    <div class="history-item-info">
                        <div class="history-item-name">${item.student_name}</div>
                        <div class="history-item-details">
                            ${item.frames} frames • Arms: ${item.arm_score} • Legs: ${item.leg_score}
                        </div>
                    </div>
                    <div class="history-item-score">
                        <div class="score" style="color: ${scoreColor}">${item.overall_score}</div>
                        <div class="time">${timeStr}</div>
                    </div>
                </div>
            `;
        }).join('');
        
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

async function clearHistory() {
    if (confirm('Are you sure you want to clear all history?')) {
        try {
            const response = await fetch('/api/clear-history', { method: 'POST' });
            const data = await response.json();

            if (!response.ok) {
                showError(data.error || 'Error clearing history');
                return;
            }

            const historyList = document.getElementById('historyList');
            historyList.innerHTML = '<p class="empty-message">No grading history yet</p>';
            showSuccess(`History cleared. Deleted ${data.deleted_files || 0} files.`);

            loadHistory();
            loadStatus();
            
        } catch (error) {
            showError('Error clearing history');
        }
    }
}

// ============================================
// UI State Management
// ============================================

function enableStudentSection() {
    studentBtn.classList.remove('disabled');
    studentBtn.disabled = false;
    document.querySelector('.student-section .info-message')?.remove();
}

function resetStudentSection() {
    studentFile.value = '';
    studentName.value = '';
    resultsSection.classList.add('hidden');
    studentUpload.scrollIntoView({ behavior: 'smooth' });
}

function showLoading(show) {
    if (show) {
        loadingSpinner.classList.remove('hidden');
    } else {
        loadingSpinner.classList.add('hidden');
    }
}

// ============================================
// Notifications
// ============================================

function showError(message) {
    const modal = document.getElementById('errorModal');
    document.getElementById('errorMessage').textContent = message;
    modal.classList.remove('hidden');
}

function showSuccess(message) {
    const modal = document.getElementById('successModal');
    document.getElementById('successMessage').textContent = message;
    modal.classList.remove('hidden');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.add('hidden');
}

// Close modals when clicking outside
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });
});

// ============================================
// Format Number Helper
// ============================================

function formatNumber(num) {
    return num.toFixed(2);
}
