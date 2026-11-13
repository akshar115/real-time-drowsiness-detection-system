const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('camera-overlay');

const statusBadge = document.getElementById('status-badge');
const earValue = document.getElementById('ear-value');
const circularProgress = document.querySelector('.circular-progress');
const valueContainer = document.querySelector('.value-container');
const eyeScoreBar = document.getElementById('eye-score-bar');
const mouthScoreBar = document.getElementById('mouth-score-bar');
const eyeScoreVal = document.getElementById('eye-score-val');
const mouthScoreVal = document.getElementById('mouth-score-val');

const statDuration = document.getElementById('stat-duration');
const statAlerts = document.getElementById('stat-alerts');
const statYawns = document.getElementById('stat-yawns');

const alarmSound = document.getElementById('alarm-sound');

let drowsyStartTime = null;
let isAlarmPlaying = false;
let isProcessing = false;

// Initialize Webcam
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;
        
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                overlay.style.opacity = '0';
                setTimeout(() => overlay.style.display = 'none', 500);
                resolve(video);
            };
        });
    } catch (e) {
        overlay.innerText = "Error accessing webcam. Please allow permissions.";
        console.error("Webcam access error:", e);
    }
}

function updateUI(data) {
    // 1. Status Badge
    statusBadge.innerText = data.status;
    statusBadge.className = 'badge'; // reset
    if (data.status === 'DROWSY') {
        statusBadge.classList.add('drowsy');
    } else if (data.status === 'YAWN') {
        statusBadge.classList.add('yawn');
    } else {
        statusBadge.classList.add('alert');
    }

    // 2. Alarm Logic
    if (data.status === 'DROWSY') {
        if (!drowsyStartTime) {
            drowsyStartTime = Date.now();
        }
        // Play if > 1 second of continuous DROWSY status
        if (Date.now() - drowsyStartTime > 1000 && !isAlarmPlaying) {
            // Need user interaction first to play audio in most browsers
            alarmSound.play().catch(e => console.log("Audio play prevented by browser policy", e));
            isAlarmPlaying = true;
        }
    } else {
        drowsyStartTime = null;
        if (isAlarmPlaying) {
            alarmSound.pause();
            alarmSound.currentTime = 0;
            isAlarmPlaying = false;
        }
    }

    // 3. EAR Value
    earValue.innerText = data.ear.toFixed(3);
    
    // 4. Circular Progress (Drowsiness Level)
    const pct = data.drowsiness_pct;
    valueContainer.textContent = `${pct}%`;
    
    let color = 'var(--alert-green)';
    if (pct > 50 && pct < 80) color = 'var(--yawn-orange)';
    if (pct >= 80) color = 'var(--drowsy-red)';
    
    circularProgress.style.background = `conic-gradient(${color} ${pct * 3.6}deg, #2a2d3e 0deg)`;

    // 5. Confidence Bars
    let eyePct = Math.round(data.eye_score * 100);
    let mouthPct = Math.round(data.mouth_score * 100);
    
    // Filter out baseline mathematical noise so it looks perfectly empty when resting
    if (eyePct < 25) eyePct = 0;
    if (mouthPct < 25) mouthPct = 0;
    
    eyeScoreBar.style.width = `${eyePct}%`;
    eyeScoreVal.innerText = `${eyePct}%`;
    
    mouthScoreBar.style.width = `${mouthPct}%`;
    mouthScoreVal.innerText = `${mouthPct}%`;
    
    // Change bar colors dynamically based on severity
    eyeScoreBar.style.background = eyePct > 70 ? 'linear-gradient(90deg, #ef4444, #dc2626)' : 'linear-gradient(90deg, #60a5fa, #3b82f6)';
}

async function sendFrame() {
    if (isProcessing) return;
    if (overlay.style.display !== 'none') return; // Wait until camera is initialized
    
    isProcessing = true;
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg', 0.8);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL })
        });
        const result = await response.json();
        
        if (result.error) {
            console.error("Backend Error:", result.error);
        } else {
            updateUI(result);
        }
    } catch (err) {
        console.error("Fetch Error:", err);
    } finally {
        isProcessing = false;
    }
}

async function fetchStats() {
    try {
        const response = await fetch('/stats');
        const stats = await response.json();
        statDuration.innerText = stats.session_duration;
        statAlerts.innerText = stats.total_alerts;
        statYawns.innerText = stats.total_yawns;
    } catch (err) {
        console.error("Error fetching stats:", err);
    }
}

async function start() {
    // Need user gesture to allow audio play, attach click event listener anywhere on screen initially
    document.body.addEventListener('click', () => {
        // Just load audio to unlock it
        alarmSound.load();
    }, { once: true });

    await setupCamera();
    
    // Capture frame every 300ms
    setInterval(sendFrame, 300);
    
    // Fetch stats every 1 second
    setInterval(fetchStats, 1000);
}

window.onload = start;

const modelSelect = document.getElementById('model-select');
modelSelect.addEventListener('change', async (e) => {
    try {
        await fetch('/set_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_type: e.target.value })
        });
    } catch(err) {
        console.error("Error switching model:", err);
    }
});

const imageUpload = document.getElementById('image-upload');
imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = async (event) => {
        const base64Image = event.target.result;
        
        isProcessing = true; // Pause webcam inference
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image })
            });
            const data = await response.json();
            if (!data.error) {
                updateUI(data);
                
                // Show uploaded image on video wrapper
                const videoWrapper = document.querySelector('.video-wrapper');
                let preview = document.getElementById('uploaded-preview');
                if (!preview) {
                    preview = document.createElement('img');
                    preview.id = 'uploaded-preview';
                    preview.style.position = 'absolute';
                    preview.style.top = '0';
                    preview.style.left = '0';
                    preview.style.width = '100%';
                    preview.style.height = '100%';
                    preview.style.objectFit = 'cover';
                    preview.style.zIndex = '5';
                    videoWrapper.appendChild(preview);
                }
                preview.src = base64Image;
                preview.style.display = 'block';
                
                // Show clear button
                document.getElementById('clear-image').style.display = 'inline-block';
            }
        } catch(err) {
            console.error(err);
            isProcessing = false;
        }
    };
    reader.readAsDataURL(file);
});

const clearImageBtn = document.getElementById('clear-image');
if (clearImageBtn) {
    clearImageBtn.addEventListener('click', () => {
        let preview = document.getElementById('uploaded-preview');
        if (preview) preview.style.display = 'none';
        document.getElementById('image-upload').value = '';
        clearImageBtn.style.display = 'none';
        isProcessing = false; // Resume webcam
    });
}
