// ============================================
// IRON UTILITY MONOPOLY - INTERACTIVE APP
// Squat Form Analysis with Monopoly Game Mechanics!
// ============================================

class MonopolySquatAnalyzer {
    constructor() {
        this.videoPlayer = document.getElementById('videoPlayer');
        this.analysisCanvas = document.getElementById('analysisCanvas');
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('fileInput');
        this.resultsBoard = document.getElementById('resultsBoard');
        this.webcamBtn = document.getElementById('webcamBtn');

        // Game stats
        this.totalSessions = 0;
        this.perfectReps = 0;
        this.housesBuilt = 0;
        this.currentScore = 0;

        this.initializeEventListeners();
        this.playMonopolyMusic();
        this.startGameAnimations();
    }

    initializeEventListeners() {
        // File upload handlers
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));

        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadZone.classList.add('shake');
        });

        this.uploadZone.addEventListener('dragleave', () => {
            this.uploadZone.classList.remove('shake');
        });

        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('shake');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.processVideo(files[0]);
            }
        });

        // Webcam button
        this.webcamBtn.addEventListener('click', () => this.startWebcamAnalysis());

        // Power buttons
        const powerButtons = document.querySelectorAll('.power-button');
        powerButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                this.playSound('dice-roll');
                this.showMonopolyMessage(btn.textContent);
            });
        });
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.processVideo(file);
        }
    }

    processVideo(file) {
        // Hide upload zone, show video player
        this.uploadZone.style.display = 'none';
        this.videoPlayer.style.display = 'block';

        // Create video URL
        const videoURL = URL.createObjectURL(file);
        this.videoPlayer.src = videoURL;
        this.videoPlayer.play();

        // Show loading animation
        this.showMonopolyMessage('🎲 Rolling the dice... Analyzing your squat form!', 'info');

        // Simulate API call to backend
        setTimeout(() => {
            this.performAnalysis();
        }, 2000);
    }

    async performAnalysis() {
        try {
            // In a real app, this would call the Flask backend
            // For now, we'll simulate the analysis with mock data

            this.showMonopolyMessage('🏗️ Building your form analysis empire...', 'info');

            // Simulate processing time
            await this.sleep(3000);

            // Generate mock results
            const results = this.generateMockResults();

            // Display results with Monopoly flair!
            this.displayResults(results);

            // Update game stats
            this.updateGameStats(results);

            // Play success sound
            this.playSound('cash-register');

            this.showMonopolyMessage('🎉 Analysis Complete! Collect your form report!', 'success');

        } catch (error) {
            this.showMonopolyMessage('⚠️ Bankrupt! Analysis failed. Please try again.', 'error');
            console.error('Analysis error:', error);
        }
    }

    generateMockResults() {
        return {
            good_rep: Math.floor(Math.random() * 20),
            knee_valgus: Math.floor(Math.random() * 5),
            not_deep_enough: Math.floor(Math.random() * 8),
            butt_wink: Math.floor(Math.random() * 6),
            spinal_flexion: Math.floor(Math.random() * 4),
            confidence: 0.85 + Math.random() * 0.15
        };
    }

    displayResults(results) {
        // Show results board
        this.resultsBoard.style.display = 'block';
        this.resultsBoard.classList.add('bounce-in');

        // Calculate total reps
        const totalReps = results.good_rep + results.knee_valgus +
                         results.not_deep_enough + results.butt_wink +
                         results.spinal_flexion;

        // Update result values with animation
        this.animateValue('goodReps', 0, results.good_rep, 1500);
        this.animateValue('kneeValgus', 0, results.knee_valgus, 1500);
        this.animateValue('depthIssues', 0, results.not_deep_enough, 1500);
        this.animateValue('buttWink', 0, results.butt_wink, 1500);
        this.animateValue('spinalFlexion', 0, results.spinal_flexion, 1500);

        // Update progress bars
        setTimeout(() => {
            this.updateProgressBar('goodProgress', (results.good_rep / totalReps) * 100);
            this.updateProgressBar('valgusProgress', (results.knee_valgus / totalReps) * 100);
            this.updateProgressBar('depthProgress', (results.not_deep_enough / totalReps) * 100);
            this.updateProgressBar('winkProgress', (results.butt_wink / totalReps) * 100);
            this.updateProgressBar('spinalProgress', (results.spinal_flexion / totalReps) * 100);
            this.updateProgressBar('confidenceProgress', results.confidence * 100);
        }, 500);

        // Update confidence
        this.animateValue('confidence', 0, Math.round(results.confidence * 100), 1500, '%');

        // Calculate and show score
        const score = this.calculateMonopolyScore(results);
        this.animateValue('scoreAmount', 0, score, 2000, '$');

        // Scroll to results
        setTimeout(() => {
            this.resultsBoard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 500);
    }

    calculateMonopolyScore(results) {
        // Perfect reps are worth $200 (like passing GO!)
        let score = results.good_rep * 200;

        // Deduct for form issues (like paying rent!)
        score -= results.knee_valgus * 50;
        score -= results.not_deep_enough * 30;
        score -= results.butt_wink * 40;
        score -= results.spinal_flexion * 60;

        // Confidence bonus
        score += Math.round(results.confidence * 100);

        return Math.max(0, score);
    }

    updateGameStats(results) {
        this.totalSessions++;
        this.perfectReps += results.good_rep;

        // Build houses based on good reps (1 house per 5 good reps)
        const newHouses = Math.floor(results.good_rep / 5);
        this.housesBuilt += newHouses;

        // Update UI
        this.animateValue('totalSessions', this.totalSessions - 1, this.totalSessions, 1000);
        this.animateValue('perfectReps', this.perfectReps - results.good_rep, this.perfectReps, 1000);

        // Update houses with emoji
        const housesElement = document.getElementById('housesBuilt');
        let houseEmojis = '🏠 '.repeat(Math.min(this.housesBuilt, 5));
        if (this.housesBuilt > 5) {
            houseEmojis = '🏨 ' + (this.housesBuilt - 5);
        }
        housesElement.textContent = houseEmojis;

        // Trigger celebration if milestone reached
        if (newHouses > 0) {
            this.celebrateHouse(newHouses);
        }
    }

    celebrateHouse(count) {
        const messages = [
            `🏠 You built ${count} house${count > 1 ? 's' : ''}! Rent is going up!`,
            `🎊 Monopoly! You're dominating the squat game!`,
            `🏆 Upgrade to a hotel soon! Keep crushing those squats!`
        ];
        this.showMonopolyMessage(messages[Math.floor(Math.random() * messages.length)], 'success');
        this.createConfetti();
    }

    animateValue(elementId, start, end, duration, suffix = '') {
        const element = document.getElementById(elementId);
        const range = end - start;
        const increment = range / (duration / 16);
        let current = start;

        const timer = setInterval(() => {
            current += increment;
            if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                current = end;
                clearInterval(timer);
            }
            element.textContent = Math.round(current) + suffix;
        }, 16);
    }

    updateProgressBar(elementId, percentage) {
        const progressBar = document.getElementById(elementId);
        progressBar.style.width = percentage + '%';
    }

    async startWebcamAnalysis() {
        try {
            // Lower resolution for faster processing
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30, max: 30 }
                }
            });

            this.uploadZone.style.display = 'none';
            this.videoPlayer.style.display = 'block';
            this.videoPlayer.srcObject = stream;
            this.videoPlayer.play();

            this.showMonopolyMessage('📹 Webcam activated! Start your squats!', 'info');

            // Start real-time analysis
            this.startRealtimeAnalysis();

        } catch (error) {
            this.showMonopolyMessage('⚠️ Could not access webcam. Please check permissions!', 'error');
            console.error('Webcam error:', error);
        }
    }

    startRealtimeAnalysis() {
        this.showMonopolyMessage('🔥 Real-time analysis active! Form check every rep!', 'info');

        // Create canvas for frame capture
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // Initialize squat tracking display
        this.initializeRealtimeDisplay();

        // Performance optimization: process frames at lower FPS for backend processing
        let isProcessing = false;
        let lastFrameTime = 0;
        const targetFPS = 10; // Lower to 10fps to match backend processing capability
        const frameInterval = 1000 / targetFPS;

        // Use smaller resolution for processing
        const processingWidth = 320;  // Much smaller for faster processing
        const processingHeight = 240;

        const processFrame = async (currentTime) => {
            // Continue animation loop
            this.analysisRAF = requestAnimationFrame(processFrame);

            // Throttle to target FPS
            const elapsed = currentTime - lastFrameTime;
            if (elapsed < frameInterval) {
                return;
            }

            lastFrameTime = currentTime - (elapsed % frameInterval);

            // Skip if still processing previous frame
            if (isProcessing || this.videoPlayer.paused || this.videoPlayer.ended) {
                return;
            }

            isProcessing = true;

            try {
                // Capture current frame at reduced resolution
                canvas.width = processingWidth;
                canvas.height = processingHeight;
                ctx.drawImage(this.videoPlayer, 0, 0, processingWidth, processingHeight);

                // Convert to base64 with very low quality for faster transmission
                const frameData = canvas.toDataURL('image/jpeg', 0.5);

                // Send to backend for analysis
                const response = await fetch('/api/webcam/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ frame: frameData })
                });

                if (response.ok) {
                    const data = await response.json();
                    this.updateRealtimeDisplay(data);

                    // Check if rep was completed
                    if (data.rep_completed) {
                        this.onRepCompleted(data);
                    }
                }
            } catch (error) {
                console.error('Analysis error:', error);
            } finally {
                isProcessing = false;
            }
        };

        // Start animation loop
        this.analysisRAF = requestAnimationFrame(processFrame);
    }

    initializeRealtimeDisplay() {
        // Create real-time stats overlay
        const overlay = document.createElement('div');
        overlay.id = 'realtimeOverlay';
        overlay.style.cssText = `
            position: fixed;
            top: 120px;
            left: 30px;
            background: rgba(0, 0, 0, 0.85);
            border: 5px solid var(--monopoly-green);
            border-radius: 20px;
            padding: 25px;
            z-index: 9000;
            color: white;
            font-family: 'Fredoka One', cursive;
            min-width: 300px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.5);
        `;

        overlay.innerHTML = `
            <h2 style="margin: 0 0 20px 0; color: var(--monopoly-gold); text-shadow: 2px 2px 4px #000;">
                🎩 LIVE SQUAT TRACKER 🎩
            </h2>
            <div style="font-size: 48px; text-align: center; margin: 20px 0; color: var(--monopoly-green); text-shadow: 3px 3px 6px #000;">
                <span id="liveSquatCount">0</span>
            </div>
            <div style="font-size: 18px; margin: 10px 0;">
                Stage: <span id="liveStage" style="color: var(--monopoly-yellow);">READY</span>
            </div>
            <div style="font-size: 18px; margin: 10px 0;">
                Angle: <span id="liveAngle" style="color: var(--monopoly-blue);">0°</span>
            </div>
            <div style="font-size: 18px; margin: 10px 0;">
                Good Reps: <span id="liveGoodReps" style="color: var(--monopoly-green);">0</span> / <span id="liveTotalReps">0</span>
            </div>
            <div style="font-size: 14px; margin: 10px 0; color: #888;">
                Processing: <span id="liveProcessingTime" style="color: var(--monopoly-blue);">0ms</span>
            </div>
            <div id="liveFormErrors" style="margin-top: 15px; padding: 10px; background: rgba(226,35,26,0.2); border-radius: 10px; min-height: 40px;">
                <strong style="color: var(--monopoly-red);">Form Errors:</strong>
                <div id="errorList" style="margin-top: 8px; font-size: 14px;">None</div>
            </div>
            <div style="margin-top: 20px; text-align: center;">
                <button id="resetTrackerBtn" style="
                    background: var(--monopoly-red);
                    color: white;
                    border: 3px solid white;
                    border-radius: 10px;
                    padding: 10px 20px;
                    font-family: 'Fredoka One', cursive;
                    font-size: 16px;
                    cursor: pointer;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                ">
                    🔄 RESET
                </button>
            </div>
        `;

        document.body.appendChild(overlay);

        // Add reset button handler
        document.getElementById('resetTrackerBtn').addEventListener('click', () => this.resetTracker());
    }

    updateRealtimeDisplay(data) {
        // Update squat count
        document.getElementById('liveSquatCount').textContent = data.squat_count;

        // Update stage
        const stageElement = document.getElementById('liveStage');
        stageElement.textContent = data.stage;
        stageElement.style.color = data.stage === 'UP' ? 'var(--monopoly-green)' :
                                    data.stage === 'DOWN' ? 'var(--monopoly-orange)' :
                                    'var(--monopoly-yellow)';

        // Update angle
        document.getElementById('liveAngle').textContent = data.current_angle + '°';

        // Update good reps
        document.getElementById('liveGoodReps').textContent = data.good_reps;
        document.getElementById('liveTotalReps').textContent = data.total_reps;

        // Update processing time
        if (data.processing_time_ms) {
            const processingTimeEl = document.getElementById('liveProcessingTime');
            processingTimeEl.textContent = data.processing_time_ms + 'ms';
            // Color code based on performance
            if (data.processing_time_ms < 50) {
                processingTimeEl.style.color = 'var(--monopoly-green)';
            } else if (data.processing_time_ms < 100) {
                processingTimeEl.style.color = 'var(--monopoly-orange)';
            } else {
                processingTimeEl.style.color = 'var(--monopoly-red)';
            }
        }

        // Update form errors with detailed feedback
        const errorList = document.getElementById('errorList');
        if (data.feedback_messages && data.feedback_messages.length > 0) {
            errorList.innerHTML = data.feedback_messages.map(message =>
                `<div style="color: #FFF; margin: 5px 0; font-size: 13px; line-height: 1.4;">${message}</div>`
            ).join('');
        } else if (data.form_errors && data.form_errors.length > 0) {
            // Fallback to basic error display if no feedback messages
            errorList.innerHTML = data.form_errors.map(error =>
                `<div style="color: #FFF; margin: 5px 0;">⚠️ ${error.replace(/_/g, ' ')}</div>`
            ).join('');
        } else {
            errorList.innerHTML = '<div style="color: var(--monopoly-green);">✓ Perfect Form!</div>';
        }

        // Display annotated frame with skeleton overlay
        if (data.annotated_frame) {
            this.videoPlayer.style.display = 'none';

            // Create or update skeleton overlay image
            let skeletonImg = document.getElementById('skeletonOverlay');
            if (!skeletonImg) {
                skeletonImg = document.createElement('img');
                skeletonImg.id = 'skeletonOverlay';
                skeletonImg.style.cssText = `
                    width: 100%;
                    max-width: 800px;
                    border-radius: 20px;
                    border: 5px solid var(--monopoly-green);
                    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                `;
                this.videoPlayer.parentElement.appendChild(skeletonImg);
            }
            skeletonImg.src = data.annotated_frame;
        }
    }

    onRepCompleted(data) {
        // Check if it was a good rep
        const isGoodRep = !data.form_errors || data.form_errors.length === 0;

        if (isGoodRep) {
            this.showMonopolyMessage('🎉 Perfect Rep! Pass GO and collect $200!', 'success');
            this.createConfetti();
            this.playSound('cash-register');

            // Update stats
            this.perfectReps++;
            this.currentScore += 200;
        } else {
            // Show first feedback message as toast notification
            const feedbackMsg = data.feedback_messages && data.feedback_messages.length > 0
                ? data.feedback_messages[0]
                : '⚠️ Form errors detected! Check the feedback panel!';

            this.showMonopolyMessage(feedbackMsg, 'warning');
            this.playSound('error-buzz');

            // Deduct monopoly money for bad form
            this.currentScore = Math.max(0, this.currentScore - 50);
        }
    }

    async resetTracker() {
        try {
            const response = await fetch('/api/webcam/reset', {
                method: 'POST'
            });

            if (response.ok) {
                this.showMonopolyMessage('🔄 Tracker reset! Start fresh!', 'info');
                document.getElementById('liveSquatCount').textContent = '0';
                document.getElementById('liveGoodReps').textContent = '0';
                document.getElementById('liveTotalReps').textContent = '0';
                document.getElementById('errorList').innerHTML = '<div style="color: var(--monopoly-green);">✓ Ready to squat!</div>';
            }
        } catch (error) {
            console.error('Reset error:', error);
        }
    }

    stopRealtimeAnalysis() {
        // Cancel animation frame
        if (this.analysisRAF) {
            cancelAnimationFrame(this.analysisRAF);
            this.analysisRAF = null;
        }

        // Remove overlay
        const overlay = document.getElementById('realtimeOverlay');
        if (overlay) {
            overlay.remove();
        }

        // Remove skeleton overlay
        const skeletonImg = document.getElementById('skeletonOverlay');
        if (skeletonImg) {
            skeletonImg.remove();
        }

        // Stop video stream
        if (this.videoPlayer.srcObject) {
            this.videoPlayer.srcObject.getTracks().forEach(track => track.stop());
            this.videoPlayer.srcObject = null;
        }
    }

    showMonopolyMessage(message, type = 'info') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `monopoly-toast ${type}`;
        messageDiv.innerHTML = `
            <div class="toast-icon">${this.getToastIcon(type)}</div>
            <div class="toast-message">${message}</div>
        `;

        // Add styles
        messageDiv.style.cssText = `
            position: fixed;
            top: 100px;
            right: 30px;
            background: white;
            border: 4px solid var(--monopoly-dark);
            border-radius: 15px;
            padding: 20px 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            z-index: 10000;
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 18px;
            font-weight: 700;
            animation: slideInRight 0.5s ease, slideOutRight 0.5s ease 3.5s;
        `;

        document.body.appendChild(messageDiv);

        setTimeout(() => {
            messageDiv.remove();
        }, 4000);
    }

    getToastIcon(type) {
        const icons = {
            'info': '💡',
            'success': '🎉',
            'error': '⚠️',
            'warning': '⚡'
        };
        return icons[type] || icons.info;
    }

    createConfetti() {
        const colors = ['#E2231A', '#1FB25A', '#0072BB', '#FEF200', '#F7941D'];
        const confettiCount = 50;

        for (let i = 0; i < confettiCount; i++) {
            const confetti = document.createElement('div');
            confetti.style.cssText = `
                position: fixed;
                width: 10px;
                height: 10px;
                background: ${colors[Math.floor(Math.random() * colors.length)]};
                top: -10px;
                left: ${Math.random() * window.innerWidth}px;
                z-index: 9999;
                animation: confettiFall ${2 + Math.random() * 3}s linear forwards;
                transform: rotate(${Math.random() * 360}deg);
            `;
            document.body.appendChild(confetti);

            setTimeout(() => confetti.remove(), 5000);
        }
    }

    playSound(soundType) {
        // In a real app, you'd play actual sound files
        console.log(`🔊 Playing sound: ${soundType}`);
    }

    playMonopolyMusic() {
        // Background music would go here
        console.log('🎵 Monopoly theme music playing...');
    }

    startGameAnimations() {
        // Add extra floating monopoly tokens
        setInterval(() => {
            this.addFloatingToken();
        }, 3000);

        // Update time-based elements
        this.startClock();
    }

    addFloatingToken() {
        const tokens = ['🎩', '🚗', '🐕', '⛵', '🛴', '💎', '🎸'];
        const token = document.createElement('div');
        token.textContent = tokens[Math.floor(Math.random() * tokens.length)];
        token.style.cssText = `
            position: fixed;
            font-size: 40px;
            left: ${Math.random() * window.innerWidth}px;
            top: ${window.innerHeight}px;
            z-index: 100;
            pointer-events: none;
            animation: floatUp 5s linear forwards;
        `;
        document.body.appendChild(token);

        setTimeout(() => token.remove(), 5000);
    }

    startClock() {
        // Could add a game timer here
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Add CSS animations dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }

    @keyframes confettiFall {
        0% {
            top: -10px;
            transform: translateX(0) rotate(0deg);
        }
        100% {
            top: 100vh;
            transform: translateX(${Math.random() * 200 - 100}px) rotate(720deg);
        }
    }

    @keyframes floatUp {
        0% {
            bottom: -50px;
            opacity: 1;
        }
        100% {
            bottom: 110vh;
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new MonopolySquatAnalyzer();
    console.log('🎲 Iron Utility Monopoly - Ready to build your fitness empire!');

    // Show welcome message
    setTimeout(() => {
        app.showMonopolyMessage('🎩 Welcome to Iron Utility Monopoly! Upload your squat video to begin!', 'info');
    }, 1000);
});
