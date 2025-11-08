// media.js - Manages camera and main loop
(function(){
  const videoElement = document.getElementById('videoElement');
  const outputCanvas = document.getElementById('outputCanvas');
  const startButton = document.getElementById('startButton');
  const introScreen = document.getElementById('introScreen');
  const skipIntroBtn = document.getElementById('skipIntro');
  let animationFrameId = null;
  let cameraStream = null;
  let workoutActive = false;

  // Preload audio elements for sound effects
  const diceSound = new Audio('assets/sounds/dice-roll.mp3');
  const startSound = new Audio('assets/sounds/game-start.mp3');  // e.g., a small fanfare or dice roll
  const endSound = new Audio('assets/sounds/cash-register.mp3'); // sound for workout finish (cash register "cha-ching")

  // Handle intro skip
  skipIntroBtn.addEventListener('click', () => {
    introScreen.classList.add('fade-out');
    // Play a quick dice sound when skipping intro
    diceSound.play().catch(e => console.warn('Dice sound play error:', e));
  });

  // Start button event: begin video + hide intro if showing
  startButton.addEventListener('click', async () => {
    if (workoutActive) return;
    workoutActive = true;
    // Hide intro screen if still visible
    if (introScreen.style.display !== 'none') {
      introScreen.classList.add('fade-out');
      setTimeout(() => { introScreen.style.display = 'none'; }, 500);
    }
    try {
      await startWebcam();
      startSound.play().catch(e => {});  // play start sound (dice roll or similar)
      UIManager.onWorkoutStart();       // inform UI manager (e.g., reset score, animations)
      predictWebcam();                  // begin the main loop
    } catch(err) {
      alert('Unable to access camera: ' + err);
      workoutActive = false;
    }
  });

  async function startWebcam() {
    // Request video stream from camera
    const constraints = { video: true, audio: false };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoElement.srcObject = stream;
    cameraStream = stream;
    return new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        videoElement.play();
        resolve();
      };
    });
  }

  function predictWebcam() {
    // This function runs on each animation frame
    // It calls MediaPipe pose estimation and then passes results to formChecker and UIManager
    const ctx = outputCanvas.getContext('2d');
    // Example: copy video frame to canvas (assuming videoElement is hidden but playing)
    ctx.drawImage(videoElement, 0, 0, outputCanvas.width, outputCanvas.height);

    // TODO: Integrate MediaPipe Pose (poseLandmarker) to get landmarks.
    // For demonstration, we'll simulate with a dummy landmarks or skip if no AI:
    let landmarks = null;
    // if (poseLandmarker) { landmarks = poseLandmarker.detectForVideo(...); }

    // Pass landmarks to form checker (if null, formChecker will handle gracefully)
    const analysis = formChecker.analyze(landmarks);
    // Update UI based on analysis
    UIManager.update(analysis);

    // Loop again
    animationFrameId = requestAnimationFrame(predictWebcam);
  }

  function stopWorkout() {
    // Call this when workout session is complete (user finished or manually stopped)
    if (!workoutActive) return;
    workoutActive = false;
    cancelAnimationFrame(animationFrameId);
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
    }
    // Trigger end-of-workout UI events
    endSound.play().catch(e => {});
    UIManager.onWorkoutEnd();           // e.g., show "Passed GO" celebration
    aiCore.fetchCoachingReport();      // Fetch AI coaching feedback (Claude backend)
  }

  // Expose stopWorkout globally so it can be called elsewhere (e.g., by UI or automatically)
  window.stopWorkout = stopWorkout;
})();
