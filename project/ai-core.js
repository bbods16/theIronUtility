// ai-core.js - Main orchestrator for The Iron Utility app (imports all modules and runs the loop)

import { setupCamera, stopCamera } from './media.js';
import { calculateAngle } from './utils.js';
import { checkErrors } from './formChecker.js';
import { RepCounter } from './RepCounter.js';
import { ErrorDetector } from './ErrorDetector.js';
import * as UIManager from './UIManager.js';
import * as api from './api.js';

// Initialize pose model variables
let poseLandmarker = null;
let lastVideoTime = -1;
let isActive = false;
let currentExercise = 'squat'; // Default exercise

// ... (state and HTML elements)
const exerciseSelect = document.getElementById('exerciseSelect');

// ... (loadPoseModel function)

// Main prediction loop
async function processFrame() {
  // ... (frame processing logic)
  if (result.landmarks && result.landmarks.length > 0) {
      const landmarks = result.landmarks[0];
      drawSkeleton(landmarks);

      const errors = checkErrors(landmarks, currentExercise);
      errorDetector.recordFrameErrors(errors);

      // ... (immediate feedback logic)

      const repCompleted = repCounter.update(landmarks, currentExercise);
      if (repCompleted) {
        // ... (rep completion logic)
      } else {
        // ... (no rep completed logic)
      }
    }
    // ... (rest of frame processing)
}

// ... (drawSkeleton, startButton, finishButton, coachButton event listeners)

// Event listener for exercise selection
exerciseSelect.addEventListener('change', (event) => {
    currentExercise = event.target.value;
    repCounter.count = 0; // Reset rep count when exercise changes
    UIManager.updateRepCount(0);
    UIManager.updateScore(0);
});
