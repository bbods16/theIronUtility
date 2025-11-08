// formChecker.js - Analyzes pose landmarks to provide squat form feedback
const formChecker = (function(){
  let mlModel = null; // Placeholder for the loaded ML model (TensorFlow.js or ONNX.js)
  let repCount = 0;
  let currentRepStage = 'initial'; // 'initial', 'descending', 'bottom', 'ascending', 'top'
  let repStarted = false;
  let repCompletedThisFrame = false;

  // Configuration for squat analysis (angles, thresholds)
  const SQUAT_THRESHOLDS = {
    KNEE_ANGLE_BOTTOM: 80,  // Angle at the bottom of the squat (e.g., < 80 degrees)
    HIP_ANGLE_BOTTOM: 90,   // Angle at the bottom of the squat (e.g., < 90 degrees)
    MIN_REP_DEPTH: 0.6,     // Ratio of hip_y / knee_y for depth check
    KNEE_VALGUS_THRESHOLD: 0.1, // Threshold for knee caving (e.g., distance between knees vs ankles)
    SPINAL_FLEXION_THRESHOLD: 0.1 // Threshold for back rounding
  };

  // Map integer class IDs from ML model to human-readable error strings
  const ERROR_LABELS = {
    0: "good_rep",
    1: "KNEE_VALGUS", // Knees caving in
    2: "NOT_DEEP_ENOUGH",
    3: "BUTT_WINK", // Posterior pelvic tilt
    4: "SPINAL_FLEXION" // Back rounding
  };

  // --- Helper Functions for Keypoint Processing ---
  // These will be more robustly implemented once the exact keypoint format from MediaPipe is confirmed
  function getKeypoint(landmarks, index) {
    if (!landmarks || !landmarks[index]) return null;
    // Assuming landmarks are in normalized [0,1] coordinates or similar
    return landmarks[index]; // {x, y, z, visibility}
  }

  function calculateAngle(p1, p2, p3) {
    // Calculates angle between three points (p2 is the vertex)
    if (!p1 || !p2 || !p3) return null;
    const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
    const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
    const dot = v1.x * v2.x + v1.y * v2.y;
    const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
    const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
    if (mag1 === 0 || mag2 === 0) return null;
    const angleRad = Math.acos(dot / (mag1 * mag2));
    return angleRad * 180 / Math.PI; // Convert to degrees
  }

  function normalizeKeypoints(landmarks) {
    // This function should mirror the normalization done in the Python training pipeline
    // For now, a simple placeholder.
    if (!landmarks || landmarks.length === 0) return null;

    // Example: Center around hip and scale by torso length
    const leftHip = getKeypoint(landmarks, 11);
    const rightHip = getKeypoint(landmarks, 12);
    const leftShoulder = getKeypoint(landmarks, 23);
    const rightShoulder = getKeypoint(landmarks, 24);

    if (!leftHip || !rightHip || !leftShoulder || !rightShoulder) return null;

    const hipCenter = { x: (leftHip.x + rightHip.x) / 2, y: (leftHip.y + rightHip.y) / 2, z: (leftHip.z + rightHip.z) / 2 };
    const shoulderCenter = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2, z: (leftShoulder.z + rightShoulder.z) / 2 };

    const scaleFactor = Math.sqrt(
      Math.pow(hipCenter.x - shoulderCenter.x, 2) +
      Math.pow(hipCenter.y - shoulderCenter.y, 2) +
      Math.pow(hipCenter.z - shoulderCenter.z, 2)
    );

    if (scaleFactor === 0) return null;

    const normalized = landmarks.map(kp => ({
      x: (kp.x - hipCenter.x) / scaleFactor,
      y: (kp.y - hipCenter.y) / scaleFactor,
      z: (kp.z - hipCenter.z) / scaleFactor,
      visibility: kp.visibility
    }));

    // Flatten to a 1D array of features (x, y, z for each keypoint)
    return normalized.flatMap(kp => [kp.x, kp.y, kp.z]); // Ignoring visibility for now
  }

  // --- Rep Counting Logic (Heuristic for now, ML model will refine) ---
  function updateRepStatus(landmarks) {
    repCompletedThisFrame = false;
    if (!landmarks) return;

    const hip = getKeypoint(landmarks, 24); // Assuming right hip for simplicity
    const knee = getKeypoint(landmarks, 26); // Assuming right knee
    const ankle = getKeypoint(landmarks, 28); // Assuming right ankle

    if (!hip || !knee || !ankle) return;

    // Simple heuristic: hip_y relative to knee_y to detect squat depth
    const depthRatio = hip.y / knee.y; // Lower y means deeper in normalized coords

    if (currentRepStage === 'initial' && depthRatio > SQUAT_THRESHOLDS.MIN_REP_DEPTH + 0.1) {
      // User is standing, ready to start
      repStarted = false;
    } else if (!repStarted && depthRatio < SQUAT_THRESHOLDS.MIN_REP_DEPTH) {
      // Started descending
      repStarted = true;
      currentRepStage = 'descending';
    } else if (repStarted && currentRepStage === 'descending' && depthRatio >= SQUAT_THRESHOLDS.MIN_REP_DEPTH) {
      // Reached bottom (or passed it)
      currentRepStage = 'bottom';
    } else if (repStarted && currentRepStage === 'bottom' && depthRatio > SQUAT_THRESHOLDS.MIN_REP_DEPTH + 0.05) {
      // Started ascending
      currentRepStage = 'ascending';
    } else if (repStarted && currentRepStage === 'ascending' && depthRatio > SQUAT_THRESHOLDS.MIN_REP_DEPTH + 0.1) {
      // Rep completed (returned to top)
      repCount++;
      repStarted = false;
      repCompletedThisFrame = true;
      currentRepStage = 'initial';
    }
  }

  // --- ML Model Inference ---
  async function loadMlModel() {
    // TODO: Load the exported TensorFlow.js or ONNX.js model here
    // Example for TensorFlow.js:
    // mlModel = await tf.loadGraphModel('path/to/your/model.json');
    // Example for ONNX.js:
    // mlModel = new onnx.InferenceSession();
    // await mlModel.loadModel('path/to/your/model.onnx');
    console.log("ML Model loading placeholder...");
    // For now, mlModel remains null, and we'll rely on heuristics.
  }

  async function runMlInference(processedKeypoints) {
    if (!mlModel) {
      console.warn("ML Model not loaded. Skipping inference.");
      return null;
    }
    // TODO: Prepare input tensor, run inference, and process output
    // Example for TensorFlow.js:
    // const inputTensor = tf.tensor(processedKeypoints, [1, SEQUENCE_LENGTH, NUM_FEATURES]);
    // const predictions = mlModel.predict(inputTensor);
    // const outputArray = predictions.array(); // Get raw probabilities
    // return outputArray;
    return null; // Return null if model not loaded or for placeholder
  }

  // --- Public Interface ---
  return {
    init: async function() {
      await loadMlModel();
      console.log("FormChecker initialized. ML Model status:", mlModel ? "Loaded" : "Not Loaded (using heuristics)");
    },

    analyze: async function(landmarks) {
      repCompletedThisFrame = false; // Reset for current frame

      if (!landmarks) {
        return { repCompleted: false, errors: [] };
      }

      // 1. Pre-process landmarks (normalization, flattening)
      const processedKeypoints = normalizeKeypoints(landmarks);
      if (!processedKeypoints) {
        return { repCompleted: false, errors: [] };
      }

      // 2. Run ML model inference (if loaded)
      let mlPredictions = null;
      if (mlModel) {
        mlPredictions = await runMlInference(processedKeypoints);
      }

      // 3. Interpret ML predictions or use heuristics
      let errors = [];
      let predictedClassId = null;
      let confidence = 0;

      if (mlPredictions) {
        // Assuming mlPredictions is an array of probabilities for each class
        predictedClassId = mlPredictions.indexOf(Math.max(...mlPredictions));
        confidence = mlPredictions[predictedClassId];

        // Apply confidence threshold from Python config (e.g., 0.90)
        if (confidence > 0.90 && predictedClassId !== ERROR_LABELS.good_rep) {
          errors.push(ERROR_LABELS[predictedClassId]);
        }
      } else {
        // Fallback to heuristic-based error checking if ML model not loaded
        // This is a simplified example and needs proper implementation
        const kneeAngle = calculateAngle(getKeypoint(landmarks, 24), getKeypoint(landmarks, 26), getKeypoint(landmarks, 28));
        if (kneeAngle && kneeAngle < SQUAT_THRESHOLDS.KNEE_ANGLE_BOTTOM) {
          // This is just a depth check, not a specific error like KNEE_VALGUS
          // For a real heuristic, we'd need more complex geometry checks.
        }
        // Example heuristic for KNEE_VALGUS (knees caving in)
        const leftKnee = getKeypoint(landmarks, 25);
        const rightKnee = getKeypoint(landmarks, 26);
        const leftAnkle = getKeypoint(landmarks, 27);
        const rightAnkle = getKeypoint(landmarks, 28);

        if (leftKnee && rightKnee && leftAnkle && rightAnkle) {
          const kneeDist = Math.abs(leftKnee.x - rightKnee.x);
          const ankleDist = Math.abs(leftAnkle.x - rightAnkle.x);
          // If knees are significantly closer than ankles (simplified)
          if (kneeDist < ankleDist * (1 - SQUAT_THRESHOLDS.KNEE_VALGUS_THRESHOLD)) {
            errors.push("KNEE_VALGUS");
          }
        }
      }

      // 4. Update rep status (can be driven by ML model or heuristics)
      updateRepStatus(landmarks); // This heuristic is still active for rep counting

      return {
        repCompleted: repCompletedThisFrame,
        errors: errors,
        currentRepCount: repCount,
        // Add other relevant analysis data here, e.g., confidence, current stage
      };
    },

    getRepCount: function() {
      return repCount;
    },

    reset: function() {
      repCount = 0;
      currentRepStage = 'initial';
      repStarted = false;
      repCompletedThisFrame = false;
      // Optionally unload ML model or reset its state if needed
    }
  };
})();

// Initialize FormChecker when the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  formChecker.init();
});
