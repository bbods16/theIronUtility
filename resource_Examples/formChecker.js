// formChecker.js - Analyzes pose landmarks to provide squat form feedback
const formChecker = (function(){
  let mlModelSession = null; // ONNX Runtime InferenceSession
  let repCount = 0;
  let currentRepStage = 'initial'; // 'initial', 'descending', 'bottom', 'ascending', 'top'
  let repStarted = false;
  let repCompletedThisFrame = false;
  let keypointSequence = []; // Stores a sequence of processed keypoints for the ML model
  const SEQUENCE_LENGTH = 100; // Must match the sequence_length used in Python training
  const NUM_KEYPOINTS = 33; // MediaPipe Pose
  const NUM_COORDS = 3; // x, y, z

  // Configuration for squat analysis (angles, thresholds) - mostly for rep counting now
  const SQUAT_THRESHOLDS = {
    MIN_REP_DEPTH: 0.6,     // Ratio of hip_y / knee_y for depth check
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
  function getKeypoint(landmarks, index) {
    if (!landmarks || !landmarks[index]) return null;
    // MediaPipe landmarks are typically {x, y, z, visibility, presence}
    return landmarks[index];
  }

  function normalizeKeypoints(landmarks) {
    // This function must mirror the normalization done in the Python training pipeline
    if (!landmarks || landmarks.length === 0) return null;

    // Assuming MediaPipe keypoint indices:
    // 11: left_hip, 12: right_hip
    // 23: left_shoulder, 24: right_shoulder

    const leftHip = getKeypoint(landmarks, 11);
    const rightHip = getKeypoint(landmarks, 12);
    const leftShoulder = getKeypoint(landmarks, 23);
    const rightShoulder = getKeypoint(landmarks, 24);

    if (!leftHip || !rightHip || !leftShoulder || !rightShoulder ||
        leftHip.visibility < 0.5 || rightHip.visibility < 0.5 ||
        leftShoulder.visibility < 0.5 || rightShoulder.visibility < 0.5) {
      // If critical keypoints are not visible, return null or a zero array
      return new Array(NUM_KEYPOINTS * NUM_COORDS).fill(0);
    }

    const hipCenter = {
      x: (leftHip.x + rightHip.x) / 2,
      y: (leftHip.y + rightHip.y) / 2,
      z: (leftHip.z + rightHip.z) / 2
    };
    const shoulderCenter = {
      x: (leftShoulder.x + rightShoulder.x) / 2,
      y: (leftShoulder.y + rightShoulder.y) / 2,
      z: (leftShoulder.z + rightShoulder.z) / 2
    };

    const scaleFactor = Math.sqrt(
      Math.pow(hipCenter.x - shoulderCenter.x, 2) +
      Math.pow(hipCenter.y - shoulderCenter.y, 2) +
      Math.pow(hipCenter.z - shoulderCenter.z, 2)
    );

    if (scaleFactor === 0) return new Array(NUM_KEYPOINTS * NUM_COORDS).fill(0); // Avoid division by zero

    const processed = [];
    for (let i = 0; i < NUM_KEYPOINTS; i++) {
      const kp = getKeypoint(landmarks, i);
      if (kp && kp.visibility > 0.5) { // Only use visible keypoints
        processed.push((kp.x - hipCenter.x) / scaleFactor);
        processed.push((kp.y - hipCenter.y) / scaleFactor);
        processed.push((kp.z - hipCenter.z) / scaleFactor);
      } else {
        processed.push(0, 0, 0); // Fill with zeros if keypoint not visible
      }
    }
    return processed; // Flattened array of (x,y,z) for each keypoint
  }

  // --- Rep Counting Logic ---
  function updateRepStatus(landmarks) {
    repCompletedThisFrame = false;
    if (!landmarks) return;

    const hip = getKeypoint(landmarks, 24); // Assuming right hip for simplicity
    const knee = getKeypoint(landmarks, 26); // Assuming right knee

    if (!hip || !knee || hip.visibility < 0.5 || knee.visibility < 0.5) return;

    // Simple heuristic: hip_y relative to knee_y to detect squat depth
    // Note: In normalized coordinates, lower y means higher position if origin is top-left.
    // If normalized by hip-to-shoulder, y-axis might be inverted or centered.
    // Assuming lower y-value means deeper squat for now.
    const depthRatio = hip.y - knee.y; // Relative y-position of hip to knee

    // This rep counting logic is simplified. A more robust one would track
    // velocity, acceleration, and specific joint angles over time.
    if (currentRepStage === 'initial' && depthRatio > SQUAT_THRESHOLDS.MIN_REP_DEPTH) {
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
    try {
      // Path to your exported ONNX model
      const modelPath = 'models/squat_form_model.onnx'; // Assuming model is in resource_Examples/models/
      mlModelSession = await ort.InferenceSession.create(modelPath);
      console.log("ONNX ML Model loaded successfully.");
    } catch (e) {
      console.error("Failed to load ONNX ML Model:", e);
      mlModelSession = null;
    }
  }

  async function runMlInference(sequence) {
    if (!mlModelSession) {
      console.warn("ML Model not loaded. Skipping inference.");
      return null;
    }

    // ONNX model expects input shape [1, SEQUENCE_LENGTH, NUM_KEYPOINTS * NUM_COORDS]
    // The sequence is already flattened (NUM_KEYPOINTS * NUM_COORDS)
    const inputTensor = new ort.Tensor('float32', new Float32Array(sequence), [1, SEQUENCE_LENGTH, NUM_KEYPOINTS * NUM_COORDS]);

    try {
      const feeds = { input: inputTensor }; // 'input' is the name defined during ONNX export
      const results = await mlModelSession.run(feeds);
      const output = results.output.data; // 'output' is the name defined during ONNX export

      // Convert raw output (logits) to probabilities
      const probabilities = softmax(Array.from(output));
      return probabilities;
    } catch (e) {
      console.error("ONNX inference failed:", e);
      return null;
    }
  }

  function softmax(arr) {
    const maxVal = Math.max(...arr);
    const expArr = arr.map(x => Math.exp(x - maxVal)); // Subtract max for numerical stability
    const sumExpArr = expArr.reduce((a, b) => a + b, 0);
    return expArr.map(x => x / sumExpArr);
  }

  // --- Public Interface ---
  return {
    init: async function() {
      await loadMlModel();
      console.log("FormChecker initialized. ML Model status:", mlModelSession ? "Loaded" : "Not Loaded (using heuristics)");
    },

    analyze: async function(landmarks) {
      repCompletedThisFrame = false; // Reset for current frame
      let errors = [];
      let mlPredictedClassId = null;
      let mlConfidence = 0;

      if (!landmarks) {
        // If no landmarks, clear sequence and return
        keypointSequence = [];
        return { repCompleted: false, errors: [] };
      }

      // 1. Pre-process current frame's landmarks (normalization, flattening)
      const processedKeypoints = normalizeKeypoints(landmarks);
      if (!processedKeypoints) {
        // If processing fails for current frame, append zeros or skip
        keypointSequence.push(new Array(NUM_KEYPOINTS * NUM_COORDS).fill(0));
      } else {
        keypointSequence.push(processedKeypoints);
      }

      // Maintain fixed sequence length
      if (keypointSequence.length > SEQUENCE_LENGTH) {
        keypointSequence.shift(); // Remove oldest frame
      } else if (keypointSequence.length < SEQUENCE_LENGTH) {
        // Pad with zeros if not enough frames yet
        while (keypointSequence.length < SEQUENCE_LENGTH) {
          keypointSequence.unshift(new Array(NUM_KEYPOINTS * NUM_COORDS).fill(0));
        }
      }

      // Only run ML inference if sequence is full
      if (keypointSequence.length === SEQUENCE_LENGTH && mlModelSession) {
        // Flatten the 2D array of keypoints into a 1D array for the ONNX input
        const flatSequence = keypointSequence.flat();
        const mlPredictions = await runMlInference(flatSequence);

        if (mlPredictions) {
          // Find the class with the highest probability
          mlPredictedClassId = mlPredictions.indexOf(Math.max(...mlPredictions));
          mlConfidence = mlPredictions[mlPredictedClassId];

          // Apply confidence threshold from Python config (e.g., 0.90)
          // Only report error if confidence is high and it's not 'good_rep'
          if (mlConfidence > 0.90 && mlPredictedClassId !== ERROR_LABELS.good_rep) {
            errors.push(ERROR_LABELS[mlPredictedClassId]);
          }
        }
      } else if (!mlModelSession) {
        // Fallback to heuristic-based error checking if ML model not loaded
        // This is a simplified example and needs proper implementation
        // For demo, we'll just show a dummy error if ML model isn't loaded
        // In a real app, you'd have more robust heuristics here.
        // if (Math.random() < 0.01) { // 1% chance of a dummy error
        //   errors.push("KNEE_VALGUS");
        // }
      }

      // Update rep status (still using heuristic for rep counting)
      updateRepStatus(landmarks);

      return {
        repCompleted: repCompletedThisFrame,
        errors: errors,
        currentRepCount: repCount,
        mlConfidence: mlConfidence,
        mlPredictedClass: ERROR_LABELS[mlPredictedClassId] || "N/A",
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
      keypointSequence = []; // Clear sequence on reset
      // Optionally unload ML model or reset its state if needed
    }
  };
})();

// Initialize FormChecker when the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  formChecker.init();
});
