// formChecker.js - Heuristic rules to detect squat form errors

import { calculateAngle } from './utils.js';

// Training data for squat form validation
const squatData = [
  { input: { action: 'squat', angle: 170 }, output: { feedback: 'good' } },
  { input: { action: 'squat', angle: 160 }, output: { feedback: 'good' } },
  { input: { action: 'squat', angle: 150 }, output: { feedback: 'good' } },
  { input: { action: 'squat', angle: 140 }, output: { feedback: 'good' } },
  { input: { action: 'squat', angle: 130 }, output: { feedback: 'good' } },
  { input: { action: 'squat', angle: 120 }, output: { feedback: 'good' } },
  { input: { action: 'squat', angle: 110 }, output: { feedback: 'good' } },
  { input: { action: 'squat', angle: 100 }, output: { feedback: 'good' } },
  { input: { action: 'squat', angle: 90 }, output: { feedback: 'good' } },
  { input: { action: 'squat', angle: 80 }, output: { feedback: 'too low' } },
  { input: { action: 'squat', angle: 70 }, output: { feedback: 'too low' } },
  { input: { action: 'squat', angle: 60 }, output: { feedback: 'too low' } },
];

/**
 * Analyze pose landmarks for form errors.
 * @param {Array} landmarks - Array of 33 pose landmark points (each with x,y,z [normalized]).
 * @param {string} exercise - The exercise being performed.
 * @returns {Array<string>} List of error identifiers detected (empty if none).
 */
export function checkErrors(landmarks, exercise) {
  const errors = [];
  if (!landmarks || landmarks.length === 0) {
    return errors; // no pose detected
  }

  if (exercise === 'squat') {
    const LEFT_HIP = 23, LEFT_KNEE = 25, LEFT_ANKLE = 27;
    const RIGHT_HIP = 24, RIGHT_KNEE = 26, RIGHT_ANKLE = 28;

    const leftKneeAngle = calculateAngle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE]);
    const rightKneeAngle = calculateAngle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE]);

    // Use the smaller angle to be more conservative
    const kneeAngle = Math.min(leftKneeAngle, rightKneeAngle);

    // Find the closest data point from our training data
    let closestEntry = squatData.reduce((prev, curr) => {
      return (Math.abs(curr.input.angle - kneeAngle) < Math.abs(prev.input.angle - kneeAngle) ? curr : prev);
    });

    if (closestEntry.output.feedback === 'too low') {
      errors.push('squat_too_low');
    }

  } else if (exercise === 'bicep_curl') {
    const LEFT_SHOULDER = 11, LEFT_ELBOW = 13, LEFT_WRIST = 15;
    const RIGHT_SHOULDER = 12, RIGHT_ELBOW = 14, RIGHT_WRIST = 16;

    const leftElbowAngle = calculateAngle(landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST]);
    const rightElbowAngle = calculateAngle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST]);

    if (leftElbowAngle > 60 || rightElbowAngle > 60) {
        errors.push('incomplete_curl');
    }
  }

  return errors;
}
