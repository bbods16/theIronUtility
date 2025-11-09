// RepCounter.js - Tracks completed repetitions using a state machine
import { calculateAngle } from './utils.js';

export class RepCounter {
  constructor() {
    this.count = 0;
    this.inRep = false;
  }

  /**
   * Update rep counter state based on current pose.
   * @param {Array} landmarks - Pose landmarks array.
   * @param {string} exercise - The exercise being performed.
   * @returns {boolean} True if a rep was just completed on this update.
   */
  update(landmarks, exercise) {
    if (!landmarks || landmarks.length === 0) {
      return false;
    }

    if (exercise === 'squat') {
        // Squat detection using knee angle
        const LEFT_HIP = 23, LEFT_KNEE = 25, LEFT_ANKLE = 27;
        const RIGHT_HIP = 24, RIGHT_KNEE = 26, RIGHT_ANKLE = 28;

        const leftKneeAngle = calculateAngle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE]);
        const rightKneeAngle = calculateAngle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE]);
        const kneeAngle = Math.min(leftKneeAngle, rightKneeAngle);

        if (!this.inRep && kneeAngle < 100) { // Squatting down (knees bent)
            this.inRep = true;
        } else if (this.inRep && kneeAngle > 140) { // Standing up (knees extended)
            this.inRep = false;
            this.count++;
            return true;
        }
    } else if (exercise === 'bicep_curl') {
        const LEFT_SHOULDER = 11, LEFT_ELBOW = 13, LEFT_WRIST = 15;
        const RIGHT_SHOULDER = 12, RIGHT_ELBOW = 14, RIGHT_WRIST = 16;

        const leftElbowAngle = calculateAngle(landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST]);
        const rightElbowAngle = calculateAngle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST]);
        const elbowAngle = Math.min(leftElbowAngle, rightElbowAngle);

        if (!this.inRep && elbowAngle < 70) { // Arm is curled
            this.inRep = true;
        } else if (this.inRep && elbowAngle > 140) { // Arm is extended
            this.inRep = false;
            this.count++;
            return true;
        }
    }
    return false;
  }
}
