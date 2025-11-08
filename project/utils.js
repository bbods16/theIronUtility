// utils.js - Utility functions for math/geometry on landmarks

/**
 * Calculate the angle (in degrees) formed at point p2 by the line segments p1-p2 and p3-p2.
 * @param {Object} p1 - First point with coordinates {x, y, z}.
 * @param {Object} p2 - Vertex point (angle at this point).
 * @param {Object} p3 - Third point with coordinates {x, y, z}.
 * @returns {number} Angle in degrees between p1->p2 and p3->p2.
 */
export function calculateAngle(p1, p2, p3) {
  // Vector from p2 to p1
  const v1 = { x: p1.x - p2.x, y: p1.y - p2.y, z: p1.z - p2.z || 0 };
  // Vector from p2 to p3
  const v2 = { x: p3.x - p2.x, y: p3.y - p2.y, z: p3.z - p2.z || 0 };
  // Calculate dot product v1·v2
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  // Calculate magnitudes |v1| and |v2|
  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
  if (mag1 * mag2 === 0) {
    return 0; // prevent division by zero (if points overlap)
  }
  // Compute cosine of angle
  let cosTheta = dot / (mag1 * mag2);
  // Clamp cosTheta to [-1,1] to avoid precision errors causing NaN
  cosTheta = Math.max(-1, Math.min(1, cosTheta));
  const angleRad = Math.acos(cosTheta);
  const angleDeg = angleRad * (180 / Math.PI);
  return angleDeg;
}
