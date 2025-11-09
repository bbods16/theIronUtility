"""
Squat Tracker with MediaPipe Pose Estimation
Real-time skeletal tracking and squat counting
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

class SquatTracker:
    """
    Real-time squat tracking using MediaPipe pose estimation
    Counts squats and detects form errors
    """

    def __init__(self, model_complexity=0, enable_gpu=True):
        """
        Initialize Squat Tracker

        Args:
            model_complexity (int): 0=Lite (fastest), 1=Balanced, 2=Heavy (most accurate)
            enable_gpu (bool): Enable GPU acceleration if available (RTX 4070)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # GPU ACCELERATION: MediaPipe automatically uses GPU (CUDA/TensorRT) when available
        # on systems with NVIDIA GPUs like RTX 4070. No additional configuration needed.
        #
        # model_complexity=0 is fastest (Lite model) - USE THIS FOR WEBCAM
        # model_complexity=1 is balanced (default) - GOOD FOR VIDEO PROCESSING
        # model_complexity=2 is most accurate but slowest
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=model_complexity,  # Configurable based on use case
            enable_segmentation=False,  # Disable segmentation for speed
            smooth_landmarks=True  # Enable smoothing for better tracking
        )

        self.model_complexity = model_complexity
        print(f"[SQUAT TRACKER] Initialized with model_complexity={model_complexity} (GPU: {enable_gpu})")

        self.squat_count = 0
        self.is_down = False
        self.depth_reached_in_current_rep = False
        self.hip_angle_threshold_down = 100
        self.hip_angle_threshold_rep_count = 90
        self.hip_angle_threshold_up = 160
        self.torso_angle_threshold_butt_wink = 80
        self.torso_angle_threshold_spinal_flexion = 170

        self.form_errors = {
            'knee_valgus': 0,
            'not_deep_enough': 0,
            'butt_wink': 0,
            'spinal_flexion': 0
        }

        self.angle_history = deque(maxlen=10)
        self.torso_angle_history = deque(maxlen=10)

        self.good_reps = 0
        self.total_reps = 0

    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """
        Calculate angle between three points
        a, b, c are (x, y) coordinates
        b is the vertex
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])

        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def get_hip_angle(self, landmarks) -> Optional[float]:
        """Calculate hip angle (shoulder-hip-knee) for both sides and average them, considering visibility."""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]

        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]

        angles = []

        if all(lm.visibility > 0.7 for lm in [left_shoulder, left_hip, left_knee]):
            angles.append(self.calculate_angle(
                (left_shoulder.x, left_shoulder.y),
                (left_hip.x, left_hip.y),
                (left_knee.x, left_knee.y)
            ))

        if all(lm.visibility > 0.7 for lm in [right_shoulder, right_hip, right_knee]):
            angles.append(self.calculate_angle(
                (right_shoulder.x, right_shoulder.y),
                (right_hip.x, right_hip.y),
                (right_knee.x, right_knee.y)
            ))

        if angles:
            return float(np.mean(angles))
        return None

    def get_knee_angle(self, landmarks) -> Optional[float]:
        """Calculate knee angle (hip-knee-ankle) for both sides and average them, considering visibility."""
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        angles = []

        if all(lm.visibility > 0.7 for lm in [left_hip, left_knee, left_ankle]):
            angles.append(self.calculate_angle(
                (left_hip.x, left_hip.y),
                (left_knee.x, left_knee.y),
                (left_ankle.x, left_ankle.y)
            ))

        if all(lm.visibility > 0.7 for lm in [right_hip, right_knee, right_ankle]):
            angles.append(self.calculate_angle(
                (right_hip.x, right_hip.y),
                (right_knee.x, right_knee.y),
                (right_ankle.x, right_ankle.y)
            ))

        if angles:
            return float(np.mean(angles))
        return None

    def get_torso_angle(self, landmarks) -> Optional[float]:
        """Calculate torso angle (shoulder-hip-heel) for both sides and average them, considering visibility."""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        left_heel = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value]

        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_heel = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value]

        angles = []

        if all(lm.visibility > 0.7 for lm in [left_shoulder, left_hip, left_heel]):
            angles.append(self.calculate_angle(
                (left_shoulder.x, left_shoulder.y),
                (left_hip.x, left_hip.y),
                (left_heel.x, left_heel.y)
            ))

        if all(lm.visibility > 0.7 for lm in [right_shoulder, right_hip, right_heel]):
            angles.append(self.calculate_angle(
                (right_shoulder.x, right_shoulder.y),
                (right_hip.x, right_hip.y),
                (right_heel.x, right_heel.y)
            ))

        if angles:
            return float(np.mean(angles))
        return None

    def check_knee_valgus(self, landmarks) -> bool:
        """
        Check if knees are caving in (knee valgus)
        Compares the horizontal distance between knees and ankles.
        If knees are significantly closer than ankles, it suggests valgus.
        """
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        if not all(lm.visibility > 0.7 for lm in [left_knee, right_knee, left_ankle, right_ankle]):
            return False

        knee_center_x = (left_knee.x + right_knee.x) / 2
        ankle_center_x = (left_ankle.x + right_ankle.x) / 2

        valgus_threshold = 0.05

        if abs(knee_center_x - ankle_center_x) > valgus_threshold:
            return True

        return False

    def check_depth(self, hip_angle: float) -> bool:
        """Check if squat is deep enough (hip below knee level)"""
        return hip_angle < 100

    def check_butt_wink(self, torso_angle: float) -> bool:
        """Check for butt wink (lower back rounding)"""
        return torso_angle < self.torso_angle_threshold_butt_wink

    def check_spinal_flexion(self, torso_angle: float) -> bool:
        """Check for excessive spinal flexion (forward lean)"""
        return torso_angle > self.torso_angle_threshold_spinal_flexion

    def detect_form_errors(self, landmarks, hip_angle: float, torso_angle: float, is_bottom_position: bool) -> List[str]:
        """Detect all form errors in current frame"""
        errors = []

        if is_bottom_position:
            if self.check_knee_valgus(landmarks):
                errors.append('KNEE_VALGUS')

            if not self.check_depth(hip_angle):
                errors.append('NOT_DEEP_ENOUGH')

            if self.check_butt_wink(torso_angle):
                errors.append('BUTT_WINK')

            if self.check_spinal_flexion(torso_angle):
                errors.append('SPINAL_FLEXION')

        return errors

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame - optimized for 60fps
        Returns: annotated frame and tracking data
        """
        # Convert BGR to RGB more efficiently
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process with MediaPipe (GPU accelerated if available)
        results = self.pose.process(image_rgb)

        # Convert back to BGR for OpenCV
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        tracking_data = {
            'squat_count': self.squat_count,
            'current_angle': 0,
            'torso_angle': 0,
            'stage': 'unknown',
            'form_errors': [],
            'feedback_messages': [],
            'landmarks': None,
            'good_reps': self.good_reps,
            'total_reps': self.total_reps,
            'rep_completed': False
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Simplified drawing for faster rendering
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            hip_angle = self.get_hip_angle(landmarks)
            knee_angle = self.get_knee_angle(landmarks)
            torso_angle = self.get_torso_angle(landmarks)

            if hip_angle is not None and torso_angle is not None:
                self.angle_history.append(hip_angle)
                smoothed_hip_angle = np.mean(self.angle_history)

                self.torso_angle_history.append(torso_angle)
                smoothed_torso_angle = np.mean(self.torso_angle_history)

                tracking_data['current_angle'] = int(smoothed_hip_angle)
                tracking_data['torso_angle'] = int(smoothed_torso_angle)

                rep_completed = False

                if smoothed_hip_angle <= self.hip_angle_threshold_down:
                    if not self.is_down:
                        self.is_down = True
                        tracking_data['stage'] = 'DOWN'

                    if smoothed_hip_angle <= self.hip_angle_threshold_rep_count:
                        if not self.depth_reached_in_current_rep:
                            self.squat_count += 1
                            self.total_reps += 1
                            self.depth_reached_in_current_rep = True
                            rep_completed = True

                            errors = self.detect_form_errors(landmarks, smoothed_hip_angle, smoothed_torso_angle, True)
                            tracking_data['form_errors'] = errors

                            # Generate detailed feedback messages
                            tracking_data['feedback_messages'] = self.get_form_feedback(errors)

                            for error in errors:
                                if error in self.form_errors:
                                    self.form_errors[error] += 1

                            if not tracking_data['form_errors']:
                                self.good_reps += 1

                elif smoothed_hip_angle >= self.hip_angle_threshold_up:
                    if self.is_down:
                        self.is_down = False
                        self.depth_reached_in_current_rep = False
                    tracking_data['stage'] = 'UP'

                else:
                    tracking_data['stage'] = 'TRANSITION'

                tracking_data['squat_count'] = self.squat_count
                tracking_data['rep_completed'] = rep_completed

            tracking_data['landmarks'] = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                for lm in landmarks
            ]

            self._draw_info(image, tracking_data)

        return image, tracking_data

    def _draw_info(self, image: np.ndarray, data: Dict) -> None:
        """Draw tracking info on the frame - optimized for small resolution"""
        h, w = image.shape[:2]

        # Scale overlay based on image size (much smaller for 320x240)
        box_width = min(180, int(w * 0.55))
        box_height = min(120, int(h * 0.5))

        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (box_width, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # Border
        cv2.rectangle(image, (5, 5), (box_width, box_height), (0, 255, 0), 2)

        # Smaller font sizes
        font_scale_small = 0.35
        font_scale_medium = 0.45
        font_scale_large = 0.6
        thickness = 1

        y_offset = 18
        line_spacing = 18

        # Title (smaller)
        cv2.putText(image, 'IRON UTILITY',
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 215, 0), thickness)
        y_offset += line_spacing

        # Squat count (larger, green)
        cv2.putText(image, f'SQUATS: {data["squat_count"]}',
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, (0, 255, 0), 2)
        y_offset += line_spacing + 2

        # Good reps
        cv2.putText(image, f'Good Reps: {data["good_reps"]}/{data["total_reps"]}',
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), thickness)
        y_offset += line_spacing

        # Hip angle
        cv2.putText(image, f'Hip: {data["current_angle"]}deg',
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), thickness)
        y_offset += line_spacing

        # Stage
        stage_color = (0, 255, 0) if data['stage'] == 'UP' else (255, 165, 0)
        cv2.putText(image, f'Stage: {data["stage"]}',
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, stage_color, thickness)

        # Depth indicator (right side, smaller)
        depth_y = int(h * 0.5)
        depth_height = int(h * 0.6)
        bar_width = 15

        cv2.rectangle(image, (w - bar_width - 5, depth_y - depth_height),
                     (w - 5, depth_y), (100, 100, 100), -1)

        if data['current_angle'] > 0:
            depth_percent = max(0, min(1, (180 - data['current_angle']) / 90))
            indicator_y = int(depth_y - depth_height * depth_percent)

            color = (0, 255, 0) if depth_percent > 0.5 else (0, 165, 255)
            cv2.circle(image, (w - int(bar_width/2) - 5, indicator_y), 5, color, -1)

    def get_form_feedback(self, errors: List[str]) -> List[str]:
        """
        Generate detailed, actionable feedback messages for form errors
        Returns Monopoly-themed feedback messages
        """
        feedback_messages = []

        for error in errors:
            if error == 'KNEE_VALGUS':
                feedback_messages.append("⚠️ KNEE VALGUS DETECTED! Your knees are caving inward. Push them outward to align with your toes!")
            elif error == 'NOT_DEEP_ENOUGH':
                feedback_messages.append("📉 GO DEEPER! Your hips aren't breaking parallel. Squat lower until your hip crease is below your knee!")
            elif error == 'BUTT_WINK':
                feedback_messages.append("🔄 BUTT WINK ALERT! Your lower back is rounding at the bottom. Brace your core and don't go deeper than your mobility allows!")
            elif error == 'SPINAL_FLEXION':
                feedback_messages.append("🦴 EXCESSIVE FORWARD LEAN! Keep your chest up and back straight. Engage your core and look forward!")

        if not feedback_messages:
            feedback_messages.append("✅ PERFECT FORM! Pass GO and collect $200! Keep it up!")

        return feedback_messages

    def get_session_stats(self) -> Dict:
        """Get current session statistics"""
        return {
            'squat_count': self.squat_count,
            'good_reps': self.good_reps,
            'total_reps': self.total_reps,
            'form_errors': self.form_errors.copy(),
            'success_rate': (self.good_reps / self.total_reps * 100) if self.total_reps > 0 else 0
        }

    def reset(self) -> None:
        """Reset the tracker"""
        self.squat_count = 0
        self.good_reps = 0
        self.total_reps = 0
        self.is_down = False
        self.depth_reached_in_current_rep = False
        self.form_errors = {
            'knee_valgus': 0,
            'not_deep_enough': 0,
            'butt_wink': 0,
            'spinal_flexion': 0
        }
        self.angle_history.clear()
        self.torso_angle_history.clear()

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'pose'):
            self.pose.close()
