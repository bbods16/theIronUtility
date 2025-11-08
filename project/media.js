// media.js - Handles webcam media capture
export let mediaStream = null;  // will hold the MediaStream for the camera

/**
 * Request access to the webcam and start the video stream.
 * @param {HTMLVideoElement} videoElement - The video element to attach the stream to.
 * @returns {Promise} Resolves when the video is loaded and ready to play.
 */
export async function setupCamera(videoElement) {
  // Define camera constraints (e.g., preferred resolution and front camera)
  const constraints = {
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user"     // use front-facing camera if available
    },
    audio: false
  };
  try {
    // Request webcam stream
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    // Attach stream to the video element
    videoElement.srcObject = mediaStream;
    // Return a promise that resolves when video metadata (dimensions) is loaded
    return new Promise((resolve) => {
      videoElement.onloadeddata = () => {
        resolve();  // video is ready
      };
    });
  } catch (err) {
    console.error("Error accessing webcam:", err);
    return Promise.reject(err);
  }
}

/**
 * Stop the webcam stream and turn off the camera.
 */
export function stopCamera() {
  if (mediaStream) {
    // Stop all tracks (video feed) to release the camera
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
}
