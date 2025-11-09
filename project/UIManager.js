// UIManager.js - Handles updating the DOM for UI elements (feedback, counts, modal, etc.)

/**
 * Update the displayed rep count.
 * @param {number} count - Current number of completed reps.
 */
export function updateRepCount(count) {
  const repCountEl = document.getElementById('repCount');
  if (repCountEl) repCountEl.textContent = count;
}

/**
 * Update the displayed score.
 * (Here, "score" can represent number of good reps or another metric as defined by the app.)
 * @param {number} score - Current score value.
 */
export function updateScore(score) {
  const scoreEl = document.getElementById('score');
  if (scoreEl) scoreEl.textContent = score;
}

/**
 * Display or update an immediate feedback message (e.g., form cues) in the UI.
 * @param {string} message - The message to show. If empty, clears the message.
 */
export function setLiveFeedback(message) {
  const liveFeedbackEl = document.getElementById('liveFeedback');
  if (!liveFeedbackEl) return;
  liveFeedbackEl.textContent = message;
}

/**
 * Append a message to the feedback log (the scrollable log of feedback/messages).
 * Optionally styles the message based on type (e.g., success, warning, danger).
 * @param {string} message - The message text to append.
 * @param {string} [type] - Message type: "success", "warning", "danger", etc. (maps to Bootstrap text color classes).
 */
export function appendLogMessage(message, type = "") {
  const logEl = document.getElementById('feedbackLog');
  if (!logEl) return;
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('message');
  if (type) {
    // Apply Bootstrap text color classes if type is provided
    if (type === "success") msgDiv.classList.add('text-success');
    else if (type === "warning") msgDiv.classList.add('text-warning');
    else if (type === "danger" || type === "error") msgDiv.classList.add('text-danger');
    else if (type === "info" || type === "primary") msgDiv.classList.add('text-primary');
  }
  msgDiv.innerText = message;
  logEl.appendChild(msgDiv);
  // Scroll to bottom so the latest message is visible
  logEl.scrollTop = logEl.scrollHeight;
}

/**
 * Enable or disable the "Get Coaching" button.
 * @param {boolean} enabled - True to enable the button, false to disable.
 */
export function setCoachingButtonEnabled(enabled) {
  const coachBtn = document.getElementById('coachButton');
  if (coachBtn) {
    coachBtn.disabled = !enabled;
  }
}

/**
 * Show the coaching modal with the given markdown content.
 * Converts markdown to HTML and displays it in a Bootstrap modal.
 * @param {string} markdownText - The markdown string returned by the AI coach.
 */
export function showCoachingModal(markdownText) {
  const contentEl = document.getElementById('coachContent');
  if (contentEl) {
    // Convert Markdown to HTML using the marked library
    contentEl.innerHTML = window.marked.parse(markdownText);
  }
  // Initialize and show the Bootstrap modal
  const modalEl = document.getElementById('coachingModal');
  if (modalEl) {
    const modal = new bootstrap.Modal(modalEl);
    modal.show();
  }
}
