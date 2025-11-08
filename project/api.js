// api.js - Handles communication with backend APIs (leaderboard submission, AI coaching request)

/**
 * Send the workout summary to the backend (e.g., to record on a leaderboard).
 * @param {Object} summaryData - An object containing the workout summary (reps and errors).
 * @returns {Promise<Response>} The fetch() promise.
 */
export async function sendSummary(summaryData) {
  try {
    const response = await fetch('/api/summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(summaryData)
    });
    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }
    return response; // you might handle response data if needed
  } catch (err) {
    console.error("Failed to send summary to backend:", err);
    throw err;
  }
}

/**
 * Request coaching feedback from the AI backend (Claude).
 * Sends the workout summary and expects a Markdown response with coaching tips.
 * @param {Object} summaryData - The workout summary to send for analysis.
 * @returns {Promise<string>} Resolves with the Markdown string from the AI coach.
 */
export async function getCoaching(summaryData) {
  try {
    const response = await fetch('/api/coaching', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(summaryData)
    });
    if (!response.ok) {
      throw new Error(`Coaching API returned ${response.status}`);
    }
    const text = await response.text();  // assuming the response is raw markdown text
    return text;
  } catch (err) {
    console.error("Failed to get coaching feedback:", err);
    throw err;
  }
}
