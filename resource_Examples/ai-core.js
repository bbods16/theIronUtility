// ai-core.js - Handles communication with backend AI (Claude) and leaderboard
const aiCore = (function(){
  const API_URL = "";  // base URL for backend API (to be filled in)
  // Sound for card flip when showing coaching
  const cardFlipSound = new Audio('assets/sounds/card-flip.mp3');

  return {
    // Fetch AI coaching report after workout
    fetchCoachingReport: async function() {
      try {
        // Prepare summary data (this should match what backend expects)
        const summary = {
          reps: formChecker.getRepCount(),
          // We could also include counts of each error type, etc.
        };
        const response = await fetch(`${API_URL}/api/coaching`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(summary)
        });
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        const data = await response.json();
        const coachingText = data.coaching; // assume {coaching: "some feedback text"}
        // Show the coaching feedback in the UI
        UIManager.showCoachingReport(coachingText);
        cardFlipSound.play().catch(e => {});
      } catch (err) {
        console.error("Failed to fetch coaching report:", err);
      }
    },

    // Leaderboard fetching (for viewing top scores)
    fetchLeaderboard: async function() {
      try {
        const response = await fetch(`${API_URL}/api/leaderboard`);
        const scores = await response.json();
        UIManager.populateLeaderboard(scores);
      } catch (err) {
        console.error("Failed to fetch leaderboard:", err);
      }
    },

    // Leaderboard submission (to submit current user score)
    submitScore: async function(name, score) {
      try {
        await fetch(`${API_URL}/api/scores`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name, score })
        });
      } catch (err) {
        console.error("Failed to submit score:", err);
      }
    }
  };
})();
