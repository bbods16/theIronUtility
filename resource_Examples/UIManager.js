// UIManager.js - Updates UI elements and triggers theme animations/sounds
const UIManager = (function(){
  const scoreDisplay = document.getElementById('scoreDisplay');
  const feedbackBox = document.getElementById('feedbackBox');
  const playerToken = document.getElementById('playerToken');
  const leaderboardList = document.getElementById('leaderboardList');
  // Sound effects
  const successSound = new Audio('assets/sounds/money-clink.mp3');  // played on rep success (coin/money sound)
  const errorSound = new Audio('assets/sounds/card-flip.mp3');      // played on error appearance (using card flip as generic error cue)

  // Token position state for mini-board (0-39 corresponding to Monopoly board spaces)
  let tokenPosition = 0;
  const totalSpaces = 40;

  // Utility: convert token index to board coordinate (percentage)
  function getTokenPositionCoords(index) {
    // Normalize index within 0-39
    const i = index % totalSpaces;
    let x = 0, y = 0;
    if (i <= 10) {
      // bottom row: 0 (GO) to 10 (Jail)
      x = 1 - (i / 10);
      y = 1;
    } else if (i <= 20) {
      // left column: 10 (Jail) to 20 (Free Parking)
      x = 0;
      y = 1 - ((i - 10) / 10);
    } else if (i <= 30) {
      // top row: 20 (Free Parking) to 30 (Go To Jail)
      x = (i - 20) / 10;
      y = 0;
    } else if (i <= 39) {
      // right column: 30 (Go To Jail) to 39 (just before GO)
      x = 1;
      y = (i - 30) / 10;
    }
    // Adjust to percentage with a small offset so token stays fully visible
    const offsetX = x * 90 + 5; // range [5%,95%]
    const offsetY = y * 90 + 5;
    return { left: offsetX + '%', top: offsetY + '%' };
  }

  return {
    update: function(analysis) {
      // Update feedback and score each frame based on analysis
      if (analysis.repCompleted) {
        // Update score display
        const newScore = parseInt(scoreDisplay.textContent) + 1;
        scoreDisplay.textContent = newScore;
        // Move token piece one space forward on mini-board
        tokenPosition = (tokenPosition + 1) % totalSpaces;
        const coords = getTokenPositionCoords(tokenPosition);
        playerToken.style.left = coords.left;
        playerToken.style.top = coords.top;
        // Flying money animation
        this.showFlyingMoney();
        // Play success sound
        successSound.play().catch(e => {});
      }
      // Handle feedback for errors
      if (analysis.errors && analysis.errors.length > 0) {
        // Display an error message in feedback box (using error code to message map)
        const errorMsgs = analysis.errors.map(code => {
          if (code === "KNEE_VALGUS") return "Knees Collapsing In!";
          // ... other error codes mapping ...
          return code;
        });
        feedbackBox.textContent = errorMsgs.join(' ');
        // Show feedback box briefly
        feedbackBox.style.display = 'block';
        setTimeout(() => { feedbackBox.style.display = 'none'; }, 2000);
        // Play error alert sound
        errorSound.play().catch(e => {});
      }
    },

    showFlyingMoney: function() {
      // Create a flying money element (💸 emoji) and animate it
      const moneyElem = document.createElement('div');
      moneyElem.className = 'money-fly';
      moneyElem.textContent = '💸';
      // Position it around the score HUD (for example)
      const scoreHud = document.getElementById('scoreHUD');
      const rect = scoreHud.getBoundingClientRect();
      moneyElem.style.left = (rect.left + rect.width/2) + 'px';
      moneyElem.style.top = (rect.top) + 'px';
      document.body.appendChild(moneyElem);
      // Remove after animation completes
      moneyElem.addEventListener('animationend', () => {
        moneyElem.remove();
      });
    },

    onWorkoutStart: function() {
      // Reset score and token position at workout start
      scoreDisplay.textContent = '0';
      tokenPosition = 0;
      const coords = getTokenPositionCoords(tokenPosition);
      playerToken.style.left = coords.left;
      playerToken.style.top = coords.top;
      // Hide any lingering feedback
      feedbackBox.style.display = 'none';
    },

    onWorkoutEnd: function() {
      // Celebration for passing GO on workout completion
      // Move token explicitly to GO space if not already
      tokenPosition = 0;
      const coords = getTokenPositionCoords(tokenPosition);
      playerToken.style.left = coords.left;
      playerToken.style.top = coords.top;
      // Flash "Passed GO" message
      const goBanner = document.createElement('div');
      goBanner.className = 'position-fixed top-50 start-50 translate-middle p-4 bg-danger text-white fw-bold passed-go';
      goBanner.style.zIndex = 2000;
      goBanner.style.border = '4px solid #000';
      goBanner.style.borderRadius = '0.5rem';
      goBanner.textContent = 'YOU PASSED GO! +$200';
      document.body.appendChild(goBanner);
      setTimeout(() => { goBanner.remove(); }, 3000);
      // Optionally, open leaderboard submission or immediate fetch of coaching
      // In this demo, we'll fetch coaching after a short delay
      setTimeout(() => {
        aiCore.fetchCoachingReport();
      }, 1000);
    },

    populateLeaderboard: function(scores) {
      // Populate the leaderboard modal list with scores [{name, score}, ...]
      leaderboardList.innerHTML = '';
      scores.forEach((entry, index) => {
        const li = document.createElement('li');
        // Apply property group color by rank (1: dark-blue, 2: dark-blue, 3: green, 4: green, 5: yellow, 6: yellow, 7: red, 8: red, 9: orange, 10: orange)
        let color = '#87ceeb'; // default light blue
        const rank = index + 1;
        if (rank <= 2) color = '#1f3d7a';       // dark blue
        else if (rank <= 4) color = '#28a745';  // green
        else if (rank <= 6) color = '#ffd700';  // gold/yellow
        else if (rank <= 8) color = '#d90429';  // red
        else if (rank <= 10) color = '#ff8c00'; // orange
        li.style.backgroundColor = '#fff'; // white card background
        li.style.borderTop = `20px solid ${color}`; // colored band at top
        // Content: name and score
        const nameSpan = document.createElement('span');
        nameSpan.className = 'name';
        nameSpan.textContent = entry.name;
        const scoreSpan = document.createElement('span');
        scoreSpan.className = 'score';
        scoreSpan.textContent = entry.score;
        li.appendChild(nameSpan);
        li.appendChild(scoreSpan);
        leaderboardList.appendChild(li);
      });
      // Show the leaderboard modal
      const leaderboardModal = new bootstrap.Modal(document.getElementById('leaderboardModal'));
      leaderboardModal.show();
    },

    showCoachingReport: function(text) {
      // Insert the coaching text into the modal and show it
      const coachTextElem = document.getElementById('coachText');
      coachTextElem.textContent = text;
      // Show the coaching modal (Community Chest card)
      const coachingModal = new bootstrap.Modal(document.getElementById('coachingModal'));
      coachingModal.show();
    }
  };
})();

// Event hookup for showing leaderboard modal via button
document.getElementById('showLeaderboardBtn').addEventListener('click', () => {
  aiCore.fetchLeaderboard();
});
