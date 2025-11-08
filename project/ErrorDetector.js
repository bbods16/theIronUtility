// ErrorDetector.js - Tracks errors per repetition for summary and feedback
export class ErrorDetector {
  constructor() {
    this.currentRepErrors = {}; // Flexible object for errors: { errorType: boolean }
    this.records = []; // Array of { rep: number, errors: { errorType: boolean } }
  }

  /**
   * Call when a new rep is about to start (resets error flags for the new rep).
   */
  startRep() {
    this.currentRepErrors = {};
  }

  /**
   * Record errors observed in the current frame (during an ongoing rep).
   * @param {Array<string>} errors - List of error identifiers from formChecker for this frame.
   */
  recordFrameErrors(errors) {
    for (const error of errors) {
      this.currentRepErrors[error] = true;
    }
  }

  /**
   * Call when a rep is completed. Stores the errors that occurred during that rep.
   * @param {number} repNumber - The count/index of the rep just completed.
   * @returns {Object} The errors object for the completed rep.
   */
  endRep(repNumber) {
    const errorsForRep = { ...this.currentRepErrors };
    this.records.push({
      rep: repNumber,
      errors: errorsForRep,
    });
    return errorsForRep;
  }

  /**
   * Get a summary of all reps and errors (total counts).
   * @returns {Object} Summary with totalReps and a dictionary of error counts.
   */
  getSummary() {
    const errorCounts = {};
    this.records.forEach(record => {
      for (const error in record.errors) {
        if (record.errors[error]) {
          errorCounts[error] = (errorCounts[error] || 0) + 1;
        }
      }
    });
    return {
      totalReps: this.records.length,
      errorCounts: errorCounts,
    };
  }
}
