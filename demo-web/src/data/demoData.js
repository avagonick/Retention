// Demo content: assets, scores, and copy used across the pages.
export const demoData = {
  question: "How do fractions work?",

  original: {
    video: "/assets/math_v1.mp4",
    brainVideo: "/assets/base_brain.mp4",
    retentionChart: "/assets/base_retention.png",
    score: 0.1545, // real math_v1 retention baseline
  },

  contextPlaceholder:
    'Add context for the model (optional) — e.g. "audience is 5th graders, slow the pacing, more visuals"',

  // Real preview slice of out/math_v1/preds.npy (first 8 rows × 12 of 20,484 cols).
  tribe: {
    shape: [37, 20484],
    rows: [
      [0.31, -0.039, 0.133, 0.055, 0.133, 0.112, 0.112, 0.014, -0.099, -0.038, 0.135, -0.105],
      [0.191, -0.138, 0.114, -0.0, 0.086, 0.02, 0.042, -0.124, -0.114, -0.057, 0.262, -0.072],
      [0.162, -0.106, 0.097, -0.013, 0.063, 0.023, 0.045, -0.127, -0.119, -0.071, 0.189, -0.085],
      [0.188, -0.07, 0.127, -0.03, 0.033, -0.001, 0.076, -0.138, -0.134, -0.057, 0.23, -0.108],
      [0.14, -0.068, 0.127, -0.132, -0.008, 0.007, 0.123, -0.162, -0.128, -0.086, 0.318, -0.123],
      [0.115, -0.034, 0.092, -0.137, -0.041, -0.01, 0.168, -0.149, -0.127, -0.073, 0.285, -0.137],
      [0.091, -0.032, 0.11, -0.155, -0.057, 0.005, 0.109, -0.16, -0.109, -0.096, 0.391, -0.089],
      [0.117, 0.008, 0.136, -0.147, -0.056, 0.058, 0.095, -0.206, -0.125, -0.08, 0.416, -0.114],
    ],
    note:
      "Each row = 1 second of video. Each column = 1 of 20,484 cortical surface vertices " +
      "(fsaverage5). Cols 0–10,241 = LEFT hemisphere, 10,242–20,483 = RIGHT. Values = z-scored " +
      "predicted BOLD (signed). We reduce this to retention = mean |z| over higher-order " +
      "association cortex — the part of the brain that encodes meaning into memory.",
  },

  // Generator/evaluator loop — rewards on the retention scale.
  loop: {
    baseline: 0.1545,
    bestIndex: 2,
    iterations: [
      { reward: 0.158, gen: "Insert a pizza-slice visual at 20s to re-anchor attention.", evalMsg: "0.158 ▲ assoc. cortex up, but t=24s still flat." },
      { reward: 0.172, gen: 'Add narration "here\'s the part most people miss" + slower pacing.', evalMsg: "0.172 ▲ language network re-engaged through the back half." },
      { reward: 0.191, gen: "Replace the dense overlay at 27s with a single animated fraction.", evalMsg: "0.191 ★ BEST — sustained engagement, late drop-off gone." },
      { reward: 0.183, gen: "Push a second visual at 30s for emphasis.", evalMsg: "0.183 ▽ overload — slight regression at 30s." },
      { reward: 0.176, gen: "Revert to single-visual, tighten the timing.", evalMsg: "0.176 ▽ below iteration 3." },
    ],
  },

  result: {
    video: "/assets/math_v1.mp4",
  },

  analytics: {
    original: {
      brainVideo: "/assets/base_brain.mp4",
      retentionChart: "/assets/base_retention.png",
      score: 0.1545,
    },
    improved: {
      brainVideo: "/assets/base_brain.mp4",
      retentionChart: "/assets/base_retention.png",
      score: 0.191,
    },
  },
};
