export function cloneReferenceAudio();
// Stores reference audio files for style cloning.
type ReferenceAudio {
  referenceClip, referenceScore;
}
function cloneReferenceAudio(referenceClip:string) {
   const score = calculateStyleMatchScore(referenceClip);
   return {score: number}; }