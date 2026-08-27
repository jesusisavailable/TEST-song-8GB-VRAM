export function setParameters();
// Holds style parameters: Genre, Mood, Voice, Instruments, Tempo.
type StyleControl {
  genre, mood, voice, instruments, tempo;
}
function setParameters(styleParams:object) {
   return styleParams; }