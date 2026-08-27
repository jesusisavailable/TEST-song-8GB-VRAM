export function generateSong();
// Defines song blocks and generates a song from lyrics.
type SongStructure {
  introBlock, verseBlock, chorusBlock, bridgeBlock, instrumentalBlock;
}
function generateSong(lyrics:string, styleParams:object) {
   // Build intro
   const intro = buildIntro(ly="introBlock");
   // Add verses etc.
   return new song; }