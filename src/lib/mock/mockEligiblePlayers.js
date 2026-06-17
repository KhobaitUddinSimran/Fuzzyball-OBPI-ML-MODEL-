const commonPositions = ["AM", "CAM", "Advanced 8", "Inside Forward"];

export const mockEligiblePlayers = Object.fromEntries(
  [
    [3794686, ["Argentina", 779, "Saudi Arabia", 801, ["Lionel Messi", "Papu Gomez", "Angel Di Maria", "Julian Alvarez"], ["Salem Al-Dawsari", "Sami Al-Najei", "Feras Al-Brikan", "Hattan Bahebri"]]],
    [3794687, ["Denmark", 776, "Tunisia", 777, ["Christian Eriksen", "Mikkel Damsgaard", "Jesper Lindstrom"], ["Wahbi Khazri", "Youssef Msakni", "Anis Ben Slimane"]]],
    [3794688, ["Mexico", 769, "Poland", 768, ["Alexis Vega", "Hirving Lozano", "Uriel Antuna"], ["Piotr Zielinski", "Sebastian Szymanski", "Arkadiusz Milik"]]],
    [3794689, ["France", 771, "Australia", 792, ["Antoine Griezmann", "Ousmane Dembele", "Kylian Mbappe", "Kingsley Coman"], ["Ajdin Hrustic", "Mathew Leckie", "Craig Goodwin"]]],
    [3794690, ["Morocco", 815, "Croatia", 785, ["Hakim Ziyech", "Sofiane Boufal", "Abdelhamid Sabiri"], ["Luka Modric", "Mateo Kovacic", "Nikola Vlasic"]]],
    [3794691, ["Germany", 770, "Japan", 778, ["Jamal Musiala", "Thomas Muller", "Serge Gnabry"], ["Daichi Kamada", "Takefusa Kubo", "Takumi Minamino"]]],
    [3794692, ["Spain", 772, "Costa Rica", 793, ["Pedri", "Gavi", "Dani Olmo", "Marco Asensio"], ["Joel Campbell", "Jewison Bennette", "Brandon Aguilera"]]],
    [3794693, ["Belgium", 782, "Canada", 794, ["Kevin De Bruyne", "Eden Hazard", "Dries Mertens"], ["Alphonso Davies", "Jonathan David", "Tajon Buchanan"]]],
    [3794694, ["Switzerland", 773, "Cameroon", 781, ["Xherdan Shaqiri", "Djibril Sow", "Ruben Vargas"], ["Bryan Mbeumo", "Eric Choupo-Moting", "Karl Toko Ekambi"]]],
    [3794695, ["Uruguay", 784, "South Korea", 783, ["Federico Valverde", "Giorgian De Arrascaeta", "Facundo Pellistri"], ["Son Heung-min", "Lee Kang-in", "Hwang Hee-chan"]]],
    [3794696, ["Portugal", 765, "Ghana", 786, ["Bruno Fernandes", "Bernardo Silva", "Joao Felix", "Rafael Leao"], ["Mohammed Kudus", "Andre Ayew", "Kamaldeen Sulemana"]]],
    [3794697, ["Brazil", 7810, "Serbia", 780, ["Neymar", "Lucas Paqueta", "Vinicius Junior", "Rodrygo"], ["Sergej Milinkovic-Savic", "Dusan Tadic", "Filip Kostic"]]],
    [3794698, ["Wales", 797, "Iran", 796, ["Gareth Bale", "Aaron Ramsey", "Harry Wilson"], ["Mehdi Taremi", "Saman Ghoddos", "Ali Gholizadeh"]]],
    [3794699, ["Qatar", 795, "Senegal", 788, ["Akram Afif", "Hassan Al-Haydos", "Almoez Ali"], ["Ismaila Sarr", "Krepin Diatta", "Iliman Ndiaye"]]],
    [3794700, ["Netherlands", 7790, "Ecuador", 790, ["Cody Gakpo", "Steven Bergwijn", "Davy Klaassen"], ["Gonzalo Plata", "Angel Mena", "Jeremy Sarmiento"]]],
    [3794701, ["England", 7680, "United States", 791, ["Phil Foden", "Mason Mount", "Bukayo Saka", "Jack Grealish"], ["Christian Pulisic", "Giovanni Reyna", "Brenden Aaronson", "Tim Weah"]]],
    [3869685, ["Argentina", 779, "France", 771, ["Lionel Messi", "Alexis Mac Allister", "Angel Di Maria", "Paulo Dybala"], ["Antoine Griezmann", "Kylian Mbappe", "Ousmane Dembele", "Kingsley Coman"]]]
  ].map(([matchId, [homeTeam, homeId, awayTeam, awayId, homePlayers, awayPlayers]]) => [
    matchId,
    [
      ...homePlayers.map((name, index) => makePlayer(matchId, homeId, homeTeam, name, index)),
      ...awayPlayers.map((name, index) => makePlayer(matchId, awayId, awayTeam, name, index + 10))
    ]
  ])
);

export function getMockEligiblePlayers(matchId) {
  return mockEligiblePlayers[matchId] || [];
}

export function getMockEligiblePlayer(matchId, playerId) {
  return getMockEligiblePlayers(matchId).find((player) => String(player.player_id) === String(playerId));
}

function makePlayer(matchId, teamId, teamName, playerName, index) {
  return {
    player_id: Number(`${teamId}${String(index + 1).padStart(2, "0")}`),
    player_name: playerName,
    team_id: teamId,
    team_name: teamName,
    position: commonPositions[index % commonPositions.length],
    minutes: [90, 82, 71, 63, 55][index % 5],
    match_id: matchId
  };
}
