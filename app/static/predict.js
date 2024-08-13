document.getElementById("areaRange").addEventListener("input", (e) => {
  document.getElementById(
    "areaRangeLabel"
  ).innerHTML = `Area: ${e.target.value}`;
});
