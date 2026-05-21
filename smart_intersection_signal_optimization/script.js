const canvas = document.getElementById("signalCanvas");
const ctx = canvas.getContext("2d");

const phases = [
  { name: "north_south_through", label: "南北直行", color: "#2d6cdf", duration: 34 },
  { name: "east_west_through", label: "东西直行", color: "#ef8a17", duration: 27 },
  { name: "north_south_left", label: "南北左转", color: "#2a9d62", duration: 14 },
  { name: "east_west_left", label: "东西左转", color: "#c44e52", duration: 13 }
];

let tick = 0;

function activePhase() {
  const cycle = phases.reduce((sum, phase) => sum + phase.duration, 0);
  let cursor = tick % cycle;
  for (const phase of phases) {
    if (cursor < phase.duration) return phase;
    cursor -= phase.duration;
  }
  return phases[0];
}

function queueLength(index, active) {
  const base = [13, 10, 6, 5][index];
  const wave = Math.sin((tick + index * 30) / 28) * 2.4;
  const relief = active.name === phases[index].name ? -5.2 : 1.8;
  return Math.max(1, Math.min(15, base + wave + relief));
}

function drawRoads() {
  ctx.fillStyle = "#f9fbfd";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#3d4654";
  ctx.fillRect(260, 0, 160, canvas.height);
  ctx.fillRect(0, 145, canvas.width, 130);
  ctx.fillStyle = "#596575";
  ctx.fillRect(280, 165, 120, 90);
  ctx.strokeStyle = "#f2e9c8";
  ctx.lineWidth = 3;
  ctx.setLineDash([18, 14]);
  ctx.beginPath();
  ctx.moveTo(340, 0);
  ctx.lineTo(340, canvas.height);
  ctx.moveTo(0, 210);
  ctx.lineTo(canvas.width, 210);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawSignal(x, y, isGreen) {
  ctx.fillStyle = "#202631";
  roundRect(x, y, 34, 72, 7);
  ctx.fillStyle = isGreen ? "#65464a" : "#d73f45";
  circle(x + 17, y + 20, 10);
  ctx.fillStyle = isGreen ? "#2fbf71" : "#46515f";
  circle(x + 17, y + 52, 10);
}

function drawQueues(active) {
  const queues = [
    { phase: phases[0], x: 295, y: 24, vertical: true },
    { phase: phases[1], x: 505, y: 178, vertical: false },
    { phase: phases[2], x: 372, y: 320, vertical: true },
    { phase: phases[3], x: 28, y: 232, vertical: false }
  ];

  queues.forEach((queue, index) => {
    const count = Math.round(queueLength(index, active));
    ctx.fillStyle = queue.phase.color;
    for (let i = 0; i < count; i += 1) {
      if (queue.vertical) {
        roundRect(queue.x, queue.y + i * 20, 32, 13, 4);
      } else {
        roundRect(queue.x + i * 35, queue.y, 25, 15, 4);
      }
    }
  });
}

function drawLabels(active) {
  ctx.fillStyle = "#ffffff";
  ctx.globalAlpha = 0.94;
  roundRect(20, 20, 230, 96, 8);
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#1f2933";
  ctx.font = "700 18px Microsoft YaHei, Arial";
  ctx.fillText("当前绿灯：" + active.label, 36, 54);
  ctx.font = "14px Microsoft YaHei, Arial";
  ctx.fillText("周期内动态相位与排队压力", 36, 84);
  ctx.fillText("优化后绿灯总时长：" + active.duration + "s", 36, 106);
}

function draw() {
  const active = activePhase();
  drawRoads();
  drawQueues(active);
  drawSignal(302, 105, active.name.includes("north_south"));
  drawSignal(430, 238, active.name.includes("east_west"));
  drawLabels(active);
  tick += 1;
  requestAnimationFrame(draw);
}

function circle(x, y, radius) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function roundRect(x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
  ctx.fill();
}

draw();
