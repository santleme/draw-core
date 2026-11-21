// Explosao de entrada: abre-se um buraco no meio da pagina, soltando estilhacos pixelados para os lados.
// O buraco e uma mascara radial em #root (so o #root cresce em raio); os estilhacos sao quadrados num canvas por cima.
const DURATION = 2000;
const SAMPLES = 96;
const CENTER_Y = 0.4;

type Shard = { x: number; y: number; vx: number; vy: number; size: number; born: number; moved: number; life: number; color: string };

const ease = (t: number) => 1 - (1 - t) ** 2;

// Velocidade de avanco por angulo: 3-4 pontas principais explodem rapido, o resto vai atras, irregular.
// Toda direcao tem velocidade >= 1, entao ao fim do tempo o buraco cobre tudo.
function makeSpeeds(): number[] {
  const spikes = Array.from({ length: 3 + Math.floor(Math.random() * 2) }, () => ({
    angle: Math.random() * Math.PI * 2,
    power: 1.1 + Math.random() * 0.9,
    width: 0.16 + Math.random() * 0.14,
  }));
  const raw = Array.from({ length: SAMPLES }, () => Math.random());
  return Array.from({ length: SAMPLES }, (_, i) => {
    const angle = (i / SAMPLES) * Math.PI * 2;
    // ruido suavizado: bordas denteadas, com lombadas pequenas
    const noise = (raw[(i + SAMPLES - 1) % SAMPLES] + raw[i] * 2 + raw[(i + 1) % SAMPLES]) / 4;
    let speed = 1 + noise * 0.5 + Math.random() * 0.12;
    for (const spike of spikes) {
      let d = Math.abs(angle - spike.angle) % (Math.PI * 2);
      if (d > Math.PI) d = Math.PI * 2 - d;
      speed += spike.power * Math.exp(-((d / spike.width) ** 2));
    }
    return speed;
  });
}

export function runBlast(onCovered: () => void): () => void {
  const root = document.getElementById('root');
  const style = getComputedStyle(document.body);
  const ink = style.getPropertyValue('--ink').trim() || '#22231f';
  const paper = style.getPropertyValue('--paper').trim() || '#f7f7f4';
  const accent = style.getPropertyValue('--accent').trim() || '#d8d08a';
  const palette = [ink, ink, paper, accent, '#ffffff'];

  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;z-index:90;pointer-events:none;image-rendering:pixelated';
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = window.innerWidth;
  const h = window.innerHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const cx = w / 2;
  const cy = h * CENTER_Y;
  const maxR = Math.hypot(Math.max(cx, w - cx), Math.max(cy, h - cy)) + 40;

  const speeds = makeSpeeds();
  const shards: Shard[] = [];
  let raf = 0;
  let start = 0;
  let last = 0;
  let covered = false;

  const radiusAt = (index: number, t: number) => ease(Math.min(1, t * speeds[index % SAMPLES])) * maxR;

  const spawn = (now: number, count: number, spanMs: number) => {
    for (let i = 0; i < count; i += 1) {
      // mais estilhacos saem das pontas (as direcoes mais rapidas)
      let index = Math.floor(Math.random() * SAMPLES);
      for (let k = 0; k < 2; k += 1) {
        const other = Math.floor(Math.random() * SAMPLES);
        if (speeds[other] > speeds[index]) index = other;
      }
      // espalha o nascimento pelo intervalo perdido: quem "saiu" antes
      // nasce na borda de entao e ja chega em voo, nao amontoado na atual
      const born = now - spanMs * Math.random();
      const age = (now - born) / 1000;
      const angle = ((index + Math.random()) / SAMPLES) * Math.PI * 2;
      const radius = radiusAt(index, Math.min(1, born / DURATION));
      if (radius >= maxR) continue;
      const speed = (160 + Math.random() * 620) * (0.7 + 0.3 * speeds[index]);
      const size = [3, 4, 6, 8, 12, 16][Math.floor(Math.random() * 6)];
      const vx = Math.cos(angle) * speed + (Math.random() - 0.5) * 60;
      const vy = Math.sin(angle) * speed + (Math.random() - 0.5) * 60;
      shards.push({
        x: cx + Math.cos(angle) * radius + vx * age,
        y: cy + Math.sin(angle) * radius + vy * age,
        vx,
        vy,
        size,
        born,
        moved: now,
        life: 450 + Math.random() * 700,
        color: palette[Math.floor(Math.random() * palette.length)],
      });
    }
  };

  const finish = () => {
    cancelAnimationFrame(raf);
    canvas.remove();
    if (root) root.style.removeProperty('clip-path');
  };

  const frame = (now: number) => {
    if (!start) { start = now; last = now; }
    const elapsed = now - start;
    const gap = (now - last) / 1000;
    last = now;
    const t = Math.min(1, elapsed / DURATION);
    const points = Array.from({ length: SAMPLES }, (_, i) => {
      const angle = (i / SAMPLES) * Math.PI * 2;
      const radius = radiusAt(i, t);
      return [cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius] as const;
    });

    if (root) {
      // clip-path vive no sistema de coordenadas do #root, que anda com a rolagem; o canvas e fixo na janela.
      const box = root.getBoundingClientRect();
      const hole = `M${points.map(([x, y]) => `${(x - box.left).toFixed(1)} ${(y - box.top).toFixed(1)}`).join('L')}Z`;
      root.style.clipPath = `path(evenodd, "M-10 -10H${w + 10}V${Math.max(box.height, h) + 10}H-10Z${hole}")`;
    }

    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      // Brilho da explosao: o miolo vai branco, como no flash do estudio.
      ctx.fillStyle = `rgba(255,255,255,${Math.min(1, t * 1.6)})`;
      ctx.beginPath();
      points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fill();
      if (t < 0.85) {
        // quantidade por segundo (nao por quadro), para a densidade ser igual em 60 e 120 Hz
        const want = (18 + 40 * (1 - t)) * 60 * gap;
        spawn(elapsed, Math.floor(want) + (Math.random() < want % 1 ? 1 : 0), gap * 1000);
      }
      for (let i = shards.length - 1; i >= 0; i -= 1) {
        const s = shards[i];
        const age = elapsed - s.born;
        if (age > s.life) { shards.splice(i, 1); continue; }
        // fisica limitada por estilhaco: quem nasceu neste frame nao anda
        // de novo (a posicao ja veio integrada ate `now`)
        const step = Math.min(0.05, (elapsed - s.moved) / 1000);
        s.x += s.vx * step;
        s.y += s.vy * step;
        s.moved = elapsed;
        ctx.globalAlpha = 1 - age / s.life;
        ctx.fillStyle = s.color;
        ctx.fillRect(Math.round(s.x / 2) * 2, Math.round(s.y / 2) * 2, s.size, s.size);
      }
      ctx.globalAlpha = 1;
    }

    if (t >= 1 && !covered) { covered = true; onCovered(); }
    if (t < 1 || shards.length) raf = requestAnimationFrame(frame);
    else finish();
  };

  raf = requestAnimationFrame(frame);
  return finish;
}
