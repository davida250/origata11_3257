/**
 * Basic Origami Folds — simple, reliable folds with your psychedelic look.
 * - Single valley/mountain folds (vertical/horizontal/diagonal)
 * - Gate fold (two vertical valleys)
 * - Accordion pleat (5 creases alternating M/V)
 *
 * Folding model: SEQUENTIAL, with CPU-propagated axes (O(N^2)), but we limit to a few creases.
 * Angle sign convention: Valley = +degrees, Mountain = -degrees (as in MIT Origami Simulator). 
 * See: origamisimulator.org → “Design Tips” (“fold angle is positive for valley, negative for mountain”).  // ref citation in README / code comments
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { FXAAShader } from 'three/addons/shaders/FXAAShader.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import GUI from 'lil-gui';

// ---------- Renderer / Scene ----------
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x050509, 5, 30);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 1.8, 5.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// ---------- Post ----------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.0, 0.6, 0.15);
composer.addPass(bloom);
const fxaa = new ShaderPass(FXAAShader);
fxaa.material.uniforms.resolution.value.set(1 / window.innerWidth, 1 / window.innerHeight);
composer.addPass(fxaa);
composer.addPass(new OutputPass());

// ---------- Paper geometry ----------
const WIDTH = 4.0, HEIGHT = 2.4;
const SEG_X = 160, SEG_Y = 120;           // crisp creases
const geo = new THREE.PlaneGeometry(WIDTH, HEIGHT, SEG_X, SEG_Y);
geo.rotateX(-0.25);

// ---------- Math helpers ----------
const tmp = {
  v1: new THREE.Vector3(),
  q: new THREE.Quaternion()
};
function signedDistance2(p /*Vec3*/, a /*Vec3*/, d /*unit Vec3*/) {
  // 2D side test in XY (paper plane)
  const px = p.x - a.x, py = p.y - a.y;
  return d.x * py - d.y * px; // z-component of 2D cross
}
function rotatePointAroundLine(p, a, axisUnit, ang) {
  tmp.v1.copy(p).sub(a).applyAxisAngle(axisUnit, ang).add(a);
  p.copy(tmp.v1);
}
function rotateVectorAxis(v, axisUnit, ang) {
  v.applyAxisAngle(axisUnit, ang);
}
function smooth01(e0, e1, x) {
  const t = THREE.MathUtils.clamp((x - e0) / Math.max(1e-9, e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}

// ---------- “Base” crease sequence (few creases = simple & reliable) ----------
const MAX_CREASES = 8;
const base = {
  count: 0,
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  amp:   new Array(MAX_CREASES).fill(0),   // |angle| in radians
  band:  new Array(MAX_CREASES).fill(0),   // soft hinge half-width
  phase: new Array(MAX_CREASES).fill(0),   // animation phase
  sign:  new Array(MAX_CREASES).fill(1)    // +1 valley, -1 mountain
};
function resetBase() {
  base.count = 0;
  for (let i=0;i<MAX_CREASES;i++){
    base.A[i].set(0,0,0);
    base.D[i].set(1,0,0);
    base.amp[i] = 0; base.band[i] = 0; base.phase[i] = 0; base.sign[i] = 1;
  }
}
function addCrease(Ax, Ay, Dx, Dy, deg=90, sign=+1, bandFrac=0.006) {
  if (base.count >= MAX_CREASES) return;
  const i = base.count++;
  const d = new THREE.Vector2(Dx, Dy).normalize();
  base.A[i].set(Ax, Ay, 0);
  base.D[i].set(d.x, d.y, 0);
  base.amp[i]  = THREE.MathUtils.degToRad(Math.max(0, Math.min(180, Math.abs(deg))));
  base.sign[i] = sign >= 0 ? +1 : -1; // +1 = valley, -1 = mountain
  base.band[i] = bandFrac * WIDTH;
  base.phase[i]= Math.random()*Math.PI*2;
}

// ---------- Effective axes (sequential propagation) ----------
const eff = {
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  ang: new Float32Array(MAX_CREASES),
  band: new Float32Array(MAX_CREASES)
};

// Drive parameters (simple & predictable)
const drive = {
  animate: false,       // toggle
  speed: 0.9,           // radians/s factor in sine
  progress: 0.7         // 0..1 slider
};

// Compute signed angle for each crease from drive (valley +=, mountain -=)
function computeAngles(tSec) {
  for (let i=0;i<base.count;i++){
    const t = drive.animate ? (0.5 + 0.5*Math.sin(tSec*drive.speed + base.phase[i])) : drive.progress;
    eff.ang[i] = base.sign[i] * base.amp[i] * t;
    eff.band[i] = base.band[i];
  }
  for (let i=base.count;i<MAX_CREASES;i++){ eff.ang[i]=0; eff.band[i]=0.001; }
}

// Propagate earlier folds into later crease axes (sequential)
function computeEffectiveAxes() {
  // start from base
  for (let i=0;i<base.count;i++){
    eff.A[i].copy(base.A[i]);
    eff.D[i].copy(base.D[i]).normalize();
  }
  // rotate later axes by earlier folds (softly across hinge band)
  for (let j=0;j<base.count;j++){
    const Aj = eff.A[j];
    const Dj = eff.D[j].clone().normalize();
    const angle = eff.ang[j];
    if (Math.abs(angle) < 1e-7) continue;

    for (let k=j+1;k<base.count;k++){
      const sd = signedDistance2(eff.A[k], Aj, Dj);
      const m = smooth01(0, eff.band[j], sd);  // right side rotates, left stays
      const ang = angle * m;
      if (Math.abs(ang) < 1e-7) continue;

      rotatePointAroundLine(eff.A[k], Aj, Dj, ang);
      rotateVectorAxis(eff.D[k], Dj, ang);
      eff.D[k].normalize();
    }
  }
}

// ---------- Uniforms (psychedelic paper shader preserved) ----------
const uniforms = {
  uTime:       { value: 0 },
  uSectors:    { value: 10.0 },
  uHueShift:   { value: 0.0 },
  uIridescence:{ value: 0.65 },
  uFilmIOR:    { value: 1.35 },
  uFilmNm:     { value: 360.0 },
  uFiber:      { value: 0.35 },
  uEdgeGlow:   { value: 0.9 },

  // folding data
  uCreaseCount: { value: 0 },
  uAeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3()) },
  uDeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3(1,0,0)) },
  uAng:   { value: new Float32Array(MAX_CREASES) },
  uBand:  { value: new Float32Array(MAX_CREASES) }
};

function pushEffToUniforms() {
  uniforms.uCreaseCount.value = base.count;
  uniforms.uAeff.value = eff.A.map(v => v.clone());
  uniforms.uDeff.value = eff.D.map(v => v.clone());
  uniforms.uAng.value  = Float32Array.from(eff.ang);
  uniforms.uBand.value = Float32Array.from(eff.band);
  mat.uniformsNeedUpdate = true; // force uniform refresh
}

// ---------- Shaders (exact visual look from previous builds) ----------
const vs = /* glsl */`
  #define MAX_CREASES ${MAX_CREASES}
  precision highp float;
  uniform int   uCreaseCount;
  uniform vec3  uAeff[MAX_CREASES];
  uniform vec3  uDeff[MAX_CREASES];
  uniform float uAng[MAX_CREASES];
  uniform float uBand[MAX_CREASES];
  varying vec3 vPos; varying vec3 vN; varying vec3 vLocal; varying vec2 vUv;

  vec3 rotateAroundLine(vec3 p, vec3 a, vec3 u, float ang){
    vec3 v = p - a; float c = cos(ang), s = sin(ang);
    return a + v*c + cross(u, v)*s + u*dot(u, v)*(1.0 - c);
  }
  vec3 rotateVector(vec3 v, vec3 u, float ang){
    float c = cos(ang), s = sin(ang);
    return v*c + cross(u, v)*s + u*dot(u, v)*(1.0 - c);
  }
  float signedDistanceToLine(vec2 p, vec2 a, vec2 d){
    return d.x*(p.y - a.y) - d.y*(p.x - a.x);
  }
  float sstep(float e0, float e1, float x){
    float t = clamp((x - e0) / max(1e-6, e1 - e0), 0.0, 1.0);
    return t*t*(3.0 - 2.0*t);
  }

  void main(){
    vUv = uv;
    vec3 p = position;
    vec3 n = normalize(normal);

    for (int i=0; i<MAX_CREASES; i++){
      if (i >= uCreaseCount) break;
      vec3 a = uAeff[i];
      vec3 d = normalize(uDeff[i]);

      float sd = signedDistanceToLine(p.xy, a.xy, d.xy);
      float m = sstep(0.0, uBand[i], sd);
      float ang = uAng[i] * m;

      p = rotateAroundLine(p, a, d, ang);
      n = normalize(rotateVector(n, d, ang));
    }

    vLocal = p;
    vec4 world = modelMatrix * vec4(p, 1.0);
    vPos = world.xyz;
    vN   = normalize(mat3(modelMatrix) * n);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const fs = /* glsl */`
  #define MAX_CREASES ${MAX_CREASES}
  precision highp float;

  uniform float uTime;
  uniform float uSectors, uHueShift;
  uniform float uIridescence, uFilmIOR, uFilmNm, uFiber, uEdgeGlow;
  uniform int   uCreaseCount;
  uniform vec3  uAeff[MAX_CREASES];
  uniform vec3  uDeff[MAX_CREASES];

  varying vec3 vPos; varying vec3 vN; varying vec3 vLocal; varying vec2 vUv;

  #define PI 3.14159265359

  float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
  float noise(vec2 p){
    vec2 i=floor(p), f=fract(p);
    float a=hash(i), b=hash(i+vec2(1,0)), c=hash(i+vec2(0,1)), d=hash(i+vec2(1,1));
    vec2 u=f*f*(3.0-2.0*f);
    return mix(a,b,u.x)+ (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
  }
  float fbm(vec2 p){
    float v=0.0, a=0.5;
    for(int i=0;i<5;i++){ v+=a*noise(p); p*=2.0; a*=0.5; }
    return v;
  }
  vec3 hsv2rgb(vec3 c){
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.,4.,2.), 6.)-3.)-1., 0., 1.);
    return c.z * mix(vec3(1.0), rgb, c.y);
  }
  vec3 thinFilm(float cosTheta, float ior, float nm){
    vec3 lambda = vec3(680.0, 550.0, 440.0);
    vec3 phase  = 4.0 * PI * ior * nm * cosTheta / lambda;
    return 0.5 + 0.5*cos(phase);
  }
  float signedDistanceToLine(vec2 p, vec2 a, vec2 d){
    return d.x*(p.y - a.y) - d.y*(p.x - a.x);
  }

  void main(){
    // Kaleidoscopic color
    float theta = atan(vPos.z, vPos.x);
    float r = length(vPos.xz) * 0.55;
    float seg = 2.0*PI / max(3.0, uSectors);
    float a = mod(theta, seg); a = abs(a - 0.5*seg);
    vec2 k = vec2(cos(a), sin(a)) * r;

    vec2 q = k*2.0 + vec2(0.15*uTime, -0.1*uTime);
    q += 0.5*vec2(noise(q+13.1), noise(q+71.7));
    float n = noise(q*2.0) * 0.75 + 0.25*noise(q*5.0);
    float hue = fract(n + 0.15*sin(uTime*0.3) + uHueShift);
    vec3 baseCol = hsv2rgb(vec3(hue, 0.9, smoothstep(0.25, 1.0, n)));

    // Paper fibers + grain
    float fiberLines = 0.0;
    {
      float warp = fbm(vLocal.xy*4.0 + vec2(0.2*uTime, -0.1*uTime));
      float l = sin(vLocal.y*420.0 + warp*8.0);
      float widthAA = fwidth(l);
      fiberLines = smoothstep(0.6, 0.6 - widthAA, abs(l));
    }
    float grain = fbm(vLocal.xy*25.0);
    baseCol *= 1.0 + uFiber*(0.06*grain - 0.03) + uFiber*0.08*fiberLines;

    // Crease glow: nearest effective crease
    float minD = 1e9;
    for (int i=0;i<MAX_CREASES;i++){
      if (i >= uCreaseCount) break;
      vec2 a2 = uAeff[i].xy;
      vec2 d2 = normalize(uDeff[i].xy);
      float sd = abs(signedDistanceToLine(vLocal.xy, a2, d2));
      minD = min(minD, sd);
    }
    float aa = fwidth(minD);
    float edge = 1.0 - smoothstep(0.0025, 0.0025 + aa, minD);

    // Iridescence (view-dependent)
    vec3 V = normalize(cameraPosition - vPos);
    vec3 N = normalize(vN);
    float cosT = clamp(dot(N, V), 0.0, 1.0);
    vec3 film = thinFilm(cosT, uFilmIOR, uFilmNm);
    float F = pow(1.0 - cosT, 5.0);
    vec3 col = mix(baseCol, mix(baseCol, film, uIridescence), F);

    col += uEdgeGlow * edge * film * 0.7;

    float vign = smoothstep(1.2, 0.2, length(vUv-0.5)*1.2);
    gl_FragColor = vec4(col*vign, 1.0);
  }
`;

const mat = new THREE.ShaderMaterial({
  vertexShader: vs,
  fragmentShader: fs,
  uniforms,
  side: THREE.DoubleSide,
  extensions: { derivatives: true }
});
const sheet = new THREE.Mesh(geo, mat);
scene.add(sheet);

// Background dome
scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(50, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0x070711, side: THREE.BackSide })
));

// ---------- GUI (optional look tuning) ----------
const gui = new GUI();
const looks = gui.addFolder('Look');
looks.add(uniforms.uSectors, 'value', 3, 24, 1).name('kaleidoSectors');
looks.add(uniforms.uHueShift, 'value', 0, 1, 0.001).name('hueShift');
looks.add(uniforms.uIridescence, 'value', 0, 1, 0.001).name('iridescence');
looks.add(uniforms.uFilmIOR, 'value', 1.0, 2.333, 0.001).name('filmIOR');
looks.add(uniforms.uFilmNm, 'value', 100, 800, 1).name('filmThickness(nm)');
looks.add(uniforms.uFiber, 'value', 0, 1, 0.001).name('paperFiber');
looks.add(uniforms.uEdgeGlow, 'value', 0.0, 2.0, 0.01).name('edgeGlow');
looks.add(bloom, 'strength', 0.0, 2.5, 0.01).name('bloomStrength');
looks.add(bloom, 'radius', 0.0, 1.5, 0.01).name('bloomRadius');
looks.open();

// ---------- Presets (simple folds that “just work”) ----------
const VALLEY = +1, MOUNTAIN = -1;
const BAND = 0.006; // soft hinge width (meters in local space)

function preset_half_vertical_valley() {
  resetBase();
  // crease at x=0, direction = vertical
  addCrease(0, 0, 0, 1, 180, VALLEY, BAND); // 180° valley = fold in half
}
function preset_half_horizontal_valley() {
  resetBase();
  addCrease(0, 0, 1, 0, 180, VALLEY, BAND);
}
function preset_diagonal_valley() {
  resetBase();
  // diagonal across sheet (bottom-left to top-right)
  addCrease(0, 0, 1, 1, 180, VALLEY, BAND);
}
function preset_gate_valley() {
  resetBase();
  // two vertical valleys that fold edges to the center (quarters)
  const x = WIDTH * 0.25;
  addCrease(+x, 0, 0, 1, 180, VALLEY, BAND);
  addCrease(-x, 0, 0, 1, 180, VALLEY, BAND);
}
function preset_accordion_5() {
  resetBase();
  // 5 parallel creases across width, alternating valley/mountain
  const n = 5;
  for (let i=1;i<=n;i++){
    const x = THREE.MathUtils.lerp(-WIDTH/2, WIDTH/2, i/(n+1));
    const sign = (i % 2 === 1) ? VALLEY : MOUNTAIN;
    addCrease(x, 0, 0, 1, 135, sign, BAND); // 135° to avoid heavy overlap
  }
}
function preset_single_vertical_mountain() {
  resetBase();
  addCrease(0, 0, 0, 1, 180, MOUNTAIN, BAND);
}

// Apply a preset by key
function applyPreset(key){
  if (key === 'half-vertical-valley') preset_half_vertical_valley();
  else if (key === 'half-horizontal-valley') preset_half_horizontal_valley();
  else if (key === 'diagonal-valley') preset_diagonal_valley();
  else if (key === 'gate-valley') preset_gate_valley();
  else if (key === 'accordion-5') preset_accordion_5();
  else if (key === 'single-vertical-mountain') preset_single_vertical_mountain();
}

// ---------- DOM controls ----------
const presetSel = document.getElementById('preset');
const btnApply  = document.getElementById('btnApply');
const btnAnim   = document.getElementById('btnAnim');
const btnSnap   = document.getElementById('btnSnap');
const progress  = document.getElementById('progress');

btnApply.onclick = () => {
  applyPreset(presetSel.value);
  // tiny feedback nudge
  camera.position.x += (Math.random()-0.5) * 0.03;
  camera.position.y += (Math.random()-0.5) * 0.03;
};
presetSel.addEventListener('change', () => btnApply.click());

btnAnim.onclick = () => {
  drive.animate = !drive.animate;
  btnAnim.textContent = 'Animate: ' + (drive.animate ? 'On' : 'Off');
};
progress.addEventListener('input', () => { drive.progress = parseFloat(progress.value); });

btnSnap.onclick = () => {
  renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'origami.png'; a.click();
    URL.revokeObjectURL(url);
  }, 'image/png', 1.0);
};

// ---------- Start with something obvious ----------
applyPreset('half-vertical-valley');
progress.value = String(drive.progress);

// ---------- Per-frame update ----------
function updateFolding(tSec){
  computeAngles(tSec);
  computeEffectiveAxes();
  pushEffToUniforms();
}

function tick(t){
  const tSec = t * 0.001;
  uniforms.uTime.value = tSec;
  updateFolding(tSec);
  controls.update();
  composer.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ---------- Resize ----------
window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.setSize(w, h); composer.setSize(w, h);
  fxaa.material.uniforms.resolution.value.set(1 / w, 1 / h);
});
