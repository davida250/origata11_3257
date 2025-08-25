/**
 * Rigid-Hinge Origami — Basic folds that behave like real paper.
 * - Piecewise-rigid: each crease is a zero-thickness hinge (no bending elsewhere).
 * - Sequential: later crease axes are transformed by earlier folds.
 * - Visually identical shader stack: ACES filmic + Bloom + FXAA + iridescent paper.
 *
 * MV & angle convention follows MIT’s Origami Simulator / FOLD:
 *   foldAngle ∈ [-180°, 180°], positive = VALLEY, negative = MOUNTAIN.  (Design Tips)  [See README/refs]
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
fxaa.material.uniforms['resolution'].value.set(1 / window.innerWidth, 1 / window.innerHeight);
composer.addPass(fxaa);
composer.addPass(new OutputPass());

// ---------- Paper geometry ----------
const WIDTH = 4.0, HEIGHT = 2.4;
const SEG_X = 180, SEG_Y = 140; // dense grid -> crisp hinges without artifacts
const geo = new THREE.PlaneGeometry(WIDTH, HEIGHT, SEG_X, SEG_Y);
geo.rotateX(-0.25);

// ---------- Math helpers ----------
const tmp = {
  v1: new THREE.Vector3(),
  v2: new THREE.Vector3()
};
function signedDistance2(p /*Vec3*/, a /*Vec3*/, d /*unit Vec3*/) {
  // 2D side test in XY (paper plane)
  const px = p.x - a.x, py = p.y - a.y;
  return d.x * py - d.y * px; // z of 2D cross
}
function rotatePointAroundLine(p, a, axisUnit, ang) {
  tmp.v1.copy(p).sub(a).applyAxisAngle(axisUnit, ang).add(a);
  p.copy(tmp.v1);
}
function rotateVectorAxis(v, axisUnit, ang) {
  v.applyAxisAngle(axisUnit, ang);
}

// ---------- Crease sequence (ordered) ----------
const MAX_CREASES = 16; // enough for our basic folds
const base = {
  count: 0,
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  amp:  new Array(MAX_CREASES).fill(0),   // |angle| in radians (target)
  sign: new Array(MAX_CREASES).fill(1),   // +1=VALLEY, -1=MOUNTAIN
  phase:new Array(MAX_CREASES).fill(0)    // for animation
};
function resetBase(){
  base.count = 0;
  for (let i=0;i<MAX_CREASES;i++){
    base.A[i].set(0,0,0);
    base.D[i].set(1,0,0);
    base.amp[i]=0; base.sign[i]=1; base.phase[i]=0;
  }
}
function addCrease(Ax, Ay, Dx, Dy, deg=180, sign=+1){
  if (base.count >= MAX_CREASES) return;
  const i = base.count++;
  const d = new THREE.Vector2(Dx, Dy).normalize();
  base.A[i].set(Ax, Ay, 0);
  base.D[i].set(d.x, d.y, 0);
  base.amp[i]  = THREE.MathUtils.degToRad(Math.max(0, Math.min(180, Math.abs(deg))));
  base.sign[i] = sign >= 0 ? +1 : -1;
  base.phase[i]= Math.random()*Math.PI*2;
}

// ---------- Effective axes (sequential propagation) ----------
const eff = {
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  ang: new Float32Array(MAX_CREASES)
};

const drive = { animate:false, speed:0.9, progress:0.7 };
function computeAngles(tSec){
  for (let i=0;i<base.count;i++){
    const t = drive.animate ? (0.5 + 0.5*Math.sin(tSec*drive.speed + base.phase[i])) : drive.progress;
    eff.ang[i] = base.sign[i] * base.amp[i] * t;
  }
  for (let i=base.count;i<MAX_CREASES;i++) eff.ang[i] = 0;
}
function computeEffectiveAxes(){
  // start from base
  for (let i=0;i<base.count;i++){
    eff.A[i].copy(base.A[i]);
    eff.D[i].copy(base.D[i]).normalize();
  }
  // propagate: rotate later crease frames by earlier folds
  for (let j=0;j<base.count;j++){
    const Aj = eff.A[j];
    const Dj = eff.D[j].clone().normalize();
    const angle = eff.ang[j];
    if (Math.abs(angle) < 1e-8) continue;

    for (let k=j+1;k<base.count;k++){
      const sd = signedDistance2(eff.A[k], Aj, Dj);
      // CRISP HINGE: rotate only the positive side; negative side fixed
      if (sd > 0.0){
        rotatePointAroundLine(eff.A[k], Aj, Dj, angle);
        rotateVectorAxis(eff.D[k], Dj, angle);
        eff.D[k].normalize();
      }
    }
  }
}

// ---------- Uniforms (kept look; folding data is rigid & crisp) ----------
const uniforms = {
  uTime:       { value: 0 },
  uSectors:    { value: 10.0 },
  uHueShift:   { value: 0.0 },
  uIridescence:{ value: 0.65 },
  uFilmIOR:    { value: 1.35 },
  uFilmNm:     { value: 360.0 },
  uFiber:      { value: 0.35 },
  uEdgeGlow:   { value: 0.9 },

  uCreaseCount: { value: 0 },
  uAeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3()) },
  uDeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3(1,0,0)) },
  uAng:   { value: new Float32Array(MAX_CREASES) }
};

function pushEffToUniforms(){
  uniforms.uCreaseCount.value = base.count;
  uniforms.uAeff.value = eff.A.map(v => v.clone());
  uniforms.uDeff.value = eff.D.map(v => v.clone());
  uniforms.uAng.value  = Float32Array.from(eff.ang);
  mat.uniformsNeedUpdate = true; // force re-upload
}

// ---------- Shaders ----------
// Vertex: piecewise-rigid folding (no smoothing band, like real paper)
// Fragment: same psychedelic paper (iridescence+fibers+bloom)
const vs = /* glsl */`
  #define MAX_CREASES ${MAX_CREASES}
  precision highp float;
  uniform int   uCreaseCount;
  uniform vec3  uAeff[MAX_CREASES];
  uniform vec3  uDeff[MAX_CREASES];
  uniform float uAng[MAX_CREASES];
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
    // z of 2D cross, sign is which side of the crease we're on
    return d.x*(p.y - a.y) - d.y*(p.x - a.x);
  }

  void main(){
    vUv = uv;
    vec3 p = position;
    vec3 n = normalize(normal);

    for (int i=0; i<MAX_CREASES; i++){
      if (i >= uCreaseCount) break;
      vec3 a = uAeff[i];
      vec3 d = normalize(uDeff[i]);

      // CRISP HINGE: rotate ONLY the positive side; negative side remains
      float sd = signedDistanceToLine(p.xy, a.xy, d.xy);
      if (sd > 0.0){
        p = rotateAroundLine(p, a, d, uAng[i]);
        n = normalize(rotateVector(n, d, uAng[i]));
      }
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
    // Kaleidoscopic mapping in XZ (same look)
    float theta = atan(vPos.z, vPos.x);
    float r = length(vPos.xz) * 0.55;
    float seg = 2.0*PI / max(3.0, uSectors);
    float a = mod(theta, seg);
    a = abs(a - 0.5*seg);
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

    // Crease glow (distance to nearest effective crease)
    float minD = 1e9;
    for (int i=0; i<MAX_CREASES; i++){
      if (i >= uCreaseCount) break;
      vec2 a2 = uAeff[i].xy;
      vec2 d2 = normalize(uDeff[i].xy);
      float sd = abs(signedDistanceToLine(vLocal.xy, a2, d2));
      minD = min(minD, sd);
    }
    float aa = fwidth(minD);
    float edge = 1.0 - smoothstep(0.0025, 0.0025 + aa, minD);

    // Iridescence
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

// ---------- GUI (look tuning) ----------
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

// ---------- Simple presets (paper-like) ----------
const VALLEY = +1, MOUNTAIN = -1;

function preset_half_vertical_valley(){
  resetBase();
  addCrease(0, 0, 0, 1, 180, VALLEY); // x=0 vertical hinge
}
function preset_half_horizontal_valley(){
  resetBase();
  addCrease(0, 0, 1, 0, 180, VALLEY); // y=0 horizontal hinge
}
function preset_diagonal_valley(){
  resetBase();
  addCrease(0, 0, 1, 1, 180, VALLEY); // main diagonal
}
function preset_gate_valley(){
  resetBase();
  const x = WIDTH * 0.25;        // quarter folds to center
  addCrease(+x, 0, 0, 1, 180, VALLEY);
  addCrease(-x, 0, 0, 1, 180, VALLEY);
}
function preset_accordion_5(){
  resetBase();
  const n = 5;
  for (let i=1;i<=n;i++){
    const x = THREE.MathUtils.lerp(-WIDTH/2, WIDTH/2, i/(n+1));
    const sign = (i % 2 === 1) ? VALLEY : MOUNTAIN;
    addCrease(x, 0, 0, 1, 180, sign);
  }
}
function preset_single_vertical_mountain(){
  resetBase();
  addCrease(0, 0, 0, 1, 180, MOUNTAIN);
}
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

// ---------- Start with an obvious fold ----------
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
  fxaa.material.uniforms['resolution'].value.set(1 / w, 1 / h);
});
