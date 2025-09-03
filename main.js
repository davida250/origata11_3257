// Prism Fold — Voronoi “retessellation”, iridescent film, and ghosted motion.
// WebGL1-friendly shaders (no #version 300 es). Uses import map for addons.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { AfterimagePass } from 'three/addons/postprocessing/AfterimagePass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RGBShiftShader } from 'three/addons/shaders/RGBShiftShader.js';

// -----------------------------------------------------------------------------
// Renderer
// -----------------------------------------------------------------------------
const container = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;       // linear workflow with post; OutputPass at end handles conversion
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.debug.checkShaderErrors = true;
container.appendChild(renderer.domElement);

// -----------------------------------------------------------------------------
// Scene & Camera
// -----------------------------------------------------------------------------
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// -----------------------------------------------------------------------------
// Geometry + Shaders
// -----------------------------------------------------------------------------
const geo = new THREE.IcosahedronGeometry(1.0, 1);

// seed uniforms (up to 16 for WebGL1-friendly loops)
const MAX_SEEDS = 16;
const seedArray = Array.from({ length: MAX_SEEDS }, () => new THREE.Vector3());
function randomOnSphere(r = 1.2) {
  // Marsaglia method
  let x1, x2, s;
  do { x1 = Math.random() * 2 - 1; x2 = Math.random() * 2 - 1; s = x1 * x1 + x2 * x2; } while (s >= 1 || s === 0);
  const m = 2 * Math.sqrt(1 - s);
  return new THREE.Vector3(x1 * m, x2 * m, 1 - 2 * s).multiplyScalar(r);
}
function reseed(count, mode = 'random') {
  for (let i = 0; i < MAX_SEEDS; i++) {
    const v = (mode === 'regularize')
      ? new THREE.Vector3().setFromSphericalCoords(1.2, Math.acos(THREE.MathUtils.randFloatSpread(2)), 2 * Math.PI * (i / count))
      : randomOnSphere(1.2);
    seedArray[i].copy(v);
  }
  uniforms.uSeedCount.value = count | 0;
  // copy into uniform array
  for (let i = 0; i < MAX_SEEDS; i++) uniforms.uSeeds.value[i].copy(seedArray[i]);
}

// Uniforms
const uniforms = {
  uTime:           { value: 0 },
  uFold:           { value: 0.8 },       // legacy plane fold (mouse)
  uStripeFreq:     { value: 11.0 },
  uStripeMove:     { value: 1.25 },
  uThicknessBase:  { value: 420.0 },     // nm
  uIorFilm:        { value: 1.38 },
  uBaseColor:      { value: new THREE.Color(0x0a0f08) },

  // Voronoi remesh controls
  uSeedCount:      { value: 10 },
  uSeeds:          { value: Array.from({ length: MAX_SEEDS }, () => new THREE.Vector3()) },
  uVoronoiFold:    { value: 0.65 },
  uEdgeWidth:      { value: 0.18 },
  uEdgeGain:       { value: 1.4 }
};

// Vertex shader: legacy 3-plane fold + Voronoi cell folding (bisector plane reflect)
const vertexShader = `
precision highp float;

uniform float uTime;
uniform float uFold;

// Voronoi
uniform int   uSeedCount;
uniform vec3  uSeeds[${MAX_SEEDS}];
uniform float uVoronoiFold;
uniform float uEdgeWidth;

varying vec3 vNormal;
varying vec3 vPosView;
varying vec3 vPosWorld;
varying float vCellEdge;   // edge strength for shading
varying float vCellId;     // normalized cell id (for subtle tinting)

void foldPlane(inout vec3 p, inout vec3 nrm, vec3 pn, float d, float k){
  float s = dot(p, pn) + d;
  float m = smoothstep(0.0, 1.0, k * max(-s, 0.0));
  p  -= 2.0*m*s*pn;
  nrm = normalize(nrm - 2.0*m*dot(nrm, pn)*pn);
}

mat3 rotY(float a){
  float c = cos(a), s = sin(a);
  return mat3(c,0.,s, 0.,1.,0., -s,0.,c);
}

void voronoiFold(inout vec3 p, inout vec3 nrm, out float edge, out float idNorm) {
  float best1 = 1e9; float best2 = 1e9;
  vec3 s1 = vec3(0.0), s2 = vec3(0.0);
  int idx1 = 0;

  for (int i = 0; i < ${MAX_SEEDS}; i++) {
    if (i >= uSeedCount) break;
    vec3 s = uSeeds[i];
    float d = length(p - s);
    if (d < best1) {
      best2 = best1; s2 = s;
      best1 = d; s1 = s; idx1 = i;
    } else if (d < best2) {
      best2 = d; s2 = s;
    }
  }

  // distance delta between nearest and second-nearest → edge intensity
  float dv = max(1e-4, best2 - best1);
  edge = 1.0 - smoothstep(uEdgeWidth, uEdgeWidth * 2.0, dv);

  // reflect across the bisector plane between s1 and s2, scaled by edge
  vec3 pn = normalize(s2 - s1);
  vec3 mid = 0.5 * (s1 + s2);
  float sd = dot(p - mid, pn);
  p   -= 2.0 * uVoronoiFold * edge * sd * pn;
  nrm  = normalize(nrm - 2.0 * uVoronoiFold * edge * dot(nrm, pn) * pn);

  idNorm = float(idx1 + 1) / float(uSeedCount + 1);
}

void main(){
  vec3 p = position;
  vec3 nrm = normal;

  // subtle rotation for life
  float a = uTime * 0.25;
  mat3 R = rotY(a);
  p = R * p;
  nrm = R * nrm;

  // 3 animated macro folds (gives the big star-like deformations)
  vec3 p1 = normalize(vec3( 0.7, 0.0,  0.7));
  vec3 p2 = normalize(vec3(-0.3, 0.9,  0.1));
  vec3 p3 = normalize(vec3( 0.0, 0.7, -0.7));
  float d = 0.22 * sin(uTime * 0.6);

  foldPlane(p, nrm, p1,  d, uFold);
  foldPlane(p, nrm, p2, -d, uFold);
  foldPlane(p, nrm, p3,  d, uFold);

  // Voronoi-driven local folding (emulates retessellation)
  float edge, idn;
  voronoiFold(p, nrm, edge, idn);
  vCellEdge = edge;
  vCellId   = idn;

  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vPosView  = mv.xyz;
  vPosWorld = (modelMatrix * vec4(p,1.0)).xyz;
  vNormal   = normalize(normalMatrix * nrm);
  gl_Position = projectionMatrix * mv;
}
`;

// Fragment shader: thin-film + stripe bands + cell-edge boost + subtle backface glow
const fragmentShader = `
precision highp float;

uniform float uTime;
uniform float uStripeFreq;
uniform float uStripeMove;
uniform float uThicknessBase;
uniform float uIorFilm;
uniform vec3  uBaseColor;
uniform float uEdgeGain;

varying vec3 vNormal;
varying vec3 vPosView;
varying vec3 vPosWorld;
varying float vCellEdge;
varying float vCellId;

const float PI = 3.141592653589793;

// thin-film interference (approximate RGB wavelengths)
vec3 thinFilmIridescence(float thickness, float n1, float n2, float n3, float cosTheta1){
  vec3 lambda = vec3(680.0, 550.0, 440.0); // nm
  float sinTheta1 = sqrt(max(0.0, 1.0 - cosTheta1*cosTheta1));
  float sinTheta2 = n1 / n2 * sinTheta1;
  float cosTheta2 = sqrt(max(0.0, 1.0 - sinTheta2*sinTheta2));
  vec3 phase = 4.0 * PI * n2 * thickness * cosTheta2 / lambda;
  return 0.5 + 0.5 * cos(phase);
}

void main(){
  vec3 N = normalize(vNormal);
  vec3 V = normalize(-vPosView);
  float NdotV = clamp(dot(N, V), 0.0, 1.0);

  // Diagonal bands modulating thickness (slides over time)
  vec3 dir = normalize(vec3(0.7, 0.0, 0.3));
  float coord = dot(vPosWorld, dir) * uStripeFreq + uTime * uStripeMove;
  float s = 0.5 + 0.5 * sin(coord);
  float stripe = smoothstep(0.70, 0.98, s);

  // Film thickness varies with stripes (adds color flow)
  float thickness = uThicknessBase * (0.7 + 0.3 * sin(coord + uTime * 0.5));
  vec3 film = thinFilmIridescence(thickness, 1.0, uIorFilm, 1.0, NdotV);

  // Simple lighting + fresnel
  vec3 L = normalize(vec3(0.4, 0.8, 0.2));
  float diff = max(dot(N, L), 0.0);
  float f0 = 0.06;
  float fresnel = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);

  // Base + film
  vec3 color = uBaseColor * diff;
  color = mix(color, film, 0.75);
  color += film * stripe * 1.5;
  color += fresnel * film;

  // Voronoi cell edges pop (ghosty edge shine)
  color += film * vCellEdge * uEdgeGain;

  // Subtle backface glow (adds inner "ghost" feel)
  #ifdef GL_FRAGMENT_PRECISION_HIGH
    float back = gl_FrontFacing ? 0.0 : 1.0;
  #else
    float back = gl_FrontFacing ? 0.0 : 1.0;
  #endif
  color += back * film * 0.12;

  gl_FragColor = vec4(color, 1.0);
}
`;

const material = new THREE.ShaderMaterial({
  uniforms, vertexShader, fragmentShader, side: THREE.DoubleSide, transparent: false
});
const mesh = new THREE.Mesh(geo, material);
scene.add(mesh);

// -----------------------------------------------------------------------------
// Post-processing: Render → Bloom → Afterimage (ghosting) → RGBShift → Output
// -----------------------------------------------------------------------------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.1, 0.55, 0.15);
composer.addPass(bloomPass);

const afterPass = new AfterimagePass();
afterPass.damp = 0.82; // 0..1 (lower = longer trails). See example.  :contentReference[oaicite:5]{index=5}
composer.addPass(afterPass);

const rgbShiftPass = new ShaderPass(RGBShiftShader);
rgbShiftPass.uniforms['amount'].value = 0.0020;
rgbShiftPass.uniforms['angle'].value  = Math.PI / 4;
composer.addPass(rgbShiftPass);

composer.addPass(new OutputPass());

// -----------------------------------------------------------------------------
// UI wiring
// -----------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const ui = {
  followMouse: $('followMouse'),
  autoSpin: $('autoSpin'),
  spinSpeed: $('spinSpeed'), spinSpeedVal: $('spinSpeedVal'),

  seedCount: $('seedCount'), seedCountVal: $('seedCountVal'),
  vFold: $('vFold'), vFoldVal: $('vFoldVal'),
  edgeWidth: $('edgeWidth'), edgeWidthVal: $('edgeWidthVal'),
  edgeGain: $('edgeGain'), edgeGainVal: $('edgeGainVal'),
  reseed: $('reseed'), regularize: $('regularize'),

  fold: $('vFold'), // alias for mouse-drag fold is separate (uFold)

  stripeFreq: $('stripeFreq'), stripeFreqVal: $('stripeFreqVal'),
  stripeSpeed: $('stripeSpeed'), stripeSpeedVal: $('stripeSpeedVal'),
  thickness: $('thickness'), thicknessVal: $('thicknessVal'),
  ior: $('ior'), iorVal: $('iorVal'),

  after: $('after'), afterVal: $('afterVal'),
  rgbAmount: $('rgbAmount'), rgbAmountVal: $('rgbAmountVal'),
  bloomStrength: $('bloomStrength'), bloomStrengthVal: $('bloomStrengthVal'),
  bloomRadius: $('bloomRadius'), bloomRadiusVal: $('bloomRadiusVal'),
  bloomThreshold: $('bloomThreshold'), bloomThresholdVal: $('bloomThresholdVal'),

  reset: $('reset'), toggleBloom: $('toggleBloom')
};

const defaults = {
  followMouse: true,
  autoSpin: true,
  spinSpeed: 0.002,

  seedCount: 10,
  vFold: 0.65,
  edgeWidth: 0.18,
  edgeGain: 1.4,

  stripeFreq: 11.0,
  stripeSpeed: 1.25,
  thickness: 420,
  ior: 1.38,

  after: 0.82,
  rgbAmount: 0.0020,
  bloomStrength: 1.10,
  bloomRadius: 0.55,
  bloomThreshold: 0.15
};
const state = { ...defaults };

function setVal(el, v, digits=2){ el.textContent = (typeof v === 'number') ? v.toFixed(digits) : String(v); }
function syncUI(){
  $('followMouse').checked = state.followMouse;
  $('autoSpin').checked = state.autoSpin;
  $('spinSpeed').value = state.spinSpeed; setVal($('spinSpeedVal'), state.spinSpeed, 4);

  $('seedCount').value = state.seedCount; setVal($('seedCountVal'), state.seedCount, 0);
  $('vFold').value = state.vFold; setVal($('vFoldVal'), state.vFold);
  $('edgeWidth').value = state.edgeWidth; setVal($('edgeWidthVal'), state.edgeWidth, 2);
  $('edgeGain').value = state.edgeGain; setVal($('edgeGainVal'), state.edgeGain, 2);

  $('stripeFreq').value = state.stripeFreq; setVal($('stripeFreqVal'), state.stripeFreq, 1);
  $('stripeSpeed').value = state.stripeSpeed; setVal($('stripeSpeedVal'), state.stripeSpeed, 2);
  $('thickness').value = state.thickness; setVal($('thicknessVal'), state.thickness, 0);
  $('ior').value = state.ior; setVal($('iorVal'), state.ior, 3);

  $('after').value = state.after; setVal($('afterVal'), state.after, 3);
  $('rgbAmount').value = state.rgbAmount; setVal($('rgbAmountVal'), state.rgbAmount, 4);
  $('bloomStrength').value = state.bloomStrength; setVal($('bloomStrengthVal'), state.bloomStrength, 2);
  $('bloomRadius').value = state.bloomRadius; setVal($('bloomRadiusVal'), state.bloomRadius, 2);
  $('bloomThreshold').value = state.bloomThreshold; setVal($('bloomThresholdVal'), state.bloomThreshold, 2);
}

function applyState(){
  // Voronoi + fold
  uniforms.uVoronoiFold.value = state.vFold;
  uniforms.uEdgeWidth.value   = state.edgeWidth;
  uniforms.uEdgeGain.value    = state.edgeGain;
  uniforms.uFold.value        = THREE.MathUtils.lerp(0.2, 1.2, 0.5); // default mouse‑fold midpoint

  // Bands/film
  uniforms.uStripeFreq.value    = state.stripeFreq;
  uniforms.uStripeMove.value    = state.stripeSpeed;
  uniforms.uThicknessBase.value = state.thickness;
  uniforms.uIorFilm.value       = state.ior;

  // FX
  afterPass.damp = state.after; // Afterimage trail length (0..1), per example. :contentReference[oaicite:6]{index=6}
  bloomPass.strength  = state.bloomStrength;
  bloomPass.radius    = state.bloomRadius;
  bloomPass.threshold = state.bloomThreshold;
  rgbShiftPass.uniforms['amount'].value = state.rgbAmount;
}

syncUI();
reseed(state.seedCount, 'random');
applyState();

// Interactions
$('followMouse').addEventListener('change', () => state.followMouse = $('followMouse').checked);
$('autoSpin').addEventListener('change',   () => state.autoSpin   = $('autoSpin').checked);
$('spinSpeed').addEventListener('input', () => { state.spinSpeed = parseFloat($('spinSpeed').value); setVal($('spinSpeedVal'), state.spinSpeed, 4); });

$('seedCount').addEventListener('input', () => {
  state.seedCount = parseInt($('seedCount').value, 10);
  setVal($('seedCountVal'), state.seedCount, 0);
  reseed(state.seedCount, 'random');
});

$('vFold').addEventListener('input', () => { state.vFold = parseFloat($('vFold').value); setVal($('vFoldVal'), state.vFold); applyState(); });
$('edgeWidth').addEventListener('input', () => { state.edgeWidth = parseFloat($('edgeWidth').value); setVal($('edgeWidthVal'), state.edgeWidth, 2); applyState(); });
$('edgeGain').addEventListener('input', () => { state.edgeGain = parseFloat($('edgeGain').value); setVal($('edgeGainVal'), state.edgeGain, 2); applyState(); });

$('reseed').addEventListener('click', () => reseed(state.seedCount, 'random'));
$('regularize').addEventListener('click', () => reseed(state.seedCount, 'regularize'));

$('stripeFreq').addEventListener('input', () => { state.stripeFreq = parseFloat($('stripeFreq').value); setVal($('stripeFreqVal'), state.stripeFreq, 1); applyState(); });
$('stripeSpeed').addEventListener('input', () => { state.stripeSpeed = parseFloat($('stripeSpeed').value); setVal($('stripeSpeedVal'), state.stripeSpeed, 2); applyState(); });
$('thickness').addEventListener('input', () => { state.thickness = parseFloat($('thickness').value); setVal($('thicknessVal'), state.thickness, 0); applyState(); });
$('ior').addEventListener('input', () => { state.ior = parseFloat($('ior').value); setVal($('iorVal'), state.ior, 3); applyState(); });

$('after').addEventListener('input', () => { state.after = parseFloat($('after').value); setVal($('afterVal'), state.after, 3); applyState(); });
$('rgbAmount').addEventListener('input', () => { state.rgbAmount = parseFloat($('rgbAmount').value); setVal($('rgbAmountVal'), state.rgbAmount, 4); applyState(); });
$('bloomStrength').addEventListener('input', () => { state.bloomStrength = parseFloat($('bloomStrength').value); setVal($('bloomStrengthVal'), state.bloomStrength, 2); applyState(); });
$('bloomRadius').addEventListener('input', () => { state.bloomRadius = parseFloat($('bloomRadius').value); setVal($('bloomRadiusVal'), state.bloomRadius, 2); applyState(); });
$('bloomThreshold').addEventListener('input', () => { state.bloomThreshold = parseFloat($('bloomThreshold').value); setVal($('bloomThresholdVal'), state.bloomThreshold, 2); applyState(); });

$('reset').addEventListener('click', () => { Object.assign(state, defaults); syncUI(); reseed(state.seedCount, 'random'); applyState(); });
$('toggleBloom').addEventListener('click', () => { bloomPass.enabled = !bloomPass.enabled; });

// Pointer → macro fold amount (legacy)
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!state.followMouse) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  uniforms.uFold.value = THREE.MathUtils.lerp(0.2, 1.4, THREE.MathUtils.clamp(x, 0, 1));
});

// Wheel → stripe frequency quick-tune
renderer.domElement.addEventListener('wheel', (e) => {
  const delta = e.deltaY > 0 ? -0.5 : 0.5;
  state.stripeFreq = THREE.MathUtils.clamp(state.stripeFreq + delta, 2, 24);
  $('stripeFreq').value = state.stripeFreq; setVal($('stripeFreqVal'), state.stripeFreq, 1);
  applyState();
}, { passive: true });

// -----------------------------------------------------------------------------
// Resize & Animate
// -----------------------------------------------------------------------------
function onResize(){
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', onResize);

const clock = new THREE.Clock();
(function animate(){
  uniforms.uTime.value = clock.getElapsedTime();
  if (state.autoSpin) mesh.rotation.y += state.spinSpeed;
  controls.update();
  composer.render();
  requestAnimationFrame(animate);
})();
