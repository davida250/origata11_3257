// Prism Fold — Cohesive folding, thin-film, subtle ghosting (WebGL1-friendly).
// Uses ShaderMaterial (built-in attributes/uniforms injected by three.js).

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
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
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
// Geometry (higher tessellation for smoother folds)
// -----------------------------------------------------------------------------
const geo = new THREE.IcosahedronGeometry(1.0, 3); // more vertices = nicer folding

// -----------------------------------------------------------------------------
// Symmetry planes: ring + diagonals (gives cohesive starry folds)
// -----------------------------------------------------------------------------
const MAX_PLANES = 9;
const planeArray = new Array(MAX_PLANES).fill(0).map(() => new THREE.Vector3());

function buildPlanes(count = 7) {
  const list = [];
  // 5 evenly spaced around Y (pentagonal vibe)
  const tilt = 0.35;
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    const n = new THREE.Vector3(Math.cos(a), tilt, Math.sin(a)).normalize();
    list.push(n);
  }
  // 2 diagonals for cohesion
  list.push(new THREE.Vector3(1, 1, 1).normalize());
  list.push(new THREE.Vector3(-1, 1, 1).normalize());
  // pad / trim to requested count (max 9)
  while (list.length < count) list.push(new THREE.Vector3(0, 1, 0));
  return list.slice(0, Math.min(count, MAX_PLANES));
}
function loadPlanesToUniform(count) {
  const arr = buildPlanes(count);
  for (let i = 0; i < MAX_PLANES; i++) planeArray[i].copy(arr[i % arr.length]);
  uniforms.uPlaneCount.value = count | 0;
  for (let i = 0; i < MAX_PLANES; i++) uniforms.uPlanes.value[i].copy(planeArray[i]);
}

// -----------------------------------------------------------------------------
// Uniforms & Shaders
// -----------------------------------------------------------------------------
const uniforms = {
  uTime:           { value: 0 },
  // Fold
  uFoldStrength:   { value: 0.90 },   // overall blend amount
  uFoldSoft:       { value: 0.18 },   // softness around planes (cohesion)
  uPlaneCount:     { value: 7 },
  uPlanes:         { value: Array.from({ length: MAX_PLANES }, () => new THREE.Vector3()) },
  // Material
  uStripeFreq:     { value: 12.0 },
  uStripeMove:     { value: 1.10 },
  uThicknessBase:  { value: 430.0 },  // nm
  uIorFilm:        { value: 1.38 },
  uBaseColor:      { value: new THREE.Color(0x0a0f08) }
};

// Vertex shader — blended reflections across multiple planes (2 iterations).
// WebGL1: no #version 300 es; rely on built-in position/normal/matrices.
const vertexShader = `
precision highp float;

uniform float uTime;
uniform float uFoldStrength;
uniform float uFoldSoft;
uniform int   uPlaneCount;
uniform vec3  uPlanes[${MAX_PLANES}];

varying vec3 vNormal;
varying vec3 vPosView;
varying vec3 vPosWorld;
varying float vCrease; // highlight near folds

// Mirror a point across a plane n·x + d = 0 (here d is animated).
vec3 reflectPoint(vec3 p, vec3 n, float d){
  float s = dot(p, n) + d;
  return p - 2.0 * s * n;
}
// Mirror a normal across plane normal n.
vec3 reflectNormal(vec3 nor, vec3 n){
  return normalize(nor - 2.0 * dot(nor, n) * n);
}

// Blend reflection with original based on smooth weight around the plane.
// This keeps the surface cohesive.
void blendedFold(inout vec3 p, inout vec3 nrm, vec3 n, float d, float strength, float softness, inout float creaseAcc){
  float s = dot(p, n) + d;
  float w = exp(-abs(s) / max(1e-3, softness)); // smooth, continuous
  w = clamp(w * strength, 0.0, 1.0);
  vec3 pr = reflectPoint(p, n, d);
  vec3 nr = reflectNormal(nrm, n);
  p = mix(p, pr, w);
  nrm = normalize(mix(nrm, nr, w));
  // accumulate crease indicator (for shading)
  creaseAcc = max(creaseAcc, w);
}

mat3 rotY(float a){ float c = cos(a), s = sin(a); return mat3(c,0.,s, 0.,1.,0., -s,0.,c); }

void main(){
  vec3 p   = position;
  vec3 nrm = normal;

  // gentle rotation for life
  float a = uTime * 0.25;
  mat3 R = rotY(a);
  p = R * p;
  nrm = R * nrm;

  float crease = 0.0;

  // Two passes of plane folds = deeper "fold into itself" while cohesive.
  // Planes drift slightly over time to create breathing motion.
  for(int iter=0; iter<2; iter++){
    for(int i=0; i<${MAX_PLANES}; i++){
      if(i >= uPlaneCount) break;
      vec3 n = normalize(uPlanes[i]);
      float phase = float(i) * 1.618 + float(iter) * 0.73; // golden-ish offsets
      float d = 0.18 * sin(uTime * 0.6 + phase);
      blendedFold(p, nrm, n, d, uFoldStrength, uFoldSoft, crease);
    }
  }

  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vPosView  = mv.xyz;
  vPosWorld = (modelMatrix * vec4(p,1.0)).xyz;
  vNormal   = normalize(normalMatrix * nrm);
  vCrease   = crease;

  gl_Position = projectionMatrix * mv;
}
`;

// Fragment shader — thin-film iridescence + bands + crease/rim emphasis.
const fragmentShader = `
precision highp float;

uniform float uTime;
uniform float uStripeFreq;
uniform float uStripeMove;
uniform float uThicknessBase;
uniform float uIorFilm;
uniform vec3  uBaseColor;

varying vec3 vNormal;
varying vec3 vPosView;
varying vec3 vPosWorld;
varying float vCrease;

const float PI = 3.141592653589793;

// Thin-film interference (approximate RGB wavelengths)
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

  // Diagonal bands modulating film thickness
  vec3 dir = normalize(vec3(0.7, 0.0, 0.3));
  float coord = dot(vPosWorld, dir) * uStripeFreq + uTime * uStripeMove;
  float s = 0.5 + 0.5 * sin(coord);
  float stripe = smoothstep(0.72, 0.985, s);

  float thickness = uThicknessBase * (0.72 + 0.28 * sin(coord + uTime * 0.5));
  vec3 film = thinFilmIridescence(thickness, 1.0, uIorFilm, 1.0, NdotV);

  // Simple lighting + fresnel
  vec3 L = normalize(vec3(0.4, 0.8, 0.2));
  float diff = max(dot(N, L), 0.0);
  float f0 = 0.06;
  float fresnel = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);

  vec3 color = uBaseColor * diff;
  color = mix(color, film, 0.75);
  color += film * stripe * 1.3;
  color += fresnel * film;

  // Crease highlight (subtle) - reads as 'ghost' edges in your refs
  color += film * vCrease * 0.45;

  // Slight backface glow to enhance cohesion
  float back = gl_FrontFacing ? 0.0 : 1.0;
  color += back * film * 0.08;

  gl_FragColor = vec4(color, 1.0);
}
`;

const material = new THREE.ShaderMaterial({
  uniforms,
  vertexShader,
  fragmentShader,
  side: THREE.DoubleSide,
  transparent: false
});

const mesh = new THREE.Mesh(geo, material);
scene.add(mesh);

// -----------------------------------------------------------------------------
// Post-processing (Render → Bloom → Afterimage → RGBShift → Output)
// -----------------------------------------------------------------------------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.90, // strength (lower by default, can increase from panel)
  0.45, // radius
  0.18  // threshold
);
composer.addPass(bloomPass);

const afterPass = new AfterimagePass();
composer.addPass(afterPass);

const rgbShiftPass = new ShaderPass(RGBShiftShader);
rgbShiftPass.uniforms['amount'].value = 0.0010;
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

  foldStrength: $('foldStrength'), foldStrengthVal: $('foldStrengthVal'),
  cohesion: $('cohesion'), cohesionVal: $('cohesionVal'),
  planeCount: $('planeCount'), planeCountVal: $('planeCountVal'),

  stripeFreq: $('stripeFreq'), stripeFreqVal: $('stripeFreqVal'),
  stripeSpeed: $('stripeSpeed'), stripeSpeedVal: $('stripeSpeedVal'),
  thickness: $('thickness'), thicknessVal: $('thicknessVal'),
  ior: $('ior'), iorVal: $('iorVal'),

  afterBase: $('afterBase'), afterBaseVal: $('afterBaseVal'),
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

  foldStrength: 0.90,
  cohesion: 0.18,
  planeCount: 7,

  stripeFreq: 12.0,
  stripeSpeed: 1.10,
  thickness: 430,
  ior: 1.38,

  afterBase: 0.96,       // base damp; per-frame adjusted for FPS consistency
  rgbAmount: 0.0010,
  bloomStrength: 0.90,
  bloomRadius: 0.45,
  bloomThreshold: 0.18
};
const state = { ...defaults };

function setVal(el, v, digits=2){ el.textContent = (typeof v === 'number') ? v.toFixed(digits) : String(v); }
function syncUI(){
  $('followMouse').checked = state.followMouse;
  $('autoSpin').checked = state.autoSpin;
  $('spinSpeed').value = state.spinSpeed; setVal($('spinSpeedVal'), state.spinSpeed, 4);

  $('foldStrength').value = state.foldStrength; setVal($('foldStrengthVal'), state.foldStrength);
  $('cohesion').value = state.cohesion; setVal($('cohesionVal'), state.cohesion, 2);
  $('planeCount').value = state.planeCount; setVal($('planeCountVal'), state.planeCount, 0);

  $('stripeFreq').value = state.stripeFreq; setVal($('stripeFreqVal'), state.stripeFreq, 1);
  $('stripeSpeed').value = state.stripeSpeed; setVal($('stripeSpeedVal'), state.stripeSpeed, 2);
  $('thickness').value = state.thickness; setVal($('thicknessVal'), state.thickness, 0);
  $('ior').value = state.ior; setVal($('iorVal'), state.ior, 3);

  $('afterBase').value = state.afterBase; setVal($('afterBaseVal'), state.afterBase, 3);
  $('rgbAmount').value = state.rgbAmount; setVal($('rgbAmountVal'), state.rgbAmount, 4);
  $('bloomStrength').value = state.bloomStrength; setVal($('bloomStrengthVal'), state.bloomStrength, 2);
  $('bloomRadius').value = state.bloomRadius; setVal($('bloomRadiusVal'), state.bloomRadius, 2);
  $('bloomThreshold').value = state.bloomThreshold; setVal($('bloomThresholdVal'), state.bloomThreshold, 2);
}
function applyState(){
  uniforms.uFoldStrength.value = state.foldStrength;
  uniforms.uFoldSoft.value     = state.cohesion;
  loadPlanesToUniform(state.planeCount);

  uniforms.uStripeFreq.value    = state.stripeFreq;
  uniforms.uStripeMove.value    = state.stripeSpeed;
  uniforms.uThicknessBase.value = state.thickness;
  uniforms.uIorFilm.value       = state.ior;

  bloomPass.strength  = state.bloomStrength;
  bloomPass.radius    = state.bloomRadius;
  bloomPass.threshold = state.bloomThreshold;

  rgbShiftPass.uniforms['amount'].value = state.rgbAmount;
}
syncUI(); applyState();

// Controls
$('followMouse').addEventListener('change', () => state.followMouse = $('followMouse').checked);
$('autoSpin').addEventListener('change',   () => state.autoSpin   = $('autoSpin').checked);
$('spinSpeed').addEventListener('input', () => { state.spinSpeed = parseFloat($('spinSpeed').value); setVal($('spinSpeedVal'), state.spinSpeed, 4); });

$('foldStrength').addEventListener('input', () => { state.foldStrength = parseFloat($('foldStrength').value); setVal($('foldStrengthVal'), state.foldStrength); applyState(); });
$('cohesion').addEventListener('input', () => { state.cohesion = parseFloat($('cohesion').value); setVal($('cohesionVal'), state.cohesion, 2); applyState(); });
$('planeCount').addEventListener('input', () => { state.planeCount = parseInt($('planeCount').value, 10); setVal($('planeCountVal'), state.planeCount, 0); applyState(); });

$('stripeFreq').addEventListener('input', () => { state.stripeFreq = parseFloat($('stripeFreq').value); setVal($('stripeFreqVal'), state.stripeFreq, 1); applyState(); });
$('stripeSpeed').addEventListener('input', () => { state.stripeSpeed = parseFloat($('stripeSpeed').value); setVal($('stripeSpeedVal'), state.stripeSpeed, 2); applyState(); });
$('thickness').addEventListener('input', () => { state.thickness = parseFloat($('thickness').value); setVal($('thicknessVal'), state.thickness, 0); applyState(); });
$('ior').addEventListener('input', () => { state.ior = parseFloat($('ior').value); setVal($('iorVal'), state.ior, 3); applyState(); });

$('afterBase').addEventListener('input', () => { state.afterBase = parseFloat($('afterBase').value); setVal($('afterBaseVal'), state.afterBase, 3); });
$('rgbAmount').addEventListener('input', () => { state.rgbAmount = parseFloat($('rgbAmount').value); setVal($('rgbAmountVal'), state.rgbAmount, 4); applyState(); });
$('bloomStrength').addEventListener('input', () => { state.bloomStrength = parseFloat($('bloomStrength').value); setVal($('bloomStrengthVal'), state.bloomStrength, 2); applyState(); });
$('bloomRadius').addEventListener('input', () => { state.bloomRadius = parseFloat($('bloomRadius').value); setVal($('bloomRadiusVal'), state.bloomRadius, 2); applyState(); });
$('bloomThreshold').addEventListener('input', () => { state.bloomThreshold = parseFloat($('bloomThreshold').value); setVal($('bloomThresholdVal'), state.bloomThreshold, 2); applyState(); });

$('reset').addEventListener('click', () => { Object.assign(state, defaults); syncUI(); applyState(); });
$('toggleBloom').addEventListener('click', () => { bloomPass.enabled = !bloomPass.enabled; });

// Pointer → fold strength modulation (kept very subtle to preserve cohesion)
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!state.followMouse) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  uniforms.uFoldStrength.value = THREE.MathUtils.lerp(0.5, 1.25, THREE.MathUtils.clamp(x, 0, 1));
  $('foldStrength').value = uniforms.uFoldStrength.value;
  setVal($('foldStrengthVal'), uniforms.uFoldStrength.value);
});

// Wheel → quick band density tweak
renderer.domElement.addEventListener('wheel', (e) => {
  const delta = e.deltaY > 0 ? -0.5 : 0.5;
  state.stripeFreq = THREE.MathUtils.clamp(state.stripeFreq + delta, 2, 24);
  $('stripeFreq').value = state.stripeFreq; setVal($('stripeFreqVal'), state.stripeFreq, 1);
  applyState();
}, { passive: true });

// -----------------------------------------------------------------------------
// Resize & Animate (framerate-aware afterimage damp)
// -----------------------------------------------------------------------------
function onResize(){
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', onResize);

const clock = new THREE.Clock();
function animate(){
  const dt = clock.getDelta();
  uniforms.uTime.value += dt;

  // framerate-independent ghosting: adjust damp per-frame around a 60Hz base
  const base = state.afterBase; // e.g. 0.96
  const dampThisFrame = Math.pow(base, dt * 60.0);
  // AfterimagePass exposes a 'uniforms' bag in examples; set it here.
  afterPass.uniforms['damp'].value = dampThisFrame;

  if (state.autoSpin) mesh.rotation.y += state.spinSpeed;
  controls.update();
  composer.render();
  requestAnimationFrame(animate);
}
animate();

// Initialize plane uniforms
loadPlanesToUniform(state.planeCount);
