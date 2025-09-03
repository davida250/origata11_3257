// Prism Fold — Three.js (WebGL1-compatible shaders)
// Faceted geometry "folding in on itself" with thin‑film iridescence and post‑FX.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RGBShiftShader } from 'three/addons/shaders/RGBShiftShader.js';

// --- renderer --------------------------------------------------------------
const container = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({
  antialias: false,
  powerPreference: 'high-performance'
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.debug.checkShaderErrors = true; // helpful if something goes wrong
container.appendChild(renderer.domElement);

// --- scene & camera --------------------------------------------------------
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  45, window.innerWidth / window.innerHeight, 0.1, 100
);
camera.position.set(0, 0, 5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// --- geometry + shaders ----------------------------------------------------
const geo = new THREE.IcosahedronGeometry(1.0, 1);

const uniforms = {
  uTime:           { value: 0 },
  uFold:           { value: 0.8 },
  uStripeFreq:     { value: 11.0 },
  uStripeMove:     { value: 1.25 },
  uThicknessBase:  { value: 420.0 },  // nm
  uIorFilm:        { value: 1.38 },
  uBaseColor:      { value: new THREE.Color(0x0a0f08) }
};

// GLSL 1.00 versions (no "#version 300 es") for WebGL1 compatibility.
const vertexShader = `
precision highp float;

uniform float uTime;
uniform float uFold;

attribute vec3 position;
attribute vec3 normal;

uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

varying vec3 vNormal;
varying vec3 vPosView;
varying vec3 vPosWorld;

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

void main(){
  vec3 p = position;
  vec3 nrm = normal;

  // subtle rotation for life
  float a = uTime * 0.25;
  mat3 R = rotY(a);
  p = R * p;
  nrm = R * nrm;

  // three animated folding planes
  vec3 p1 = normalize(vec3( 0.7, 0.0,  0.7));
  vec3 p2 = normalize(vec3(-0.3, 0.9,  0.1));
  vec3 p3 = normalize(vec3( 0.0, 0.7, -0.7));
  float d = 0.22 * sin(uTime * 0.6);

  foldPlane(p, nrm, p1,  d, uFold);
  foldPlane(p, nrm, p2, -d, uFold);
  foldPlane(p, nrm, p3,  d, uFold);

  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vPosView  = mv.xyz;
  vPosWorld = (modelMatrix * vec4(p,1.0)).xyz;
  vNormal   = normalize(normalMatrix * nrm);

  gl_Position = projectionMatrix * mv;
}
`;

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

const float PI = 3.141592653589793;

// Simplified thin‑film interference (per‑channel phase vs. wavelength)
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

  // diagonal stripes in object/world space
  vec3 dir = normalize(vec3(0.7, 0.0, 0.3));
  float coord = dot(vPosWorld, dir) * uStripeFreq + uTime * uStripeMove;
  float s = 0.5 + 0.5 * sin(coord);
  float stripe = smoothstep(0.70, 0.98, s); // thin bright lines

  float thickness = uThicknessBase * (0.7 + 0.3 * sin(coord + uTime * 0.5));
  vec3 film = thinFilmIridescence(thickness, 1.0, uIorFilm, 1.0, NdotV);

  // simple lambert + fresnel & rim
  vec3 L = normalize(vec3(0.4, 0.8, 0.2));
  float diff = max(dot(N, L), 0.0);
  float f0 = 0.06;
  float fresnel = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);

  vec3 color = uBaseColor * diff;
  color = mix(color, film, 0.75);
  color += film * stripe * 1.5;
  color += fresnel * film;
  color += pow(1.0 - NdotV, 2.0) * film * 0.2;

  gl_FragColor = vec4(color, 1.0);
}
`;

const material = new THREE.ShaderMaterial({
  uniforms,
  vertexShader,
  fragmentShader,
  // GLSL1 by default; no glslVersion set → works on WebGL1 renderers
  side: THREE.DoubleSide,
  transparent: false
});

const mesh = new THREE.Mesh(geo, material);
scene.add(mesh);

// --- post-processing -------------------------------------------------------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  1.05, // strength
  0.5,  // radius
  0.15  // threshold
);
composer.addPass(bloomPass);

const rgbShift = new ShaderPass(RGBShiftShader);
rgbShift.uniforms.amount.value = 0.0015;
rgbShift.uniforms.angle.value  = Math.PI / 4;
composer.addPass(rgbShift);

composer.addPass(new OutputPass()); // ensures tone mapping & output color space

// --- UI wiring -------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const ui = {
  followMouse: $('followMouse'),
  autoSpin: $('autoSpin'),
  spinSpeed: $('spinSpeed'), spinSpeedVal: $('spinSpeedVal'),
  fold: $('fold'), foldVal: $('foldVal'),
  stripeFreq: $('stripeFreq'), stripeFreqVal: $('stripeFreqVal'),
  stripeSpeed: $('stripeSpeed'), stripeSpeedVal: $('stripeSpeedVal'),
  thickness: $('thickness'), thicknessVal: $('thicknessVal'),
  ior: $('ior'), iorVal: $('iorVal'),
  bloomStrength: $('bloomStrength'), bloomStrengthVal: $('bloomStrengthVal'),
  bloomRadius: $('bloomRadius'), bloomRadiusVal: $('bloomRadiusVal'),
  bloomThreshold: $('bloomThreshold'), bloomThresholdVal: $('bloomThresholdVal'),
  rgbAmount: $('rgbAmount'), rgbAmountVal: $('rgbAmountVal'),
  reset: $('reset'),
  toggleBloom: $('toggleBloom')
};

const defaults = {
  followMouse: true,
  autoSpin: true,
  spinSpeed: 0.002,
  fold: 0.8,
  stripeFreq: 11.0,
  stripeSpeed: 1.25,
  thickness: 420,
  ior: 1.38,
  bloomStrength: 1.05,
  bloomRadius: 0.5,
  bloomThreshold: 0.15,
  rgbAmount: 0.0015
};
const state = { ...defaults };

function setVal(el, v, digits=2) { el.textContent = (typeof v === 'number') ? v.toFixed(digits) : String(v); }
function syncUI() {
  ui.followMouse.checked = state.followMouse;
  ui.autoSpin.checked = state.autoSpin;
  ui.spinSpeed.value = state.spinSpeed; setVal(ui.spinSpeedVal, state.spinSpeed, 4);

  ui.fold.value = state.fold; setVal(ui.foldVal, state.fold);
  ui.stripeFreq.value = state.stripeFreq; setVal(ui.stripeFreqVal, state.stripeFreq, 1);
  ui.stripeSpeed.value = state.stripeSpeed; setVal(ui.stripeSpeedVal, state.stripeSpeed, 2);

  ui.thickness.value = state.thickness; setVal(ui.thicknessVal, state.thickness, 0);
  ui.ior.value = state.ior; setVal(ui.iorVal, state.ior, 3);

  ui.bloomStrength.value = state.bloomStrength; setVal(ui.bloomStrengthVal, state.bloomStrength, 2);
  ui.bloomRadius.value = state.bloomRadius; setVal(ui.bloomRadiusVal, state.bloomRadius, 2);
  ui.bloomThreshold.value = state.bloomThreshold; setVal(ui.bloomThresholdVal, state.bloomThreshold, 2);

  ui.rgbAmount.value = state.rgbAmount; setVal(ui.rgbAmountVal, state.rgbAmount, 4);
}
function applyStateToScene() {
  uniforms.uFold.value          = state.fold;
  uniforms.uStripeFreq.value    = state.stripeFreq;
  uniforms.uStripeMove.value    = state.stripeSpeed;
  uniforms.uThicknessBase.value = state.thickness;
  uniforms.uIorFilm.value       = state.ior;

  bloomPass.strength  = state.bloomStrength;
  bloomPass.radius    = state.bloomRadius;
  bloomPass.threshold = state.bloomThreshold;

  rgbShift.uniforms.amount.value = state.rgbAmount;
}
syncUI(); applyStateToScene();

ui.followMouse.addEventListener('change', () => { state.followMouse = ui.followMouse.checked; });
ui.autoSpin.addEventListener('change',   () => { state.autoSpin   = ui.autoSpin.checked; });
ui.spinSpeed.addEventListener('input', () => { state.spinSpeed = parseFloat(ui.spinSpeed.value); setVal(ui.spinSpeedVal, state.spinSpeed, 4); });

ui.fold.addEventListener('input', () => { state.fold = parseFloat(ui.fold.value); setVal(ui.foldVal, state.fold); applyStateToScene(); });
ui.stripeFreq.addEventListener('input', () => { state.stripeFreq = parseFloat(ui.stripeFreq.value); setVal(ui.stripeFreqVal, state.stripeFreq, 1); applyStateToScene(); });
ui.stripeSpeed.addEventListener('input', () => { state.stripeSpeed = parseFloat(ui.stripeSpeed.value); setVal(ui.stripeSpeedVal, state.stripeSpeed, 2); applyStateToScene(); });
ui.thickness.addEventListener('input', () => { state.thickness = parseFloat(ui.thickness.value); setVal(ui.thicknessVal, state.thickness, 0); applyStateToScene(); });
ui.ior.addEventListener('input', () => { state.ior = parseFloat(ui.ior.value); setVal(ui.iorVal, state.ior, 3); applyStateToScene(); });
ui.bloomStrength.addEventListener('input', () => { state.bloomStrength = parseFloat(ui.bloomStrength.value); setVal(ui.bloomStrengthVal, state.bloomStrength, 2); applyStateToScene(); });
ui.bloomRadius.addEventListener('input', () => { state.bloomRadius = parseFloat(ui.bloomRadius.value); setVal(ui.bloomRadiusVal, state.bloomRadius, 2); applyStateToScene(); });
ui.bloomThreshold.addEventListener('input', () => { state.bloomThreshold = parseFloat(ui.bloomThreshold.value); setVal(ui.bloomThresholdVal, state.bloomThreshold, 2); applyStateToScene(); });
ui.rgbAmount.addEventListener('input', () => { state.rgbAmount = parseFloat(ui.rgbAmount.value); setVal(ui.rgbAmountVal, state.rgbAmount, 4); applyStateToScene(); });

ui.reset.addEventListener('click', () => { Object.assign(state, defaults); syncUI(); applyStateToScene(); });
ui.toggleBloom.addEventListener('click', () => { bloomPass.enabled = !bloomPass.enabled; });

// pointer → fold amount (if enabled)
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!state.followMouse) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  state.fold = THREE.MathUtils.lerp(0.2, 1.4, THREE.MathUtils.clamp(x, 0, 1));
  ui.fold.value = state.fold; ui.foldVal.textContent = state.fold.toFixed(2);
  applyStateToScene();
});
// wheel on canvas adjusts stripes
renderer.domElement.addEventListener('wheel', (e) => {
  const delta = e.deltaY > 0 ? -0.5 : 0.5;
  state.stripeFreq = THREE.MathUtils.clamp(state.stripeFreq + delta, 2, 24);
  ui.stripeFreq.value = state.stripeFreq; ui.stripeFreqVal.textContent = state.stripeFreq.toFixed(1);
  applyStateToScene();
}, { passive: true });

// --- resize & animate -------------------------------------------------------
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
