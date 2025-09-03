// Cohesive folding + convex hull (straight edges) + thin-film & "weird" reflections + full-range ghosting.
// WebGL1-friendly ShaderMaterial (derivatives enabled for flat facets). Imports via import map.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { AfterimagePass } from 'three/addons/postprocessing/AfterimagePass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RGBShiftShader } from 'three/addons/shaders/RGBShiftShader.js';

// -----------------------------------------------------------------------------
// Renderer / Scene / Camera
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

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// -----------------------------------------------------------------------------
// Dynamic Convex Hull (straight-edged geometry)
// -----------------------------------------------------------------------------
const MAX_POINTS = 40;
let pointCount = 18;
let spike = 0.45;
let animatePts = true;
let ptSpeed = 0.25;

const seeds = new Array(MAX_POINTS).fill(0).map(() => new THREE.Vector3());
const base = new Array(MAX_POINTS).fill(0).map(() => new THREE.Vector3());

function reseed(mode = 'random') {
  const R = 1.0;
  for (let i = 0; i < MAX_POINTS; i++) {
    const t = (i / MAX_POINTS) * Math.PI * 2;
    const ring = new THREE.Vector3(Math.cos(t), 0, Math.sin(t));
    if (mode === 'regularize') {
      base[i].copy(ring.multiplyScalar(R));
    } else {
      // jittered rings with some vertical distribution
      base[i].set(
        (Math.random() * 2 - 1) * R,
        (Math.random() * 2 - 1) * 0.6,
        (Math.random() * 2 - 1) * R
      ).normalize().multiplyScalar(R);
    }
  }
}
reseed('random');

function updatePoints(t) {
  for (let i = 0; i < pointCount; i++) {
    const b = base[i];
    const amp = spike;                          // spikiness
    const w = animatePts ? ptSpeed : 0.0;       // angular speed factor
    const off = i * 0.37;
    const y = Math.sin(t * w + off) * 0.5;
    const radial = 1.0 + amp * Math.sin(t * (0.7 + 0.23 * Math.sin(off)) + off * 1.618);
    seeds[i].set(b.x * radial, b.y + 0.3 * y, b.z * radial);
  }
}

// Build / swap hull geometry + edges
let mesh, edgeLines;
function rebuildHull() {
  // dispose previous
  if (mesh) {
    mesh.geometry.dispose();
    scene.remove(mesh);
  }
  const pts = seeds.slice(0, pointCount);
  const geom = new ConvexGeometry(pts); // QuickHull under the hood (straight edges) :contentReference[oaicite:2]{index=2}

  mesh = new THREE.Mesh(geom, material);
  scene.add(mesh);

  if (edgeLines) {
    edgeLines.geometry.dispose();
    scene.remove(edgeLines);
  }
  const eGeo = new THREE.EdgesGeometry(geom, 15);
  const eMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: state.edgeOpacity });
  edgeLines = new THREE.LineSegments(eGeo, eMat);
  scene.add(edgeLines);
}

// -----------------------------------------------------------------------------
// Symmetry planes for cohesive folding
// -----------------------------------------------------------------------------
const MAX_PLANES = 9;
const planeArray = new Array(MAX_PLANES).fill(0).map(() => new THREE.Vector3());

function buildPlanes(count = 7) {
  const arr = [];
  const tilt = 0.35;
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    arr.push(new THREE.Vector3(Math.cos(a), tilt, Math.sin(a)).normalize());
  }
  arr.push(new THREE.Vector3(1, 1, 1).normalize());
  arr.push(new THREE.Vector3(-1, 1, 1).normalize());
  while (arr.length < count) arr.push(new THREE.Vector3(0, 1, 0));
  return arr.slice(0, Math.min(count, MAX_PLANES));
}

function loadPlanes(count) {
  const arr = buildPlanes(count);
  for (let i = 0; i < MAX_PLANES; i++) planeArray[i].copy(arr[i % arr.length]);
  uniforms.uPlaneCount.value = count | 0;
  for (let i = 0; i < MAX_PLANES; i++) uniforms.uPlanes.value[i].copy(planeArray[i]);
}

// -----------------------------------------------------------------------------
// Shader (GLSL1, flat-shaded facets via derivatives)
// -----------------------------------------------------------------------------
const uniforms = {
  uTime:           { value: 0 },
  // folding
  uFoldStrength:   { value: 1.0 },
  uFoldSoft:       { value: 0.16 },
  uPlaneCount:     { value: 7 },
  uPlanes:         { value: Array.from({ length: MAX_PLANES }, () => new THREE.Vector3()) },
  // material patterns
  uPatMix:         { value: 0.55 },  // 0=stripes, 1=noise
  uNoiseScale:     { value: 2.0 },
  uStripeFreq:     { value: 12.0 },
  uStripeMove:     { value: 1.10 },
  uThicknessBase:  { value: 430.0 },
  uIorFilm:        { value: 1.38 },
  uBaseColor:      { value: new THREE.Color(0x0a0f08) }
};

// Vertex shader: blended plane reflections (two passes) for cohesive folding
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
varying float vCrease;

vec3 reflectPoint(vec3 p, vec3 n, float d){
  float s = dot(p, n) + d;
  return p - 2.0 * s * n;
}
vec3 reflectNormal(vec3 nor, vec3 n){
  return normalize(nor - 2.0 * dot(nor, n) * n);
}
void blendedFold(inout vec3 p, inout vec3 nrm, vec3 n, float d, float strength, float softness, inout float creaseAcc){
  float s = dot(p, n) + d;
  float w = exp(-abs(s) / max(1e-3, softness));
  w = clamp(w * strength, 0.0, 1.0);
  vec3 pr = reflectPoint(p, n, d);
  vec3 nr = reflectNormal(nrm, n);
  p = mix(p, pr, w);
  nrm = normalize(mix(nrm, nr, w));
  creaseAcc = max(creaseAcc, w);
}
mat3 rotY(float a){ float c=cos(a), s=sin(a); return mat3(c,0.,s, 0.,1.,0., -s,0.,c); }

void main(){
  vec3 p   = position;
  vec3 nrm = normal;

  float a = uTime * 0.25;
  mat3 R = rotY(a);
  p = R * p;
  nrm = R * nrm;

  float crease = 0.0;
  for(int iter=0; iter<2; iter++){
    for(int i=0; i<${MAX_PLANES}; i++){
      if(i >= uPlaneCount) break;
      vec3 n = normalize(uPlanes[i]);
      float phase = float(i) * 1.618 + float(iter) * 0.73;
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

// Fragment shader: thin-film iridescence + noise/stripe hybrid + anisotropic glints.
// Flat shading via face normal from derivatives (OES_standard_derivatives).
const fragmentShader = `
#extension GL_OES_standard_derivatives : enable
precision highp float;

uniform float uTime;
uniform float uPatMix;
uniform float uNoiseScale;
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

// hash/noise utilities (cheap value noise)
float hash11(float n){ return fract(sin(n)*43758.5453123); }
float hash31(vec3 p){ return fract(sin(dot(p, vec3(127.1,311.7,74.7)))*43758.5453123); }
float noise3(vec3 x){
  vec3 i = floor(x), f = fract(x);
  float n = dot(i, vec3(1.0, 57.0, 113.0));
  float a = hash11(n + 0.0);
  float b = hash11(n + 1.0);
  float c = hash11(n + 57.0);
  float d = hash11(n + 58.0);
  float e = hash11(n + 113.0);
  float f1 = hash11(n + 114.0);
  float g = hash11(n + 170.0);
  float h = hash11(n + 171.0);
  vec3 u = f*f*(3.0-2.0*f);
  float xy1 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
  float xy2 = mix(mix(e,f1,u.x), mix(g,h,u.x), u.y);
  return mix(xy1, xy2, u.z);
}
float fbm(vec3 p){
  float s = 0.0, a = 0.5;
  for(int i=0;i<5;i++){ s += a * noise3(p); p *= 2.02; a *= 0.5; }
  return s;
}

// thin-film interference (approximate RGB wavelengths)  :contentReference[oaicite:3]{index=3}
vec3 thinFilmIridescence(float thickness, float n1, float n2, float n3, float cosTheta1){
  vec3 lambda = vec3(680.0, 550.0, 440.0); // nm
  float sinTheta1 = sqrt(max(0.0, 1.0 - cosTheta1*cosTheta1));
  float sinTheta2 = n1 / n2 * sinTheta1;
  float cosTheta2 = sqrt(max(0.0, 1.0 - sinTheta2*sinTheta2));
  vec3 phase = 4.0 * PI * n2 * thickness * cosTheta2 / lambda;
  return 0.5 + 0.5 * cos(phase);
}

void main(){
  // Face normal for flat-shaded facets (straight-edged look)
  vec3 Ng = normalize(cross(dFdx(vPosWorld), dFdy(vPosWorld)));
  if(dot(Ng, vNormal) < 0.0) Ng = -Ng;
  vec3 N = Ng;

  vec3 V = normalize(-vPosView);
  float NdotV = clamp(dot(N, V), 0.0, 1.0);

  // Hybrid pattern: diagonal bands + fbm noise (not just stripes)
  vec3 dir = normalize(vec3(0.7, 0.0, 0.3));
  float coord = dot(vPosWorld, dir) * uStripeFreq + uTime * uStripeMove;
  float stripes = smoothstep(0.72, 0.985, 0.5 + 0.5*sin(coord));
  float n = fbm(vPosWorld * uNoiseScale + vec3(0.0, uTime*0.15, 0.0));
  float pat = mix(stripes, n, uPatMix);

  // Film thickness varies with pattern → oily, weird reflections
  float thickness = uThicknessBase * (0.70 + 0.30 * pat);
  vec3 film = thinFilmIridescence(thickness, 1.0, uIorFilm, 1.0, NdotV);

  // Simple lighting + fresnel
  vec3 L = normalize(vec3(0.35, 0.9, 0.15));
  float diff = max(dot(N, L), 0.0);
  float f0 = 0.06;
  float fresnel = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);

  // Anisotropic "weird" glints based on reflection vector
  vec3 R = reflect(-V, N);
  vec3 A1 = normalize(vec3(0.2, 1.0, 0.0));
  vec3 A2 = normalize(vec3(-0.7, 0.3, 0.6));
  float aniso = pow(abs(dot(R, A1)), 24.0) + 0.5 * pow(abs(dot(R, A2)), 36.0);

  // Base + film + effects
  vec3 color = uBaseColor * diff;
  color = mix(color, film, 0.75);
  color += film * pat * 1.2;
  color += fresnel * film;
  color += aniso * film * 0.45;       // subtle glints
  color += vCrease * film * 0.35;     // crease highlight near folds

  gl_FragColor = vec4(color, 1.0);
}
`;

const material = new THREE.ShaderMaterial({
  uniforms,
  vertexShader,
  fragmentShader,
  side: THREE.DoubleSide,
  transparent: false,
  extensions: { derivatives: true } // enables dFdx/dFdy for flat facets
});

// Mesh placeholders (created in rebuildHull)
rebuildHull();

// -----------------------------------------------------------------------------
// Post-processing: Render → Bloom → Afterimage (ghosting) → RGBShift → Output
// -----------------------------------------------------------------------------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.90, 0.45, 0.18);
composer.addPass(bloomPass);

const afterPass = new AfterimagePass(); // exposes 'uniforms.damp' (see examples / forum) :contentReference[oaicite:4]{index=4}
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
const state = {
  // hull
  ptCount: 18, spike: 0.45, animatePts: true, ptSpeed: 0.25, edgeOpacity: 0.12,
  // fold
  foldStr: 1.0, foldSoft: 0.16, planeCount: 7,
  // material
  patMix: 0.55, noiseScale: 2.0, thick: 430, ior: 1.38, bandFreq: 12.0, bandSpeed: 1.10,
  // fx
  after: 0.96, rgb: 0.0010, bloomStr: 0.90, bloomRad: 0.45, bloomThr: 0.18
};

function setVal(id, v, d=2){ $(id).textContent = (typeof v === 'number') ? v.toFixed(d) : String(v); }
function syncUI(){
  $('#ptCount').value = state.ptCount; setVal('ptCountVal', state.ptCount, 0);
  $('#spike').value = state.spike; setVal('spikeVal', state.spike, 2);
  $('#animatePts').checked = state.animatePts;
  $('#ptSpeed').value = state.ptSpeed; setVal('ptSpeedVal', state.ptSpeed, 2);
  $('#edgeOpacity').value = state.edgeOpacity; setVal('edgeOpacityVal', state.edgeOpacity, 2);

  $('#foldStr').value = state.foldStr; setVal('foldStrVal', state.foldStr);
  $('#foldSoft').value = state.foldSoft; setVal('foldSoftVal', state.foldSoft, 2);
  $('#planeCount').value = state.planeCount; setVal('planeCountVal', state.planeCount, 0);

  $('#patMix').value = state.patMix; setVal('patMixVal', state.patMix, 2);
  $('#noiseScale').value = state.noiseScale; setVal('noiseScaleVal', state.noiseScale, 2);
  $('#thick').value = state.thick; setVal('thickVal', state.thick, 0);
  $('#ior').value = state.ior; setVal('iorVal', state.ior, 3);
  $('#bandFreq').value = state.bandFreq; setVal('bandFreqVal', state.bandFreq, 1);
  $('#bandSpeed').value = state.bandSpeed; setVal('bandSpeedVal', state.bandSpeed, 2);

  $('#after').value = state.after; setVal('afterVal', state.after, 4);
  $('#rgb').value = state.rgb; setVal('rgbVal', state.rgb, 4);
  $('#bloomStr').value = state.bloomStr; setVal('bloomStrVal', state.bloomStr, 2);
  $('#bloomRad').value = state.bloomRad; setVal('bloomRadVal', state.bloomRad, 2);
  $('#bloomThr').value = state.bloomThr; setVal('bloomThrVal', state.bloomThr, 2);
}
function applyState(){
  pointCount = state.ptCount;
  spike      = state.spike;
  animatePts = state.animatePts;
  ptSpeed    = state.ptSpeed;
  if (edgeLines) edgeLines.material.opacity = state.edgeOpacity;

  uniforms.uFoldStrength.value = state.foldStr;
  uniforms.uFoldSoft.value     = state.foldSoft;
  loadPlanes(state.planeCount);

  uniforms.uPatMix.value        = state.patMix;
  uniforms.uNoiseScale.value    = state.noiseScale;
  uniforms.uStripeFreq.value    = state.bandFreq;
  uniforms.uStripeMove.value    = state.bandSpeed;
  uniforms.uThicknessBase.value = state.thick;
  uniforms.uIorFilm.value       = state.ior;

  bloomPass.strength  = state.bloomStr;
  bloomPass.radius    = state.bloomRad;
  bloomPass.threshold = state.bloomThr;
  rgbShiftPass.uniforms['amount'].value = state.rgb;
}
syncUI(); applyState();

$('#ptCount').addEventListener('input', () => { state.ptCount = parseInt($('#ptCount').value,10); setVal('ptCountVal', state.ptCount, 0); rebuildHull(); });
$('#spike').addEventListener('input', () => { state.spike = parseFloat($('#spike').value); setVal('spikeVal', state.spike, 2); });
$('#animatePts').addEventListener('change', () => { state.animatePts = $('#animatePts').checked; });
$('#ptSpeed').addEventListener('input', () => { state.ptSpeed = parseFloat($('#ptSpeed').value); setVal('ptSpeedVal', state.ptSpeed, 2); });
$('#edgeOpacity').addEventListener('input', () => { state.edgeOpacity = parseFloat($('#edgeOpacity').value); setVal('edgeOpacityVal', state.edgeOpacity, 2); if(edgeLines) edgeLines.material.opacity = state.edgeOpacity; });

$('#reseed').addEventListener('click', () => { reseed('random'); rebuildHull(); });
$('#regularize').addEventListener('click', () => { reseed('regularize'); rebuildHull(); });

$('#foldStr').addEventListener('input', () => { state.foldStr = parseFloat($('#foldStr').value); setVal('foldStrVal', state.foldStr); applyState(); });
$('#foldSoft').addEventListener('input', () => { state.foldSoft = parseFloat($('#foldSoft').value); setVal('foldSoftVal', state.foldSoft, 2); applyState(); });
$('#planeCount').addEventListener('input', () => { state.planeCount = parseInt($('#planeCount').value,10); setVal('planeCountVal', state.planeCount, 0); applyState(); });

$('#patMix').addEventListener('input', () => { state.patMix = parseFloat($('#patMix').value); setVal('patMixVal', state.patMix, 2); applyState(); });
$('#noiseScale').addEventListener('input', () => { state.noiseScale = parseFloat($('#noiseScale').value); setVal('noiseScaleVal', state.noiseScale, 2); applyState(); });
$('#thick').addEventListener('input', () => { state.thick = parseFloat($('#thick').value); setVal('thickVal', state.thick, 0); applyState(); });
$('#ior').addEventListener('input', () => { state.ior = parseFloat($('#ior').value); setVal('iorVal', state.ior, 3); applyState(); });
$('#bandFreq').addEventListener('input', () => { state.bandFreq = parseFloat($('#bandFreq').value); setVal('bandFreqVal', state.bandFreq, 1); applyState(); });
$('#bandSpeed').addEventListener('input', () => { state.bandSpeed = parseFloat($('#bandSpeed').value); setVal('bandSpeedVal', state.bandSpeed, 2); applyState(); });

$('#after').addEventListener('input', () => { state.after = parseFloat($('#after').value); setVal('afterVal', state.after, 4); });
$('#rgb').addEventListener('input', () => { state.rgb = parseFloat($('#rgb').value); setVal('rgbVal', state.rgb, 4); rgbShiftPass.uniforms['amount'].value = state.rgb; });
$('#bloomStr').addEventListener('input', () => { state.bloomStr = parseFloat($('#bloomStr').value); setVal('bloomStrVal', state.bloomStr, 2); applyState(); });
$('#bloomRad').addEventListener('input', () => { state.bloomRad = parseFloat($('#bloomRad').value); setVal('bloomRadVal', state.bloomRad, 2); applyState(); });
$('#bloomThr').addEventListener('input', () => { state.bloomThr = parseFloat($('#bloomThr').value); setVal('bloomThrVal', state.bloomThr, 2); applyState(); });

$('#reset').addEventListener('click', () => {
  Object.assign(state, {
    ptCount: 18, spike: 0.45, animatePts: true, ptSpeed: 0.25, edgeOpacity: 0.12,
    foldStr: 1.0, foldSoft: 0.16, planeCount: 7,
    patMix: 0.55, noiseScale: 2.0, thick: 430, ior: 1.38, bandFreq: 12.0, bandSpeed: 1.10,
    after: 0.96, rgb: 0.0010, bloomStr: 0.90, bloomRad: 0.45, bloomThr: 0.18
  });
  syncUI(); applyState(); rebuildHull();
});
$('#toggleBloom').addEventListener('click', () => { bloomPass.enabled = !bloomPass.enabled; });

// Pointer gently modulates fold strength (kept subtle)
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!state.animatePts) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  uniforms.uFoldStrength.value = THREE.MathUtils.lerp(0.6, 1.3, THREE.MathUtils.clamp(x, 0, 1));
  $('#foldStr').value = uniforms.uFoldStrength.value;
  setVal('foldStrVal', uniforms.uFoldStrength.value);
});

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
rebuildHull(); // initial hull built once
loadPlanes(state.planeCount);

function animate(){
  const dt = clock.getDelta();
  const t  = (uniforms.uTime.value += dt);

  // Framerate-aware ghosting: consistent trail length across refresh rates. :contentReference[oaicite:5]{index=5}
  // AfterimagePass expects a 'damp' in [0..~1]. We adjust per-frame.
  const dampThisFrame = Math.pow(state.after, dt * 60.0);
  afterPass.uniforms['damp'].value = dampThisFrame;

  // Update hull points and rebuild geometry at a modest cadence
  updatePoints(t);
  rebuildHull();

  // Spin + render
  if (controls.enabled) controls.update();
  composer.render();
  requestAnimationFrame(animate);
}
animate();
