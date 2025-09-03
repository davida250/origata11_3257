// Convex hull (straight edges) + cohesive folding + thin‑film + subtle ghosting.
// Fixes QuickHull error by populating valid points BEFORE building hull.

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

const { clamp, lerp } = THREE.MathUtils;

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
// UI helpers
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
  after: 0.96, rgb: 0.0010, bloomStr: 0.90, bloomRad: 0.45, bloomThr: 0.18,
  autoSpin: true, spinSpeed: 0.002
};
function setVal(id, v, d = 2){ $(id).textContent = (typeof v === 'number') ? v.toFixed(d) : String(v); }
function syncUI(){
  $('ptCount').value = state.ptCount; setVal('ptCountVal', state.ptCount, 0);
  $('spike').value   = state.spike;   setVal('spikeVal', state.spike, 2);
  $('animatePts').checked = state.animatePts;
  $('ptSpeed').value = state.ptSpeed; setVal('ptSpeedVal', state.ptSpeed, 2);
  $('edgeOpacity').value = state.edgeOpacity; setVal('edgeOpacityVal', state.edgeOpacity, 2);

  $('foldStr').value = state.foldStr; setVal('foldStrVal', state.foldStr);
  $('foldSoft').value = state.foldSoft; setVal('foldSoftVal', state.foldSoft, 2);
  $('planeCount').value = state.planeCount; setVal('planeCountVal', state.planeCount, 0);

  $('patMix').value = state.patMix; setVal('patMixVal', state.patMix, 2);
  $('noiseScale').value = state.noiseScale; setVal('noiseScaleVal', state.noiseScale, 2);
  $('thick').value = state.thick; setVal('thickVal', state.thick, 0);
  $('ior').value = state.ior; setVal('iorVal', state.ior, 3);
  $('bandFreq').value = state.bandFreq; setVal('bandFreqVal', state.bandFreq, 1);
  $('bandSpeed').value = state.bandSpeed; setVal('bandSpeedVal', state.bandSpeed, 2);

  $('after').value = state.after; setVal('afterVal', state.after, 4);
  $('rgb').value = state.rgb; setVal('rgbVal', state.rgb, 4);
  $('bloomStr').value = state.bloomStr; setVal('bloomStrVal', state.bloomStr, 2);
  $('bloomRad').value = state.bloomRad; setVal('bloomRadVal', state.bloomRad, 2);
  $('bloomThr').value = state.bloomThr; setVal('bloomThrVal', state.bloomThr, 2);
}
syncUI();

// -----------------------------------------------------------------------------
// Dynamic Convex Hull (straight-edged geometry) — robust build
// -----------------------------------------------------------------------------
const MAX_POINTS = 40;
const seeds = Array.from({ length: MAX_POINTS }, () => new THREE.Vector3());
const base  = Array.from({ length: MAX_POINTS }, () => new THREE.Vector3());

// initialize base points (non-degenerate)
function reseed(mode = 'random') {
  const R = 1.0;
  for (let i = 0; i < MAX_POINTS; i++) {
    if (mode === 'regularize') {
      // points roughly on a ring (then animated)
      const a = (i / MAX_POINTS) * Math.PI * 2;
      base[i].set(Math.cos(a), 0, Math.sin(a)).multiplyScalar(R);
    } else {
      // jittered sphere sampling
      const u = Math.random(), v = Math.random();
      const th = 2 * Math.PI * u, ph = Math.acos(2 * v - 1);
      base[i].set(
        Math.sin(ph) * Math.cos(th),
        Math.cos(ph) * 0.6,                     // squash Y a bit
        Math.sin(ph) * Math.sin(th)
      ).normalize().multiplyScalar(R);
    }
  }
}
reseed('random');

function updatePoints(t) {
  const n = Math.max(4, Math.min(state.ptCount, MAX_POINTS));
  const amp = state.spike;
  const speed = state.animatePts ? state.ptSpeed : 0.0;

  for (let i = 0; i < n; i++) {
    const off = i * 0.37;
    const b = base[i];
    const radial = 1.0 + amp * Math.sin(t * (0.7 + 0.23 * Math.sin(off)) + off * 1.618);
    const y = 0.3 * Math.sin(t * speed + off);
    seeds[i].set(b.x * radial, b.y + y, b.z * radial);
    // add tiny jitter so no two points are exactly equal (avoids hull degeneracy)
    seeds[i].x += 1e-4 * Math.sin(999.1 * (i + 0.123));
    seeds[i].y += 1e-4 * Math.sin(777.7 * (i + 0.456));
    seeds[i].z += 1e-4 * Math.sin(555.5 * (i + 0.789));
  }
}

// produce a clean array of points for ConvexGeometry
function gatherHullPoints() {
  const n = Math.max(4, Math.min(state.ptCount, MAX_POINTS));
  const uniq = new Map();
  const pts = [];
  for (let i = 0; i < n; i++) {
    const p = seeds[i];
    if (!isFinite(p.x) || !isFinite(p.y) || !isFinite(p.z)) continue;
    const key = `${p.x.toFixed(6)},${p.y.toFixed(6)},${p.z.toFixed(6)}`;
    if (!uniq.has(key)) { uniq.set(key, true); pts.push(p.clone()); }
  }
  return pts;
}

let mesh, edgeLines;

// (Re)build the hull; guard against invalid/degenerate point sets
function rebuildHull() {
  const pts = gatherHullPoints();
  if (pts.length < 4) return; // not enough distinct points yet

  // dispose previous
  if (mesh) { mesh.geometry.dispose(); scene.remove(mesh); }
  if (edgeLines) { edgeLines.geometry.dispose(); scene.remove(edgeLines); }

  // Convex hull via QuickHull (examples addon)  :contentReference[oaicite:4]{index=4}
  const geom = new ConvexGeometry(pts);

  mesh = new THREE.Mesh(geom, material);
  scene.add(mesh);

  const eGeo = new THREE.EdgesGeometry(geom, 15);
  const eMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: state.edgeOpacity });
  edgeLines = new THREE.LineSegments(eGeo, eMat);
  scene.add(edgeLines);
}

// -----------------------------------------------------------------------------
// Symmetry planes for cohesive folding
// -----------------------------------------------------------------------------
const MAX_PLANES = 9;
const planeArray = Array.from({ length: MAX_PLANES }, () => new THREE.Vector3());
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

// -----------------------------------------------------------------------------
// Shader (GLSL1, flat-shaded facets via derivatives)
// -----------------------------------------------------------------------------
const uniforms = {
  uTime:           { value: 0 },
  // folding
  uFoldStrength:   { value: state.foldStr },
  uFoldSoft:       { value: state.foldSoft },
  uPlaneCount:     { value: state.planeCount },
  uPlanes:         { value: Array.from({ length: MAX_PLANES }, () => new THREE.Vector3()) },
  // material patterns
  uPatMix:         { value: state.patMix },   // 0=stripes, 1=noise
  uNoiseScale:     { value: state.noiseScale },
  uStripeFreq:     { value: state.bandFreq },
  uStripeMove:     { value: state.bandSpeed },
  uThicknessBase:  { value: state.thick },
  uIorFilm:        { value: state.ior },
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

vec3 reflectPoint(vec3 p, vec3 n, float d){ float s = dot(p, n) + d; return p - 2.0 * s * n; }
vec3 reflectNormal(vec3 nor, vec3 n){ return normalize(nor - 2.0 * dot(nor, n) * n); }
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

// Fragment shader: thin-film + hybrid pattern + flat facets (derivatives)
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

// cheap value noise + fbm
float hash11(float n){ return fract(sin(n)*43758.5453123); }
float noise3(vec3 x){
  vec3 i = floor(x), f = fract(x);
  float n = dot(i, vec3(1.0, 57.0, 113.0));
  float a=hash11(n), b=hash11(n+1.0), c=hash11(n+57.0), d=hash11(n+58.0);
  float e=hash11(n+113.0), f1=hash11(n+114.0), g=hash11(n+170.0), h=hash11(n+171.0);
  vec3 u = f*f*(3.0-2.0*f);
  float xy1 = mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
  float xy2 = mix(mix(e,f1,u.x), mix(g,h,u.x), u.y);
  return mix(xy1, xy2, u.z);
}
float fbm(vec3 p){ float s = 0.0, a = 0.5; for(int i=0;i<5;i++){ s += a * noise3(p); p *= 2.02; a *= 0.5; } return s; }

// thin-film interference (approx RGB wavelengths)
vec3 thinFilmIridescence(float thickness, float n1, float n2, float n3, float cosTheta1){
  vec3 lambda = vec3(680.0, 550.0, 440.0); // nm
  float sinTheta1 = sqrt(max(0.0, 1.0 - cosTheta1*cosTheta1));
  float sinTheta2 = n1 / n2 * sinTheta1;
  float cosTheta2 = sqrt(max(0.0, 1.0 - sinTheta2*sinTheta2));
  vec3 phase = 4.0 * PI * n2 * thickness * cosTheta2 / lambda;
  return 0.5 + 0.5 * cos(phase);
}

void main(){
  // flat-shaded facet normal from derivatives (needs OES_standard_derivatives)
  vec3 Ng = normalize(cross(dFdx(vPosWorld), dFdy(vPosWorld)));
  if(dot(Ng, vNormal) < 0.0) Ng = -Ng;
  vec3 N = Ng;

  vec3 V = normalize(-vPosView);
  float NdotV = clamp(dot(N, V), 0.0, 1.0);

  // hybrid pattern (bands + fbm) → nontrivial reflections
  vec3 dir = normalize(vec3(0.7, 0.0, 0.3));
  float coord = dot(vPosWorld, dir) * uStripeFreq + uTime * uStripeMove;
  float stripes = smoothstep(0.72, 0.985, 0.5 + 0.5*sin(coord));
  float n = fbm(vPosWorld * uNoiseScale + vec3(0.0, uTime*0.15, 0.0));
  float pat = mix(stripes, n, uPatMix);

  float thickness = uThicknessBase * (0.70 + 0.30 * pat);
  vec3 film = thinFilmIridescence(thickness, 1.0, uIorFilm, 1.0, NdotV);

  vec3 L = normalize(vec3(0.35, 0.9, 0.15));
  float diff = max(dot(N, L), 0.0);
  float f0 = 0.06;
  float fresnel = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);

  // anisotropic glints
  vec3 R = reflect(-V, N);
  vec3 A1 = normalize(vec3(0.2, 1.0, 0.0));
  vec3 A2 = normalize(vec3(-0.7, 0.3, 0.6));
  float aniso = pow(abs(dot(R, A1)), 24.0) + 0.5 * pow(abs(dot(R, A2)), 36.0);

  vec3 color = uBaseColor * diff;
  color = mix(color, film, 0.75);
  color += film * pat * 1.2;
  color += fresnel * film;
  color += aniso * film * 0.45;
  color += vCrease * film * 0.35;

  gl_FragColor = vec4(color, 1.0);
}
`;

const material = new THREE.ShaderMaterial({
  uniforms, vertexShader, fragmentShader, side: THREE.DoubleSide, transparent: false,
  extensions: { derivatives: true } // enable dFdx/dFdy for flat facets  :contentReference[oaicite:5]{index=5}
});

// -----------------------------------------------------------------------------
// Post-processing: Render → Bloom → Afterimage (ghosting) → RGBShift → Output
// -----------------------------------------------------------------------------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), state.bloomStr, state.bloomRad, state.bloomThr);
composer.addPass(bloomPass);

const afterPass = new AfterimagePass(); // has uniforms.damp  :contentReference[oaicite:6]{index=6}
composer.addPass(afterPass);

const rgbShiftPass = new ShaderPass(RGBShiftShader);
rgbShiftPass.uniforms['amount'].value = state.rgb;
rgbShiftPass.uniforms['angle'].value  = Math.PI / 4;
composer.addPass(rgbShiftPass);

composer.addPass(new OutputPass());

// -----------------------------------------------------------------------------
// Plane uniforms & initial points → build FIRST, then render
// -----------------------------------------------------------------------------
function loadPlanes(count) {
  const arr = buildPlanes(count);
  uniforms.uPlaneCount.value = count | 0;
  for (let i = 0; i < MAX_PLANES; i++) {
    uniforms.uPlanes.value[i].copy(arr[i % arr.length]);
  }
}
loadPlanes(state.planeCount);

// **Important**: populate seeds BEFORE first hull
updatePoints(0);
rebuildHull(); // safe: we now have ≥4 distinct, finite points

// add mesh after material is ready
// (mesh created inside rebuildHull)

// -----------------------------------------------------------------------------
// UI events
// -----------------------------------------------------------------------------
$('ptCount').addEventListener('input', () => { state.ptCount = parseInt($('ptCount').value,10); setVal('ptCountVal', state.ptCount, 0); });
$('spike').addEventListener('input', () => { state.spike = parseFloat($('spike').value); setVal('spikeVal', state.spike, 2); });
$('animatePts').addEventListener('change', () => { state.animatePts = $('animatePts').checked; });
$('ptSpeed').addEventListener('input', () => { state.ptSpeed = parseFloat($('ptSpeed').value); setVal('ptSpeedVal', state.ptSpeed, 2); });
$('edgeOpacity').addEventListener('input', () => { state.edgeOpacity = parseFloat($('edgeOpacity').value); setVal('edgeOpacityVal', state.edgeOpacity, 2); if (edgeLines) edgeLines.material.opacity = state.edgeOpacity; });

$('reseed').addEventListener('click', () => { reseed('random'); updatePoints(uniforms.uTime.value); rebuildHull(); });
$('regularize').addEventListener('click', () => { reseed('regularize'); updatePoints(uniforms.uTime.value); rebuildHull(); });

$('foldStr').addEventListener('input', () => { state.foldStr = parseFloat($('foldStr').value); setVal('foldStrVal', state.foldStr); uniforms.uFoldStrength.value = state.foldStr; });
$('foldSoft').addEventListener('input', () => { state.foldSoft = parseFloat($('foldSoft').value); setVal('foldSoftVal', state.foldSoft, 2); uniforms.uFoldSoft.value = state.foldSoft; });
$('planeCount').addEventListener('input', () => { state.planeCount = parseInt($('planeCount').value, 10); setVal('planeCountVal', state.planeCount, 0); loadPlanes(state.planeCount); });

$('patMix').addEventListener('input', () => { state.patMix = parseFloat($('patMix').value); setVal('patMixVal', state.patMix, 2); uniforms.uPatMix.value = state.patMix; });
$('noiseScale').addEventListener('input', () => { state.noiseScale = parseFloat($('noiseScale').value); setVal('noiseScaleVal', state.noiseScale, 2); uniforms.uNoiseScale.value = state.noiseScale; });
$('thick').addEventListener('input', () => { state.thick = parseFloat($('thick').value); setVal('thickVal', state.thick, 0); uniforms.uThicknessBase.value = state.thick; });
$('ior').addEventListener('input', () => { state.ior = parseFloat($('ior').value); setVal('iorVal', state.ior, 3); uniforms.uIorFilm.value = state.ior; });
$('bandFreq').addEventListener('input', () => { state.bandFreq = parseFloat($('bandFreq').value); setVal('bandFreqVal', state.bandFreq, 1); uniforms.uStripeFreq.value = state.bandFreq; });
$('bandSpeed').addEventListener('input', () => { state.bandSpeed = parseFloat($('bandSpeed').value); setVal('bandSpeedVal', state.bandSpeed, 2); uniforms.uStripeMove.value = state.bandSpeed; });

$('after').addEventListener('input', () => { state.after = parseFloat($('after').value); setVal('afterVal', state.after, 4); });
$('rgb').addEventListener('input', () => { state.rgb = parseFloat($('rgb').value); setVal('rgbVal', state.rgb, 4); rgbShiftPass.uniforms['amount'].value = state.rgb; });
$('bloomStr').addEventListener('input', () => { state.bloomStr = parseFloat($('bloomStr').value); setVal('bloomStrVal', state.bloomStr, 2); bloomPass.strength = state.bloomStr; });
$('bloomRad').addEventListener('input', () => { state.bloomRad = parseFloat($('bloomRad').value); setVal('bloomRadVal', state.bloomRad, 2); bloomPass.radius = state.bloomRad; });
$('bloomThr').addEventListener('input', () => { state.bloomThr = parseFloat($('bloomThr').value); setVal('bloomThrVal', state.bloomThr, 2); bloomPass.threshold = state.bloomThr; });

// -----------------------------------------------------------------------------
// Resize & Animate (rebuild hull at modest cadence to avoid hitches)
// -----------------------------------------------------------------------------
function onResize(){
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', onResize);

const clock = new THREE.Clock();
let rebuildAccumulator = 0;

function animate(){
  const dt = clock.getDelta();
  const t  = (uniforms.uTime.value += dt);

  // Framerate‑aware ghosting (consistent trail length across refresh rates) :contentReference[oaicite:7]{index=7}
  afterPass.uniforms['damp'].value = Math.pow(state.after, dt * 60.0);

  // Update points then (throttled) hull rebuild
  updatePoints(t);
  rebuildAccumulator += dt;
  if (rebuildAccumulator > 0.05) { // ~20 Hz rebuilds
    rebuildHull();
    rebuildAccumulator = 0;
  }

  // Spin + render
  if (state.autoSpin && mesh) mesh.rotation.y += state.spinSpeed;
  controls.update();
  composer.render();
  requestAnimationFrame(animate);
}
animate();
