// Convex hull (straight edges) + cohesive folding + improved textures + view dropdown.
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

// ------------------------------------- Renderer / Scene / Camera
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

// ------------------------------------- UI state
const $ = (id) => document.getElementById(id);
const state = {
  viewMode: 'textured',
  ptCount: 18, spike: 0.45, animatePts: true, ptSpeed: 0.25, edgeOpacity: 0.12,
  foldStr: 1.0, foldSoft: 0.16, planeCount: 7,
  patMix: 0.55, noiseScale: 2.0, thick: 430, ior: 1.38, bandFreq: 12.0, bandSpeed: 1.10,
  after: 0.96, rgb: 0.0010, bloomStr: 0.90, bloomRad: 0.45, bloomThr: 0.18,
  autoSpin: true, spinSpeed: 0.002
};
function setVal(id, v, d = 2){ $(id).textContent = (typeof v === 'number') ? v.toFixed(d) : String(v); }
function syncUI(){
  $('viewMode').value = state.viewMode;
  $('ptCount').value = state.ptCount; setVal('ptCountVal', state.ptCount, 0);
  $('spike').value = state.spike; setVal('spikeVal', state.spike, 2);
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

// ------------------------------------- Dynamic Convex Hull (straight edges)
const MAX_POINTS = 40;
const seeds = Array.from({ length: MAX_POINTS }, () => new THREE.Vector3());
const base  = Array.from({ length: MAX_POINTS }, () => new THREE.Vector3());

function reseed(mode = 'random') {
  const R = 1.0;
  for (let i = 0; i < MAX_POINTS; i++) {
    if (mode === 'regularize') {
      const a = (i / MAX_POINTS) * Math.PI * 2;
      base[i].set(Math.cos(a), 0, Math.sin(a)).multiplyScalar(R);
    } else {
      const u = Math.random(), v = Math.random();
      const th = 2 * Math.PI * u, ph = Math.acos(2 * v - 1);
      base[i].set(
        Math.sin(ph) * Math.cos(th),
        Math.cos(ph) * 0.6,
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
    seeds[i].x += 1e-4 * Math.sin(999.1 * (i + 0.123));
    seeds[i].y += 1e-4 * Math.sin(777.7 * (i + 0.456));
    seeds[i].z += 1e-4 * Math.sin(555.5 * (i + 0.789));
  }
}

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
function rebuildHull() {
  const pts = gatherHullPoints();
  if (pts.length < 4) return;

  if (mesh) { mesh.geometry.dispose(); scene.remove(mesh); }
  if (edgeLines) { edgeLines.geometry.dispose(); scene.remove(edgeLines); }

  const geom = new ConvexGeometry(pts); // convex hull of moving points
  mesh = new THREE.Mesh(geom, material);
  scene.add(mesh);

  const eGeo = new THREE.EdgesGeometry(geom, 15);
  const eMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: state.edgeOpacity });
  edgeLines = new THREE.LineSegments(eGeo, eMat);
  scene.add(edgeLines);

  applyViewMode();
}

// ------------------------------------- Symmetry planes (single loadPlanes)
const MAX_PLANES = 9;
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
  uniforms.uPlaneCount.value = count | 0;
  for (let i = 0; i < MAX_PLANES; i++) {
    uniforms.uPlanes.value[i].copy(arr[i % arr.length]);
  }
}

// ------------------------------------- Shader (flat facets + iridescent texture)
const uniforms = {
  uTime:           { value: 0 },
  uTexEnabled:     { value: 1.0 },    // 0=shaded, 1=textured
  uFoldStrength:   { value: state.foldStr },
  uFoldSoft:       { value: state.foldSoft },
  uPlaneCount:     { value: state.planeCount },
  uPlanes:         { value: Array.from({ length: MAX_PLANES }, () => new THREE.Vector3()) },
  uPatMix:         { value: state.patMix },
  uNoiseScale:     { value: state.noiseScale },
  uStripeFreq:     { value: state.bandFreq },
  uStripeMove:     { value: state.bandSpeed },
  uThicknessBase:  { value: state.thick },
  uIorFilm:        { value: state.ior },
  uBaseColor:      { value: new THREE.Color(0x0a0f08) }
};

const vertexShader = `
precision highp float;
uniform float uTime, uFoldStrength, uFoldSoft;
uniform int uPlaneCount;
uniform vec3 uPlanes[${MAX_PLANES}];
varying vec3 vNormal, vPosView, vPosWorld;
varying float vCrease;

vec3 reflectPoint(vec3 p, vec3 n, float d){ float s = dot(p,n)+d; return p - 2.0*s*n; }
vec3 reflectNormal(vec3 nor, vec3 n){ return normalize(nor - 2.0*dot(nor,n)*n); }
void blendedFold(inout vec3 p, inout vec3 nrm, vec3 n, float d, float strength, float softness, inout float creaseAcc){
  float s = dot(p,n)+d;
  float w = exp(-abs(s)/max(1e-3,softness));
  w = clamp(w*strength,0.0,1.0);
  vec3 pr = reflectPoint(p,n,d);
  vec3 nr = reflectNormal(nrm,n);
  p = mix(p,pr,w);
  nrm = normalize(mix(nrm,nr,w));
  creaseAcc = max(creaseAcc,w);
}
mat3 rotY(float a){ float c=cos(a), s=sin(a); return mat3(c,0.,s, 0.,1.,0., -s,0.,c); }

void main(){
  vec3 p = position;
  vec3 nrm = normal;

  float a = uTime * 0.25;
  mat3 R = rotY(a);
  p = R*p; nrm = R*nrm;

  float crease = 0.0;
  for(int iter=0; iter<2; iter++){
    for(int i=0; i<${MAX_PLANES}; i++){
      if(i >= uPlaneCount) break;
      vec3 n = normalize(uPlanes[i]);
      float phase = float(i)*1.618 + float(iter)*0.73;
      float d = 0.18 * sin(uTime*0.6 + phase);
      blendedFold(p, nrm, n, d, uFoldStrength, uFoldSoft, crease);
    }
  }

  vec4 mv = modelViewMatrix * vec4(p,1.0);
  vPosView = mv.xyz;
  vPosWorld = (modelMatrix*vec4(p,1.0)).xyz;
  vNormal = normalize(normalMatrix*nrm);
  vCrease = crease;
  gl_Position = projectionMatrix * mv;
}
`;

const fragmentShader = `
#extension GL_OES_standard_derivatives : enable
precision highp float;

uniform float uTime, uTexEnabled, uPatMix, uNoiseScale, uStripeFreq, uStripeMove, uThicknessBase, uIorFilm;
uniform vec3  uBaseColor;
varying vec3 vNormal, vPosView, vPosWorld;
varying float vCrease;

float hash11(float n){ return fract(sin(n)*43758.5453123); }
float noise3(vec3 x){
  vec3 i=floor(x), f=fract(x);
  float n=dot(i,vec3(1.,57.,113.));
  float a=hash11(n), b=hash11(n+1.), c=hash11(n+57.), d=hash11(n+58.);
  float e=hash11(n+113.), f1=hash11(n+114.), g=hash11(n+170.), h=hash11(n+171.);
  vec3 u=f*f*(3.-2.*f);
  float xy1=mix(mix(a,b,u.x),mix(c,d,u.x),u.y);
  float xy2=mix(mix(e,f1,u.x),mix(g,h,u.x),u.y);
  return mix(xy1,xy2,u.z);
}
float fbm(vec3 p){ float s=0., a=0.5; for(int i=0;i<5;i++){ s+=a*noise3(p); p*=2.02; a*=0.5; } return s; }
float tri(float x){ return abs(fract(x)-0.5)*2.0; }

vec3 thinFilm(float thickness, float n1, float n2, float n3, float cosTheta1){
  const float PI = 3.141592653589793;
  vec3 lambda=vec3(680.,550.,440.);
  float sinTheta1=sqrt(max(0.,1.-cosTheta1*cosTheta1));
  float sinTheta2=n1/n2*sinTheta1;
  float cosTheta2=sqrt(max(0.,1.-sinTheta2*sinTheta2));
  vec3 phase=4.0*PI*n2*thickness*cosTheta2/lambda;
  return 0.5+0.5*cos(phase);
}

void main(){
  vec3 Ng = normalize(cross(dFdx(vPosWorld), dFdy(vPosWorld)));
  if(dot(Ng, vNormal) < 0.0) Ng = -Ng;
  vec3 N = Ng;
  vec3 V = normalize(-vPosView);
  float NdotV = clamp(dot(N,V), 0.0, 1.0);

  vec3 L = normalize(vec3(0.35,0.9,0.15));
  float diff = max(dot(N,L), 0.0);
  float rim = pow(1.0 - NdotV, 2.0);
  vec3 baseCol = uBaseColor*diff + uBaseColor*rim*0.15;

  if (uTexEnabled < 0.5) {
    vec3 color = baseCol + vCrease * 0.15;
    gl_FragColor = vec4(color, 1.0);
    return;
  }

  vec3 dir = normalize(vec3(0.7,0.,0.3));
  float coord = dot(vPosWorld, dir) * uStripeFreq + uTime * uStripeMove;
  float stripes = smoothstep(0.72,0.985,0.5+0.5*sin(coord));
  float grating = tri(coord * 0.5);
  float n = fbm(vPosWorld * uNoiseScale + vec3(0.0, uTime*0.15, 0.0));
  float pat = mix(mix(stripes, grating, 0.35), n, uPatMix);

  float thickness = uThicknessBase * (0.70 + 0.30*pat);
  vec3 film = thinFilm(thickness, 1.0, uIorFilm, 1.0, NdotV);

  float f0 = 0.06;
  float fresnel = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);
  vec3 R = reflect(-V, N);
  vec3 A1 = normalize(vec3(0.2,1.0,0.0));
  vec3 A2 = normalize(vec3(-0.7,0.3,0.6));
  float aniso = pow(abs(dot(R,A1)),24.0) + 0.5*pow(abs(dot(R,A2)),36.0);

  vec3 color = baseCol;
  color = mix(color, film, 0.75);
  color += film * pat * 1.2;
  color += fresnel * film;
  color += aniso * film * 0.45;
  color += vCrease * film * 0.35;

  gl_FragColor = vec4(color, 1.0);
}
`;

const material = new THREE.ShaderMaterial({
  uniforms, vertexShader, fragmentShader, side: THREE.DoubleSide,
  extensions: { derivatives: true }
});

// ------------------------------------- Post-processing
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), state.bloomStr, state.bloomRad, state.bloomThr);
composer.addPass(bloomPass);
const afterPass = new AfterimagePass();
composer.addPass(afterPass);
const rgbShiftPass = new ShaderPass(RGBShiftShader);
rgbShiftPass.uniforms['amount'].value = state.rgb;
rgbShiftPass.uniforms['angle'].value  = Math.PI / 4;
composer.addPass(rgbShiftPass);
composer.addPass(new OutputPass());

// ------------------------------------- View modes & initial setup
function applyViewMode(){
  if (!mesh || !edgeLines) return;
  switch (state.viewMode) {
    case 'wireframe':
      uniforms.uTexEnabled.value = 0.0;
      mesh.visible = false;
      edgeLines.visible = true;
      edgeLines.material.opacity = Math.max(0.5, state.edgeOpacity);
      break;
    case 'shaded':
      uniforms.uTexEnabled.value = 0.0;
      mesh.visible = true;
      edgeLines.visible = true;
      edgeLines.material.opacity = state.edgeOpacity;
      break;
    default:
    case 'textured':
      uniforms.uTexEnabled.value = 1.0;
      mesh.visible = true;
      edgeLines.visible = true;
      edgeLines.material.opacity = state.edgeOpacity;
      break;
  }
}

loadPlanes(state.planeCount); // **single** loadPlanes

// populate seeds BEFORE first hull
updatePoints(0);
rebuildHull();

// ------------------------------------- UI events
$('viewMode').addEventListener('change', () => { state.viewMode = $('viewMode').value; applyViewMode(); });
$('ptCount').addEventListener('input', () => { state.ptCount = parseInt($('ptCount').value,10); setVal('ptCountVal', state.ptCount, 0); });
$('spike').addEventListener('input', () => { state.spike = parseFloat($('spike').value); setVal('spikeVal', state.spike, 2); });
$('animatePts').addEventListener('change', () => { state.animatePts = $('animatePts').checked; });
$('ptSpeed').addEventListener('input', () => { state.ptSpeed = parseFloat($('ptSpeed').value); setVal('ptSpeedVal', state.ptSpeed, 2); });
$('edgeOpacity').addEventListener('input', () => {
  state.edgeOpacity = parseFloat($('edgeOpacity').value); setVal('edgeOpacityVal', state.edgeOpacity, 2);
  if (edgeLines) edgeLines.material.opacity = state.edgeOpacity;
});
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

// ------------------------------------- Resize & Animate
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

  afterPass.uniforms['damp'].value = Math.pow(state.after, dt * 60.0);
  updatePoints(t);
  rebuildAccumulator += dt;
  if (rebuildAccumulator > 0.05) { rebuildHull(); rebuildAccumulator = 0; }

  if (state.autoSpin && mesh) mesh.rotation.y += state.spinSpeed;
  controls.update();
  composer.render();
  requestAnimationFrame(animate);
}
animate();
