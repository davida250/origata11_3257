// Convex hull + cohesive folding + bold patterns & distortions (no mouse coupling).

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
renderer.toneMappingExposure = 1.1;
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
  // view
  viewMode: 'textured',
  edgesOverlay: false,
  // hull
  ptCount: 18, spike: 0.45, animatePts: true, ptSpeed: 0.25, edgeOpacity: 0.12, smooth: 0.25,
  // fold
  foldStr: 1.0, foldSoft: 0.16, foldSpeed: 1.0, planeCount: 7,
  // material / texture
  texPreset: 'refA',
  pattern: 0,                 // 0 Bands, 1 CrossGrid, 2 Spiral, 3 Kaleido, 4 Cells
  patScale: 12.0, warp: 0.80, warpSpd: 1.0, contrast: 1.40, refl: 0.30,
  patMix: 0.65, noiseScale: 2.2, bandFreq: 10.0, bandSpeed: 0.9, // legacy knobs still useful
  thick: 420, ior: 1.37,
  texStr: 0.85, desat: 0.75, glint: 0.45,
  // fx
  after: 0.96, rgb: 0.0010, bloomStr: 0.90, bloomRad: 0.45, bloomThr: 0.18,
  // motion
  autoSpin: true, spinSpeed: 0.002
};
function setVal(id, v, d = 2){ const el = document.getElementById(id); if (el) el.textContent = (typeof v === 'number') ? v.toFixed(d) : String(v); }
function syncUI(){
  $('viewMode').value = state.viewMode;
  $('edgesOverlay').checked = state.edgesOverlay;

  $('ptCount').value = state.ptCount; setVal('ptCountVal', state.ptCount, 0);
  $('spike').value = state.spike; setVal('spikeVal', state.spike, 2);
  $('animatePts').checked = state.animatePts;
  $('ptSpeed').value = state.ptSpeed; setVal('ptSpeedVal', state.ptSpeed, 2);
  $('edgeOpacity').value = state.edgeOpacity; setVal('edgeOpacityVal', state.edgeOpacity, 2);
  $('smooth').value = state.smooth; setVal('smoothVal', state.smooth, 2);

  $('foldStr').value = state.foldStr; setVal('foldStrVal', state.foldStr);
  $('foldSoft').value = state.foldSoft; setVal('foldSoftVal', state.foldSoft, 2);
  $('foldSpd').value = state.foldSpeed; setVal('foldSpdVal', state.foldSpeed, 2);
  $('planeCount').value = state.planeCount; setVal('planeCountVal', state.planeCount, 0);

  $('texPreset').value = state.texPreset;
  $('pattern').value = String(state.pattern);
  $('patScale').value = state.patScale; setVal('patScaleVal', state.patScale, 1);
  $('warp').value = state.warp; setVal('warpVal', state.warp, 2);
  $('warpSpd').value = state.warpSpd; setVal('warpSpdVal', state.warpSpd, 2);
  $('contrast').value = state.contrast; setVal('contrastVal', state.contrast, 2);
  $('refl').value = state.refl; setVal('reflVal', state.refl, 2);

  $('patMix').value = state.patMix; setVal('patMixVal', state.patMix, 2);
  $('noiseScale').value = state.noiseScale; setVal('noiseScaleVal', state.noiseScale, 2);
  $('bandFreq').value = state.bandFreq; setVal('bandFreqVal', state.bandFreq, 1);
  $('bandSpeed').value = state.bandSpeed; setVal('bandSpeedVal', state.bandSpeed, 2);
  $('thick').value = state.thick; setVal('thickVal', state.thick, 0);
  $('ior').value = state.ior; setVal('iorVal', state.ior, 3);
  $('texStr').value = state.texStr; setVal('texStrVal', state.texStr, 2);
  $('desat').value = state.desat; setVal('desatVal', state.desat, 2);
  $('glint').value = state.glint; setVal('glintVal', state.glint, 2);

  $('after').value = state.after; setVal('afterVal', state.after, 4);
  $('rgb').value = state.rgb; setVal('rgbVal', state.rgb, 4);
  $('bloomStr').value = state.bloomStr; setVal('bloomStrVal', state.bloomStr, 2);
  $('bloomRad').value = state.bloomRad; setVal('bloomRadVal', state.bloomRad, 2);
  $('bloomThr').value = state.bloomThr; setVal('bloomThrVal', state.bloomThr, 2);
}
syncUI();

// ------------------------------------- Dynamic Convex Hull (smoother)
const MAX_POINTS = 40;
const seeds     = Array.from({ length: MAX_POINTS }, () => new THREE.Vector3());
const targets   = Array.from({ length: MAX_POINTS }, () => new THREE.Vector3());
const base      = Array.from({ length: MAX_POINTS }, () => new THREE.Vector3());

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
    // initialize seed and target coherently
    seeds[i].copy(base[i]);
    targets[i].copy(base[i]);
  }
}
reseed('random');

function updateTargets(t) {
  const n = Math.max(4, Math.min(state.ptCount, MAX_POINTS));
  const amp = state.spike;
  const speed = state.animatePts ? state.ptSpeed : 0.0;
  for (let i = 0; i < n; i++) {
    const off = i * 0.37;
    const b = base[i];
    const radial = 1.0 + amp * Math.sin(t * (0.7 + 0.23 * Math.sin(off)) + off * 1.618);
    const y = 0.3 * Math.sin(t * speed + off);
    targets[i].set(b.x * radial, b.y + y, b.z * radial);
  }
}

function integrateSeeds(alpha) {
  const n = Math.max(4, Math.min(state.ptCount, MAX_POINTS));
  for (let i = 0; i < n; i++) {
    seeds[i].lerp(targets[i], clamp(alpha, 0, 1)); // low-pass filter
  }
}

function gatherHullPoints() {
  const n = Math.max(4, Math.min(state.ptCount, MAX_POINTS));
  const pts = [];
  const seen = new Map();
  for (let i = 0; i < n; i++) {
    const p = seeds[i];
    const k = `${p.x.toFixed(6)},${p.y.toFixed(6)},${p.z.toFixed(6)}`;
    if (!seen.has(k)) { seen.set(k, 1); pts.push(p.clone()); }
  }
  return pts;
}

let mesh, edgeLines;
function rebuildHull() {
  const pts = gatherHullPoints();
  if (pts.length < 4) return;

  if (mesh) { mesh.geometry.dispose(); scene.remove(mesh); }
  if (edgeLines) { edgeLines.geometry.dispose(); scene.remove(edgeLines); }

  const geom = new ConvexGeometry(pts);
  mesh = new THREE.Mesh(geom, material);
  scene.add(mesh);

  const eGeo = new THREE.EdgesGeometry(geom, 15);
  const eMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: state.edgeOpacity });
  edgeLines = new THREE.LineSegments(eGeo, eMat);
  scene.add(edgeLines);

  applyViewMode();
}

// ------------------------------------- Symmetry planes
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

// ------------------------------------- Shader (flat facets + bold patterns)
const uniforms = {
  uTime:           { value: 0 },
  // view
  uTexEnabled:     { value: 1.0 },    // 0=shaded, 1=textured
  // folding
  uFoldStrength:   { value: state.foldStr },
  uFoldSoft:       { value: state.foldSoft },
  uFoldSpeed:      { value: state.foldSpeed },
  uPlaneCount:     { value: state.planeCount },
  uPlanes:         { value: Array.from({ length: MAX_PLANES }, () => new THREE.Vector3()) },
  // material patterns & bold controls
  uPatternType:    { value: state.pattern | 0 },
  uPatternScale:   { value: state.patScale },
  uWarp:           { value: state.warp },
  uWarpSpeed:      { value: state.warpSpd },
  uContrast:       { value: state.contrast },
  uReflMix:        { value: state.refl },
  // legacy/extra
  uPatMix:         { value: state.patMix },
  uNoiseScale:     { value: state.noiseScale },
  uStripeFreq:     { value: state.bandFreq },
  uStripeMove:     { value: state.bandSpeed },
  uThicknessBase:  { value: state.thick },
  uIorFilm:        { value: state.ior },
  uTexStr:         { value: state.texStr },
  uDesat:          { value: state.desat },
  uGlint:          { value: state.glint },
  uBaseColor:      { value: new THREE.Color(0x070809) }
};

const vertexShader = `
precision highp float;
uniform float uTime, uFoldStrength, uFoldSoft, uFoldSpeed;
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
  vec3 p = position, nrm = normal;
  float a = uTime * 0.20; // slow rigid spin; visual focus on folding speed slider
  mat3 R = rotY(a);
  p = R*p; nrm = R*nrm;

  float crease = 0.0;
  for(int iter=0; iter<2; iter++){
    for(int i=0; i<${MAX_PLANES}; i++){
      if(i >= uPlaneCount) break;
      vec3 n = normalize(uPlanes[i]);
      float phase = float(i)*1.618 + float(iter)*0.73;
      float d = 0.20 * sin(uTime * uFoldSpeed + phase);
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

// NOTE: no #extension directives; derivatives are core in WebGL2/ESSL3 via three.js pipeline.
const fragmentShader = `
precision highp float;

uniform float uTime, uTexEnabled;
uniform int   uPatternType;
uniform float uPatternScale, uWarp, uWarpSpeed, uContrast, uReflMix;
uniform float uPatMix, uNoiseScale, uStripeFreq, uStripeMove, uThicknessBase, uIorFilm;
uniform float uTexStr, uDesat, uGlint;
uniform vec3  uBaseColor;

varying vec3 vNormal, vPosView, vPosWorld;
varying float vCrease;

// -------- utilities
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
float luma(vec3 c){ return dot(c, vec3(0.2126,0.7152,0.0722)); }
vec3  ortho(vec3 n){ return normalize(abs(n.x)>0.5? vec3(-n.y,n.x,0.): vec3(0.,-n.z,n.y)); }

// cheap Worley-like 2D cells on arbitrary plane
float cells2(vec2 uv){
  vec2 g = floor(uv), f = fract(uv);
  float dmin = 1e9;
  for(int j=-1;j<=1;j++){
    for(int i=-1;i<=1;i++){
      vec2 o = vec2(float(i), float(j));
      float id = dot(g+o, vec2(127.1,311.7));
      vec2 r = vec2(hash11(id), hash11(id+1.23));
      vec2 p = o + r - f + 0.12 * sin(uTime*0.6 + id); // animated jitter
      dmin = min(dmin, dot(p,p));
    }
  }
  return exp(-3.0 * dmin); // bright cell centers
}

// thin-film interference (approx wavelengths)
vec3 thinFilm(float thickness, float n1, float n2, float n3, float cosTheta1){
  const float PI = 3.141592653589793;
  vec3 lambda=vec3(680.,550.,440.);
  float sinTheta1=sqrt(max(0.,1.-cosTheta1*cosTheta1));
  float sinTheta2=n1/n2*sinTheta1;
  float cosTheta2=sqrt(max(0.,1.-sinTheta2*sinTheta2));
  vec3 phase=4.0*PI*n2*thickness*cosTheta2/lambda;
  return 0.5+0.5*cos(phase);
}

// simple procedural env from reflection vector
vec3 envColor(vec3 R){
  float v = clamp(R.y*0.5+0.5, 0.0, 1.0);
  float hStripes = 0.5 + 0.5 * sin((R.x+R.z)*24.0 + uTime*0.6);
  float rings = 0.5 + 0.5 * sin(acos(clamp(R.y, -1., 1.)) * 12.0 - uTime*0.5);
  vec3 sky = mix(vec3(0.03,0.04,0.05), vec3(0.12,0.15,0.18), v);
  return sky + 0.25*hStripes*vec3(0.8,0.9,1.0) + 0.2*rings*vec3(1.0,0.85,0.6);
}

// pattern generator (bold)
float patternValue(vec3 pos, vec3 N){
  // build local 2D space on the facet plane
  vec3 T = ortho(N);
  vec3 B = normalize(cross(N, T));
  vec2 uv = vec2(dot(pos, T), dot(pos, B)) * uPatternScale;

  // domain warp
  vec3 pw = pos;
  vec3 q = pos * (0.35 + 0.15*uPatternScale*0.05);
  vec3 warp = vec3(
    fbm(q*0.8 + vec3(0.0,  uTime*uWarpSpeed, 10.0)),
    fbm(q*0.8 + vec3(10.0, uTime*uWarpSpeed, 0.0)),
    fbm(q*0.8 + vec3(20.0, uTime*uWarpSpeed, 5.0))
  );
  pw += (warp - 0.5) * uWarp;
  uv += (warp.xz - 0.5*vec2(1.0)) * (uWarp*0.75) * uPatternScale;

  float t = uTime;

  if (uPatternType == 0) {
    // Bands
    float s = sin(uv.x + 0.6*sin(uv.y*0.5 + t*0.7));
    return 0.5 + 0.5*s;
  } else if (uPatternType == 1) {
    // CrossGrid
    float gx = 0.5 + 0.5*sin(uv.x + t*0.9);
    float gy = 0.5 + 0.5*sin(uv.y*1.03 - t*0.8);
    return pow(gx*gy, 0.8);
  } else if (uPatternType == 2) {
    // Spiral
    float r = length(uv)*0.25;
    float a = atan(uv.y, uv.x);
    return 0.5 + 0.5*sin(10.0*a + 6.0*r - t*1.2);
  } else if (uPatternType == 3) {
    // Kaleido (6-fold angle folding)
    float a = atan(uv.y, uv.x);
    float k = 3.141592653589793/3.0;
    a = mod(a, k);
    float r = length(uv)*0.22;
    return 0.5 + 0.5*sin(12.0*a + 10.0*r - t*1.0);
  } else {
    // Cells (Worley-ish)
    float c = cells2(uv*0.15);
    return c;
  }
}

void main(){
  // flat facet normal via derivatives
  vec3 Ng = normalize(cross(dFdx(vPosWorld), dFdy(vPosWorld)));
  if(dot(Ng, vNormal) < 0.0) Ng = -Ng;
  vec3 N = Ng;

  vec3 V = normalize(-vPosView);
  float NdotV = clamp(dot(N,V), 0.0, 1.0);

  // base lambert + rim
  vec3 L = normalize(vec3(0.35,0.9,0.15));
  float diff = max(dot(N,L), 0.0);
  float rim  = pow(1.0 - NdotV, 2.0);
  vec3 baseCol = uBaseColor * (0.25 + 0.75*diff) + uBaseColor * rim * 0.12;

  if (uTexEnabled < 0.5) {
    vec3 color = baseCol + vCrease * 0.12;
    gl_FragColor = vec4(color, 1.0);
    return;
  }

  // bold pattern
  float pat = patternValue(vPosWorld, N);

  // optional extra blend with legacy hybrid (for subtle micro-structure)
  vec3 d1 = normalize(vec3(0.7, 0.0, 0.3));
  float coord = dot(vPosWorld, d1) * uStripeFreq + uTime * uStripeMove;
  float grating = tri((coord) * 0.5);
  float n = fbm(vPosWorld * uNoiseScale + vec3(0.0, uTime*0.12, 0.0));
  float micro = mix(grating, n, uPatMix);
  pat = clamp(mix(pat, (pat*0.7 + micro*0.3), 0.25), 0.0, 1.0);

  // contrast
  pat = clamp(0.5 + (pat - 0.5) * uContrast, 0.0, 1.0);

  // thin-film
  float thickness = uThicknessBase * (0.70 + 0.30 * pat) * (0.92 + 0.08 * vCrease);
  vec3 film = thinFilm(thickness, 1.0, uIorFilm, 1.0, NdotV);

  // reflections (procedural env)
  vec3 R = reflect(-V, N);
  vec3 env = envColor(R);

  // fresnel + anisotropic glints
  float f0 = 0.05 + 0.02 * pat;
  float fresnel = f0 + (1.0 - f0) * pow(1.0 - NdotV, 5.0);
  vec3 A1 = normalize(vec3(0.2, 1.0, 0.0));
  vec3 A2 = normalize(vec3(-0.7, 0.3, 0.6));
  float aniso = pow(abs(dot(R,A1)), 24.0) + 0.5 * pow(abs(dot(R,A2)), 36.0);

  // combine
  vec3 color = baseCol;
  vec3 irid = film * (0.6 + 0.7*pat);
  color = mix(color, irid, uTexStr);
  color += fresnel * (mix(env, irid, 0.5)) * (0.25 + 0.5*uReflMix);
  color += aniso * irid * (0.6 * uGlint);
  color += vCrease * irid * 0.25;

  // desaturate toward luma for the dark look
  float Y = luma(color);
  color = mix(color, vec3(Y), clamp(uDesat, 0.0, 1.0));

  gl_FragColor = vec4(color, 1.0);
}
`;

const material = new THREE.ShaderMaterial({
  uniforms, vertexShader, fragmentShader,
  side: THREE.DoubleSide,
  transparent: false,
  wireframe: false
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

// ------------------------------------- View & presets
function applyViewMode(){
  if (!mesh || !edgeLines) return;
  material.wireframe = false;
  mesh.visible = true;
  // edges overlay only if requested (and never in pure wireframe)
  edgeLines.visible = state.edgesOverlay && state.viewMode !== 'wireframe';
  edgeLines.material.opacity = state.edgeOpacity;

  switch (state.viewMode) {
    case 'wireframe':
      material.wireframe = true;
      uniforms.uTexEnabled.value = 0.0;
      edgeLines.visible = false;
      break;
    case 'shaded':
      uniforms.uTexEnabled.value = 0.0;
      break;
    default:
    case 'textured':
      uniforms.uTexEnabled.value = 1.0;
      break;
  }
}

function applyMaterialState(){
  uniforms.uPatternType.value  = state.pattern | 0;
  uniforms.uPatternScale.value = state.patScale;
  uniforms.uWarp.value         = state.warp;
  uniforms.uWarpSpeed.value    = state.warpSpd;
  uniforms.uContrast.value     = state.contrast;
  uniforms.uReflMix.value      = state.refl;

  uniforms.uPatMix.value       = state.patMix;
  uniforms.uNoiseScale.value   = state.noiseScale;
  uniforms.uStripeFreq.value   = state.bandFreq;
  uniforms.uStripeMove.value   = state.bandSpeed;
  uniforms.uThicknessBase.value = state.thick;
  uniforms.uIorFilm.value      = state.ior;
  uniforms.uTexStr.value       = state.texStr;
  uniforms.uDesat.value        = state.desat;
  uniforms.uGlint.value        = state.glint;
}

function applyPreset(name){
  state.texPreset = name;
  if (name === 'refA'){           // dark iridescent
    Object.assign(state, {
      patMix: 0.65, noiseScale: 2.2, bandFreq: 10.0, bandSpeed: 0.9,
      thick: 420, ior: 1.37, texStr: 0.85, desat: 0.75, glint: 0.45
    });
  } else if (name === 'refB'){    // ghost grid
    Object.assign(state, {
      patMix: 0.35, noiseScale: 1.3, bandFreq: 18.0, bandSpeed: 1.2,
      thick: 400, ior: 1.36, texStr: 0.80, desat: 0.70, glint: 0.35
    });
  } else if (name === 'refC'){    // marble bands
    Object.assign(state, {
      patMix: 0.80, noiseScale: 3.2, bandFreq: 7.0, bandSpeed: 0.6,
      thick: 520, ior: 1.40, texStr: 0.75, desat: 0.60, glint: 0.30
    });
  }
  syncUI(); applyMaterialState();
}

// ------------------------------------- initial setup
loadPlanes(state.planeCount);
applyMaterialState();
updateTargets(0);
integrateSeeds(1.0);
rebuildHull();

// ------------------------------------- UI events
$('viewMode').addEventListener('change', () => { state.viewMode = $('viewMode').value; applyViewMode(); });
$('edgesOverlay').addEventListener('change', () => { state.edgesOverlay = $('edgesOverlay').checked; applyViewMode(); });
$('edgeOpacity').addEventListener('input', () => {
  state.edgeOpacity = parseFloat($('edgeOpacity').value); setVal('edgeOpacityVal', state.edgeOpacity, 2);
  if (edgeLines) edgeLines.material.opacity = state.edgeOpacity;
});

$('ptCount').addEventListener('input', () => { state.ptCount = parseInt($('ptCount').value,10); setVal('ptCountVal', state.ptCount, 0); });
$('spike').addEventListener('input', () => { state.spike = parseFloat($('spike').value); setVal('spikeVal', state.spike, 2); });
$('animatePts').addEventListener('change', () => { state.animatePts = $('animatePts').checked; });
$('ptSpeed').addEventListener('input', () => { state.ptSpeed = parseFloat($('ptSpeed').value); setVal('ptSpeedVal', state.ptSpeed, 2); });
$('smooth').addEventListener('input', () => { state.smooth = parseFloat($('smooth').value); setVal('smoothVal', state.smooth, 2); });

$('reseed').addEventListener('click', () => { reseed('random'); updateTargets(uniforms.uTime.value); integrateSeeds(1.0); rebuildHull(); });
$('regularize').addEventListener('click', () => { reseed('regularize'); updateTargets(uniforms.uTime.value); integrateSeeds(1.0); rebuildHull(); });

$('foldStr').addEventListener('input', () => { state.foldStr = parseFloat($('foldStr').value); setVal('foldStrVal', state.foldStr); uniforms.uFoldStrength.value = state.foldStr; });
$('foldSoft').addEventListener('input', () => { state.foldSoft = parseFloat($('foldSoft').value); setVal('foldSoftVal', state.foldSoft, 2); uniforms.uFoldSoft.value = state.foldSoft; });
$('foldSpd').addEventListener('input', () => { state.foldSpeed = parseFloat($('foldSpd').value); setVal('foldSpdVal', state.foldSpeed, 2); uniforms.uFoldSpeed.value = state.foldSpeed; });
$('planeCount').addEventListener('input', () => { state.planeCount = parseInt($('planeCount').value, 10); setVal('planeCountVal', state.planeCount, 0); loadPlanes(state.planeCount); });

$('texPreset').addEventListener('change', () => { applyPreset($('texPreset').value); });
$('pattern').addEventListener('change', () => { state.pattern = parseInt($('pattern').value,10); applyMaterialState(); });
$('patScale').addEventListener('input', () => { state.patScale = parseFloat($('patScale').value); setVal('patScaleVal', state.patScale, 1); applyMaterialState(); });
$('warp').addEventListener('input', () => { state.warp = parseFloat($('warp').value); setVal('warpVal', state.warp, 2); applyMaterialState(); });
$('warpSpd').addEventListener('input', () => { state.warpSpd = parseFloat($('warpSpd').value); setVal('warpSpdVal', state.warpSpd, 2); applyMaterialState(); });
$('contrast').addEventListener('input', () => { state.contrast = parseFloat($('contrast').value); setVal('contrastVal', state.contrast, 2); applyMaterialState(); });
$('refl').addEventListener('input', () => { state.refl = parseFloat($('refl').value); setVal('reflVal', state.refl, 2); applyMaterialState(); });

$('patMix').addEventListener('input', () => { state.patMix = parseFloat($('patMix').value); setVal('patMixVal', state.patMix, 2); applyMaterialState(); });
$('noiseScale').addEventListener('input', () => { state.noiseScale = parseFloat($('noiseScale').value); setVal('noiseScaleVal', state.noiseScale, 2); applyMaterialState(); });
$('bandFreq').addEventListener('input', () => { state.bandFreq = parseFloat($('bandFreq').value); setVal('bandFreqVal', state.bandFreq, 1); applyMaterialState(); });
$('bandSpeed').addEventListener('input', () => { state.bandSpeed = parseFloat($('bandSpeed').value); setVal('bandSpeedVal', state.bandSpeed, 2); applyMaterialState(); });
$('thick').addEventListener('input', () => { state.thick = parseFloat($('thick').value); setVal('thickVal', state.thick, 0); applyMaterialState(); });
$('ior').addEventListener('input', () => { state.ior = parseFloat($('ior').value); setVal('iorVal', state.ior, 3); applyMaterialState(); });
$('texStr').addEventListener('input', () => { state.texStr = parseFloat($('texStr').value); setVal('texStrVal', state.texStr, 2); applyMaterialState(); });
$('desat').addEventListener('input', () => { state.desat = parseFloat($('desat').value); setVal('desatVal', state.desat, 2); applyMaterialState(); });
$('glint').addEventListener('input', () => { state.glint = parseFloat($('glint').value); setVal('glintVal', state.glint, 2); applyMaterialState(); });

$('after').addEventListener('input', () => { state.after = parseFloat($('after').value); setVal('afterVal', state.after, 4); });
$('rgb').addEventListener('input', () => { state.rgb = parseFloat($('rgb').value); setVal('rgbVal', state.rgb, 4); rgbShiftPass.uniforms['amount'].value = state.rgb; });
$('bloomStr').addEventListener('input', () => { state.bloomStr = parseFloat($('bloomStr').value); setVal('bloomStrVal', state.bloomStr, 2); bloomPass.strength = state.bloomStr; });
$('bloomRad').addEventListener('input', () => { state.bloomRad = parseFloat($('bloomRad').value); setVal('bloomRadVal', state.bloomRad, 2); bloomPass.radius = state.bloomRad; });
$('bloomThr').addEventListener('input', () => { state.bloomThr = parseFloat($('bloomThr').value); setVal('bloomThrVal', state.bloomThr, 2); bloomPass.threshold = state.bloomThr; });

$('reset').addEventListener('click', () => {
  Object.assign(state, {
    viewMode: 'textured', edgesOverlay: false,
    ptCount: 18, spike: 0.45, animatePts: true, ptSpeed: 0.25, edgeOpacity: 0.12, smooth: 0.25,
    foldStr: 1.0, foldSoft: 0.16, foldSpeed: 1.0, planeCount: 7,
    texPreset: 'refA',
    pattern: 0, patScale: 12.0, warp: 0.80, warpSpd: 1.0, contrast: 1.40, refl: 0.30,
    patMix: 0.65, noiseScale: 2.2, bandFreq: 10.0, bandSpeed: 0.9,
    thick: 420, ior: 1.37, texStr: 0.85, desat: 0.75, glint: 0.45,
    after: 0.96, rgb: 0.0010, bloomStr: 0.90, bloomRad: 0.45, bloomThr: 0.18,
    autoSpin: true, spinSpeed: 0.002
  });
  syncUI(); loadPlanes(state.planeCount); applyMaterialState(); applyViewMode();
});

$('toggleBloom').addEventListener('click', () => { bloomPass.enabled = !bloomPass.enabled; });

// ------------------------------------- Resize & Animate (more frequent hull rebuild to avoid strobing)
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
  uniforms.uTime.value += dt;

  afterPass.uniforms['damp'].value = Math.pow(state.after, dt * 60.0);

  updateTargets(uniforms.uTime.value);
  integrateSeeds(state.smooth); // smoothing reduces jitter

  rebuildAccumulator += dt;
  if (rebuildAccumulator > 0.016) { // ~60 Hz rebuild for stable motion
    rebuildHull();
    rebuildAccumulator = 0;
  }

  if (state.autoSpin && mesh) mesh.rotation.y += state.spinSpeed;
  controls.update();
  composer.render();
  requestAnimationFrame(animate);
}
animate();
