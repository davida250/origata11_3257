/**
 * Psychedelic Origami — Paper-like Folds + Crane + Shader Animation
 *
 * ✅ Rigid hinges (piecewise rigid) with sequential propagation (later creases move with paper)
 * ✅ Crane (Demo) sequence with regional masks
 * ✅ Crane (MIT FOLD import): file picker, drag-and-drop, and ./crane.fold fetch (GitHub Pages friendly)
 * ✅ Shader animation sliders: base, amplitude, speed for Hue/Iridescence/Film nm/EdgeGlow/Sectors
 * ✅ Lower default bloom; PNG download
 *
 * FOLD requirements match Origami Simulator conventions: vertices/edges/assignments/faces; foldAngle degrees
 * with valley positive (+) and mountain negative (–). (Loader handles either folded meshes or CPs.) 
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
import earcut from 'earcut';

// ---------- Renderer / Scene ----------
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x050509, 6, 36);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 1.8, 5.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// ---------- Post ----------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
// lowered default bloom for subtlety
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.35, 0.6, 0.2);
composer.addPass(bloom);
const fxaa = new ShaderPass(FXAAShader);
fxaa.material.uniforms['resolution'].value.set(1 / window.innerWidth, 1 / window.innerHeight);
composer.addPass(fxaa);
composer.addPass(new OutputPass());

// ---------- Paper geometry (square sheet for crane) ----------
const SIZE = 3.0;
const SEG = 180;
const sheetGeo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
sheetGeo.rotateX(-0.25);

// ---------- Math helpers ----------
const tmp = { v: new THREE.Vector3(), u: new THREE.Vector3() };
function sdLine2(p, a, d){ const px=p.x-a.x, py=p.y-a.y; return d.x*py - d.y*px; }
function rotPointAroundAxis(p, a, axisUnit, ang){ tmp.v.copy(p).sub(a).applyAxisAngle(axisUnit, ang).add(a); p.copy(tmp.v); }
function rotVecAxis(v, axisUnit, ang){ v.applyAxisAngle(axisUnit, ang); }
function clamp01(x){ return Math.max(0, Math.min(1, x)); }
const Ease = {
  linear: t => t,
  smoothstep: t => t*t*(3-2*t),
  easeInOutCubic: t => (t<0.5? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2)
};

// ---------- Creases + Masks (ordered sequence) ----------
const MAX_CREASES = 32;
const MAX_MASKS_PER = 4;
const VALLEY = +1, MOUNTAIN = -1;

const base = {
  count: 0,
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  amp:  new Array(MAX_CREASES).fill(0),   // |angle| in radians
  sign: new Array(MAX_CREASES).fill(1),   // +1 valley, -1 mountain
  phase:new Array(MAX_CREASES).fill(0),   // per-crease anim phase
  mCount: new Array(MAX_CREASES).fill(0),
  mA: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3())),
  mD: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3(1,0,0)))
};

function resetBase(){
  base.count=0;
  for (let i=0;i<MAX_CREASES;i++){
    base.A[i].set(0,0,0); base.D[i].set(1,0,0);
    base.amp[i]=0; base.sign[i]=1; base.phase[i]=0; base.mCount[i]=0;
    for (let m=0;m<MAX_MASKS_PER;m++){ base.mA[i][m].set(0,0,0); base.mD[i][m].set(1,0,0); }
  }
}

function addCrease({ Ax=0, Ay=0, Dx=1, Dy=0, deg=180, sign=VALLEY, masks=[] }){
  if (base.count >= MAX_CREASES) return;
  const i = base.count++;
  const d = new THREE.Vector2(Dx, Dy).normalize();
  base.A[i].set(Ax, Ay, 0);
  base.D[i].set(d.x, d.y, 0);
  base.amp[i]  = THREE.MathUtils.degToRad(Math.max(0, Math.min(180, Math.abs(deg))));
  base.sign[i] = sign >= 0 ? VALLEY : MOUNTAIN;
  base.phase[i]= Math.random()*Math.PI*2;
  base.mCount[i] = Math.min(MAX_MASKS_PER, masks.length);
  for (let m=0;m<base.mCount[i];m++){
    const mk = masks[m]; const dd = new THREE.Vector2(mk.Dx, mk.Dy).normalize();
    base.mA[i][m].set(mk.Ax, mk.Ay, 0);
    base.mD[i][m].set(dd.x, dd.y, 0);
  }
}

// ---------- Effective frames (sequential propagation, crisp hinges) ----------
const eff = {
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  ang: new Float32Array(MAX_CREASES),
  mCount: new Int32Array(MAX_CREASES),
  mA: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3())),
  mD: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3(1,0,0)))
};

const drive = { play:false, speed:0.9, progress:0.7, easing:'smoothstep', step:0, steps:1 };

function computeAngles(tSec){
  const E = Ease[drive.easing] || Ease.linear;
  for (let i=0;i<base.count;i++){
    let t = 0;
    if (i < drive.step) t = 1;
    else if (i === drive.step) t = drive.play ? (0.5 + 0.5*Math.sin(tSec*drive.speed + base.phase[i])) : drive.progress;
    else t = 0;
    eff.ang[i] = base.sign[i] * base.amp[i] * E(clamp01(t));
    eff.mCount[i] = base.mCount[i];
  }
  for (let i=base.count;i<MAX_CREASES;i++){ eff.ang[i]=0; eff.mCount[i]=0; }
}

function computeFrames(){
  for (let i=0;i<base.count;i++){
    eff.A[i].copy(base.A[i]); eff.D[i].copy(base.D[i]).normalize();
    for (let m=0;m<MAX_MASKS_PER;m++){ eff.mA[i][m].copy(base.mA[i][m]); eff.mD[i][m].copy(base.mD[i][m]).normalize(); }
  }
  for (let j=0;j<base.count;j++){
    const Aj = eff.A[j]; const Dj = eff.D[j].clone().normalize(); const ang = eff.ang[j]; if (Math.abs(ang)<1e-7) continue;
    for (let k=j+1;k<base.count;k++){
      const sd = sdLine2(eff.A[k], Aj, Dj);
      if (sd > 0.0){
        rotPointAroundAxis(eff.A[k], Aj, Dj, ang);
        rotVecAxis(eff.D[k], Dj, ang); eff.D[k].normalize();
        for (let m=0;m<MAX_MASKS_PER;m++){
          const sdM = sdLine2(eff.mA[k][m], Aj, Dj);
          if (sdM > 0.0){
            rotPointAroundAxis(eff.mA[k][m], Aj, Dj, ang);
            rotVecAxis(eff.mD[k][m], Dj, ang); eff.mD[k][m].normalize();
          }
        }
      }
    }
  }
}

// ---------- Uniforms (folding + shader look) ----------
const uniforms = {
  uTime:       { value: 0 },
  uSectors:    { value: 10.0 },
  uHueShift:   { value: 0.0 },
  uIridescence:{ value: 0.65 },
  uFilmIOR:    { value: 1.35 },
  uFilmNm:     { value: 360.0 },
  uFiber:      { value: 0.35 },
  uEdgeGlow:   { value: 0.8 },

  // folding data
  uCreaseCount: { value: 0 },
  uAeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3()) },
  uDeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3(1,0,0)) },
  uAng:   { value: new Float32Array(MAX_CREASES) },

  // masks (flattened)
  uMaskA: { value: Array.from({length: MAX_CREASES*MAX_MASKS_PER}, () => new THREE.Vector3()) },
  uMaskD: { value: Array.from({length: MAX_CREASES*MAX_MASKS_PER}, () => new THREE.Vector3(1,0,0)) },
  uMaskOn:{ value: new Float32Array(MAX_CREASES*MAX_MASKS_PER) }
};

function pushToUniforms(){
  uniforms.uCreaseCount.value = base.count;
  uniforms.uAeff.value = eff.A.map(v => v.clone());
  uniforms.uDeff.value = eff.D.map(v => v.clone());
  uniforms.uAng.value  = Float32Array.from(eff.ang);

  const flatA = [], flatD = [], on = [];
  for (let i=0;i<base.count;i++){
    for (let m=0;m<MAX_MASKS_PER;m++){
      flatA.push(eff.mA[i][m].clone()); flatD.push(eff.mD[i][m].clone());
      on.push(m < eff.mCount[i] ? 1 : 0);
    }
  }
  const pad = MAX_CREASES*MAX_MASKS_PER - flatA.length;
  for (let p=0;p<pad;p++){ flatA.push(new THREE.Vector3()); flatD.push(new THREE.Vector3(1,0,0)); on.push(0); }
  uniforms.uMaskA.value = flatA;
  uniforms.uMaskD.value = flatD;
  uniforms.uMaskOn.value = Float32Array.from(on);
}

// ---------- Shaders (vertex = rigid hinges + masks; fragment = psychedelic paper) ----------
const vs = /* glsl */`
  #define MAX_CREASES ${MAX_CREASES}
  #define MAX_MASKS_PER ${MAX_MASKS_PER}
  precision highp float;

  uniform int   uCreaseCount;
  uniform vec3  uAeff[MAX_CREASES];
  uniform vec3  uDeff[MAX_CREASES];
  uniform float uAng[MAX_CREASES];

  uniform vec3  uMaskA[MAX_CREASES*MAX_MASKS_PER];
  uniform vec3  uMaskD[MAX_CREASES*MAX_MASKS_PER];
  uniform float uMaskOn[MAX_CREASES*MAX_MASKS_PER];

  varying vec3 vPos; varying vec3 vN; varying vec3 vLocal; varying vec2 vUv;

  vec3 rotAroundLine(vec3 p, vec3 a, vec3 u, float ang){
    vec3 v = p - a; float c = cos(ang), s = sin(ang);
    return a + v*c + cross(u, v)*s + u*dot(u, v)*(1.0 - c);
  }
  vec3 rotVec(vec3 v, vec3 u, float ang){ float c=cos(ang), s=sin(ang); return v*c + cross(u, v)*s + u*dot(u,v)*(1.0-c); }
  float sdLine(vec2 p, vec2 a, vec2 d){ return d.x*(p.y - a.y) - d.y*(p.x - a.x); }
  bool inMask(int i, vec2 p){
    for (int m=0; m<MAX_MASKS_PER; m++){
      int idx = i*MAX_MASKS_PER + m;
      if (uMaskOn[idx] > 0.5){
        vec2 a = uMaskA[idx].xy, d = normalize(uMaskD[idx].xy);
        if (sdLine(p, a, d) <= 0.0) return false;
      }
    }
    return true;
  }

  void main(){
    vUv = uv;
    vec3 p = position;
    vec3 n = normalize(normal);

    for (int i=0; i<MAX_CREASES; i++){
      if (i >= uCreaseCount) break;
      vec3 a = uAeff[i];
      vec3 d = normalize(uDeff[i]);

      float sd = sdLine(p.xy, a.xy, d.xy);
      if (sd > 0.0 && inMask(i, p.xy)){
        p = rotAroundLine(p, a, d, uAng[i]);
        n = normalize(rotVec(n, d, uAng[i]));
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
  float sdLine(vec2 p, vec2 a, vec2 d){ return d.x*(p.y - a.y) - d.y*(p.x - a.x); }

  void main(){
    // Kaleidoscopic mapping
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

    // fibers + grain
    float fiberLines = 0.0;
    {
      float warp = fbm(vLocal.xy*4.0 + vec2(0.2*uTime, -0.1*uTime));
      float l = sin(vLocal.y*420.0 + warp*8.0);
      float widthAA = fwidth(l);
      fiberLines = smoothstep(0.6, 0.6 - widthAA, abs(l));
    }
    float grain = fbm(vLocal.xy*25.0);
    baseCol *= 1.0 + uFiber*(0.06*grain - 0.03) + uFiber*0.08*fiberLines;

    // crease glow
    float minD = 1e9;
    for (int i=0; i<MAX_CREASES; i++){
      if (i >= uCreaseCount) break;
      vec2 a2 = uAeff[i].xy;
      vec2 d2 = normalize(uDeff[i].xy);
      float sd = abs(sdLine(vLocal.xy, a2, d2));
      minD = min(minD, sd);
    }
    float aa = fwidth(minD);
    float edge = 1.0 - smoothstep(0.0025, 0.0025 + aa, minD);

    // iridescence
    vec3 V = normalize(cameraPosition - vPos);
    vec3 N = normalize(vN);
    float cosT = clamp(dot(N, V), 0.0, 1.0);
    vec3 film = thinFilm(cosT, uFilmIOR, uFilmNm);
    float F = pow(1.0 - cosT, 5.0);
    vec3 col = mix(baseCol, mix(baseCol, film, uIridescence), F);

    col += uEdgeGlow * edge * film * 0.6;

    float vign = smoothstep(1.2, 0.2, length(vUv-0.5)*1.2);
    gl_FragColor = vec4(col*vign, 1.0);
  }
`;

// ---------- Material + Mesh ----------
const mat = new THREE.ShaderMaterial({
  vertexShader: vs, fragmentShader: fs, uniforms,
  side: THREE.DoubleSide, extensions: { derivatives: true }
});
const sheet = new THREE.Mesh(sheetGeo, mat);
scene.add(sheet);

// background dome
scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(50, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0x070711, side: THREE.BackSide })
));

// ---------- GUI (optional look) ----------
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
looks.close();

// ---------- Presets ----------
function preset_half_vertical_valley(){
  resetBase();
  addCrease({ Ax:0, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
}
function preset_half_horizontal_valley(){
  resetBase();
  addCrease({ Ax:0, Ay:0, Dx:1, Dy:0, deg:180, sign:VALLEY });
}
function preset_diagonal_valley(){
  resetBase();
  addCrease({ Ax:0, Ay:0, Dx:1, Dy:1, deg:180, sign:VALLEY });
}
function preset_gate_valley(){
  resetBase();
  const x = SIZE*0.25;
  addCrease({ Ax:+x, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
  addCrease({ Ax:-x, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
}
function preset_accordion_5(){
  resetBase();
  const n = 5;
  for (let i=1;i<=n;i++){
    const x = THREE.MathUtils.lerp(-SIZE/2, SIZE/2, i/(n+1));
    const sign = (i % 2 === 1) ? VALLEY : MOUNTAIN;
    addCrease({ Ax:x, Ay:0, Dx:0, Dy:1, deg:180, sign });
  }
}
function preset_single_vertical_mountain(){
  resetBase();
  addCrease({ Ax:0, Ay:0, Dx:0, Dy:1, deg:180, sign:MOUNTAIN });
}

// Crane (Demo): sequential masked folds (visual, paper-like)
function preset_crane_demo(){
  resetBase();
  const s = SIZE/2;

  // Pre-creases: two diagonals
  addCrease({ Ax:0, Ay:0, Dx:1, Dy:1,  deg:180, sign:VALLEY });
  addCrease({ Ax:0, Ay:0, Dx:1, Dy:-1, deg:180, sign:VALLEY });

  // Vertical split (masked top / bottom for neck/tail)
  addCrease({
    Ax:0, Ay:0, Dx:0, Dy:1, deg:150, sign:VALLEY,
    masks:[
      { Ax:0, Ay:0.00, Dx:0, Dy:1 },
      { Ax:-0.0001, Ay:-0.0001, Dx: 1, Dy: 1 },
      { Ax: 0.0001, Ay:-0.0001, Dx:-1, Dy: 1 }
    ]
  });
  addCrease({
    Ax:0, Ay:0, Dx:0, Dy:1, deg:150, sign:MOUNTAIN,
    masks:[
      { Ax:0, Ay:0.00, Dx:0, Dy:-1 },
      { Ax:-0.0001, Ay: 0.0001, Dx: 1, Dy:-1 },
      { Ax: 0.0001, Ay: 0.0001, Dx:-1, Dy:-1 }
    ]
  });

  // Wings
  addCrease({
    Ax:0, Ay:0, Dx:1, Dy:-1, deg:120, sign:VALLEY,
    masks:[ { Ax:0, Ay:0, Dx:0, Dy:1 }, { Ax:0, Ay:0, Dx:1, Dy:0 } ]
  });
  addCrease({
    Ax:0, Ay:0, Dx:1, Dy:1, deg:120, sign:VALLEY,
    masks:[ { Ax:0, Ay:0, Dx:0, Dy:1 }, { Ax:0, Ay:0, Dx:-1, Dy:0 } ]
  });

  // Tail / Neck approximations
  addCrease({
    Ax:0.0, Ay:-0.3*s, Dx:1, Dy:-0.15, deg:140, sign:VALLEY,
    masks:[ { Ax:0, Ay:-0.1, Dx:0, Dy:-1 }, { Ax: 0.0, Ay:0.0, Dx:1, Dy:0 } ]
  });
  addCrease({
    Ax:0.0, Ay:-0.3*s, Dx:-1, Dy:-0.15, deg:140, sign:VALLEY,
    masks:[ { Ax:0, Ay:-0.1, Dx:0, Dy:-1 }, { Ax: 0.0, Ay:0.0, Dx:-1, Dy:0 } ]
  });

  // Head tweak
  addCrease({
    Ax:-0.45*s, Ay:-0.65*s, Dx:1, Dy:-0.2, deg:90, sign:MOUNTAIN,
    masks:[ { Ax:-0.1, Ay:-0.2, Dx:-1, Dy:-1 }, { Ax:-0.2, Ay:-0.2, Dx:-1, Dy:0 } ]
  });

  drive.step = 0; drive.steps = base.count;
}

// ---------- FOLD (MIT) import ----------
let craneMesh = null;

async function fetchCraneFoldInline(){
  // Try to fetch ./crane.fold relative to the page (works on GitHub Pages if file is committed)
  try{
    const res = await fetch('./crane.fold', { cache: 'no-store' });
    if (!res.ok) throw new Error('No crane.fold at site root');
    return await res.json();
  }catch(e){ return null; }
}

function parseFOLDFromText(text){
  try{ return JSON.parse(text); }catch(e){ return null; }
}

function triangulateFace3D(faceIndices, verts){
  // Project polygon to 2D using a simple local basis for earcut
  if (faceIndices.length < 3) return [];
  const i0 = faceIndices[0];
  const p0 = new THREE.Vector3().fromArray(verts[i0]);
  const p1 = new THREE.Vector3().fromArray(verts[faceIndices[1]]);
  const p2 = new THREE.Vector3().fromArray(verts[faceIndices[2]]);
  const e1 = p1.clone().sub(p0).normalize();
  const e2 = p2.clone().sub(p0);
  e2.addScaledVector(e1, -e2.dot(e1)).normalize(); // Gram-Schmidt
  const flat = [];
  for (const idx of faceIndices){
    const p = new THREE.Vector3().fromArray(verts[idx]);
    const v = p.clone().sub(p0);
    flat.push(v.dot(e1), v.dot(e2));
  }
  const ears = earcut(flat);
  const tris = [];
  for (let t=0;t<ears.length;t+=3){
    const a = faceIndices[ears[t]];
    const b = faceIndices[ears[t+1]];
    const c = faceIndices[ears[t+2]];
    tris.push([a,b,c]);
  }
  return tris;
}

function buildMeshFromFOLD(fold){
  const verts = (fold.vertices_coords3d || fold.vertices_coords || []).map(v => [v[0], v[1], v[2] || 0]);
  const faces = fold.faces_vertices || [];
  if (!verts.length || !faces.length) throw new Error('FOLD missing faces/vertices');

  const pos = [];
  for (const f of faces){
    if (f.length === 3){
      for (const i of f){ const v = verts[i]; pos.push(v[0], v[1], v[2]); }
    } else if (f.length > 3){
      const tris = triangulateFace3D(f, verts);
      for (const tri of tris){ for (const i of tri){ const v=verts[i]; pos.push(v[0], v[1], v[2]); } }
    }
  }
  if (pos.length < 9) throw new Error('No triangles after triangulation.');

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();

  const mesh = new THREE.Mesh(g, new THREE.ShaderMaterial({
    vertexShader: vs, fragmentShader: fs, uniforms, side: THREE.DoubleSide
  }));
  return mesh;
}

async function handleFOLDObject(obj){
  // If it's a folded mesh (faces+verts present): display it directly
  const hasFaces = Array.isArray(obj.faces_vertices) && obj.faces_vertices.length>0;
  const hasVerts = Array.isArray(obj.vertices_coords) && obj.vertices_coords.length>0;
  if (hasFaces && hasVerts){
    try{
      const mesh = buildMeshFromFOLD(obj);
      if (craneMesh) scene.remove(craneMesh);
      craneMesh = mesh; craneMesh.position.set(0,0,0.002); // nudge to avoid z-fight
      scene.add(craneMesh);
      sheet.visible = false;
      setFoldStatus('Loaded FOLD (faces+verts). Showing folded mesh.', false);
      return;
    }catch(e){
      setFoldStatus('FOLD faces could not be triangulated.', true);
    }
  }

  // Otherwise, treat as crease pattern (CP) and preview kinematically
  const Vc = obj.vertices_coords, Ev = obj.edges_vertices, Ea = obj.edges_assignment, Ef = obj.edges_foldAngle;
  if (!Vc || !Ev || !Ea){
    setFoldStatus('FOLD missing CP fields (vertices_coords, edges_vertices, edges_assignment).', true);
    return;
  }

  resetBase();
  // Fit CP to our sheet (square SIZE×SIZE)
  let minX=+Infinity,minY=+Infinity,maxX=-Infinity,maxY=-Infinity;
  for (const v of Vc){ const x=v[0], y=v[1]; if(x<minX)minX=x; if(x>maxX)maxX=x; if(y<minY)minY=y; if(y>maxY)maxY=y; }
  const w=maxX-minX||1, h=maxY-minY||1; const s = Math.min(SIZE/w, SIZE/h); const cx=(minX+maxX)/2, cy=(minY+maxY)/2;

  const N = Math.min(Ev.length, MAX_CREASES);
  for (let i=0;i<N;i++){
    const assign = (Ea[i]||'').toUpperCase();
    if (assign!=='M' && assign!=='V') continue;
    const [ia,ib] = Ev[i] || [];
    const A = Vc[ia], B = Vc[ib]; if (!A || !B) continue;
    const Ax=(A[0]-cx)*s, Ay=(A[1]-cy)*s; const Dx=B[0]-A[0], Dy=B[1]-A[1];
    let deg = 120;
    let sign = assign==='M' ? MOUNTAIN : VALLEY;
    if (Ef && typeof Ef[i]==='number'){ deg = Math.min(180, Math.abs(Ef[i])); sign = (Ef[i]>=0)? VALLEY : MOUNTAIN; }
    addCrease({ Ax, Ay, Dx, Dy, deg, sign });
  }
  sheet.visible = true;
  if (craneMesh){ scene.remove(craneMesh); craneMesh = null; }
  setFoldStatus('Loaded FOLD crease pattern (kinematic preview).', false);
}

// ---------- DOM wiring ----------
const presetSel   = document.getElementById('preset');
const btnApply    = document.getElementById('btnApply');
const btnPlay     = document.getElementById('btnPlay');
const btnReset    = document.getElementById('btnReset');
const btnPrev     = document.getElementById('btnStepPrev');
const btnNext     = document.getElementById('btnStepNext');
const progress    = document.getElementById('progress');
const speed       = document.getElementById('speed');
const easingSel   = document.getElementById('easing');
const stepInfo    = document.getElementById('stepInfo');
const btnSnap     = document.getElementById('btnSnap');
const foldFile    = document.getElementById('foldFile');
const foldStatus  = document.getElementById('foldStatus');
const dropHint    = document.getElementById('dropHint');

function setFoldStatus(msg, isErr){ foldStatus.textContent = msg; foldStatus.className = isErr? 'muted danger' : 'muted'; }

btnApply.onclick = async () => {
  const v = presetSel.value;
  if (v === 'crane-fold'){
    const json = await fetchCraneFoldInline();
    if (json){ await handleFOLDObject(json); }
    else {
      setFoldStatus('No ./crane.fold found. Use “Load FOLD” or drag & drop a .fold file.', true);
      sheet.visible = true;
      if (craneMesh){ scene.remove(craneMesh); craneMesh = null; }
    }
    drive.play = false; btnPlay.textContent = 'Play';
    return;
  }
  // folding presets
  sheet.visible = true; if (craneMesh){ scene.remove(craneMesh); craneMesh = null; }
  if (v==='half-vertical-valley') preset_half_vertical_valley();
  else if (v==='half-horizontal-valley') preset_half_horizontal_valley();
  else if (v==='diagonal-valley') preset_diagonal_valley();
  else if (v==='gate-valley') preset_gate_valley();
  else if (v==='accordion-5') preset_accordion_5();
  else if (v==='single-vertical-mountain') preset_single_vertical_mountain();
  else if (v==='crane-demo') preset_crane_demo();

  drive.step = 0; drive.steps = base.count || 1;
  updateStepInfo();
  camera.position.x += (Math.random()-0.5)*0.03;
  camera.position.y += (Math.random()-0.5)*0.03;
};
presetSel.addEventListener('change', () => btnApply.click());

btnPlay.onclick = () => { drive.play = !drive.play; btnPlay.textContent = drive.play ? 'Pause' : 'Play'; };
btnReset.onclick = () => { drive.play=false; btnPlay.textContent='Play'; drive.progress=0; progress.value='0'; drive.step=0; updateStepInfo(); };
btnPrev.onclick = () => { if (drive.step>0) drive.step--; updateStepInfo(); };
btnNext.onclick = () => { if (drive.step<drive.steps-1) drive.step++; updateStepInfo(); };
function updateStepInfo(){ stepInfo.textContent = `${Math.min(drive.step+1, drive.steps)}/${drive.steps}`; }

progress.addEventListener('input', () => { drive.progress = parseFloat(progress.value); });
speed.addEventListener('input', () => { drive.speed = parseFloat(speed.value); });
easingSel.addEventListener('change', () => { drive.easing = easingSel.value; });

btnSnap.onclick = () => {
  renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'origami.png'; a.click();
    URL.revokeObjectURL(url);
  }, 'image/png', 1.0);
};

foldFile.addEventListener('change', async (e) => {
  const file = e.target.files?.[0]; if (!file) return;
  const text = await file.text(); const obj = parseFOLDFromText(text);
  if (!obj){ setFoldStatus('Invalid JSON in FOLD file.', true); return; }
  await handleFOLDObject(obj);
});
document.addEventListener('dragover', e => { e.preventDefault(); dropHint.style.opacity = '1'; });
document.addEventListener('dragleave', e => { dropHint.style.opacity = '0.7'; });
document.addEventListener('drop', async e => {
  e.preventDefault(); dropHint.style.opacity = '0.7';
  const file = e.dataTransfer?.files?.[0]; if (!file) return;
  const text = await file.text(); const obj = parseFOLDFromText(text);
  if (!obj){ setFoldStatus('Invalid JSON in FOLD file.', true); return; }
  await handleFOLDObject(obj);
});

// ---------- Shader Animation (sliders) ----------
const shaderAuto = document.getElementById('shaderAuto');
const shaderGlobalSpeed = document.getElementById('shaderGlobalSpeed');
const hueBase = document.getElementById('hueBase'), hueAmp = document.getElementById('hueAmp'), hueSpeed = document.getElementById('hueSpeed');
const iriBase = document.getElementById('iriBase'), iriAmp = document.getElementById('iriAmp'), iriSpeed = document.getElementById('iriSpeed');
const filmBase = document.getElementById('filmBase'), filmAmp = document.getElementById('filmAmp'), filmSpeed = document.getElementById('filmSpeed');
const edgeBase = document.getElementById('edgeBase'), edgeAmp = document.getElementById('edgeAmp'), edgeSpeed = document.getElementById('edgeSpeed');
const secBase = document.getElementById('secBase'), secAmp = document.getElementById('secAmp'), secSpeed = document.getElementById('secSpeed');

function updateShaderAnim(t){
  const auto = shaderAuto.value === 'on' ? 1 : 0;
  const gs = parseFloat(shaderGlobalSpeed.value);

  const HB = parseFloat(hueBase.value),  HA = parseFloat(hueAmp.value),  HS = parseFloat(hueSpeed.value);
  const IB = parseFloat(iriBase.value),  IA = parseFloat(iriAmp.value),  IS = parseFloat(iriSpeed.value);
  const FB = parseFloat(filmBase.value), FA = parseFloat(filmAmp.value), FS = parseFloat(filmSpeed.value);
  const EB = parseFloat(edgeBase.value), EA = parseFloat(edgeAmp.value), ES = parseFloat(edgeSpeed.value);
  const SB = parseFloat(secBase.value),  SA = parseFloat(secAmp.value),  SS = parseFloat(secSpeed.value);

  uniforms.uHueShift.value     = clamp01(HB + (auto? HA*Math.sin((HS+gs)*t) : 0));
  uniforms.uIridescence.value  = clamp01(IB + (auto? IA*Math.sin((IS+0.2*gs)*t+1.3) : 0));
  uniforms.uFilmNm.value       = THREE.MathUtils.clamp(FB + (auto? FA*Math.sin((FS+0.15*gs)*t+0.7) : 0), 100, 800);
  uniforms.uEdgeGlow.value     = THREE.MathUtils.clamp(EB + (auto? EA*Math.sin((ES+0.25*gs)*t+2.1) : 0), 0.0, 2.0);

  const sVal = SB + (auto? SA*Math.sin((SS+0.12*gs)*t+0.4) : 0);
  uniforms.uSectors.value = THREE.MathUtils.clamp(Math.round(sVal), 3, 24);
}

// ---------- Start ----------
function startWithCraneDemo(){
  presetSel.value = 'crane-demo';
  btnApply.click();
  progress.value = String(drive.progress);
}
startWithCraneDemo();

// ---------- Per-frame update ----------
function tick(tMs){
  const t = tMs * 0.001;
  uniforms.uTime.value = t;

  // Folding
  computeAngles(t);
  computeFrames();
  pushToUniforms();

  // Shader animation
  updateShaderAnim(t);

  // Render
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
