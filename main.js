/**
 * Origami — 3 Presets (No File Loads)
 *
 * 1) Half Vertical (Valley)   — simple working fold
 * 2) Diagonal (Valley)        — simple working fold
 * 3) Flapping Bird (built-in) — embedded low-poly origami-style bird with animated wings
 *
 * Notes:
 * - The fold engine uses a tiny uniform budget (MAX_CREASES=2), so it renders reliably everywhere.
 * - Valley = +°, Mountain = −° (same sign convention as Origami Simulator’s FOLD docs). 
 *   See: "fold angle is positive for valley folds, negative for mountain folds".  [origamisimulator.org → Design Tips] 
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
renderer.toneMappingExposure = 1.06;
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
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.3, 0.6, 0.2);
composer.addPass(bloom);
const fxaa = new ShaderPass(FXAAShader);
fxaa.material.uniforms.resolution.value.set(1 / window.innerWidth, 1 / window.height || 1 / window.innerHeight);
composer.addPass(fxaa);
composer.addPass(new OutputPass());

// ---------- Paper sheet for fold presets ----------
const SIZE = 3.0;
const SEG = 160;
const sheetGeo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
sheetGeo.rotateX(-0.25);

// ---------- Math helpers ----------
const tmp = { v: new THREE.Vector3() };
function sdLine2(p, a, d){ const px=p.x-a.x, py=p.y-a.y; return d.x*py - d.y*px; }
function rotPointAroundAxis(p, a, axisUnit, ang){ tmp.v.copy(p).sub(a).applyAxisAngle(axisUnit, ang).add(a); p.copy(tmp.v); }
function rotVecAxis(v, axisUnit, ang){ v.applyAxisAngle(axisUnit, ang); }
function clamp01(x){ return x<0?0:x>1?1:x; }

// ---------- Minimal crease engine ----------
const MAX_CREASES = 2;
const VALLEY = +1, MOUNTAIN = -1;

const base = {
  count: 0,
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  amp:  new Array(MAX_CREASES).fill(0),   // |angle| in radians
  sign: new Array(MAX_CREASES).fill(1)    // +1 valley, -1 mountain
};
function resetBase(){
  base.count=0;
  for (let i=0;i<MAX_CREASES;i++){
    base.A[i].set(0,0,0); base.D[i].set(1,0,0); base.amp[i]=0; base.sign[i]=1;
  }
}
function addCrease({ Ax=0, Ay=0, Dx=1, Dy=0, deg=180, sign=VALLEY }){
  if (base.count >= MAX_CREASES) return;
  const i = base.count++;
  const d = new THREE.Vector2(Dx, Dy).normalize();
  base.A[i].set(Ax, Ay, 0);
  base.D[i].set(d.x, d.y, 0);
  base.amp[i]  = THREE.MathUtils.degToRad(Math.max(0, Math.min(180, Math.abs(deg))));
  base.sign[i] = sign >= 0 ? VALLEY : MOUNTAIN;
}

const eff = {
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  ang: new Float32Array(MAX_CREASES)
};
const drive = { play:false, progress:0.7, speed:0.9 };

function computeAngles(tSec){
  const t = drive.play ? (0.5 + 0.5*Math.sin(tSec*drive.speed)) : drive.progress;
  for (let i=0;i<base.count;i++){
    eff.ang[i] = base.sign[i] * base.amp[i] * clamp01(t);
  }
  for (let i=base.count;i<MAX_CREASES;i++) eff.ang[i]=0;
}
function computeFrames(){
  for (let i=0;i<base.count;i++){ eff.A[i].copy(base.A[i]); eff.D[i].copy(base.D[i]).normalize(); }
  for (let j=0;j<base.count;j++){
    const Aj=eff.A[j], Dj=eff.D[j].clone().normalize(), ang=eff.ang[j]; if (Math.abs(ang)<1e-7) continue;
    for (let k=j+1;k<base.count;k++){
      const sd = sdLine2(eff.A[k], Aj, Dj);
      if (sd > 0.0){ rotPointAroundAxis(eff.A[k], Aj, Dj, ang); rotVecAxis(eff.D[k], Dj, ang); eff.D[k].normalize(); }
    }
  }
}

// ---------- Uniforms + Shaders ----------
const uniforms = {
  uTime:       { value: 0 },
  uSectors:    { value: 10.0 },
  uHueShift:   { value: 0.0 },
  uIridescence:{ value: 0.65 },
  uFilmIOR:    { value: 1.35 },
  uFilmNm:     { value: 360.0 },
  uFiber:      { value: 0.35 },
  uEdgeGlow:   { value: 0.7 },

  uCreaseCount: { value: 0 },
  uAeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3()) },
  uDeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3(1,0,0)) },
  uAng:   { value: new Float32Array(MAX_CREASES) }
};
function pushToUniforms(){
  uniforms.uCreaseCount.value = base.count;
  uniforms.uAeff.value = eff.A.map(v => v.clone());
  uniforms.uDeff.value = eff.D.map(v => v.clone());
  uniforms.uAng.value  = Float32Array.from(eff.ang);
  mat.uniformsNeedUpdate = true;
}

const vs = /* glsl */`
  #define MAX_CREASES ${MAX_CREASES}
  precision highp float;

  uniform int   uCreaseCount;
  uniform vec3  uAeff[MAX_CREASES];
  uniform vec3  uDeff[MAX_CREASES];
  uniform float uAng[MAX_CREASES];

  varying vec3 vPos; varying vec3 vN; varying vec3 vLocal; varying vec2 vUv;

  vec3 rotAroundLine(vec3 p, vec3 a, vec3 u, float ang){
    vec3 v = p - a; float c = cos(ang), s = sin(ang);
    return a + v*c + cross(u, v)*s + u*dot(u, v)*(1.0 - c);
  }
  vec3 rotVec(vec3 v, vec3 u, float ang){ float c=cos(ang), s=sin(ang); return v*c + cross(u, v)*s + u*dot(u,v)*(1.0-c); }
  float sdLine(vec2 p, vec2 a, vec2 d){ return d.x*(p.y - a.y) - d.y*(p.x - a.x); }

  void main(){
    vUv = uv;
    vec3 p = position;
    vec3 n = normalize(normal);

    // crisp hinge: rotate only the positive side of each crease
    for (int i=0; i<MAX_CREASES; i++){
      if (i >= uCreaseCount) break;
      vec3 a = uAeff[i];
      vec3 d = normalize(uDeff[i]);
      float sd = sdLine(p.xy, a.xy, d.xy);
      if (sd > 0.0){
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
    // kaleidoscopic mapping
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

    // crease glow (affects folds; harmless for bird)
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

// Material shared by sheet and bird (keeps the look unified)
const mat = new THREE.ShaderMaterial({
  vertexShader: vs, fragmentShader: fs, uniforms,
  side: THREE.DoubleSide, extensions: { derivatives: true }
});

// Paper sheet mesh
const sheet = new THREE.Mesh(sheetGeo, mat);
scene.add(sheet);

// Background dome
scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(50, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0x070711, side: THREE.BackSide })
));

// ---------- GUI (optional look tweaks) ----------
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
function preset_diagonal_valley(){
  resetBase();
  addCrease({ Ax:0, Ay:0, Dx:1, Dy:1, deg:180, sign:VALLEY });
}

// ---------- Built-in Flapping Bird (no external files) ----------
const bird = {
  group: null,
  body: null,
  leftWing: null,
  rightWing: null,
  auto: true,
  amp: 0.6,
  speed: 1.2,
  base: 0.2,
};

function makeTriGeometry(pts /*array of [x,y,z] triples (multiple of 3)*/){
  const pos = new Float32Array(pts.flat());
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return g;
}

function buildFlappingBird(){
  const gBody = makeTriGeometry([
    // body diamond (two tris)
    [ 0.00,  0.30, 0], [-0.15,  0.05, 0], [ 0.15,  0.05, 0],
    [ 0.00, -0.20, 0], [-0.15,  0.05, 0], [ 0.15,  0.05, 0],

    // tail wedge (two tris)
    [-0.15,  0.05, 0], [-0.38,  0.00, 0], [-0.28, -0.06, 0],
    [-0.15,  0.05, 0], [-0.28, -0.06, 0], [-0.20,  0.12, 0],

    // neck/head (two tris)
    [ 0.15,  0.05, 0], [ 0.32,  0.12, 0], [ 0.40,  0.05, 0],
    [ 0.32,  0.12, 0], [ 0.42,  0.16, 0], [ 0.40,  0.05, 0],
  ]);

  // Wing geometry built around a hinge at local origin (0,0,0)
  const gWingL = makeTriGeometry([
    [ 0.00,  0.00, 0], [-0.95,  0.35, 0], [-0.90, -0.28, 0],
  ]);
  const gWingR = makeTriGeometry([
    [ 0.00,  0.00, 0], [ 0.90, -0.28, 0], [ 0.95,  0.35, 0],
  ]);

  const body = new THREE.Mesh(gBody, mat);
  const leftWing = new THREE.Mesh(gWingL, mat);
  const rightWing = new THREE.Mesh(gWingR, mat);

  // Position wings at “shoulders” (hinge points); geometry is defined relative to hinge
  leftWing.position.set(-0.12, 0.08, 0.0);
  rightWing.position.set( 0.12, 0.08, 0.0);

  const group = new THREE.Group();
  group.add(body, leftWing, rightWing);

  // Global transform so it reads well in the scene
  group.rotation.x = -0.25; // match paper tilt
  group.scale.set(1.9, 1.9, 1.0);

  // Slight out-of-plane tilt to add depth
  leftWing.rotation.x = 0.12;
  rightWing.rotation.x = 0.12;

  return { group, body, leftWing, rightWing };
}

function ensureBird(){
  if (bird.group) return;
  const b = buildFlappingBird();
  bird.group = b.group; bird.body = b.body; bird.leftWing = b.leftWing; bird.rightWing = b.rightWing;
  scene.add(bird.group);
}

// ---------- DOM wiring ----------
const presetSel   = document.getElementById('preset');
const btnApply    = document.getElementById('btnApply');
const btnPlay     = document.getElementById('btnPlay');
const btnSnap     = document.getElementById('btnSnap');
const progress    = document.getElementById('progress');

const wingAmp     = document.getElementById('wingAmp');
const wingSpeed   = document.getElementById('wingSpeed');
const wingBase    = document.getElementById('wingBase');

const shaderAuto = document.getElementById('shaderAuto');
const shaderGlobalSpeed = document.getElementById('shaderGlobalSpeed');
const hueBase = document.getElementById('hueBase'), hueAmp = document.getElementById('hueAmp'), hueSpeed = document.getElementById('hueSpeed');
const filmBase = document.getElementById('filmBase'), filmAmp = document.getElementById('filmAmp'), filmSpeed = document.getElementById('filmSpeed');
const edgeBase = document.getElementById('edgeBase'), edgeAmp = document.getElementById('edgeAmp'), edgeSpeed = document.getElementById('edgeSpeed');

let currentPreset = 'half-vertical-valley';

btnApply.onclick = () => {
  currentPreset = presetSel.value;

  if (currentPreset === 'flapping-bird'){
    // Show bird, hide sheet & zero out creases
    ensureBird();
    sheet.visible = false;
    resetBase(); pushToUniforms();
  } else {
    // Show sheet, hide bird (if present)
    sheet.visible = true;
    if (bird.group) bird.group.visible = false;

    if (currentPreset==='half-vertical-valley') preset_half_vertical_valley();
    else if (currentPreset==='diagonal-valley') preset_diagonal_valley();

    // small camera nudge for visual feedback
    camera.position.x += (Math.random()-0.5) * 0.02;
    camera.position.y += (Math.random()-0.5) * 0.02;
  }
};

presetSel.addEventListener('change', () => btnApply.click());

btnPlay.onclick = () => {
  if (currentPreset === 'flapping-bird'){
    bird.auto = !bird.auto;
    btnPlay.textContent = bird.auto ? 'Pause' : 'Play';
  } else {
    drive.play = !drive.play;
    btnPlay.textContent = drive.play ? 'Pause' : 'Play';
  }
};

btnSnap.onclick = () => {
  renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'origami.png'; a.click();
    URL.revokeObjectURL(url);
  }, 'image/png', 1.0);
};

progress.addEventListener('input', () => {
  if (currentPreset === 'flapping-bird'){
    // When paused, use progress to scrub wing pose (−1..+1 mapped from 0..1)
    const t = parseFloat(progress.value)*2 - 1;
    const angle = bird.base + bird.amp * t;
    if (bird.leftWing){ bird.leftWing.rotation.z =  angle; }
    if (bird.rightWing){ bird.rightWing.rotation.z = -angle; }
  } else {
    drive.progress = parseFloat(progress.value);
  }
});
wingAmp.addEventListener('input', () => bird.amp = parseFloat(wingAmp.value));
wingSpeed.addEventListener('input', () => bird.speed = parseFloat(wingSpeed.value));
wingBase.addEventListener('input', () => bird.base = parseFloat(wingBase.value));

// ---------- Shader Animation (texture look) ----------
function updateShaderAnim(t){
  const auto = shaderAuto.value === 'on';
  const gs = parseFloat(shaderGlobalSpeed.value);

  const HB = parseFloat(hueBase.value),  HA = parseFloat(hueAmp.value),  HS = parseFloat(hueSpeed.value);
  const FB = parseFloat(filmBase.value), FA = parseFloat(filmAmp.value), FS = parseFloat(filmSpeed.value);
  const EB = parseFloat(edgeBase.value), EA = parseFloat(edgeAmp.value), ES = parseFloat(edgeSpeed.value);

  uniforms.uHueShift.value = THREE.MathUtils.clamp(HB + (auto? HA*Math.sin((HS+gs)*t) : 0), 0, 1);
  uniforms.uFilmNm.value   = THREE.MathUtils.clamp(FB + (auto? FA*Math.sin((FS+0.15*gs)*t+0.7) : 0), 100, 800);
  uniforms.uEdgeGlow.value = THREE.MathUtils.clamp(EB + (auto? EA*Math.sin((ES+0.25*gs)*t+2.1) : 0), 0.0, 2.0);
}

// ---------- Start ----------
function start(){
  // Bird not created until first time it’s selected
  presetSel.value = 'half-vertical-valley';
  btnApply.click();
  progress.value = String(drive.progress);
}
start();

// ---------- Frame loop ----------
function tick(tMs){
  const t = tMs * 0.001;
  uniforms.uTime.value = t;

  if (currentPreset === 'flapping-bird'){
    // show bird
    if (bird.group) bird.group.visible = true;

    // live wing animation when “auto” is on
    if (bird.auto && bird.leftWing && bird.rightWing){
      const a = bird.base + bird.amp * Math.sin(t * bird.speed);
      bird.leftWing.rotation.z  =  a;
      bird.rightWing.rotation.z = -a;
    }

    // disable the fold engine so shader runs with uCreaseCount=0
    resetBase(); pushToUniforms();
  } else {
    // folds: compute + upload crease transforms
    computeAngles(t);
    computeFrames();
    pushToUniforms();
  }

  // shader look
  updateShaderAnim(t);

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
