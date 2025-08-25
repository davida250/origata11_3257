/**
 * Origami — 3 Working Models (Crane + 2 basics)
 * - Ultra-low uniforms to avoid WebGL failures (MAX_CREASES=4; no masks)
 * - Shader look preserved (iridescent thin film + fibers + bloom toned down)
 * - Texture animation sliders (Hue, Film nm, EdgeGlow)
 * - Crane loader for MIT FOLD exports (faces+verts or CP); valley=+°, mountain=−°. 
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
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.3, 0.6, 0.2); // conservative default
composer.addPass(bloom);
const fxaa = new ShaderPass(FXAAShader);
fxaa.material.uniforms.resolution.value.set(1 / window.innerWidth, 1 / window.innerHeight);
composer.addPass(fxaa);
composer.addPass(new OutputPass());

// ---------- Paper geometry (square) ----------
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

// ---------- Simple crease engine (no masks) ----------
const MAX_CREASES = 4; // stays under uniform limits everywhere
const VALLEY = +1, MOUNTAIN = -1;

const base = {
  count: 0,
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  amp:  new Array(MAX_CREASES).fill(0),   // |angle| in radians
  sign: new Array(MAX_CREASES).fill(1),   // +1 valley, -1 mountain
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

// effective frames (for later creases moving with the paper)
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

// ---------- Uniforms (small footprint) ----------
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

// ---------- Shaders (vertex = rigid hinges; fragment = psychedelic paper) ----------
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

// ---------- Materials & meshes ----------
const mat = new THREE.ShaderMaterial({
  vertexShader: vs, fragmentShader: fs, uniforms,
  side: THREE.DoubleSide, extensions: { derivatives: true }
});
const sheet = new THREE.Mesh(sheetGeo, mat);
scene.add(sheet);

scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(50, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0x070711, side: THREE.BackSide })
));

// ---------- Presets (exactly 2 folds + Crane loader) ----------
function preset_half_vertical_valley(){
  resetBase();
  addCrease({ Ax:0, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
}
function preset_diagonal_valley(){
  resetBase();
  addCrease({ Ax:0, Ay:0, Dx:1, Dy:1, deg:180, sign:VALLEY });
}

// FOLD loader (Crane)
let craneMesh = null;

function triangulateFace2D(faceIndices, verts){
  // verts: [ [x,y,(z)], ... ] — if 3D, project to local 2D plane around first 3 points
  const idx0 = faceIndices[0];
  const p0 = verts[idx0];
  const is3D = p0.length >= 3;
  let flat = [];
  if (!is3D){
    for (const i of faceIndices){ flat.push(verts[i][0], verts[i][1]); }
  } else {
    const P0 = new THREE.Vector3(...verts[faceIndices[0]], 0).setZ(verts[faceIndices[0]][2]||0);
    const P1 = new THREE.Vector3(...verts[faceIndices[1]], 0).setZ(verts[faceIndices[1]][2]||0);
    const P2 = new THREE.Vector3(...verts[faceIndices[2]], 0).setZ(verts[faceIndices[2]][2]||0);
    const e1 = P1.clone().sub(P0).normalize();
    const e2 = P2.clone().sub(P0).addScaledVector(e1, -P2.clone().sub(P0).dot(e1)).normalize();
    for (const i of faceIndices){
      const P = new THREE.Vector3(...verts[i], 0).setZ(verts[i][2]||0);
      const v = P.clone().sub(P0);
      flat.push(v.dot(e1), v.dot(e2));
    }
  }
  const ears = earcut(flat);
  const tris = [];
  for (let t=0;t<ears.length;t+=3){
    tris.push([faceIndices[ears[t]], faceIndices[ears[t+1]], faceIndices[ears[t+2]]]);
  }
  return tris;
}

function buildMeshFromFOLD(fold){
  const verts = (fold.vertices_coords3d || fold.vertices_coords || []);
  const faces = fold.faces_vertices || [];
  if (!verts.length || !faces.length) throw new Error('FOLD has no faces/vertices.');
  const pos = [];
  for (const f of faces){
    if (f.length === 3){
      for (const i of f){ const v=verts[i]; pos.push(v[0], v[1], v[2]||0); }
    } else if (f.length > 3){
      const tris = triangulateFace2D(f, verts);
      for (const tri of tris){ for (const i of tri){ const v=verts[i]; pos.push(v[0], v[1], v[2]||0); } }
    }
  }
  if (pos.length < 9) throw new Error('Triangulation produced no triangles.');
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return new THREE.Mesh(g, new THREE.ShaderMaterial({
    vertexShader: vs, fragmentShader: fs, uniforms, side: THREE.DoubleSide
  }));
}

function showCraneMesh(mesh){
  if (craneMesh) scene.remove(craneMesh);
  craneMesh = mesh;
  craneMesh.position.set(0, 0, 0.002);
  scene.add(craneMesh);
  // ensure sheet is hidden and crease count is zero (so crane isn’t accidentally “folded”)
  sheet.visible = false;
  base.count = 0; pushToUniforms();
}

async function tryFetchCraneFold(){
  try{
    const res = await fetch('./crane.fold', { cache:'no-store' });
    if (!res.ok) throw new Error('No crane.fold in repo root.');
    return await res.json();
  }catch(e){ return null; }
}

async function handleFOLD(obj){
  try{
    // If folded mesh (faces+verts), just build mesh
    if (Array.isArray(obj.faces_vertices) && obj.faces_vertices.length && Array.isArray(obj.vertices_coords) && obj.vertices_coords.length){
      const mesh = buildMeshFromFOLD(obj);
      showCraneMesh(mesh);
      setFoldStatus('Loaded folded mesh from FOLD (faces+verts).');
      return;
    }

    // Otherwise, attempt CP preview by mapping edges to creases (best-effort)
    const Vc = obj.vertices_coords, Ev = obj.edges_vertices, Ea = obj.edges_assignment, Ef = obj.edges_foldAngle;
    if (!Vc || !Ev || !Ea){ throw new Error('FOLD missing CP fields (vertices_coords, edges_vertices, edges_assignment).'); }

    // Fit CP to our sheet
    let minX=+Infinity,minY=+Infinity,maxX=-Infinity,maxY=-Infinity;
    for (const v of Vc){ const x=v[0], y=v[1]; if(x<minX)minX=x; if(x>maxX)maxX=x; if(y<minY)minY=y; if(y>maxY)maxY=y; }
    const w=maxX-minX||1, h=maxY-minY||1; const s = Math.min(SIZE/w, SIZE/h); const cx=(minX+maxX)/2, cy=(minY+maxY)/2;

    resetBase();
    const N = Math.min(Ev.length, MAX_CREASES);
    for (let i=0;i<N;i++){
      const assign = (Ea[i]||'').toUpperCase();
      if (assign!=='M' && assign!=='V') continue;
      const [ia,ib] = Ev[i] || [];
      const A = Vc[ia], B = Vc[ib]; if (!A || !B) continue;
      const Ax=(A[0]-cx)*s, Ay=(A[1]-cy)*s; const Dx=B[0]-A[0], Dy=B[1]-A[1];
      let deg = 120; let sign = assign==='M'? MOUNTAIN : VALLEY;
      if (Ef && typeof Ef[i]==='number'){ deg = Math.min(180, Math.abs(Ef[i])); sign = (Ef[i]>=0)? VALLEY : MOUNTAIN; } // valley +, mountain −
      addCrease({ Ax, Ay, Dx, Dy, deg, sign });
    }
    sheet.visible = true; if (craneMesh) { scene.remove(craneMesh); craneMesh = null; }
    setFoldStatus('Loaded CP preview from FOLD (best-effort).');
  }catch(err){
    sheet.visible = true; if (craneMesh){ scene.remove(craneMesh); craneMesh = null; }
    setFoldStatus(err.message || 'Failed to load FOLD.');
  }
}

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

// ---------- DOM wiring ----------
const presetSel   = document.getElementById('preset');
const btnApply    = document.getElementById('btnApply');
const btnPlay     = document.getElementById('btnPlay');
const btnSnap     = document.getElementById('btnSnap');
const progress    = document.getElementById('progress');
const foldFile    = document.getElementById('foldFile');
const foldStatus  = document.getElementById('foldStatus');

function setFoldStatus(msg){ foldStatus.textContent = msg; }

btnApply.onclick = async () => {
  const v = presetSel.value;
  if (v === 'crane-fold'){
    // Show folded mesh if crane.fold exists; otherwise prompt to load
    const json = await tryFetchCraneFold();
    if (json){ await handleFOLD(json); }
    else {
      setFoldStatus('No ./crane.fold found — click “Load FOLD” and choose a crane .fold exported from origamisimulator.org.');
      // Keep current sheet visible; do not hide content.
    }
    return;
  }

  // Basic folds
  sheet.visible = true; if (craneMesh){ scene.remove(craneMesh); craneMesh = null; }
  if (v==='half-vertical-valley') preset_half_vertical_valley();
  else if (v==='diagonal-valley') preset_diagonal_valley();
  drive.progress = parseFloat(progress.value);
  setFoldStatus('Using built-in fold preset.');
};

presetSel.addEventListener('change', () => btnApply.click());

btnPlay.onclick = () => { drive.play = !drive.play; btnPlay.textContent = drive.play ? 'Pause' : 'Play'; };

btnSnap.onclick = () => {
  renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'origami.png'; a.click();
    URL.revokeObjectURL(url);
  }, 'image/png', 1.0);
};

progress.addEventListener('input', () => { drive.progress = parseFloat(progress.value); });

// File picker for FOLD
foldFile.addEventListener('change', async (e) => {
  const file = e.target.files?.[0]; if (!file) return;
  const text = await file.text(); let obj;
  try { obj = JSON.parse(text); } catch(_){ setFoldStatus('Invalid JSON in FOLD.'); return; }
  await handleFOLD(obj);
});

// ---------- Shader Animation (sliders) ----------
const shaderAuto = document.getElementById('shaderAuto');
const shaderGlobalSpeed = document.getElementById('shaderGlobalSpeed');
const hueBase = document.getElementById('hueBase'), hueAmp = document.getElementById('hueAmp'), hueSpeed = document.getElementById('hueSpeed');
const filmBase = document.getElementById('filmBase'), filmAmp = document.getElementById('filmAmp'), filmSpeed = document.getElementById('filmSpeed');
const edgeBase = document.getElementById('edgeBase'), edgeAmp = document.getElementById('edgeAmp'), edgeSpeed = document.getElementById('edgeSpeed');

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
  presetSel.value = 'half-vertical-valley';
  btnApply.click();
  progress.value = String(drive.progress);
}
start();

// ---------- Per-frame update ----------
function tick(tMs){
  const t = tMs * 0.001;
  uniforms.uTime.value = t;

  computeAngles(t);
  computeFrames();
  pushToUniforms();
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
