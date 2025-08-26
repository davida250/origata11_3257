/**
 * Psychedelic Origami — Simple Folds + Reflections (no external files)
 *
 * Presets (3):
 *  - Half Vertical (Valley)
 *  - Diagonal (Valley)
 *  - Gate (2× Valley)
 *
 * Global Animation:
 *  - One slider controls the base speed for folds + shader.
 *  - Each shader parameter has a "Speed Var" that multiplies around the global speed.
 *
 * Notes:
 *  - Fold angle sign matches Origami Simulator (valley +°, mountain −°). [Design Tips]
 *  - Dynamic reflections via CubeCamera (updated every frame). 
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
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.35, 0.6, 0.2);
composer.addPass(bloom);
const fxaa = new ShaderPass(FXAAShader);
fxaa.material.uniforms.resolution.value.set(1 / window.innerWidth, 1 / window.innerHeight);
composer.addPass(fxaa);
composer.addPass(new OutputPass());

// ---------- Background & Sheet ----------
const SIZE = 3.0;
const SEG = 160;
const sheetGeo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
sheetGeo.rotateX(-0.25);

const background = new THREE.Mesh(
  new THREE.SphereGeometry(50, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0x070711, side: THREE.BackSide })
);
scene.add(background);

// ---------- Dynamic Reflection Setup ----------
const cubeRT = new THREE.WebGLCubeRenderTarget(256, { generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter });
const cubeCam = new THREE.CubeCamera(0.1, 200, cubeRT);
scene.add(cubeCam);

// ---------- Math helpers ----------
const tmp = { v: new THREE.Vector3() };
function sdLine2(p, a, d){ const px=p.x-a.x, py=p.y-a.y; return d.x*py - d.y*px; }
function rotPointAroundAxis(p, a, axisUnit, ang){ tmp.v.copy(p).sub(a).applyAxisAngle(axisUnit, ang).add(a); p.copy(tmp.v); }
function rotVecAxis(v, axisUnit, ang){ v.applyAxisAngle(axisUnit, ang); }
function clamp01(x){ return x<0?0:x>1?1:x; }
const Ease = {
  linear: t => t,
  smoothstep: t => t*t*(3-2*t),
  easeInOutCubic: t => (t<0.5? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2)
};

// ---------- Minimal crease engine with tiny mask budget ----------
const MAX_CREASES = 6;
const MAX_MASKS_PER = 2;
const VALLEY = +1, MOUNTAIN = -1;

const base = {
  count: 0,
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  amp:  new Array(MAX_CREASES).fill(0),     // |angle| in radians
  sign: new Array(MAX_CREASES).fill(1),     // +1 valley, -1 mountain
  mCount: new Array(MAX_CREASES).fill(0),
  mA: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3())),
  mD: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3(1,0,0)))
};
function resetBase(){
  base.count=0;
  for (let i=0;i<MAX_CREASES;i++){
    base.A[i].set(0,0,0); base.D[i].set(1,0,0);
    base.amp[i]=0; base.sign[i]=1; base.mCount[i]=0;
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
  base.mCount[i] = Math.min(MAX_MASKS_PER, masks.length);
  for (let m=0;m<base.mCount[i];m++){
    const mk = masks[m]; const dd = new THREE.Vector2(mk.Dx, mk.Dy).normalize();
    base.mA[i][m].set(mk.Ax, mk.Ay, 0);
    base.mD[i][m].set(dd.x, dd.y, 0);
  }
}

const eff = {
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  ang: new Float32Array(MAX_CREASES),
  mCount: new Int32Array(MAX_CREASES),
  mA: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3())),
  mD: Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3(1,0,0)))
};

const drive = { play:false, progress:0.65, easing:'smoothstep', globalSpeed:1.0 };

// ---------- Uniforms ----------
const uniforms = {
  uTime:          { value: 0 },
  uSectors:       { value: 10.0 },
  uHueShift:      { value: 0.05 },
  uIridescence:   { value: 0.65 },
  uFilmIOR:       { value: 1.35 },
  uFilmNm:        { value: 360.0 },
  uFiber:         { value: 0.35 },
  uEdgeGlow:      { value: 0.7 },

  // reflections & lighting
  uEnvMap:        { value: cubeRT.texture },
  uReflectivity:  { value: 0.25 },
  uSpecIntensity: { value: 0.7 },
  uSpecPower:     { value: 24.0 },
  uRimIntensity:  { value: 0.5 },
  uLightDir:      { value: new THREE.Vector3(0.5, 1.0, 0.25).normalize() },

  // folding data
  uCreaseCount:   { value: 0 },
  uAeff:          { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3()) },
  uDeff:          { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3(1,0,0)) },
  uAng:           { value: new Float32Array(MAX_CREASES) },

  // masks (flattened)
  uMaskA:         { value: Array.from({length: MAX_CREASES*MAX_MASKS_PER}, () => new THREE.Vector3()) },
  uMaskD:         { value: Array.from({length: MAX_CREASES*MAX_MASKS_PER}, () => new THREE.Vector3(1,0,0)) },
  uMaskOn:        { value: new Float32Array(MAX_CREASES*MAX_MASKS_PER) }
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

// ---------- Shaders ----------
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

    // crisp hinge: rotate only the positive side of each crease, within masks
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

  // reflections & lighting
  uniform samplerCube uEnvMap;
  uniform float uReflectivity;
  uniform float uSpecIntensity, uSpecPower, uRimIntensity;
  uniform vec3  uLightDir;

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
    // Kaleidoscopic mapping (psychedelic base)
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

    // crease glow (distance to nearest crease)
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

    // viewing terms
    vec3 V = normalize(cameraPosition - vPos);
    vec3 N = normalize(vN);

    // thin-film iridescence (Schlick blend)
    float cosT = clamp(dot(N, V), 0.0, 1.0);
    vec3 film = thinFilm(cosT, uFilmIOR, uFilmNm);
    float F = pow(1.0 - cosT, 5.0);
    vec3 col = mix(baseCol, mix(baseCol, film, uIridescence), F);

    // specular + rim highlights
    vec3 L = normalize(uLightDir);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), uSpecPower) * uSpecIntensity;
    float rim  = pow(1.0 - max(dot(N, V), 0.0), 2.0) * uRimIntensity;
    col += spec + rim;

    // dynamic reflection (cube map)
    vec3 R = reflect(-V, N);
    vec3 env = textureCube(uEnvMap, R).rgb;
    col = mix(col, env, clamp(uReflectivity, 0.0, 1.0));

    // edge glow with thin-film tint
    col += uEdgeGlow * edge * film * 0.6;

    // vignette
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

// ---------- GUI (optional visual tuning) ----------
const gui = new GUI();
const looks = gui.addFolder('Look');
looks.add(uniforms.uSectors, 'value', 3, 24, 1).name('kaleidoSectors');
looks.add(uniforms.uHueShift, 'value', 0, 1, 0.001).name('hueShift');
looks.add(uniforms.uIridescence, 'value', 0, 1, 0.001).name('iridescence');
looks.add(uniforms.uFilmIOR, 'value', 1.0, 2.333, 0.001).name('filmIOR');
looks.add(uniforms.uFiber, 'value', 0, 1, 0.001).name('paperFiber');
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
function preset_gate_valley(){
  resetBase();
  const x = SIZE*0.25;
  addCrease({ Ax:+x, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
  addCrease({ Ax:-x, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
}

// ---------- DOM wiring ----------
const presetSel   = document.getElementById('preset');
const btnApply    = document.getElementById('btnApply');
const btnPlay     = document.getElementById('btnPlay');
const btnReset    = document.getElementById('btnReset');
const progress    = document.getElementById('progress');
const easingSel   = document.getElementById('easing');
const btnSnap     = document.getElementById('btnSnap');

// global + shader anim controls
const shaderAuto  = document.getElementById('shaderAuto');
const globalSpeed = document.getElementById('globalSpeed');

const hueBase = document.getElementById('hueBase'), hueAmp = document.getElementById('hueAmp'), hueVar = document.getElementById('hueVar');
const filmBase = document.getElementById('filmBase'), filmAmp = document.getElementById('filmAmp'), filmVar = document.getElementById('filmVar');
const edgeBase = document.getElementById('edgeBase'), edgeAmp = document.getElementById('edgeAmp'), edgeVar = document.getElementById('edgeVar');
const reflBase = document.getElementById('reflBase'), reflAmp = document.getElementById('reflAmp'), reflVar = document.getElementById('reflVar');

const specInt = document.getElementById('specInt'), specPow = document.getElementById('specPow'), rimInt = document.getElementById('rimInt');
const bloomStr = document.getElementById('bloomStr'), bloomRad = document.getElementById('bloomRad');

btnApply.onclick = () => {
  const v = presetSel.value;
  if (v==='half-vertical-valley') preset_half_vertical_valley();
  else if (v==='diagonal-valley') preset_diagonal_valley();
  else if (v==='gate-valley') preset_gate_valley();

  camera.position.x += (Math.random()-0.5) * 0.02;
  camera.position.y += (Math.random()-0.5) * 0.02;
};
presetSel.addEventListener('change', () => btnApply.click());

btnPlay.onclick = () => { drive.play = !drive.play; btnPlay.textContent = drive.play ? 'Pause' : 'Play'; };
btnReset.onclick = () => { drive.play=false; btnPlay.textContent='Play'; drive.progress=0; progress.value='0'; };
progress.addEventListener('input', () => { drive.progress = parseFloat(progress.value); });
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

// Surface + bloom
specInt.addEventListener('input', () => uniforms.uSpecIntensity.value = parseFloat(specInt.value));
specPow.addEventListener('input', () => uniforms.uSpecPower.value     = parseFloat(specPow.value));
rimInt .addEventListener('input', () => uniforms.uRimIntensity.value  = parseFloat(rimInt.value));

bloomStr.addEventListener('input', () => bloom.strength = parseFloat(bloomStr.value));
bloomRad.addEventListener('input', () => bloom.radius   = parseFloat(bloomRad.value));

// ---------- Shader Animation (global + per-parameter speed variation) ----------
function updateShaderAnim(t){
  const auto = shaderAuto.value === 'on';
  drive.globalSpeed = parseFloat(globalSpeed.value);

  const g = drive.globalSpeed;

  const HB = parseFloat(hueBase.value),  HA = parseFloat(hueAmp.value),  HV = parseFloat(hueVar.value);
  const FB = parseFloat(filmBase.value), FA = parseFloat(filmAmp.value), FV = parseFloat(filmVar.value);
  const EB = parseFloat(edgeBase.value), EA = parseFloat(edgeAmp.value), EV = parseFloat(edgeVar.value);
  const RB = parseFloat(reflBase.value), RA = parseFloat(reflAmp.value), RV = parseFloat(reflVar.value);

  const hueW  = g * (1.0 + HV);
  const filmW = g * (1.0 + FV);
  const edgeW = g * (1.0 + EV);
  const reflW = g * (1.0 + RV);

  if (auto){
    uniforms.uHueShift.value     = THREE.MathUtils.clamp(HB + HA*Math.sin(t*hueW + 0.00), 0, 1);
    uniforms.uFilmNm.value       = THREE.MathUtils.clamp(FB + FA*Math.sin(t*filmW + 0.65), 100, 800);
    uniforms.uEdgeGlow.value     = THREE.MathUtils.clamp(EB + EA*Math.sin(t*edgeW + 1.30), 0.0, 2.0);
    uniforms.uReflectivity.value = THREE.MathUtils.clamp(RB + RA*Math.sin(t*reflW + 2.10), 0.0, 1.0);
  } else {
    uniforms.uHueShift.value     = HB;
    uniforms.uFilmNm.value       = FB;
    uniforms.uEdgeGlow.value     = EB;
    uniforms.uReflectivity.value = RB;
  }
}

// ---------- Folding solver (angle + frames) ----------
function computeAngles(tSec){
  const E = Ease[drive.easing] || Ease.linear;
  const g = drive.globalSpeed;
  const animT = drive.play ? (0.5 + 0.5*Math.sin(tSec * g)) : drive.progress;

  for (let i=0;i<base.count;i++){
    eff.ang[i] = base.sign[i] * base.amp[i] * E(clamp01(animT));
    eff.mCount[i] = base.mCount[i];
    for (let m=0;m<MAX_MASKS_PER;m++){
      eff.mA[i][m].copy(base.mA[i][m]);
      eff.mD[i][m].copy(base.mD[i][m]);
    }
  }
  for (let i=base.count;i<MAX_CREASES;i++){ eff.ang[i]=0; eff.mCount[i]=0; }
}

function computeFrames(){
  // start from base frames
  for (let i=0;i<base.count;i++){
    eff.A[i].copy(base.A[i]); eff.D[i].copy(base.D[i]).normalize();
  }
  // sequential propagation of earlier folds
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

function pushAll(){
  pushToUniforms();
}

// ---------- Start ----------
presetSel.value = 'half-vertical-valley';
btnApply.click();
progress.value = String(drive.progress);

// ---------- Frame loop ----------
function tick(tMs){
  const t = tMs * 0.001;
  uniforms.uTime.value = t;

  // Fold kinematics
  computeAngles(t);
  computeFrames();
  pushAll();

  // Surface animation
  updateShaderAnim(t);

  // Update reflections (hide the sheet while capturing)
  sheet.visible = false;
  cubeCam.position.copy(sheet.position);
  cubeCam.update(renderer, scene);
  sheet.visible = true;

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
  fxaa.material.uniforms.resolution.value.set(1 / w, 1 / h);
});
