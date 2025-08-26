/**
 *   Origata v0.23
 *  - Texture types: Psychedelic · Perlin · Fractal (fBm) · Ridged
 *  - Texture Scale slider (controls spatial frequency)
 * References for algorithms (concepts): Perlin noise and improved fade; fBm & ridged fractals; canonical GLSL noise patterns. See README notes. 
 * (Perlin 2002; Quilez on fBm/ridged; Ashima GLSL-noise.) 
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { FXAAShader } from 'three/addons/shaders/FXAAShader.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

/* ---------- Renderer / Scene ---------- */
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

/* ---------- Post ---------- */
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.35, 0.6, 0.2);
composer.addPass(bloom);
const fxaa = new ShaderPass(FXAAShader);
fxaa.material.uniforms.resolution.value.set(1 / window.innerWidth, 1 / window.innerHeight);
composer.addPass(fxaa);
composer.addPass(new OutputPass());

/* ---------- Background & Paper ---------- */
const SIZE = 3.0;
const SEG = 160;
const sheetGeo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
sheetGeo.rotateX(-0.25);

const background = new THREE.Mesh(
  new THREE.SphereGeometry(50, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0x070711, side: THREE.BackSide })
);
scene.add(background);

/* ---------- Dynamic Reflection (CubeCamera) ---------- */
const cubeRT = new THREE.WebGLCubeRenderTarget(256, { generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter });
const cubeCam = new THREE.CubeCamera(0.1, 200, cubeRT);
scene.add(cubeCam);

/* ---------- Math helpers ---------- */
const tmp = { v: new THREE.Vector3() };
function sdLine2(p, a, d){ const px=p.x-a.x, py=p.y-a.y; return d.x*py - d.y*px; }
function rotPointAroundAxis(p, a, axisUnit, ang){ tmp.v.copy(p).sub(a).applyAxisAngle(axisUnit, ang).add(a); p.copy(tmp.v); }
function rotVecAxis(v, axisUnit, ang){ v.applyAxisAngle(axisUnit, ang); }
function clamp(x, lo, hi){ return x<lo?lo:x>hi?hi:x; }
function clamp01(x){ return x<0?0:x>1?1:x; }

/* ---------- Minimal crease engine ---------- */
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

const drive = { progress:0.65, globalSpeed:1.0 };

/* ---------- Uniforms ---------- */
const uniforms = {
  uTime:          { value: 0 },
  uSectors:       { value: 10.0 },

  // texture/pattern selection
  uTexMode:       { value: 0 },     // 0: psychedelic, 1: perlin, 2: fbm, 3: ridged
  uTexAmt:        { value: 0.50 },  // Amount (variability/contrast)
  uTexSpeed:      { value: 0.60 },  // Speed multiplier for texture animation
  uTexScale:      { value: 1.00 },  // NEW: spatial scale (frequency)

  // color / paper optics
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

  const flatA=[], flatD=[], on=[];
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

  mat.uniformsNeedUpdate = true;
}

/* ---------- Shaders ---------- */
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

    // sequential hinge rotations (valley/mountain)
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
  uniform float uSectors;
  uniform int   uTexMode;
  uniform float uTexAmt;
  uniform float uTexSpeed;
  uniform float uTexScale;

  uniform float uHueShift;
  uniform float uIridescence, uFilmIOR, uFilmNm, uFiber, uEdgeGlow;

  uniform samplerCube uEnvMap;
  uniform float uReflectivity;
  uniform float uSpecIntensity, uSpecPower, uRimIntensity;
  uniform vec3  uLightDir;

  uniform int   uCreaseCount;
  uniform vec3  uAeff[MAX_CREASES];
  uniform vec3  uDeff[MAX_CREASES];

  varying vec3 vPos; varying vec3 vN; varying vec3 vLocal; varying vec2 vUv;

  #define PI 3.14159265359

  /* --- utility noise (hash/value) used by fibers, etc. */
  float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
  float noiseVal(vec2 p){
    vec2 i=floor(p), f=fract(p);
    float a=hash(i), b=hash(i+vec2(1,0)), c=hash(i+vec2(0,1)), d=hash(i+vec2(1,1));
    vec2 u=f*f*(3.0-2.0*f);
    return mix(a,b,u.x)+ (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
  }
  float fbmVal(vec2 p){
    float v=0.0, a=0.5;
    for(int i=0;i<5;i++){ v+=a*noiseVal(p); p*=2.0; a*=0.5; }
    return v;
  }

  /* --- Perlin-style gradient noise (2D), with improved fade */
  vec2 grad2(vec2 p){
    float a = 6.2831853 * hash(p);  // random angle
    return vec2(cos(a), sin(a));
  }
  float fade(float t){ return t*t*t*(t*(t*6.0-15.0)+10.0); } // Perlin's 6t^5-15t^4-10t^3

  float perlin(vec2 p){
    vec2 i=floor(p), f=fract(p);
    vec2 g00=grad2(i+vec2(0,0));
    vec2 g10=grad2(i+vec2(1,0));
    vec2 g01=grad2(i+vec2(0,1));
    vec2 g11=grad2(i+vec2(1,1));
    float n00=dot(g00, f-vec2(0,0));
    float n10=dot(g10, f-vec2(1,0));
    float n01=dot(g01, f-vec2(0,1));
    float n11=dot(g11, f-vec2(1,1));
    vec2 u = vec2(fade(f.x), fade(f.y));
    return mix(mix(n00,n10,u.x), mix(n01,n11,u.x), u.y); // approx [-1,1]
  }
  float fbmPerlin(vec2 p){
    float sum=0.0, amp=0.5, freq=1.0;
    for(int i=0;i<6;i++){
      sum += amp * perlin(p*freq);
      freq*=2.0; amp*=0.5;
    }
    return sum; // range ~[-1,1]
  }
  float ridged(vec2 p){
    float sum=0.0, amp=0.5, freq=1.0;
    for(int i=0;i<6;i++){
      float n = perlin(p*freq);
      n = 1.0 - abs(n); // ridges
      n *= n;
      sum += n * amp;
      freq*=2.0; amp*=0.5;
    }
    return sum; // ~[0,1]
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

  /* --- Texture generators --- */
  vec3 texPsychedelic(vec3 worldPos, float t){
    float theta = atan(worldPos.z, worldPos.x);
    float r = length(worldPos.xz) * 0.55 * max(0.001, uTexScale);
    float seg = 2.0*PI / max(3.0, uSectors);
    float aa = mod(theta, seg); aa = abs(aa - 0.5*seg);
    vec2 k = vec2(cos(aa), sin(aa)) * r;

    vec2 q = k*2.0 + vec2(0.18*t, -0.12*t);
    q += 0.5*vec2(noiseVal(q+13.1), noiseVal(q+71.7));
    float n = noiseVal(q*2.0) * 0.75 + 0.25*noiseVal(q*5.0);
    float hue = fract(n + 0.15*sin(t*0.3) + uHueShift);
    float contrast = mix(1.0, 1.8, clamp(uTexAmt,0.0,1.0));
    vec3 baseCol = hsv2rgb(vec3(hue, 0.9, smoothstep(0.25, 1.0, n)));
    return pow(baseCol, vec3(contrast));
  }

  vec3 texPerlin(vec2 p, float t){
    p = p * max(0.001, uTexScale) + vec2(0.17*t, -0.11*t);
    float n = perlin(p);                         // [-1,1]
    float k = pow(0.5*(n+1.0), mix(1.0, 3.0, clamp(uTexAmt,0.0,1.0))); // [0,1]
    vec3 a = hsv2rgb(vec3(fract(uHueShift), 0.85, 0.9));
    vec3 b = hsv2rgb(vec3(fract(uHueShift + 0.25), 0.9, 0.95));
    return mix(a, b, k);
  }

  vec3 texFBM(vec2 p, float t){
    p = p * max(0.001, uTexScale) + vec2(0.15*t, -0.09*t);
    float f = fbmPerlin(p);                      // ~[-1,1]
    float k = pow(0.5*(f+1.0), mix(1.0, 3.0, clamp(uTexAmt,0.0,1.0)));
    vec3 a = hsv2rgb(vec3(fract(uHueShift + 0.05), 0.8, 0.92));
    vec3 b = hsv2rgb(vec3(fract(uHueShift + 0.35), 0.9, 0.95));
    return mix(a, b, k);
  }

  vec3 texRidged(vec2 p, float t){
    p = p * max(0.001, uTexScale) + vec2(0.12*t, 0.08*t);
    float r = ridged(p);                         // ~[0,1]
    float k = pow(clamp(r,0.0,1.0), mix(1.0, 2.5, clamp(uTexAmt,0.0,1.0)));
    vec3 a = hsv2rgb(vec3(fract(uHueShift + 0.10), 0.85, 0.9));
    vec3 b = hsv2rgb(vec3(fract(uHueShift + 0.55), 0.9, 0.95));
    return mix(a, b, k);
  }

  void main(){
    float tTex = uTime * uTexSpeed;

    // choose base color by texture mode
    vec3 baseCol;
    if (uTexMode == 0){
      baseCol = texPsychedelic(vPos, tTex);
    } else if (uTexMode == 1){
      baseCol = texPerlin(vLocal.xy, tTex);
    } else if (uTexMode == 2){
      baseCol = texFBM(vLocal.xy, tTex);
    } else {
      baseCol = texRidged(vLocal.xy, tTex);
    }

    // paper fibers (uses simple value-noise fbm for fine grain)
    float fiberLines = 0.0;
    {
      float warp = fbmVal(vLocal.xy*4.0 + vec2(0.2*uTime, -0.1*uTime));
      float l = sin(vLocal.y*420.0 + warp*8.0);
      float widthAA = fwidth(l);
      fiberLines = smoothstep(0.6, 0.6 - widthAA, abs(l));
    }
    float grain = fbmVal(vLocal.xy*25.0);
    baseCol *= 1.0 + uFiber*(0.06*grain - 0.03) + uFiber*0.08*fiberLines;

    // nearest crease (for glow)
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

    // view/lighting
    vec3 V = normalize(cameraPosition - vPos);
    vec3 N = normalize(vN);

    // thin-film + Fresnel blend
    float cosT = clamp(dot(N, V), 0.0, 1.0);
    vec3 film = thinFilm(cosT, uFilmIOR, uFilmNm);
    float F = pow(1.0 - cosT, 5.0);
    vec3 col = mix(baseCol, mix(baseCol, film, uIridescence), F);

    // specular + rim
    vec3 L = normalize(uLightDir);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), uSpecPower) * uSpecIntensity;
    float rim  = pow(1.0 - max(dot(N, V), 0.0), 2.0) * uRimIntensity;
    col += spec + rim;

    // dynamic reflection
    vec3 R = reflect(-V, N);
    vec3 env = textureCube(uEnvMap, R).rgb;
    col = mix(col, env, clamp(uReflectivity, 0.0, 1.0));

    // crease glow tint
    col += uEdgeGlow * edge * film * 0.6;

    // vignette
    float vign = smoothstep(1.2, 0.2, length(vUv-0.5)*1.2);
    gl_FragColor = vec4(col*vign, 1.0);
  }
`;

/* ---------- Material + Mesh ---------- */
const mat = new THREE.ShaderMaterial({
  vertexShader: vs, fragmentShader: fs, uniforms,
  side: THREE.DoubleSide, extensions: { derivatives: true }
});
const sheet = new THREE.Mesh(sheetGeo, mat);
scene.add(sheet);

/* ---------- Presets ---------- */
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

/* ---------- DOM ---------- */
const presetSel   = document.getElementById('preset');
const btnSnap     = document.getElementById('btnSnap');
const autoFx      = document.getElementById('autoFx');

const progress    = document.getElementById('progress');
const progressOut = document.getElementById('progressOut');

const globalSpeed = document.getElementById('globalSpeed');
const globalSpeedOut = document.getElementById('globalSpeedOut');

const sectors     = document.getElementById('sectors');
const sectorsOut  = document.getElementById('sectorsOut');

const btnMore     = document.getElementById('btnMore');
const drawer      = document.getElementById('drawer');

/* Texture UI */
const texMode     = document.getElementById('texMode');
const texAmp      = document.getElementById('texAmp');
const texAmpOut   = document.getElementById('texAmpOut');
const texSpeed    = document.getElementById('texSpeed');
const texSpeedOut = document.getElementById('texSpeedOut');
const texScale    = document.getElementById('texScale');
const texScaleOut = document.getElementById('texScaleOut');

/* Parameter wiring: base constants + (Amount, Speed) sliders */
const el = id => document.getElementById(id);
function out(id){ return document.getElementById(id); }

const PARAMS = {
  tex:  { base: 0.50, min:0,   max:1,   phase:1.10, amp: texAmp,               aOut: texAmpOut,               spd: texSpeed,               sOut: texSpeedOut,               set:v=>{ uniforms.uTexAmt.value=v; } },
  hue:  { base: 0.05, min:0,   max:1,   phase:0.00, amp: el('hueAmp'),         aOut: out('hueAmpOut'),        spd: el('hueSpeed'),         sOut: out('hueSpeedOut'),        set:v=>{ uniforms.uHueShift.value=v; } },
  film: { base:360.0, min:100, max:800, phase:0.65, amp: el('filmAmp'),        aOut: out('filmAmpOut'),       spd: el('filmSpeed'),        sOut: out('filmSpeedOut'),       set:v=>{ uniforms.uFilmNm.value=v; } },
  edge: { base: 0.70, min:0,   max:2,   phase:1.30, amp: el('edgeAmp'),        aOut: out('edgeAmpOut'),       spd: el('edgeSpeed'),        sOut: out('edgeSpeedOut'),       set:v=>{ uniforms.uEdgeGlow.value=v; } },
  refl: { base: 0.25, min:0,   max:1,   phase:2.10, amp: el('reflAmp'),        aOut: out('reflAmpOut'),       spd: el('reflSpeed'),        sOut: out('reflSpeedOut'),       set:v=>{ uniforms.uReflectivity.value=v; } },
  spec: { base: 0.70, min:0,   max:2,   phase:0.25, amp: el('specAmp'),        aOut: out('specAmpOut'),       spd: el('specSpeed'),        sOut: out('specSpeedOut'),       set:v=>{ uniforms.uSpecIntensity.value=v; } },
  rim:  { base: 0.50, min:0,   max:2,   phase:0.85, amp: el('rimAmp'),         aOut: out('rimAmpOut'),        spd: el('rimSpeed'),         sOut: out('rimSpeedOut'),        set:v=>{ uniforms.uRimIntensity.value=v; } },
  bstr: { base: 0.35, min:0,   max:2.5, phase:1.75, amp: el('bloomAmp'),       aOut: out('bloomAmpOut'),      spd: el('bloomSpeed'),       sOut: out('bloomSpeedOut'),      set:v=>{ bloom.strength=v; } },
  brad: { base: 0.60, min:0,   max:1.5, phase:2.50, amp: el('bloomRadAmp'),    aOut: out('bloomRadAmpOut'),   spd: el('bloomRadSpeed'),    sOut: out('bloomRadSpeedOut'),   set:v=>{ bloom.radius=v; } },
};

function setOut(el,val,dec=2){ el.textContent = Number(val).toFixed(dec); }
function bindPair(p, decA=2, decS=2){
  const syncA=()=>setOut(p.aOut, p.amp.value, decA);
  const syncS=()=>setOut(p.sOut, p.spd.value, decS);
  p.amp.addEventListener('input', ()=>{ syncA(); if(!autoFx.checked) p.set(clamp(p.base + +p.amp.value, p.min, p.max)); });
  p.spd.addEventListener('input', syncS);
  syncA(); syncS();
}
Object.values(PARAMS).forEach(bindPair);

function bindRangeWithOut(rangeEl, outEl, decimals=2, onInput){
  const sync=()=>{ outEl.textContent = Number(rangeEl.value).toFixed(decimals); if(onInput) onInput(parseFloat(rangeEl.value)); };
  rangeEl.addEventListener('input', sync); sync();
}

bindRangeWithOut(progress,    progressOut,    3, v => { drive.progress = v; });
bindRangeWithOut(globalSpeed, globalSpeedOut, 2, v => { drive.globalSpeed = v; });
bindRangeWithOut(sectors,     sectorsOut,     0, v => { uniforms.uSectors.value = v; });
bindRangeWithOut(texScale,    texScaleOut,    2, v => { uniforms.uTexScale.value = v; });

/* Toolbar actions */
btnMore.addEventListener('click', () => {
  drawer.open = !drawer.open;
  btnMore.setAttribute('aria-expanded', drawer.open ? 'true' : 'false');
});

btnSnap.addEventListener('click', () => {
  renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'origami.png'; a.click();
    URL.revokeObjectURL(url);
  }, 'image/png', 1.0);
});

presetSel.addEventListener('change', () => {
  const v = presetSel.value;
  if (v==='half-vertical-valley') preset_half_vertical_valley();
  else if (v==='diagonal-valley') preset_diagonal_valley();
  else if (v==='gate-valley') preset_gate_valley();
  camera.position.x += (Math.random()-0.5)*0.02;
  camera.position.y += (Math.random()-0.5)*0.02;
});

/* Texture selects */
texMode.addEventListener('change', () => {
  uniforms.uTexMode.value = parseInt(texMode.value, 10) | 0;
});

/* ---------- Animation model: base + amount*(sin(...) or 1) ---------- */
function updateAnimatedParameters(t){
  const g = drive.globalSpeed;
  const animOn = autoFx.checked;

  function apply(p){
    const amp = +p.amp.value;
    const spd = +p.spd.value;
    const val = animOn
      ? p.base + amp * Math.sin(g * spd * t + p.phase)
      : p.base + amp;
    p.set(clamp(val, p.min, p.max));
  }

  apply(PARAMS.tex);
  apply(PARAMS.hue);
  apply(PARAMS.film);
  apply(PARAMS.edge);
  apply(PARAMS.refl);
  apply(PARAMS.spec);
  apply(PARAMS.rim);
  apply(PARAMS.bstr);
  apply(PARAMS.brad);

  // Texture flow speed (global multiplier)
  uniforms.uTexSpeed.value = g * (+PARAMS.tex.spd.value);
}

/* ---------- Folding ---------- */
function computeAngles(){
  const t = clamp01(drive.progress);
  for (let i=0;i<base.count;i++){
    eff.ang[i] = base.sign[i] * base.amp[i] * t;
    eff.mCount[i] = base.mCount[i];
    for (let m=0;m<MAX_MASKS_PER;m++){
      eff.mA[i][m].copy(base.mA[i][m]);
      eff.mD[i][m].copy(base.mD[i][m]);
    }
  }
  for (let i=base.count;i<MAX_CREASES;i++){ eff.ang[i]=0; eff.mCount[i]=0; }
}
function computeFrames(){
  for (let i=0;i<base.count;i++){
    eff.A[i].copy(base.A[i]); eff.D[i].copy(base.D[i]).normalize();
  }
  // propagate earlier folds onto later crease frames
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
function pushAll(){ pushToUniforms(); }

/* ---------- Start ---------- */
presetSel.value = 'half-vertical-valley';
preset_half_vertical_valley(); // initialize crease set
uniforms.uTexMode.value = parseInt(texMode.value, 10) | 0;

/* ---------- Frame loop ---------- */
function tick(tMs){
  const t = (tMs * 0.001);
  uniforms.uTime.value = t;

  computeAngles();
  computeFrames();
  pushAll();

  updateAnimatedParameters(t);

  // reflections (hide sheet during capture to avoid self-reflection)
  sheet.visible = false;
  cubeCam.position.copy(sheet.position);
  cubeCam.update(renderer, scene);
  sheet.visible = true;

  controls.update();
  composer.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

/* ---------- Resize ---------- */
window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.setSize(w, h); composer.setSize(w, h);
  fxaa.material.uniforms.resolution.value.set(1 / w, 1 / h);
});
