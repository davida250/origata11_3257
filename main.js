// Origata v0.12
//
// three.js + extras (pinned versions for stability)
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.179.1/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.179.1/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'https://cdn.jsdelivr.net/npm/three@0.179.1/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'https://cdn.jsdelivr.net/npm/three@0.179.1/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'https://cdn.jsdelivr.net/npm/three@0.179.1/examples/jsm/postprocessing/UnrealBloomPass.js';
// lightweight UI
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20.0/dist/lil-gui.esm.js';

// ---------- renderer / scene / camera ----------
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x050509, 5, 30);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 1.8, 5.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// ---------- postprocessing (bloom) ----------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.9, 0.55, 0.2);
composer.addPass(bloom);

// ---------- geometry: foldable pleated sheet ----------
const WIDTH = 4.0, HEIGHT = 2.4, STRIPES = 64, SEG_Y = 120;
const geo = new THREE.PlaneGeometry(WIDTH, HEIGHT, STRIPES, SEG_Y);
geo.rotateX(-0.25); // slight tilt for depth

const uniforms = {
  uTime:     { value: 0 },
  uFold:     { value: 0.35 },     // -1..1
  uMaxAngle: { value: 1.2 },      // radians
  uWidth:    { value: WIDTH },
  uStripes:  { value: STRIPES },
  uSectors:  { value: 8.0 },
  uHueShift: { value: 0.0 }
};

const vs = /* glsl */`
  uniform float uFold, uMaxAngle, uWidth, uStripes;
  varying vec3 vPos;
  varying vec2 vUv;
  void main() {
    vUv = uv;
    float w = uWidth / uStripes;
    vec3 p = position;

    // which stripe are we in?
    float s = floor( (p.x + 0.5*uWidth) / w );
    float signFlip = (mod(s, 2.0) < 1.0) ? 1.0 : -1.0;

    // rotate each stripe around its hinge (Y axis through hingeX)
    float angle = signFlip * uFold * uMaxAngle;
    float hingeX = -0.5*uWidth + s * w;

    p.x -= hingeX;
    float c = cos(angle), s2 = sin(angle);
    vec3 r = vec3(c*p.x + s2*p.z, p.y, -s2*p.x + c*p.z);
    r.x += hingeX;

    // tiny "paper softness" bulge (visual only)
    r.z += 0.02 * sin(3.14159*uFold) * sin(3.14159*((position.x/uWidth)+0.5)) * cos(3.14159*position.y);

    vec4 world = modelMatrix * vec4(r, 1.0);
    vPos = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const fs = /* glsl */`
  precision highp float;
  uniform float uTime, uSectors, uHueShift;
  varying vec3 vPos;
  varying vec2 vUv;

  #define PI 3.14159265359

  float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
  float noise(vec2 p){
    vec2 i=floor(p), f=fract(p);
    float a=hash(i), b=hash(i+vec2(1,0)), c=hash(i+vec2(0,1)), d=hash(i+vec2(1,1));
    vec2 u=f*f*(3.0-2.0*f);
    return mix(a,b,u.x)+ (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
  }
  vec3 hsv2rgb(vec3 c){
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.,4.,2.), 6.)-3.)-1., 0., 1.);
    return c.z * mix(vec3(1.0), rgb, c.y);
  }

  void main() {
    // kaleidoscopic mapping in XZ
    float theta = atan(vPos.z, vPos.x);
    float r = length(vPos.xz) * 0.55;
    float seg = 2.0*PI / max(3.0, uSectors);
    float a = mod(theta, seg);
    a = abs(a - 0.5*seg); // mirror per segment
    vec2 k = vec2(cos(a), sin(a)) * r;

    // domain-warped noise
    vec2 q = k*2.0 + vec2(0.15*uTime, -0.1*uTime);
    q += 0.5*vec2(noise(q+13.1), noise(q+71.7));
    float n = noise(q*2.0) * 0.75 + 0.25*noise(q*5.0);

    float hue = fract(n + 0.15*sin(uTime*0.3) + uHueShift);
    float sat = 0.9;
    float val = smoothstep(0.25, 1.0, n);
    vec3 col = hsv2rgb(vec3(hue, sat, val));

    // subtle vignetting
    float vign = smoothstep(1.2, 0.2, length(vUv-0.5)*1.2);
    gl_FragColor = vec4(col*vign, 1.0);
  }
`;

const mat = new THREE.ShaderMaterial({
  vertexShader: vs,
  fragmentShader: fs,
  uniforms,
  side: THREE.DoubleSide
});
const sheet = new THREE.Mesh(geo, mat);
scene.add(sheet);

// ambient background
const bg = new THREE.Mesh(
  new THREE.SphereGeometry(50, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0x070711, side: THREE.BackSide })
);
scene.add(bg);

// ---------- GUI ----------
const gui = new GUI();
gui.add(uniforms.uFold, 'value', -1, 1, 0.001).name('fold');
gui.add(uniforms.uMaxAngle, 'value', 0, Math.PI).name('maxAngle');
gui.add(uniforms.uSectors, 'value', 3, 24, 1).name('kaleidoSectors');
gui.add(uniforms.uHueShift, 'value', 0, 1, 0.001).name('hueShift');
gui.add(bloom, 'strength', 0.0, 2.5, 0.01).name('bloomStrength');
gui.add(bloom, 'radius', 0.0, 1.5, 0.01).name('bloomRadius');

// ---------- resize ----------
function onResize() {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloom.setSize(w, h);
}
window.addEventListener('resize', onResize);

// ---------- render loop ----------
function tick(t) {
  uniforms.uTime.value = t * 0.001;
  controls.update();
  composer.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ---------- Save PNG ----------
document.getElementById('btnSnap').onclick = () => {
  renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'origami.png';
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png', 1.0);
};
