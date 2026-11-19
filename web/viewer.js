import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/OBJLoader.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

// KAMERA DIPERBAIKI: Mundur agar Monas terlihat utuh (x, y, z)
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
camera.position.set(0, 15, 40); 

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(512, 512);
renderer.outputColorSpace = THREE.SRGBColorSpace; // Colour management (sRGB)
document.getElementById('canvas-container').appendChild(renderer.domElement);

// ORBIT CONTROLS: Arahkan ke tengah Monas
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 10, 0);
controls.update();

// LIGHTING: Satu Directional Light
const light = new THREE.DirectionalLight(0xffffff, 2.5);
light.position.set(10, 20, 10);
scene.add(light);
scene.add(new THREE.AmbientLight(0x404040, 2.0));

// LOAD MODEL & TEXTURE
const loader = new OBJLoader();
loader.load('../data/mymodel.obj', function (object) {
    // Parser PPM manual untuk tekstur Monas
    fetch('../data/monas_tex.ppm')
        .then(response => response.arrayBuffer())
        .then(buffer => {
            const data = new Uint8Array(buffer);
            // Parser PPM Sederhana (Asumsi format P6)
            let headerEnd = 0, tokens = 0;
            for (let i = 0; i < data.length; i++) {
                if (data[i] === 10 || data[i] === 32) {
                    if (data[i-1] !== 10 && data[i-1] !== 32) tokens++;
                    if (tokens === 4) { headerEnd = i + 1; break; }
                }
            }
            const pixelData = data.slice(headerEnd);
            const texture = new THREE.DataTexture(pixelData, 256, 256, THREE.RGBAFormat);
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.needsUpdate = true;
            
            const material = new THREE.MeshStandardMaterial({ map: texture });
            object.traverse((child) => {
                if (child.isMesh) child.material = material;
            });
            scene.add(object);
        });
});

// Tombol Export PPM
document.getElementById('exportBtn').addEventListener('click', () => {
    renderer.render(scene, camera);
    const gl = renderer.getContext();
    const width = renderer.domElement.width;
    const height = renderer.domElement.height;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    // Konversi dari RGBA ke RGB (P6)
    let ppmStr = `P6\n${width} ${height}\n255\n`;
    const ppmData = new Uint8Array(ppmStr.length + width * height * 3);
    for (let i = 0; i < ppmStr.length; i++) ppmData[i] = ppmStr.charCodeAt(i);
    
    let offset = ppmStr.length;
    for (let i = 0; i < pixels.length; i += 4) {
        ppmData[offset++] = pixels[i];     // R
        ppmData[offset++] = pixels[i + 1]; // G
        ppmData[offset++] = pixels[i + 2]; // B
    }

    const blob = new Blob([ppmData], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'monas_three.ppm';
    a.click();
    URL.revokeObjectURL(url);
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();