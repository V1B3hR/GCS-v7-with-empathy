import * as THREE from 'three';

/**
 * Procedural High-Detail Neuro-Anatomical Cortex Surface
 * Recreates anatomical gyri and sulci convolutions based on neuro-imaging pial surfaces.
 */
export function createAnatomicalCortexGeometry(hemisphere = 'LH') {
  // High-resolution sphere base for smooth sulcal sculpting
  const geometry = new THREE.SphereGeometry(1.35, 64, 48);
  const pos = geometry.attributes.position;
  const v = new THREE.Vector3();

  const isLeft = hemisphere === 'LH';

  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);

    // Normalize to unit sphere coordinates
    const len = v.length();
    const nx = v.x / len;
    const ny = v.y / len;
    const nz = v.z / len;

    // 1. Anatomical Gross Morphology Scaling
    // Cerebral hemispheres are elongated along Z (anterior-posterior), taller in parietal, tapered at poles
    let sx = nx * 0.72; // Lateral width
    let sy = ny * 0.92; // Dorsal-ventral height
    let sz = nz * 1.32; // Anterior-posterior length

    // Temporal lobe downward/forward protrusion (inferior-lateral)
    if (ny < 0.1 && nz > -0.4 && nz < 0.5) {
      sy -= 0.18 * (1 - Math.abs(nz));
      sx += 0.12 * (1 - Math.abs(nz));
    }

    // Occipital pole narrowing and posterior taper
    if (nz < -0.4) {
      sx *= 0.88 + nz * 0.1;
      sy *= 0.92;
    }

    // Frontal pole rounding
    if (nz > 0.6) {
      sx *= 0.92;
    }

    // 2. Medial Surface Flattening (Longitudinal Interhemispheric Fissure)
    if (isLeft && sx > -0.05) {
      sx = Math.min(-0.02, sx * 0.25);
    } else if (!isLeft && sx < 0.05) {
      sx = Math.max(0.02, sx * 0.25);
    }

    // 3. Multi-Octave Gyri & Sulci Convolutions (Wrinkles)
    // Primary sulci (Sylvian fissure, central sulcus, parieto-occipital)
    const primarySulci = Math.sin(sx * 5.2 + sy * 3.8) * Math.cos(sy * 4.6 + sz * 5.1) * 0.095;
    
    // Secondary gyral folds
    const secondaryGyri = Math.sin(sx * 10.5 + sz * 9.2) * Math.cos(sy * 9.8 + sx * 8.4) * 0.045;

    // Micro-texture / tertiary sulcal fissures
    const tertiarySulci = Math.sin(sz * 20.0 + sy * 18.0) * Math.sin(sx * 22.0) * 0.018;

    const displacement = 1.0 + primarySulci + secondaryGyri + tertiarySulci;

    v.x = sx * displacement;
    v.y = sy * displacement;
    v.z = sz * displacement;

    pos.setXYZ(i, v.x, v.y, v.z);
  }

  geometry.computeVertexNormals();
  return geometry;
}

/**
 * Cerebellum with Horizontal Folia Striations
 */
export function createCerebellumGeometry() {
  const geometry = new THREE.SphereGeometry(0.68, 36, 28);
  const pos = geometry.attributes.position;
  const v = new THREE.Vector3();

  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    // Flatten dorso-ventrally, expand horizontally
    v.x *= 1.25;
    v.y *= 0.75;
    v.z *= 0.90;

    // Horizontal folia grooves (dense transverse banding)
    const folia = Math.sin(v.y * 36.0) * 0.035;
    v.multiplyScalar(1.0 + folia);

    pos.setXYZ(i, v.x, v.y, v.z);
  }

  geometry.computeVertexNormals();
  return geometry;
}

/**
 * Generates White-Matter Synaptic Point Cloud (Action Potentials)
 */
export function generateNeuralPointCloud(count = 1800) {
  const positions = new Float32Array(count * 3);
  const phases = new Float32Array(count);
  const colors = new Float32Array(count * 3);

  const c1 = new THREE.Color('#38BDF8'); // Cyan
  const c2 = new THREE.Color('#818CF8'); // Indigo
  const c3 = new THREE.Color('#10B981'); // Emerald Green

  for (let i = 0; i < count; i++) {
    // Generate scattered points inside cortical volume
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(Math.random() * 2 - 1);
    const r = 0.6 + Math.random() * 0.75;

    let x = r * Math.sin(phi) * Math.cos(theta) * 0.95;
    let y = r * Math.sin(phi) * Math.sin(theta) * 0.85;
    let z = r * Math.cos(phi) * 1.25;

    // Separate hemispheres slightly
    if (x < 0) x -= 0.15;
    else x += 0.15;

    positions[i * 3] = x;
    positions[i * 3 + 1] = y + 0.2;
    positions[i * 3 + 2] = z - 0.1;

    phases[i] = Math.random() * Math.PI * 2;

    // Assign color
    const randCol = Math.random();
    const chosenColor = randCol < 0.4 ? c1 : randCol < 0.75 ? c2 : c3;
    colors[i * 3] = chosenColor.r;
    colors[i * 3 + 1] = chosenColor.g;
    colors[i * 3 + 2] = chosenColor.b;
  }

  return { positions, phases, colors };
}
