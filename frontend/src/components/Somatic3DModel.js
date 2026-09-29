import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Cylinder, Torus } from '@react-three/drei';
import * as THREE from 'three';
import { getStatusColor } from '../constants/brainRegions';

/**
 * High-Detail 3D Holographic Somatic Mannequin with Full Anatomical Skeleton & Organs
 */
export function Somatic3DModel({
  organStates,
  selectedOrgan,
  onSelectOrgan,
  onContextMenuOrgan
}) {
  const groupRef = useRef();
  const heartMeshRef = useRef();
  const lungsMeshRef = useRef();

  useFrame(({ clock }) => {
    const time = clock.getElapsedTime();
    if (groupRef.current) {
      // Gentle floating levitation
      groupRef.current.position.y = Math.sin(time * 1.2) * 0.03 - 0.15;
    }
    // Realistic rhythmic heartbeat expansion
    if (heartMeshRef.current) {
      const beat = 1.0 + Math.pow(Math.sin(time * 4.5), 10) * 0.15;
      heartMeshRef.current.scale.set(beat, beat * 1.1, beat);
    }
    // Respiratory lung expansion
    if (lungsMeshRef.current) {
      const resp = 1.0 + Math.sin(time * 1.5) * 0.08;
      lungsMeshRef.current.scale.set(resp, resp, resp);
    }
  });

  const getOrganMaterial = (key, baseColor = '#10B981') => {
    const organ = organStates[key] || { strength: 50 };
    const status = getStatusColor(organ.strength);
    const isSelected = selectedOrgan?.id === key;

    return (
      <meshStandardMaterial
        color={status.hex}
        emissive={status.hex}
        emissiveIntensity={isSelected ? 1.2 : 0.45}
        roughness={0.25}
        metalness={0.35}
        transparent
        opacity={0.88}
      />
    );
  };

  return (
    <group ref={groupRef}>
      {/* Lights */}
      <ambientLight intensity={0.75} />
      <directionalLight position={[5, 10, 5]} intensity={1.1} />
      <directionalLight position={[-5, -5, -5]} intensity={0.5} color="#38BDF8" />
      <pointLight position={[0, 0.9, 0.5]} intensity={0.8} color="#FF5722" />

      {/* ========================================================
          1. ANATOMICAL SKELETAL WIREFRAME MATRIX
          ======================================================== */}
      <group>
        {/* Skull with Cranial Cavity */}
        <mesh position={[0, 1.62, 0]}>
          <sphereGeometry args={[0.34, 24, 24]} />
          <meshStandardMaterial color="#38BDF8" transparent opacity={0.14} wireframe />
        </mesh>
        {/* Mandible / Jaw */}
        <mesh position={[0, 1.42, 0.08]} scale={[0.8, 0.6, 0.9]}>
          <cylinderGeometry args={[0.18, 0.14, 0.18, 16]} />
          <meshStandardMaterial color="#64748B" transparent opacity={0.18} wireframe />
        </mesh>

        {/* Cervical & Thoracic Spine (Vertebral Column) */}
        <group position={[0, 0.75, -0.06]}>
          <cylinderGeometry args={[0.04, 0.05, 1.5, 16]} />
          <meshStandardMaterial color="#38BDF8" transparent opacity={0.35} emissive="#38BDF8" emissiveIntensity={0.2} />
        </group>

        {/* Clavicles (Collarbones) */}
        <mesh position={[0, 1.28, 0]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.025, 0.025, 0.85, 12]} />
          <meshStandardMaterial color="#94A3B8" transparent opacity={0.3} wireframe />
        </mesh>

        {/* Ribcage Structure (Individual Thoracic Rib Rings) */}
        <group position={[0, 0.95, 0]}>
          {[0.2, 0.1, 0.0, -0.1, -0.2, -0.3].map((yOff, i) => (
            <mesh key={i} position={[0, yOff, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[1.0 - i * 0.04, 0.75, 1.0]}>
              <torusGeometry args={[0.34 - i * 0.015, 0.018, 8, 24]} />
              <meshStandardMaterial color="#38BDF8" transparent opacity={0.16} wireframe />
            </mesh>
          ))}
          {/* Sternum / Breastbone */}
          <mesh position={[0, -0.05, 0.28]}>
            <cylinderGeometry args={[0.035, 0.02, 0.45, 8]} />
            <meshStandardMaterial color="#94A3B8" transparent opacity={0.3} />
          </mesh>
        </group>

        {/* Pelvic Girdle */}
        <group position={[0, 0.1, -0.02]} rotation={[Math.PI / 2, 0, 0]} scale={[1.1, 0.8, 0.6]}>
          <torusGeometry args={[0.32, 0.06, 12, 24]} />
          <meshStandardMaterial color="#38BDF8" transparent opacity={0.18} wireframe />
        </group>

        {/* Upper Limb Bones (Humerus, Radius/Ulna) */}
        <mesh position={[-0.55, 0.85, 0]} rotation={[0, 0, 0.25]}>
          <cylinderGeometry args={[0.035, 0.03, 0.65, 10]} />
          <meshStandardMaterial color="#64748B" transparent opacity={0.18} wireframe />
        </mesh>
        <mesh position={[0.55, 0.85, 0]} rotation={[0, 0, -0.25]}>
          <cylinderGeometry args={[0.035, 0.03, 0.65, 10]} />
          <meshStandardMaterial color="#64748B" transparent opacity={0.18} wireframe />
        </mesh>

        {/* Lower Limb Bones (Femur, Tibia) */}
        <mesh position={[-0.22, -0.65, 0]}>
          <cylinderGeometry args={[0.05, 0.04, 1.45, 12]} />
          <meshStandardMaterial color="#64748B" transparent opacity={0.18} wireframe />
        </mesh>
        <mesh position={[0.22, -0.65, 0]}>
          <cylinderGeometry args={[0.05, 0.04, 1.45, 12]} />
          <meshStandardMaterial color="#64748B" transparent opacity={0.18} wireframe />
        </mesh>

        {/* Major Vascular Pathways (Aorta Trunk & Iliac Branches) */}
        <group position={[0, 0.65, 0.02]}>
          <cylinderGeometry args={[0.02, 0.016, 1.0, 10]} />
          <meshStandardMaterial color="#EF4444" transparent opacity={0.4} emissive="#EF4444" emissiveIntensity={0.3} />
        </group>
      </group>

      {/* ========================================================
          2. DETAILED ANATOMICAL ORGANS (INTERACTIVE)
          ======================================================== */}

      {/* 1. BRAIN (Cerebral Cortex + Brainstem) */}
      <group
        position={[0, 1.64, 0]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectOrgan('brain');
        }}
        onContextMenu={(e) => {
          e.stopPropagation();
          if (e.nativeEvent) e.nativeEvent.preventDefault();
          onContextMenuOrgan('brain', e);
        }}
      >
        {/* Left & Right Hemisphere */}
        <mesh position={[-0.08, 0, 0]} scale={[0.65, 0.85, 1.1]}>
          <sphereGeometry args={[0.18, 18, 18]} />
          {getOrganMaterial('brain')}
        </mesh>
        <mesh position={[0.08, 0, 0]} scale={[0.65, 0.85, 1.1]}>
          <sphereGeometry args={[0.18, 18, 18]} />
          {getOrganMaterial('brain')}
        </mesh>
        {/* Cerebellum */}
        <mesh position={[0, -0.12, -0.1]} scale={[1.1, 0.6, 0.8]}>
          <sphereGeometry args={[0.11, 14, 14]} />
          {getOrganMaterial('brain')}
        </mesh>
      </group>

      {/* 2. HEART (Aorta Arch + Myocardial Chambers with Beating Animation) */}
      <group
        ref={heartMeshRef}
        position={[0.06, 0.95, 0.1]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectOrgan('heart');
        }}
        onContextMenu={(e) => {
          e.stopPropagation();
          if (e.nativeEvent) e.nativeEvent.preventDefault();
          onContextMenuOrgan('heart', e);
        }}
      >
        {/* Myocardial Ventricular Body */}
        <mesh scale={[1.0, 1.25, 0.9]}>
          <sphereGeometry args={[0.15, 20, 20]} />
          {getOrganMaterial('heart')}
        </mesh>
        {/* Aorta Arch */}
        <mesh position={[-0.03, 0.16, -0.02]} rotation={[0, 0, -Math.PI / 4]}>
          <torusGeometry args={[0.08, 0.025, 8, 16, Math.PI]} />
          <meshStandardMaterial color="#EF4444" emissive="#EF4444" emissiveIntensity={0.6} />
        </mesh>
      </group>

      {/* 3. LUNGS (Trachea + Dual Bronchial Trees with Breathing Animation) */}
      <group
        ref={lungsMeshRef}
        position={[0, 0.96, 0.02]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectOrgan('lungs');
        }}
        onContextMenu={(e) => {
          e.stopPropagation();
          if (e.nativeEvent) e.nativeEvent.preventDefault();
          onContextMenuOrgan('lungs', e);
        }}
      >
        {/* Trachea */}
        <mesh position={[0, 0.26, 0.05]}>
          <cylinderGeometry args={[0.025, 0.025, 0.22, 10]} />
          <meshStandardMaterial color="#94A3B8" transparent opacity={0.5} />
        </mesh>
        {/* Left Lung (2 Lobes) */}
        <mesh position={[-0.22, 0, 0]} scale={[0.75, 1.25, 0.85]}>
          <sphereGeometry args={[0.17, 18, 18]} />
          {getOrganMaterial('lungs')}
        </mesh>
        {/* Right Lung (3 Lobes) */}
        <mesh position={[0.22, 0, 0]} scale={[0.82, 1.25, 0.88]}>
          <sphereGeometry args={[0.17, 18, 18]} />
          {getOrganMaterial('lungs')}
        </mesh>
      </group>

      {/* 4. LIVER (Anatomical Wedge Lobes) */}
      <group
        position={[0.16, 0.62, 0.06]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectOrgan('liver');
        }}
        onContextMenu={(e) => {
          e.stopPropagation();
          if (e.nativeEvent) e.nativeEvent.preventDefault();
          onContextMenuOrgan('liver', e);
        }}
      >
        <mesh scale={[1.4, 0.75, 1.0]} rotation={[0, 0, -0.2]}>
          <sphereGeometry args={[0.15, 16, 16]} />
          {getOrganMaterial('liver')}
        </mesh>
      </group>

      {/* 5. STOMACH & ENTERIC GUT (Fundus & Intestinal Convolutions) */}
      <group
        position={[-0.08, 0.50, 0.08]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectOrgan('gut');
        }}
        onContextMenu={(e) => {
          e.stopPropagation();
          if (e.nativeEvent) e.nativeEvent.preventDefault();
          onContextMenuOrgan('gut', e);
        }}
      >
        {/* J-Shaped Stomach */}
        <mesh scale={[1.2, 0.8, 0.9]} rotation={[0, 0, 0.35]}>
          <sphereGeometry args={[0.14, 16, 16]} />
          {getOrganMaterial('gut')}
        </mesh>
        {/* Intestinal Mesenteric Loops */}
        <mesh position={[0.08, -0.16, 0]} scale={[1.2, 0.7, 0.8]}>
          <sphereGeometry args={[0.16, 16, 16]} />
          {getOrganMaterial('gut')}
        </mesh>
      </group>

      {/* 6. KIDNEYS (Bilateral Bean-Shaped Renal Organs) */}
      <group
        position={[0, 0.48, -0.12]}
        onClick={(e) => {
          e.stopPropagation();
          onSelectOrgan('kidneys');
        }}
        onContextMenu={(e) => {
          e.stopPropagation();
          if (e.nativeEvent) e.nativeEvent.preventDefault();
          onContextMenuOrgan('kidneys', e);
        }}
      >
        <mesh position={[-0.20, 0, 0]} scale={[0.65, 1.15, 0.7]}>
          <sphereGeometry args={[0.1, 14, 14]} />
          {getOrganMaterial('kidneys')}
        </mesh>
        <mesh position={[0.20, 0, 0]} scale={[0.65, 1.15, 0.7]}>
          <sphereGeometry args={[0.1, 14, 14]} />
          {getOrganMaterial('kidneys')}
        </mesh>
      </group>
    </group>
  );
}
