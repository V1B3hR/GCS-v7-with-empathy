import React, { useRef, useMemo, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Cylinder, Line, Html, Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';
import { CORTICAL_REGIONS_68, getStatusColor } from '../constants/brainRegions';
import {
  createAnatomicalCortexGeometry,
  createCerebellumGeometry,
  generateNeuralPointCloud
} from '../utils/brainGeometry';

/**
 * Individual Cortical Region Mesh (Desikan-Killiany Node)
 */
function CorticalRegionMesh({
  region,
  activation,
  isSelected,
  isHovered,
  selectedRegion,
  activeBandFilter,
  explodeOffset,
  onHover,
  onUnhover,
  onSelect,
  onContextMenu
}) {
  const meshRef = useRef();

  const position = useMemo(() => {
    const isLeft = region.hemisphere === 'LH';
    const explodeX = isLeft ? -explodeOffset : explodeOffset;
    return [
      region.position[0] + explodeX,
      region.position[1],
      region.position[2]
    ];
  }, [region, explodeOffset]);

  const statusInfo = useMemo(() => {
    const strength = activation?.strength ?? 50;
    return getStatusColor(strength);
  }, [activation]);

  const isBandMatched = activeBandFilter
    ? region.defaultBand.toLowerCase().includes(activeBandFilter.toLowerCase())
    : true;

  // Distance-based dissolving effect
  const opacity = useMemo(() => {
    if (isSelected) return 1.0;

    if (selectedRegion) {
      const selX = selectedRegion.position[0] + (selectedRegion.hemisphere === 'LH' ? -explodeOffset : explodeOffset);
      const selY = selectedRegion.position[1];
      const selZ = selectedRegion.position[2];

      const dx = position[0] - selX;
      const dy = position[1] - selY;
      const dz = position[2] - selZ;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (dist > 1.6) return 0.04;
      return Math.max(0.08, 0.7 - dist * 0.4);
    }

    const distCenter = Math.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2);
    const baseOpacity = Math.max(0.45, 1.05 - distCenter * 0.22);
    return isBandMatched ? baseOpacity : 0.15;
  }, [isSelected, selectedRegion, position, explodeOffset, isBandMatched]);

  // Harmonic pulsing frequency linked to EEG band
  useFrame(({ clock }) => {
    if (meshRef.current) {
      const time = clock.getElapsedTime();
      let freqMultiplier = 2.5;
      if (region.defaultBand === 'Gamma') freqMultiplier = 6.0;
      else if (region.defaultBand === 'Beta') freqMultiplier = 4.2;
      else if (region.defaultBand === 'Alpha') freqMultiplier = 2.2;
      else if (region.defaultBand === 'Theta') freqMultiplier = 1.4;
      else if (region.defaultBand === 'Delta') freqMultiplier = 0.8;

      const baseScale = isSelected ? 1.4 : isHovered ? 1.25 : 1.0;
      const pulseAmp = isSelected ? 0.08 : isHovered ? 0.05 : 0.03;
      const pulse = Math.sin(time * freqMultiplier + region.nodeId * 0.3) * pulseAmp;
      const scale = baseScale + pulse;
      meshRef.current.scale.set(scale, scale, scale);
    }
  });

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(region);
        }}
        onContextMenu={(e) => {
          e.stopPropagation();
          if (e.nativeEvent) {
            e.nativeEvent.preventDefault();
            e.nativeEvent.stopPropagation();
          }
          if (e.preventDefault) e.preventDefault();
          onContextMenu(region, e);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(region, position);
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          onUnhover();
        }}
      >
        <sphereGeometry args={[0.14, 24, 24]} />
        <meshStandardMaterial
          color={statusInfo.hex}
          emissive={statusInfo.hex}
          emissiveIntensity={isSelected ? 1.1 : isHovered ? 0.85 : 0.38}
          roughness={0.25}
          metalness={0.35}
          transparent
          opacity={opacity}
        />
      </mesh>

      {isSelected && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.20, 0.25, 32]} />
          <meshBasicMaterial color="#ffffff" side={THREE.DoubleSide} transparent opacity={0.9} />
        </mesh>
      )}

      {isBandMatched && activeBandFilter && (
        <mesh rotation={[0, Math.PI / 4, 0]}>
          <ringGeometry args={[0.18, 0.22, 24]} />
          <meshBasicMaterial color="#38BDF8" side={THREE.DoubleSide} transparent opacity={0.6} />
        </mesh>
      )}
    </group>
  );
}

/**
 * Procedural Convoluted Cerebral Cortex with Realistic Gyri & Sulci
 */
function DetailedConvolutedCortex({ explodeOffset, isIsolated }) {
  const leftCortexGeo = useMemo(() => createAnatomicalCortexGeometry('LH'), []);
  const rightCortexGeo = useMemo(() => createAnatomicalCortexGeometry('RH'), []);
  const leftCerebellumGeo = useMemo(() => createCerebellumGeometry(), []);
  const rightCerebellumGeo = useMemo(() => createCerebellumGeometry(), []);

  const cortexOpacity = isIsolated ? 0.05 : 0.18;
  const solidOpacity = isIsolated ? 0.02 : 0.08;

  return (
    <group>
      {/* 1. Left Cerebral Hemisphere (Convoluted Gyri & Sulci) */}
      <group position={[-0.85 - explodeOffset, 0.25, -0.15]}>
        <mesh geometry={leftCortexGeo}>
          <meshStandardMaterial
            color="#38BDF8"
            transparent
            opacity={cortexOpacity}
            roughness={0.35}
            metalness={0.25}
            wireframe={false}
          />
        </mesh>
        <mesh geometry={leftCortexGeo} scale={[1.002, 1.002, 1.002]}>
          <meshBasicMaterial
            color="#38BDF8"
            transparent
            opacity={0.07}
            wireframe
          />
        </mesh>
      </group>

      {/* 2. Right Cerebral Hemisphere (Convoluted Gyri & Sulci) */}
      <group position={[0.85 + explodeOffset, 0.25, -0.15]}>
        <mesh geometry={rightCortexGeo}>
          <meshStandardMaterial
            color="#38BDF8"
            transparent
            opacity={cortexOpacity}
            roughness={0.35}
            metalness={0.25}
            wireframe={false}
          />
        </mesh>
        <mesh geometry={rightCortexGeo} scale={[1.002, 1.002, 1.002]}>
          <meshBasicMaterial
            color="#38BDF8"
            transparent
            opacity={0.07}
            wireframe
          />
        </mesh>
      </group>

      {/* 3. Corpus Callosum Commisure (Interhemispheric Nerve Arch) */}
      <group position={[0, 0.45, -0.1]}>
        <Cylinder args={[0.5, 0.5, 0.4 + explodeOffset * 1.5, 32]} rotation={[0, 0, Math.PI / 2]}>
          <meshStandardMaterial
            color="#818CF8"
            transparent
            opacity={isIsolated ? 0.03 : 0.22}
            emissive="#818CF8"
            emissiveIntensity={0.2}
            roughness={0.3}
          />
        </Cylinder>
      </group>

      {/* 4. Ventricles (Fluid Core) */}
      <group position={[0, 0.25, -0.2]}>
        <Sphere args={[0.42, 20, 20]} scale={[0.55, 0.75, 1.4]}>
          <meshStandardMaterial
            color="#00F0FF"
            transparent
            opacity={isIsolated ? 0.02 : 0.16}
            emissive="#00F0FF"
            emissiveIntensity={0.25}
          />
        </Sphere>
      </group>

      {/* 5. Cerebellum (Dual Lobes with Horizontal Folia) */}
      <group position={[-0.65 - explodeOffset * 0.5, -0.85, -1.3]}>
        <mesh geometry={leftCerebellumGeo}>
          <meshStandardMaterial
            color="#A855F7"
            transparent
            opacity={isIsolated ? 0.04 : 0.25}
            wireframe
          />
        </mesh>
      </group>
      <group position={[0.65 + explodeOffset * 0.5, -0.85, -1.3]}>
        <mesh geometry={rightCerebellumGeo}>
          <meshStandardMaterial
            color="#A855F7"
            transparent
            opacity={isIsolated ? 0.04 : 0.25}
            wireframe
          />
        </mesh>
      </group>

      {/* 6. Brainstem & Pons (Midbrain, Pons, Medulla Oblongata) */}
      <group position={[0, -1.05, -0.55]}>
        {/* Pons bulge */}
        <Sphere args={[0.32, 16, 16]} position={[0, 0.2, 0]} scale={[1.1, 0.9, 1.0]}>
          <meshStandardMaterial color="#6366F1" transparent opacity={isIsolated ? 0.03 : 0.25} roughness={0.4} />
        </Sphere>
        {/* Medulla cylinder */}
        <Cylinder args={[0.22, 0.16, 1.3, 20]} rotation={[0.18, 0, 0]}>
          <meshStandardMaterial color="#6366F1" transparent opacity={isIsolated ? 0.03 : 0.25} roughness={0.4} />
        </Cylinder>
      </group>
    </group>
  );
}

/**
 * Neural Synaptic Action Potential Point Cloud
 */
function NeuralSynapticCloud({ isIsolated, explodeOffset }) {
  const pointsRef = useRef();
  const data = useMemo(() => generateNeuralPointCloud(1600), []);

  useFrame(({ clock }) => {
    if (pointsRef.current) {
      const time = clock.getElapsedTime();
      // Wave oscillation through neural cloud
      const geom = pointsRef.current.geometry;
      const positions = geom.attributes.position.array;
      for (let i = 0; i < data.phases.length; i++) {
        const p = data.phases[i];
        const wave = Math.sin(time * 3.5 + p) * 0.015;
        positions[i * 3 + 1] += wave * 0.1;
      }
      geom.attributes.position.needsUpdate = true;
    }
  });

  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(data.positions), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(data.colors), 3));
    return g;
  }, [data]);

  return (
    <points ref={pointsRef} geometry={geo}>
      <PointMaterial
        size={0.035}
        vertexColors
        transparent
        opacity={isIsolated ? 0.15 : 0.65}
        depthWrite={false}
      />
    </points>
  );
}

/**
 * Synaptic Graph Connectivity Tracks
 */
function SynapticGraphLines({ regions, explodeOffset, showSynapses, selectedRegion }) {
  const linePairs = useMemo(() => {
    const pairs = [];
    for (let i = 0; i < regions.length; i++) {
      for (let j = i + 1; j < regions.length; j++) {
        const r1 = regions[i];
        const r2 = regions[j];
        const sameHemi = r1.hemisphere === r2.hemisphere;
        const dx = r1.position[0] - r2.position[0];
        const dy = r1.position[1] - r2.position[1];
        const dz = r1.position[2] - r2.position[2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if ((sameHemi && dist < 1.1) || (!sameHemi && dist < 0.9)) {
          pairs.push({ from: r1, to: r2, dist });
        }
      }
    }
    return pairs;
  }, [regions]);

  if (!showSynapses) return null;

  return (
    <group>
      {linePairs.map((pair, idx) => {
        const isLeft1 = pair.from.hemisphere === 'LH';
        const isLeft2 = pair.to.hemisphere === 'LH';
        const off1 = isLeft1 ? -explodeOffset : explodeOffset;
        const off2 = isLeft2 ? -explodeOffset : explodeOffset;

        const p1 = [pair.from.position[0] + off1, pair.from.position[1], pair.from.position[2]];
        const p2 = [pair.to.position[0] + off2, pair.to.position[1], pair.to.position[2]];

        const isConnectedToSelected = selectedRegion &&
          (pair.from.nodeId === selectedRegion.nodeId || pair.to.nodeId === selectedRegion.nodeId);

        const opacity = selectedRegion ? (isConnectedToSelected ? 0.75 : 0.02) : 0.16;

        return (
          <Line
            key={idx}
            points={[p1, p2]}
            color={isConnectedToSelected ? '#38BDF8' : '#64748B'}
            lineWidth={isConnectedToSelected ? 1.5 : 0.75}
            transparent
            opacity={opacity}
          />
        );
      })}
    </group>
  );
}

/**
 * Main 3D Brain Model Component
 */
export function Brain3DModel({
  regionActivations,
  selectedRegion,
  activeBandFilter,
  onSelectRegion,
  onContextMenuRegion,
  explodeView,
  showSynapses
}) {
  const [hoveredRegion, setHoveredRegion] = useState(null);
  const [tooltipPos, setTooltipPos] = useState([0, 0, 0]);

  const explodeOffset = explodeView ? 0.65 : 0.0;
  const isIsolated = Boolean(selectedRegion);

  const handleHover = (region, pos) => {
    setHoveredRegion(region);
    setTooltipPos(pos);
  };

  const handleUnhover = () => {
    setHoveredRegion(null);
  };

  return (
    <group>
      {/* Lighting Architecture */}
      <ambientLight intensity={0.75} />
      <directionalLight position={[10, 15, 10]} intensity={1.2} />
      <directionalLight position={[-10, -10, -10]} intensity={0.6} color="#38BDF8" />
      <pointLight position={[0, 4, 0]} intensity={0.8} color="#ffffff" />

      {/* Procedural High-Detail Convoluted Cortex & Cerebellum */}
      <DetailedConvolutedCortex explodeOffset={explodeOffset} isIsolated={isIsolated} />

      {/* Neural Synaptic Point Cloud (Action Potentials) */}
      <NeuralSynapticCloud isIsolated={isIsolated} explodeOffset={explodeOffset} />

      {/* Structural Synaptic Connectivity Lines */}
      <SynapticGraphLines
        regions={CORTICAL_REGIONS_68}
        explodeOffset={explodeOffset}
        showSynapses={showSynapses}
        selectedRegion={selectedRegion}
      />

      {/* 68 Desikan-Killiany Cortical Regions */}
      {CORTICAL_REGIONS_68.map((region) => (
        <CorticalRegionMesh
          key={region.nodeId}
          region={region}
          activation={regionActivations[region.nodeId]}
          isSelected={selectedRegion?.nodeId === region.nodeId}
          isHovered={hoveredRegion?.nodeId === region.nodeId}
          selectedRegion={selectedRegion}
          activeBandFilter={activeBandFilter}
          explodeOffset={explodeOffset}
          onHover={handleHover}
          onUnhover={handleUnhover}
          onSelect={onSelectRegion}
          onContextMenu={onContextMenuRegion}
        />
      ))}

      {/* Hover Tooltip via Drei Html */}
      {hoveredRegion && !selectedRegion && (
        <Html
          position={[tooltipPos[0], tooltipPos[1] + 0.32, tooltipPos[2]]}
          center
          distanceFactor={10}
          zIndexRange={[100, 0]}
        >
          <div
            style={{
              background: 'rgba(12, 17, 26, 0.95)',
              border: '1px solid rgba(56, 189, 248, 0.5)',
              borderRadius: '6px',
              padding: '6px 10px',
              fontFamily: 'Inter, sans-serif',
              fontSize: '11px',
              color: '#ffffff',
              whiteSpace: 'nowrap',
              boxShadow: '0 4px 16px rgba(0,0,0,0.6)',
              pointerEvents: 'none',
              transform: 'translateY(-100%)'
            }}
          >
            <div style={{ fontWeight: 600, color: '#38BDF8' }}>
              {hoveredRegion.name}
            </div>
            <div style={{ fontSize: '10px', color: '#94A3B8' }}>
              {hoveredRegion.lobe} • {hoveredRegion.brodmann} • {hoveredRegion.defaultBand}
            </div>
            <div style={{ marginTop: '2px', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span style={{ color: '#E2E8F0' }}>Activation:</span>
              <span
                style={{
                  fontWeight: 700,
                  color: getStatusColor(regionActivations[hoveredRegion.nodeId]?.strength ?? 50).hex
                }}
              >
                {regionActivations[hoveredRegion.nodeId]?.strength ?? 50}% (
                {getStatusColor(regionActivations[hoveredRegion.nodeId]?.strength ?? 50).status})
              </span>
            </div>
            <div style={{ fontSize: '9px', color: 'var(--text-accent)', marginTop: '2px' }}>
              💡 Right-click for clinical protocols & medicine
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}
