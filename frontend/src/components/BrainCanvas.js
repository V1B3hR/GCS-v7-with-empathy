import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import {
  Brain,
  Layers,
  GitFork,
  RotateCcw,
  X,
  Eye,
  Download,
  Zap,
  Pill,
  HeartPulse,
  Sparkles,
  Maximize2
} from 'lucide-react';
import { Brain3DModel } from './Brain3DModel';
import { getStatusColor } from '../constants/brainRegions';

/**
 * Smooth Camera Rig for Cinematic Zoom-to-Region
 */
function CameraRig({ selectedRegion, controlsRef, explodeView }) {
  useFrame(({ camera }) => {
    if (!controlsRef.current) return;
    const controls = controlsRef.current;

    if (selectedRegion) {
      const explodeOffset = explodeView ? 0.65 : 0.0;
      const isLeft = selectedRegion.hemisphere === 'LH';
      const offX = isLeft ? -explodeOffset : explodeOffset;

      const targetX = selectedRegion.position[0] + offX;
      const targetY = selectedRegion.position[1];
      const targetZ = selectedRegion.position[2];

      const targetLookAt = new THREE.Vector3(targetX, targetY, targetZ);
      const targetCamPos = new THREE.Vector3(targetX + 0.8, targetY + 0.5, targetZ + 1.2);

      camera.position.lerp(targetCamPos, 0.06);
      controls.target.lerp(targetLookAt, 0.06);
      controls.update();
    }
  });

  return null;
}

export const BrainCanvas = ({
  regionActivations,
  selectedRegion,
  activeBandFilter,
  onSelectRegion,
  onTriggerNotification
}) => {
  const controlsRef = useRef();
  const canvasContainerRef = useRef();
  const [explodeView, setExplodeView] = useState(false);
  const [showSynapses, setShowSynapses] = useState(true);

  // Right-click context menu state
  const [contextMenu, setContextMenu] = useState(null);

  // Close context menu on outside click or escape
  useEffect(() => {
    const handleClick = () => setContextMenu(null);
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') setContextMenu(null);
    };
    window.addEventListener('click', handleClick);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('click', handleClick);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  // Exact mouse cursor positioning for right click (Requirement 2)
  const handleContextMenuRegion = (region, e) => {
    const native = e.nativeEvent || e;
    if (native.preventDefault) native.preventDefault();
    if (e.preventDefault) e.preventDefault();

    // Direct viewport mouse coordinates
    let x = native.clientX ?? 400;
    let y = native.clientY ?? 300;

    // Ensure menu stays within screen bounds
    const menuWidth = 300;
    const menuHeight = 340;
    if (x + menuWidth > window.innerWidth) {
      x = window.innerWidth - menuWidth - 16;
    }
    if (y + menuHeight > window.innerHeight) {
      y = window.innerHeight - menuHeight - 16;
    }

    setContextMenu({
      visible: true,
      x: Math.max(12, x),
      y: Math.max(12, y),
      region
    });
  };

  const handleExecuteAction = (action, region) => {
    if (onTriggerNotification) {
      onTriggerNotification(`Executed "${action.label}" on ${region.name}.`);
    }

    if (action.id === 'download_csv' || action.type === 'download') {
      exportRegionCSV(region);
    }
    setContextMenu(null);
  };

  const exportRegionCSV = (region) => {
    const activation = regionActivations[region.nodeId]?.strength || 50;
    const now = new Date().toISOString();
    const rows = [
      ['Timestamp', 'Node_ID', 'Region_Name', 'Hemisphere', 'Lobe', 'Brodmann_Area', 'Resonant_Band', 'Activation_Percent', 'Status'],
      [now, region.nodeId, `"${region.name}"`, region.hemisphere, `"${region.lobe}"`, `"${region.brodmann}"`, region.defaultBand, activation, getStatusColor(activation).status]
    ];

    for (let i = 29; i >= 0; i--) {
      const pastTime = new Date(Date.now() - i * 500).toISOString();
      const val = Math.min(100, Math.max(10, Math.round(activation + (Math.random() - 0.5) * 8)));
      rows.push([pastTime, region.nodeId, `"${region.name}"`, region.hemisphere, `"${region.lobe}"`, `"${region.brodmann}"`, region.defaultBand, val, getStatusColor(val).status]);
    }

    const csvContent = 'data:text/csv;charset=utf-8,' + rows.map((e) => e.join(',')).join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `gcs_region_${region.key}_epoch.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    if (onTriggerNotification) {
      onTriggerNotification(`Saved telemetry for ${region.shortName} to disk (.CSV).`);
    }
  };

  const setCameraView = (viewType) => {
    if (!controlsRef.current) return;
    const controls = controlsRef.current;
    onSelectRegion(null);

    switch (viewType) {
      case 'axial':
        controls.object.position.set(0, 6, 0.01);
        break;
      case 'sagittal':
        controls.object.position.set(6, 0.5, 0);
        break;
      case 'coronal':
        controls.object.position.set(0, 0.5, 6);
        break;
      case 'iso':
      default:
        controls.object.position.set(3.5, 3.2, 4.5);
        break;
    }
    controls.target.set(0, 0.3, 0);
    controls.update();
  };

  const handleResetCamera = () => {
    setCameraView('iso');
  };

  const selectedActivation = selectedRegion
    ? regionActivations[selectedRegion.nodeId]
    : null;
  const selectedStatus = selectedActivation
    ? getStatusColor(selectedActivation.strength)
    : null;

  return (
    <div className="dashboard-column" style={{ flex: 1 }} ref={canvasContainerRef}>
      <div className="column-card">
        {/* Column Header */}
        <div className="column-header">
          <div className="column-header-title">
            <Brain size={18} />
            <span>3D Cortical Atlas (Desikan-Killiany 68)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {selectedRegion && (
              <button
                className="hud-btn active"
                onClick={() => onSelectRegion(null)}
                style={{ padding: '0.25rem 0.6rem' }}
              >
                <Maximize2 size={12} />
                <span>Exit Focus ({selectedRegion.shortName})</span>
              </button>
            )}
            <span className="column-badge">NEURO-ANATOMICAL MESH</span>
          </div>
        </div>

        {/* Viewport Container */}
        <div
          className="brain-viewport-container"
          onContextMenu={(e) => {
            // Prevent default browser right-click menu in the canvas viewport
            e.preventDefault();
          }}
        >
          {/* Top HUD Controls Toolbar */}
          <div className="viewport-hud-toolbar">
            <div className="hud-btn-group">
              <button
                className={`hud-btn ${explodeView ? 'active' : ''}`}
                onClick={() => setExplodeView(!explodeView)}
                title="Split hemispheres to inspect medial surfaces & insula"
              >
                <Layers size={13} />
                <span>{explodeView ? 'Merge Hemispheres' : 'Explode View'}</span>
              </button>

              <button
                className={`hud-btn ${showSynapses ? 'active' : ''}`}
                onClick={() => setShowSynapses(!showSynapses)}
                title="Toggle neural structural graph pathways"
              >
                <GitFork size={13} />
                <span>Synaptic Tracks</span>
              </button>
            </div>

            <div className="hud-btn-group">
              <button className="hud-btn" onClick={() => setCameraView('iso')} title="Isometric 3D">
                <Eye size={13} />
                <span>3D</span>
              </button>
              <button className="hud-btn" onClick={() => setCameraView('axial')} title="Axial (Top View)">
                <span>Axial</span>
              </button>
              <button className="hud-btn" onClick={() => setCameraView('sagittal')} title="Sagittal (Side View)">
                <span>Sagittal</span>
              </button>
              <button className="hud-btn" onClick={() => setCameraView('coronal')} title="Coronal (Front View)">
                <span>Coronal</span>
              </button>
              <button className="hud-btn" onClick={handleResetCamera} title="Reset Camera">
                <RotateCcw size={13} />
              </button>
            </div>
          </div>

          {/* React Three Fiber Canvas with full Device Pixel Ratio */}
          <div className="canvas-3d-wrapper">
            <Canvas
              dpr={[1, 2]}
              gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
              camera={{ position: [3.5, 3.2, 4.5], fov: 45 }}
            >
              <Brain3DModel
                regionActivations={regionActivations}
                selectedRegion={selectedRegion}
                activeBandFilter={activeBandFilter}
                onSelectRegion={onSelectRegion}
                onContextMenuRegion={handleContextMenuRegion}
                explodeView={explodeView}
                showSynapses={showSynapses}
              />
              <OrbitControls
                ref={controlsRef}
                enableDamping
                dampingFactor={0.05}
                minDistance={1.2}
                maxDistance={12}
                target={[0, 0.3, 0]}
              />
              <CameraRig
                selectedRegion={selectedRegion}
                controlsRef={controlsRef}
                explodeView={explodeView}
              />
            </Canvas>
          </div>

          {/* Bottom Left HUD Color Legend */}
          <div className="hud-color-legend">
            <div className="legend-title">Cortical Status Spectrum</div>
            <div className="legend-items">
              <div className="legend-item">
                <span className="legend-swatch" style={{ backgroundColor: 'var(--status-optimal)' }} />
                <span>Normal (40-60%)</span>
              </div>
              <div className="legend-item">
                <span className="legend-swatch" style={{ backgroundColor: 'var(--status-attention)' }} />
                <span>Attention (60-75%)</span>
              </div>
              <div className="legend-item">
                <span className="legend-swatch" style={{ backgroundColor: 'var(--status-elevated)' }} />
                <span>High (75-85%)</span>
              </div>
              <div className="legend-item">
                <span className="legend-swatch" style={{ backgroundColor: 'var(--status-danger)' }} />
                <span>Danger (&gt;85%)</span>
              </div>
            </div>
          </div>

          {/* Selected Region Inspector HUD Panel */}
          {selectedRegion && selectedStatus && (
            <div className="region-inspector-hud">
              <div className="inspector-header">
                <div className="inspector-title">{selectedRegion.name}</div>
                <button
                  className="inspector-close-btn"
                  onClick={() => onSelectRegion(null)}
                  title="Close Inspector"
                >
                  <X size={14} />
                </button>
              </div>

              <div className="inspector-details">
                <div className="inspector-row">
                  <span className="inspector-label">Hemisphere / Node:</span>
                  <span className="inspector-val">{selectedRegion.hemisphereLabel} (#{selectedRegion.nodeId})</span>
                </div>
                <div className="inspector-row">
                  <span className="inspector-label">Cortical Lobe:</span>
                  <span className="inspector-val">{selectedRegion.lobe}</span>
                </div>
                <div className="inspector-row">
                  <span className="inspector-label">Brodmann Area:</span>
                  <span className="inspector-val">{selectedRegion.brodmann}</span>
                </div>
                <div className="inspector-row">
                  <span className="inspector-label">Resonant Rhythm:</span>
                  <span className="inspector-val" style={{ color: 'var(--cyan-accent)' }}>
                    {selectedRegion.defaultBand} Wave
                  </span>
                </div>
                <div className="inspector-row">
                  <span className="inspector-label">Activation Level:</span>
                  <span className="inspector-val" style={{ color: selectedStatus.hex }}>
                    {selectedActivation.strength}% ({selectedStatus.status})
                  </span>
                </div>
                <div className="inspector-desc">{selectedRegion.function}</div>

                <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
                  <button
                    className="sim-toggle-btn"
                    style={{ flex: 1, padding: '0.35rem 0.5rem', fontSize: '11px', justifyContent: 'center' }}
                    onClick={() => exportRegionCSV(selectedRegion)}
                  >
                    <Download size={12} />
                    <span>Download (.CSV)</span>
                  </button>
                  <button
                    className="sim-toggle-btn"
                    style={{
                      flex: 1,
                      padding: '0.35rem 0.5rem',
                      fontSize: '11px',
                      justifyContent: 'center',
                      background: 'rgba(16, 185, 129, 0.2)',
                      borderColor: 'rgba(16, 185, 129, 0.4)'
                    }}
                    onClick={() => {
                      if (onTriggerNotification) {
                        onTriggerNotification(`Administered neuromodulation to ${selectedRegion.shortName}.`);
                      }
                    }}
                  >
                    <Zap size={12} />
                    <span>Neuromodulate</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Right-Click Context Menu on Bubble (Opens EXACTLY at cursor position) */}
          {contextMenu && (
            <div
              className="bubble-context-menu"
              style={{
                position: 'fixed',
                top: `${contextMenu.y}px`,
                left: `${contextMenu.x}px`,
                zIndex: 9999
              }}
              onClick={(e) => e.stopPropagation()}
              onContextMenu={(e) => e.preventDefault()}
            >
              <div className="context-menu-header">
                <Brain size={14} style={{ color: 'var(--cyan-accent)' }} />
                <div>
                  <div className="context-title">{contextMenu.region.name}</div>
                  <div className="context-sub">{contextMenu.region.lobe} • {contextMenu.region.brodmann}</div>
                </div>
                <button
                  onClick={() => setContextMenu(null)}
                  style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', marginLeft: 'auto', padding: '2px' }}
                >
                  <X size={12} />
                </button>
              </div>

              <div className="context-menu-items">
                {contextMenu.region.actions && contextMenu.region.actions.map((act) => (
                  <button
                    key={act.id}
                    className="context-item"
                    onClick={() => handleExecuteAction(act, contextMenu.region)}
                  >
                    {act.type === 'medicine' ? (
                      <Pill size={14} style={{ color: '#F59E0B' }} />
                    ) : act.type === 'stress' ? (
                      <HeartPulse size={14} style={{ color: '#10B981' }} />
                    ) : (
                      <Zap size={14} style={{ color: '#38BDF8' }} />
                    )}
                    <div style={{ textAlign: 'left' }}>
                      <div className="context-item-label">{act.label}</div>
                      <div className="context-item-desc">{act.desc}</div>
                    </div>
                  </button>
                ))}

                <button
                  className="context-item"
                  onClick={() => handleExecuteAction({ id: 'download_csv', label: 'Save & Download Telemetry (.CSV)', type: 'download' }, contextMenu.region)}
                >
                  <Download size={14} style={{ color: 'var(--cyan-accent)' }} />
                  <div style={{ textAlign: 'left' }}>
                    <div className="context-item-label">Save & Download Data (.CSV)</div>
                    <div className="context-item-desc">Export high-density activation epoch</div>
                  </div>
                </button>

                <button
                  className="context-item"
                  onClick={() => {
                    onSelectRegion(contextMenu.region);
                    setContextMenu(null);
                  }}
                >
                  <Sparkles size={14} style={{ color: '#A855F7' }} />
                  <div style={{ textAlign: 'left' }}>
                    <div className="context-item-label">Isolate & Zoom-In</div>
                    <div className="context-item-desc">Focus camera and dissolve outer nodes</div>
                  </div>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
