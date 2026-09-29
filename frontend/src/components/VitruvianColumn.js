import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import {
  UserCheck,
  HeartHandshake,
  Download,
  FileText,
  Layers,
  Sparkles,
  Zap,
  Pill,
  HeartPulse,
  RotateCcw,
  X,
  History,
  Activity
} from 'lucide-react';
import { Somatic3DModel } from './Somatic3DModel';
import { getStatusColor } from '../constants/brainRegions';

export const ORGAN_DEFINITIONS = {
  brain: {
    id: 'brain',
    name: 'Cerebral Cortex & Limbic Axis',
    system: 'Central Nervous System',
    metrics: [
      { label: 'Intracranial Pressure', value: '11 mmHg', status: 'Normal' },
      { label: 'Neurovascular Coupling', value: '98%', status: 'Optimal' },
      { label: 'Cortical Coherence', value: '0.86', status: 'Optimal' }
    ],
    actions: [
      { id: 'brain_sync', label: 'Full Cortical Synchrony Sweep', desc: 'Induce 40 Hz gamma binding across both hemispheres', type: 'stimulate' },
      { id: 'admin_neuroprotect', label: 'Administer Neuroprotective Agent', desc: 'Infuse cerebrovascular antioxidant cocktail', type: 'medicine' },
      { id: 'lower_neural_strain', label: 'Dampen Hyperexcitability', desc: 'Engage slow inhibitory interneuron tone', type: 'stress' }
    ]
  },
  heart: {
    id: 'heart',
    name: 'Cardiovascular Core (Heart)',
    system: 'Cardiopulmonary System',
    metrics: [
      { label: 'Cardiac Output', value: '5.2 L/min', status: 'Optimal' },
      { label: 'Ejection Fraction', value: '64%', status: 'Optimal' },
      { label: 'Vagal Heart Rhythm', value: 'Sinus Normal', status: 'Normal' }
    ],
    actions: [
      { id: 'vagal_cardiopacing', label: 'Engage Vagal Cardiopacing', desc: 'Auricular stimulation to lower heart rate to 68 BPM', type: 'stress' },
      { id: 'admin_betablocker', label: 'Administer Beta-Adrenergic Blocker', desc: 'Mitigate sympathetic tachycardia spikes', type: 'medicine' },
      { id: 'cardiac_resonance', label: '0.1 Hz Resonance Entrainment', desc: 'Align heart rate oscillations with breathing', type: 'stimulate' }
    ]
  },
  lungs: {
    id: 'lungs',
    name: 'Pulmonary & Respiratory Tree',
    system: 'Respiratory System',
    metrics: [
      { label: 'Tidal Volume', value: '510 mL', status: 'Normal' },
      { label: 'Pulmonary Compliance', value: 'Optimal', status: 'Optimal' },
      { label: 'End-Tidal CO2', value: '38 mmHg', status: 'Normal' }
    ],
    actions: [
      { id: 'diaphragm_sync', label: 'Guided Diaphragm Pacing (4-7-8)', desc: 'Re-establish parasympathetic tidal volume', type: 'stress' },
      { id: 'admin_bronchodilator', label: 'Administer Bronchodilator', desc: 'Reduce airway resistance and optimize gas exchange', type: 'medicine' }
    ]
  },
  liver: {
    id: 'liver',
    name: 'Hepatic & Metabolic Nexus',
    system: 'Metabolic System',
    metrics: [
      { label: 'Blood Glucose', value: '94 mg/dL', status: 'Optimal' },
      { label: 'Hepatic Perfusion', value: '1.4 L/min', status: 'Normal' },
      { label: 'Lactate Clearance', value: '0.9 mmol/L', status: 'Optimal' }
    ],
    actions: [
      { id: 'metabolic_boost', label: 'Metabolic Homeostasis Optimization', desc: 'Activate cellular NAD+ replenishment protocol', type: 'medicine' }
    ]
  },
  gut: {
    id: 'gut',
    name: 'Enteric Nervous System (Second Brain)',
    system: 'Enteric-Vagal Axis',
    metrics: [
      { label: 'Subdiaphragmatic Vagal Tone', value: '86%', status: 'Optimal' },
      { label: 'Gut-Brain Serotonin Synthesis', value: 'Normal', status: 'Normal' },
      { label: 'Gastric Motility Index', value: '3.2 cpm', status: 'Normal' }
    ],
    actions: [
      { id: 'vagus_gut_stim', label: 'Subdiaphragmatic Vagus Stimulation', desc: 'Direct anti-inflammatory cholinergic pathway trigger', type: 'stress' },
      { id: 'admin_probiotic_neuro', label: 'Administer Neuro-Active Metabolite', desc: 'GABA-producing probiotic postbiotic infusion', type: 'medicine' }
    ]
  },
  kidneys: {
    id: 'kidneys',
    name: 'Renal & Electrolyte Regulation',
    system: 'Renal System',
    metrics: [
      { label: 'Estimated GFR (eGFR)', value: '>90 mL/min', status: 'Optimal' },
      { label: 'Electrolyte Osmolality', value: '288 mOsm/kg', status: 'Normal' }
    ],
    actions: [
      { id: 'renal_perfusion', label: 'Renal Microvascular Optimization', desc: 'Regulate sympathetic renal vasoconstriction', type: 'stimulate' }
    ]
  },
  limbs: {
    id: 'limbs',
    name: 'Peripheral & Musculoskeletal Axis',
    system: 'Somatomotor System',
    metrics: [
      { label: 'Electrodermal Conductance', value: '2.2 µS', status: 'Normal' },
      { label: 'Peripheral Vascular Resistance', value: 'Normal', status: 'Normal' },
      { label: 'Muscular Tension EMG', value: 'Low Basal', status: 'Optimal' }
    ],
    actions: [
      { id: 'muscle_decompress', label: 'Engage Somatic Muscle Release', desc: 'Progressive muscle relaxation pulse wave', type: 'stress' },
      { id: 'admin_spasmolytic', label: 'Administer Peripheral Relaxant', desc: 'Alleviate distal muscular hypertonia', type: 'medicine' }
    ]
  }
};

export const VitruvianColumn = ({
  affectiveState,
  affectiveHistory = [],
  somaticHotspots,
  selectedHotspot,
  onSelectHotspot,
  onTriggerNotification
}) => {
  const [viewMode, setViewMode] = useState('2D');
  const [selectedOrganKey, setSelectedOrganKey] = useState('heart');
  const [organContextMenu, setOrganContextMenu] = useState(null);

  useEffect(() => {
    const handleClick = () => setOrganContextMenu(null);
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') setOrganContextMenu(null);
    };
    window.addEventListener('click', handleClick);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('click', handleClick);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const emotionColor = getStatusColor(affectiveState.strength);

  const getOrganStrength = (key) => {
    if (key === 'brain') return somaticHotspots.head?.strength || 50;
    if (key === 'heart') return somaticHotspots.chest?.strength || 50;
    if (key === 'lungs') return somaticHotspots.lungs?.strength || 48;
    if (key === 'gut') return somaticHotspots.core?.strength || 46;
    if (key === 'limbs') return somaticHotspots.arms?.strength || 44;
    return 48;
  };

  const handleSelectOrgan = (key) => {
    setSelectedOrganKey(key);
    if (onSelectHotspot) {
      onSelectHotspot(key);
    }
  };

  // Exact mouse cursor positioning for right click (Requirement 2)
  const handleContextMenuOrgan = (key, e) => {
    const native = e.nativeEvent || e;
    if (native.preventDefault) native.preventDefault();
    if (e.preventDefault) e.preventDefault();

    let x = native.clientX ?? 400;
    let y = native.clientY ?? 300;

    const menuWidth = 290;
    const menuHeight = 320;
    if (x + menuWidth > window.innerWidth) {
      x = window.innerWidth - menuWidth - 16;
    }
    if (y + menuHeight > window.innerHeight) {
      y = window.innerHeight - menuHeight - 16;
    }

    const organDef = ORGAN_DEFINITIONS[key] || ORGAN_DEFINITIONS.heart;

    setOrganContextMenu({
      visible: true,
      x: Math.max(12, x),
      y: Math.max(12, y),
      organ: organDef
    });
  };

  const handleExecuteOrganAction = (action, organ) => {
    if (onTriggerNotification) {
      onTriggerNotification(`Executed "${action.label}" on ${organ.name}.`);
    }
    setOrganContextMenu(null);
  };

  const exportAffectiveCSV = () => {
    const rows = [
      ['Timestamp', 'Emotion_Label', 'Valence', 'Arousal', 'Confidence_Pct', 'Strength_Pct', 'Crisis_Detected', 'AI_Response']
    ];

    affectiveHistory.forEach((item) => {
      rows.push([
        item.timestamp,
        `"${item.label}"`,
        item.valence,
        item.arousal,
        (item.confidence * 100).toFixed(0),
        item.strength,
        item.crisisDetected ? 'CRISIS_ALERT' : 'NORMAL',
        `"${item.empathicResponse?.content || ''}"`
      ]);
    });

    const csvContent = 'data:text/csv;charset=utf-8,' + rows.map((e) => e.join(',')).join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `gcs_affective_history_20states.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    if (onTriggerNotification) {
      onTriggerNotification(`Saved Affective State History (.CSV) to disk.`);
    }
  };

  const exportAffectiveTXT = () => {
    let txt = `========================================================\n`;
    txt += `GCS v7.0 EMPATHIC COGNITIVE SYNC - AFFECTIVE AUDIT LOG\n`;
    txt += `Generated: ${new Date().toLocaleString()}\n`;
    txt += `Total Recorded State Transitions: ${affectiveHistory.length}\n`;
    txt += `========================================================\n\n`;

    affectiveHistory.forEach((h, i) => {
      txt += `[#${i + 1}] ${h.timestamp} | Emotion: ${h.label.toUpperCase()} (${h.icon})\n`;
      txt += `     Valence: ${h.valence > 0 ? '+' : ''}${h.valence} | Arousal: ${h.arousal} | Conf: ${(h.confidence * 100).toFixed(0)}%\n`;
      txt += `     DuetMind: "${h.empathicResponse?.content}"\n`;
      txt += `--------------------------------------------------------\n`;
    });

    const element = document.createElement('a');
    const file = new Blob([txt], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `gcs_affective_log.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);

    if (onTriggerNotification) {
      onTriggerNotification(`Saved Affective State Audit Log (.TXT) to disk.`);
    }
  };

  const currentOrganDef = ORGAN_DEFINITIONS[selectedOrganKey] || ORGAN_DEFINITIONS.heart;
  const currentOrganStrength = getOrganStrength(selectedOrganKey);
  const currentOrganStatus = getStatusColor(currentOrganStrength);

  return (
    <div className="dashboard-column right-column-container">
      {/* 1. Somatic Map: 2D Vector / 3D Hologram Toggle Card */}
      <div className="body-map-card">
        <div className="column-header">
          <div className="column-header-title">
            <UserCheck size={18} />
            <span>Somatic Biometric Map</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div className="mode-toggle-pill">
              <button
                className={`mode-btn ${viewMode === '2D' ? 'active' : ''}`}
                onClick={() => setViewMode('2D')}
                title="Detailed Da Vinci Biometric Vector Map"
              >
                2D
              </button>
              <button
                className={`mode-btn ${viewMode === '3D' ? 'active' : ''}`}
                onClick={() => setViewMode('3D')}
                title="3D Holographic Mannequin with OrbitControls"
              >
                3D
              </button>
            </div>
            <span className="column-badge">ACTIVE EMBODIMENT</span>
          </div>
        </div>

        <div
          className="body-canvas-wrap"
          onContextMenu={(e) => e.preventDefault()}
        >
          {viewMode === '3D' ? (
            <div style={{ width: '100%', height: '100%', position: 'relative' }}>
              <Canvas
                dpr={[1, 2]}
                gl={{ antialias: true, alpha: true }}
                camera={{ position: [0, 0.4, 3.4], fov: 45 }}
              >
                <Somatic3DModel
                  organStates={{
                    brain: { strength: getOrganStrength('brain') },
                    heart: { strength: getOrganStrength('heart') },
                    lungs: { strength: getOrganStrength('lungs') },
                    liver: { strength: 48 },
                    gut: { strength: getOrganStrength('gut') },
                    kidneys: { strength: 46 }
                  }}
                  selectedOrgan={currentOrganDef}
                  onSelectOrgan={handleSelectOrgan}
                  onContextMenuOrgan={handleContextMenuOrgan}
                />
                <OrbitControls enableDamping dampingFactor={0.05} minDistance={1.8} maxDistance={6} />
              </Canvas>
              <div style={{ position: 'absolute', bottom: '8px', left: '12px', fontSize: '10px', color: 'var(--text-muted)' }}>
                Rotate & Zoom 3D Mannequin • Right-click organs for features
              </div>
            </div>
          ) : (
            <svg
              className="vitruvian-svg"
              viewBox="0 0 500 500"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <defs>
                <radialGradient id="vitruvian-glow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#38BDF8" stopOpacity="0.12" />
                  <stop offset="100%" stopColor="#38BDF8" stopOpacity="0.0" />
                </radialGradient>
              </defs>

              {/* Classical Circle and Square Framing */}
              <circle cx="250" cy="250" r="230" stroke="rgba(56, 189, 248, 0.2)" strokeWidth="1.2" strokeDasharray="4 4" />
              <circle cx="250" cy="250" r="230" fill="url(#vitruvian-glow)" />
              <rect x="50" y="50" width="400" height="400" stroke="rgba(255, 255, 255, 0.08)" strokeWidth="1" />

              {/* Da Vinci Stylized Human Skeletal & Muscular Framework */}
              <g stroke="#94A3B8" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" opacity="0.38">
                {/* Cranium & Jaw */}
                <ellipse cx="250" cy="115" rx="26" ry="30" fill="none" />
                <path d="M 235 165 L 265 165 L 270 260 L 230 260 Z" fill="none" />
                
                {/* Ribcage Outline */}
                {[170, 185, 200, 215, 230].map((yRib) => (
                  <path key={yRib} d={`M 234 ${yRib} Q 215 ${yRib + 5} 230 ${yRib + 12} M 266 ${yRib} Q 285 ${yRib + 5} 270 ${yRib + 12}`} stroke="#38BDF8" strokeWidth="1" opacity="0.4" />
                ))}

                {/* Arms & Limbs */}
                <path d="M 235 175 L 140 175 L 70 175" />
                <path d="M 265 175 L 360 175 L 430 175" />
                <line x1="238" y1="260" x2="230" y2="430" />
                <line x1="262" y1="260" x2="270" y2="430" />

                {/* Descending Aorta Vessel */}
                <line x1="250" y1="188" x2="250" y2="280" stroke="#EF4444" strokeWidth="2" opacity="0.45" />
              </g>

              {/* Anatomical Organ Hotspots */}

              {/* 1. BRAIN */}
              <g
                className="body-hotspot"
                onClick={() => handleSelectOrgan('brain')}
                onContextMenu={(e) => { e.preventDefault(); handleContextMenuOrgan('brain', e); }}
              >
                <ellipse cx="250" cy="115" rx="22" ry="24" fill={getStatusColor(getOrganStrength('brain')).hex} fillOpacity="0.25" stroke={getStatusColor(getOrganStrength('brain')).hex} strokeWidth="2" />
                <circle cx="250" cy="115" r="5" fill="#ffffff" />
                <text x="250" y="82" textAnchor="middle" fill="#38BDF8" fontSize="10" fontFamily="JetBrains Mono" fontWeight="600">BRAIN / CNS</text>
              </g>

              {/* 2. HEART */}
              <g
                className="body-hotspot"
                onClick={() => handleSelectOrgan('heart')}
                onContextMenu={(e) => { e.preventDefault(); handleContextMenuOrgan('heart', e); }}
              >
                <path
                  d="M 245 180 C 235 168 220 180 220 192 C 220 205 245 220 245 220 C 245 220 270 205 270 192 C 270 180 255 168 245 180 Z"
                  fill={getStatusColor(getOrganStrength('heart')).hex}
                  fillOpacity="0.32"
                  stroke={getStatusColor(getOrganStrength('heart')).hex}
                  strokeWidth="2"
                />
                <circle cx="245" cy="192" r="4" fill="#ffffff" />
                <text x="315" y="196" textAnchor="start" fill={getStatusColor(getOrganStrength('heart')).hex} fontSize="10" fontFamily="JetBrains Mono" fontWeight="600">HEART</text>
              </g>

              {/* 3. LUNGS */}
              <g
                className="body-hotspot"
                onClick={() => handleSelectOrgan('lungs')}
                onContextMenu={(e) => { e.preventDefault(); handleContextMenuOrgan('lungs', e); }}
              >
                <ellipse cx="218" cy="188" rx="14" ry="20" fill={getStatusColor(getOrganStrength('lungs')).hex} fillOpacity="0.25" stroke={getStatusColor(getOrganStrength('lungs')).hex} strokeWidth="1.5" />
                <ellipse cx="272" cy="188" rx="14" ry="20" fill={getStatusColor(getOrganStrength('lungs')).hex} fillOpacity="0.25" stroke={getStatusColor(getOrganStrength('lungs')).hex} strokeWidth="1.5" />
                <text x="180" y="188" textAnchor="end" fill="#94A3B8" fontSize="9" fontFamily="JetBrains Mono">LUNGS</text>
              </g>

              {/* 4. LIVER */}
              <g
                className="body-hotspot"
                onClick={() => handleSelectOrgan('liver')}
                onContextMenu={(e) => { e.preventDefault(); handleContextMenuOrgan('liver', e); }}
              >
                <path d="M 252 222 L 285 222 L 280 240 L 252 235 Z" fill={getStatusColor(48).hex} fillOpacity="0.25" stroke={getStatusColor(48).hex} strokeWidth="1.5" />
                <text x="315" y="232" textAnchor="start" fill="#94A3B8" fontSize="9" fontFamily="JetBrains Mono">LIVER</text>
              </g>

              {/* 5. STOMACH / GUT (Enteric Axis) */}
              <g
                className="body-hotspot"
                onClick={() => handleSelectOrgan('gut')}
                onContextMenu={(e) => { e.preventDefault(); handleContextMenuOrgan('gut', e); }}
              >
                <circle cx="238" cy="245" r="14" fill={getStatusColor(getOrganStrength('gut')).hex} fillOpacity="0.3" stroke={getStatusColor(getOrganStrength('gut')).hex} strokeWidth="1.8" />
                <text x="175" y="248" textAnchor="end" fill="#94A3B8" fontSize="9" fontFamily="JetBrains Mono">GUT (ENTERIC)</text>
              </g>

              {/* 6. LIMBS / PERIPHERAL */}
              <g
                className="body-hotspot"
                onClick={() => handleSelectOrgan('limbs')}
                onContextMenu={(e) => { e.preventDefault(); handleContextMenuOrgan('limbs', e); }}
              >
                <circle cx="70" cy="175" r="10" fill={getStatusColor(getOrganStrength('limbs')).hex} fillOpacity="0.25" stroke={getStatusColor(getOrganStrength('limbs')).hex} strokeWidth="1.5" />
                <circle cx="430" cy="175" r="10" fill={getStatusColor(getOrganStrength('limbs')).hex} fillOpacity="0.25" stroke={getStatusColor(getOrganStrength('limbs')).hex} strokeWidth="1.5" />
              </g>
            </svg>
          )}
        </div>

        {/* Selected Organ Readings Inspector */}
        <div className="organ-readings-strip">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontWeight: 600, color: 'var(--cyan-accent)', fontSize: '12px' }}>
              {currentOrganDef.name}
            </span>
            <span
              className="vital-status-pill"
              style={{
                backgroundColor: currentOrganStatus.glow,
                color: currentOrganStatus.hex,
                border: `1px solid ${currentOrganStatus.hex}66`
              }}
            >
              {currentOrganStatus.status}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px', marginTop: '6px' }}>
            {currentOrganDef.metrics.map((m) => (
              <div key={m.label} style={{ background: 'rgba(255,255,255,0.03)', padding: '4px 6px', borderRadius: '4px' }}>
                <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>{m.label}</div>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#fff', marginTop: '1px' }}>{m.value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 2. Empathy Engine & Affective State Hub with 20-State History */}
      <div className="empathy-card">
        <div className="column-header">
          <div className="column-header-title">
            <HeartHandshake size={18} />
            <span>Empathy & Affective State</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <button
              className="hud-btn"
              onClick={exportAffectiveCSV}
              title="Download 20-State Affective History as .CSV"
              style={{ padding: '0.2rem 0.5rem', fontSize: '10px' }}
            >
              <Download size={11} />
              <span>.CSV</span>
            </button>
            <button
              className="hud-btn"
              onClick={exportAffectiveTXT}
              title="Download Affective Log Summary as .TXT"
              style={{ padding: '0.2rem 0.5rem', fontSize: '10px' }}
            >
              <FileText size={11} />
              <span>.TXT</span>
            </button>
          </div>
        </div>

        <div className="empathy-body">
          {/* Real-time Hero State */}
          <div className="affective-hero">
            <div className="emotion-icon-display">{affectiveState.icon}</div>
            <div className="emotion-text-wrap">
              <div className="emotion-title-row">
                <span className="emotion-name">{affectiveState.label}</span>
                <span className="emotion-conf">
                  Conf: {(affectiveState.confidence * 100).toFixed(0)}%
                </span>
              </div>

              <div className="strength-bar-track">
                <div
                  className="strength-bar-progress"
                  style={{
                    width: `${affectiveState.strength}%`,
                    backgroundColor: emotionColor.hex,
                    boxShadow: `0 0 10px ${emotionColor.hex}88`
                  }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '11px', fontFamily: 'JetBrains Mono' }}>
                <span style={{ color: 'var(--text-muted)' }}>Emotional Arousal</span>
                <span style={{ color: emotionColor.hex, fontWeight: 700 }}>
                  {affectiveState.strength}% ({emotionColor.status})
                </span>
              </div>
            </div>
          </div>

          {/* Valence & Arousal Circumplex Coordinates */}
          <div className="circumplex-grid-wrap">
            <div className="coord-box">
              <span className="coord-label">Valence [-1, 1]</span>
              <span
                className="coord-val"
                style={{ color: affectiveState.valence >= 0 ? 'var(--status-optimal)' : 'var(--status-danger)' }}
              >
                {affectiveState.valence > 0 ? `+${affectiveState.valence}` : affectiveState.valence}
              </span>
            </div>
            <div className="coord-box">
              <span className="coord-label">Arousal [0, 1]</span>
              <span
                className="coord-val"
                style={{ color: affectiveState.arousal > 0.75 ? 'var(--status-elevated)' : 'var(--text-primary)' }}
              >
                {affectiveState.arousal}
              </span>
            </div>
          </div>

          {/* Empathetic AI Response Bubble */}
          <div className="empathic-dialogue-bubble">
            <div className="agent-header-row">
              <span className="agent-name">
                <Sparkles size={13} />
                <span>DuetMind Agent</span>
              </span>
              <span className="agent-mode-badge">
                {affectiveState.crisisDetected ? 'CRISIS STABILIZATION' : 'VALIDATION'}
              </span>
            </div>
            <div className="agent-text">
              "{affectiveState.empathicResponse.content}"
            </div>
          </div>

          {/* 20 Previous Affective States Reel */}
          <div className="affective-history-section">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <History size={12} />
                <span>Previous States (Last {affectiveHistory.length})</span>
              </div>
              <span style={{ fontSize: '10px', color: 'var(--cyan-accent)' }}>Audit Trail</span>
            </div>

            <div className="affective-history-reel">
              {affectiveHistory.slice().reverse().map((item, idx) => (
                <div key={idx} className="history-pill" title={`${item.timestamp} - Valence: ${item.valence}, Arousal: ${item.arousal}`}>
                  <span style={{ fontSize: '13px' }}>{item.icon}</span>
                  <span style={{ fontWeight: 600, textTransform: 'capitalize' }}>{item.label}</span>
                  <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>{item.timestamp?.split(' ')[1] || item.timestamp}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Right-Click Organ Clinical Action Menu (Opens EXACTLY at cursor position) */}
      {organContextMenu && (
        <div
          className="bubble-context-menu"
          style={{
            position: 'fixed',
            top: `${organContextMenu.y}px`,
            left: `${organContextMenu.x}px`,
            zIndex: 9999
          }}
          onClick={(e) => e.stopPropagation()}
          onContextMenu={(e) => e.preventDefault()}
        >
          <div className="context-menu-header">
            <Activity size={14} style={{ color: 'var(--cyan-accent)' }} />
            <div>
              <div className="context-title">{organContextMenu.organ.name}</div>
              <div className="context-sub">{organContextMenu.organ.system}</div>
            </div>
            <button
              onClick={() => setOrganContextMenu(null)}
              style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', marginLeft: 'auto', padding: '2px' }}
            >
              <X size={12} />
            </button>
          </div>

          <div className="context-menu-items">
            {organContextMenu.organ.actions.map((act) => (
              <button
                key={act.id}
                className="context-item"
                onClick={() => handleExecuteOrganAction(act, organContextMenu.organ)}
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
              onClick={() => {
                if (onTriggerNotification) {
                  onTriggerNotification(`Exported organ telemetry for ${organContextMenu.organ.name}.`);
                }
                setOrganContextMenu(null);
              }}
            >
              <Download size={14} style={{ color: 'var(--cyan-accent)' }} />
              <div style={{ textAlign: 'left' }}>
                <div className="context-item-label">Export Organ Log (.CSV)</div>
                <div className="context-item-desc">Save continuous hemodynamic/metabolic stream</div>
              </div>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
