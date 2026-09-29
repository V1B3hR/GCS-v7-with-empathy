import React from 'react';
import { X, Sliders, CheckCircle2, ShieldAlert } from 'lucide-react';
import { PRESET_SCENARIOS } from '../hooks/useBCIData';

export const SimulationModal = ({
  isOpen,
  onClose,
  activeScenario,
  onSelectScenario
}) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sliders size={18} style={{ color: 'var(--cyan-accent)' }} />
            <span className="modal-title">BCI Emulation & Scenario Controller</span>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '4px'
            }}
          >
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            Select an operational state to emulate neural synchrony, autonomic biomarkers, 68-node Desikan-Killiany cortical activations, and Stage 1-5 Empathy responses:
          </p>

          <div className="scenario-presets">
            {Object.values(PRESET_SCENARIOS).map((scenario) => {
              const isActive = activeScenario === scenario.id;
              return (
                <button
                  key={scenario.id}
                  className={`scenario-btn ${isActive ? 'active' : ''}`}
                  onClick={() => {
                    onSelectScenario(scenario.id);
                    onClose();
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                    <span className="scenario-title" style={{ color: isActive ? 'var(--cyan-accent)' : '#fff' }}>
                      {scenario.name}
                    </span>
                    {isActive && <CheckCircle2 size={14} style={{ color: 'var(--cyan-accent)' }} />}
                  </div>
                  <span className="scenario-desc">{scenario.desc}</span>
                  <div style={{ display: 'flex', gap: '8px', marginTop: '4px', fontSize: '11px', fontFamily: 'JetBrains Mono', color: 'var(--text-muted)' }}>
                    <span>HR: {scenario.hr} bpm</span>
                    <span>HRV: {scenario.hrv} ms</span>
                    <span>Act: {scenario.regionBase}%</span>
                  </div>
                </button>
              );
            })}
          </div>

          <div
            style={{
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '10px 12px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px'
            }}
          >
            <ShieldAlert size={18} style={{ color: 'var(--status-optimal)' }} />
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              WebSocket auto-sync is active at <code style={{ color: 'var(--text-accent)' }}>ws://localhost:8000/ws</code>. When a live hardware feed is connected, real-time sensor packets override the simulation seamlessly.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
