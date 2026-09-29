import React from 'react';
import { Activity, Wifi, Radio, Sliders, Shield, User, HardDrive, UserPlus, ChevronDown } from 'lucide-react';

export const HeaderBar = ({
  connectionStatus,
  networkPing,
  pushHz,
  channelCount = 64,
  patientData,
  onOpenSimModal,
  onOpenPatientModal,
  onOpenSessionMenu,
  activeScenarioName
}) => {
  const isConnected = connectionStatus === 'CONNECTED';
  const isSimulated = connectionStatus === 'SIMULATED';

  return (
    <header className="header-bar">
      {/* Left: Brand Name (Clickable for Session Menu - Requirement 10) & Subject Tag */}
      <div className="header-left">
        <button
          className="brand-badge-btn"
          onClick={onOpenSessionMenu}
          title="Click to Open Session Operations & Storage Menu"
        >
          <div className="brand-icon">
            <Activity size={20} />
          </div>
          <div className="brand-titles">
            <span className="brand-title">Grand Council Sentient</span>
            <span className="brand-sub">GCS v7.0 • EMPATHIC COGNITIVE SYNC ▼</span>
          </div>
        </button>

        {/* Subject Tag (Clickable to edit patient details) */}
        <button
          className="subject-tag-btn"
          onClick={onOpenPatientModal}
          title="View & Edit Subject Records"
        >
          <User size={13} className="subject-label" />
          <span className="subject-label">Subject:</span>
          <span className="subject-val">{patientData?.subjectId || 'V1B3hr'}</span>
          <span style={{ color: 'rgba(255,255,255,0.2)' }}>|</span>
          <span className="subject-label">Cap:</span>
          <span style={{ color: '#cbd5e1', fontWeight: 500 }}>{channelCount}-CH</span>
        </button>
      </div>

      {/* Center: System Status & Network Metrics */}
      <div className="header-center">
        <div
          className={`system-status-pill ${
            isConnected
              ? 'status-connected'
              : isSimulated
              ? 'status-simulated'
              : 'status-disconnected'
          }`}
        >
          <span
            className={`status-dot ${
              isConnected
                ? 'dot-optimal pulse-optimal'
                : isSimulated
                ? 'dot-simulated pulse-optimal'
                : 'dot-danger pulse-danger'
            }`}
          />
          <span>
            {isConnected
              ? 'LIVE HARDWARE SYNC'
              : isSimulated
              ? `SIMULATED LIVE (${activeScenarioName || 'ACTIVE'})`
              : 'OFFLINE / RECONNECTING'}
          </span>
        </div>

        <div className="network-metrics">
          <div className="metric-pill">
            <Wifi size={12} style={{ display: 'inline', marginRight: '4px' }} />
            <span>{networkPing}ms</span>
          </div>
          <div className="metric-pill">
            <Radio size={12} style={{ display: 'inline', marginRight: '4px' }} />
            <span>{pushHz.toFixed(1)} Hz</span>
          </div>
          <div className="metric-pill">
            <Shield size={12} style={{ display: 'inline', marginRight: '4px', color: '#10B981' }} />
            <span>68 NODES • DESIKAN</span>
          </div>
        </div>
      </div>

      {/* Right: Demographics, Patient Upload (Requirement 9), and Scenario Controller */}
      <div className="header-right">
        <div className="demographics-panel">
          <div className="demo-item">
            <span>{patientData?.sex || 'Male'}, {patientData?.age || 32}</span>
          </div>
          <span className="demo-divider">•</span>
          <div className="demo-item">
            <span>{patientData?.handedness || 'Right-Handed'}</span>
          </div>
          <span className="demo-divider">•</span>
          <div className="demo-item">
            <span style={{ color: 'var(--text-accent)' }}>{channelCount}-CH Array</span>
          </div>
        </div>

        {/* Patient Details Upload / Profile Button (Requirement 9) */}
        <button
          className="sim-toggle-btn"
          onClick={onOpenPatientModal}
          title="Upload or Edit Patient Details & Calibrations"
          style={{ background: 'rgba(99, 102, 241, 0.2)', borderColor: 'rgba(99, 102, 241, 0.4)' }}
        >
          <UserPlus size={14} />
          <span>Patient File</span>
        </button>

        {/* Scenarios Button */}
        <button
          className="sim-toggle-btn"
          onClick={onOpenSimModal}
          title="Switch Neural Scenarios or Emulation Modes"
        >
          <Sliders size={14} />
          <span>Scenarios</span>
        </button>
      </div>
    </header>
  );
};
