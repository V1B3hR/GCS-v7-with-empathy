import React, { useState } from 'react';
import {
  Heart,
  Activity,
  Wind,
  Droplet,
  Zap,
  Thermometer,
  Waves,
  Brain,
  ChevronDown,
  Download,
  Calendar,
  Layers,
  Sparkles,
  X
} from 'lucide-react';
import { Sparkline } from './Sparkline';

export const CHANNEL_PRESETS = [
  { channels: 8, label: '8-ch Consumer BCI', rate: '250 Hz', type: 'Dry Electrode' },
  { channels: 16, label: '16-ch Research EEG', rate: '500 Hz', type: 'Gel Saline' },
  { channels: 32, label: '32-ch Clinical 10-20', rate: '1000 Hz', type: 'Clinical Ag/AgCl' },
  { channels: 64, label: '64-ch High-Density OpenBCI', rate: '1000 Hz', type: 'Active Shielded' },
  { channels: 128, label: '128-ch Geodesic Sensor Net', rate: '2000 Hz', type: 'Hydrocel' },
  { channels: 256, label: '256-ch Cortical Mapping Array', rate: '4000 Hz', type: 'High Density' },
  { channels: 512, label: '512-ch Subdural µECoG Grid', rate: '8000 Hz', type: 'Invasive Surface' },
  { channels: 1024, label: '1024-ch Neuralink / Utah Array', rate: '20 kHz', type: 'Micro-Electrode Probe' }
];

export const PhysiologicalColumn = ({
  vitals,
  eegBands,
  channelCount = 64,
  onChangeChannels,
  activeBandFilter,
  onSelectBandFilter,
  onTriggerNotification
}) => {
  const [showChannelModal, setShowChannelModal] = useState(false);
  const [selectedWaveDetail, setSelectedWaveDetail] = useState(null); // 'alpha', 'beta', etc.

  // Helper for status badge
  const getStatusBadge = (status) => {
    switch (status) {
      case 'Normal':
        return {
          bg: 'rgba(16, 185, 129, 0.15)',
          color: 'var(--status-optimal)',
          border: 'rgba(16, 185, 129, 0.3)'
        };
      case 'Attention':
        return {
          bg: 'rgba(245, 158, 11, 0.15)',
          color: 'var(--status-attention)',
          border: 'rgba(245, 158, 11, 0.3)'
        };
      case 'Elevated':
        return {
          bg: 'rgba(234, 88, 12, 0.18)',
          color: 'var(--status-elevated)',
          border: 'rgba(234, 88, 12, 0.35)'
        };
      case 'Danger':
      default:
        return {
          bg: 'rgba(239, 68, 68, 0.2)',
          color: 'var(--status-danger)',
          border: 'rgba(239, 68, 68, 0.4)'
        };
    }
  };

  const getSparkColor = (status) => {
    if (status === 'Normal') return '#10B981';
    if (status === 'Attention') return '#F59E0B';
    if (status === 'Elevated') return '#EA580C';
    return '#EF4444';
  };

  const vitalConfig = [
    {
      key: 'heartRate',
      name: 'Heart Rate',
      unit: 'BPM',
      icon: <Heart size={14} />,
      data: vitals.heartRate,
      rangeText: 'Safe: 60-90 BPM'
    },
    {
      key: 'hrv',
      name: 'HRV (RMSSD)',
      unit: 'ms',
      icon: <Activity size={14} />,
      data: vitals.hrv,
      rangeText: 'Normal: >45 ms'
    },
    {
      key: 'respiration',
      name: 'Respiration',
      unit: 'BrPM',
      icon: <Wind size={14} />,
      data: vitals.respiration,
      rangeText: 'Safe: 12-20 BrPM'
    },
    {
      key: 'spo2',
      name: 'SpO2 Saturation',
      unit: '%',
      icon: <Droplet size={14} />,
      data: vitals.spo2,
      rangeText: 'Optimal: >95%'
    },
    {
      key: 'gsr',
      name: 'Skin Conductance',
      unit: 'µS',
      icon: <Zap size={14} />,
      data: vitals.gsr,
      rangeText: 'Baseline: 1.0-4.5 µS'
    },
    {
      key: 'skinTemp',
      name: 'Skin Temperature',
      unit: '°C',
      icon: <Thermometer size={14} />,
      data: vitals.skinTemp,
      rangeText: 'Normal: 36.0-37.2 °C'
    }
  ];

  const bandsList = [
    { key: 'delta', name: 'Delta (δ)', freq: '0.5 - 4 Hz', data: eegBands.delta, color: '#38BDF8', desc: 'Slow-wave deep restorative state' },
    { key: 'theta', name: 'Theta (θ)', freq: '4 - 8 Hz', data: eegBands.theta, color: '#818CF8', desc: 'Memory consolidation, drowsiness, meditation' },
    { key: 'alpha', name: 'Alpha (α)', freq: '8 - 12 Hz', data: eegBands.alpha, color: '#10B981', desc: 'Calm presence, relaxed alertness, bridge to flow' },
    { key: 'beta', name: 'Beta (β)', freq: '12 - 30 Hz', data: eegBands.beta, color: '#F59E0B', desc: 'Active analytical thought, executive problem solving' },
    { key: 'gamma', name: 'Gamma (γ)', freq: '30 - 100 Hz', data: eegBands.gamma, color: '#EC4899', desc: 'Cross-modal sensory binding, eureka insights' }
  ];

  // Save 7-Day EEG Telemetry to SSD (.CSV)
  const export7DayEEG = () => {
    const days = ['Day -6', 'Day -5', 'Day -4', 'Day -3', 'Day -2', 'Yesterday', 'Today'];
    const rows = [
      ['Timeline', 'Channel_Config', 'Delta_uV2', 'Theta_uV2', 'Alpha_uV2', 'Beta_uV2', 'Gamma_uV2', 'FAA_Index']
    ];

    days.forEach((day, idx) => {
      const delta = Math.round(18 + Math.sin(idx) * 4);
      const theta = Math.round(22 + Math.cos(idx) * 5);
      const alpha = Math.round(38 + Math.sin(idx * 2) * 6);
      const beta = Math.round(16 + Math.cos(idx * 1.5) * 4);
      const gamma = Math.round(6 + Math.sin(idx) * 2);
      const faa = (0.12 + Math.sin(idx) * 0.05).toFixed(3);
      rows.push([day, `${channelCount}-CH`, delta, theta, alpha, beta, gamma, faa]);
    });

    const csvContent = 'data:text/csv;charset=utf-8,' + rows.map((e) => e.join(',')).join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `gcs_eeg_${channelCount}ch_7day_history.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    if (onTriggerNotification) {
      onTriggerNotification(`Exported 7-Day ${channelCount}-Channel EEG Telemetry to SSD.`);
    }
  };

  return (
    <div className="dashboard-column">
      <div className="column-card">
        {/* Column Header */}
        <div className="column-header">
          <div className="column-header-title">
            <Activity size={18} />
            <span>Physiological Telemetry</span>
          </div>

          {/* Clickable Channel Count Selector (Requirement 4) */}
          <button
            className="channel-selector-btn"
            onClick={() => setShowChannelModal(true)}
            title="Configure Electrode Channel Density (8 to 1024+ Ch)"
          >
            <Layers size={13} style={{ color: 'var(--cyan-accent)' }} />
            <span>{channelCount}-CH</span>
            <ChevronDown size={12} />
          </button>
        </div>

        {/* Scrollable Container with Guaranteed Clearance & No Overlap */}
        <div className="scrollable-content">
          {/* Section: Autonomic Biomarkers */}
          <div className="section-divider-title">
            <Heart size={13} />
            <span>Autonomic Biomarkers</span>
          </div>

          {/* Biomarkers Cards */}
          <div className="vitals-cards-stack">
            {vitalConfig.map((item) => {
              const badge = getStatusBadge(item.data.status);
              const sparkColor = getSparkColor(item.data.status);

              return (
                <div key={item.key} className="vital-card">
                  <div className="vital-card-top">
                    <div className="vital-meta">
                      <div className="vital-icon-wrap">{item.icon}</div>
                      <span className="vital-name">{item.name}</span>
                    </div>
                    <span
                      className="vital-status-pill"
                      style={{
                        backgroundColor: badge.bg,
                        color: badge.color,
                        border: `1px solid ${badge.border}`
                      }}
                    >
                      {item.data.status}
                    </span>
                  </div>

                  <div className="vital-card-middle">
                    <div className="vital-value-wrap">
                      <span className="vital-val">{item.data.value}</span>
                      <span className="vital-unit">{item.unit}</span>
                    </div>
                    <span className="vital-target-range">{item.rangeText}</span>
                  </div>

                  {/* Trend History Sparkline */}
                  <Sparkline
                    data={item.data.history}
                    color={sparkColor}
                    height={26}
                    strokeWidth={1.8}
                  />
                </div>
              );
            })}
          </div>

          {/* Section: EEG Spectral Power Bands */}
          <div className="section-divider-title" style={{ marginTop: '0.8rem' }}>
            <Waves size={13} />
            <span>EEG Bands ({channelCount}-Ch HD Array)</span>
            <span style={{ fontSize: '10px', color: 'var(--cyan-accent)', marginLeft: 'auto' }}>
              Click to Filter 3D
            </span>
          </div>

          {/* Interactive EEG Bands - Linked with 3D Bubbles */}
          <div className="eeg-bands-stack">
            {bandsList.map((band) => {
              const isSelected = activeBandFilter && band.name.toLowerCase().includes(activeBandFilter.toLowerCase());

              return (
                <div
                  key={band.key}
                  className={`eeg-band-row ${isSelected ? 'active-band-filter' : ''}`}
                  onClick={() => {
                    const next = isSelected ? null : band.key;
                    onSelectBandFilter(next);
                  }}
                  title={`Click to link ${band.name} with 3D cortical regions`}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="eeg-band-top">
                    <span className="eeg-band-name">
                      <span
                        style={{
                          width: '8px',
                          height: '8px',
                          borderRadius: '50%',
                          backgroundColor: band.color,
                          display: 'inline-block',
                          boxShadow: `0 0 6px ${band.color}`
                        }}
                      />
                      {band.name}
                      <span className="eeg-band-freq">{band.freq}</span>
                    </span>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="eeg-band-val" style={{ color: band.color }}>
                        {band.data.power}%
                      </span>
                      <button
                        className="micro-inspect-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedWaveDetail(band);
                        }}
                        title="View Detailed Wave & 7-Day Retrospective"
                      >
                        <Calendar size={11} />
                      </button>
                    </div>
                  </div>

                  {/* Relative Power Progress Bar */}
                  <div className="eeg-bar-track">
                    <div
                      className="eeg-bar-fill"
                      style={{
                        width: `${Math.min(100, band.data.power * 2)}%`,
                        backgroundColor: band.color,
                        boxShadow: `0 0 8px ${band.color}66`
                      }}
                    />
                  </div>

                  {/* Sparkline for band dynamics */}
                  <Sparkline
                    data={band.data.history}
                    color={band.color}
                    height={18}
                    strokeWidth={1.5}
                  />
                </div>
              );
            })}
          </div>

          {/* Frontal Alpha Asymmetry (FAA Index) */}
          <div
            className="vital-card"
            style={{
              background: 'linear-gradient(135deg, rgba(56, 189, 248, 0.05), rgba(139, 92, 246, 0.05))',
              border: '1px solid rgba(56, 189, 248, 0.25)',
              marginTop: '0.4rem'
            }}
          >
            <div className="vital-card-top">
              <div className="vital-meta">
                <div className="vital-icon-wrap" style={{ color: 'var(--cyan-accent)' }}>
                  <Brain size={14} />
                </div>
                <span className="vital-name">Frontal Alpha Asymmetry</span>
              </div>
              <span
                className="vital-status-pill"
                style={{
                  backgroundColor: 'rgba(16, 185, 129, 0.15)',
                  color: 'var(--status-optimal)',
                  border: '1px solid rgba(16, 185, 129, 0.3)'
                }}
              >
                {eegBands.faa.status}
              </span>
            </div>
            <div className="vital-card-middle" style={{ marginTop: '0.2rem' }}>
              <div className="vital-value-wrap">
                <span className="vital-val" style={{ color: 'var(--cyan-accent)' }}>
                  {eegBands.faa.power > 0 ? `+${eegBands.faa.power}` : eegBands.faa.power}
                </span>
                <span className="vital-unit">ln(F4) - ln(F3)</span>
              </div>
              <span className="vital-target-range">Approach / Prosocial Bias</span>
            </div>
          </div>

          {/* 7-Day History Export Button (Requirement 4) */}
          <div style={{ marginTop: '0.6rem', paddingBottom: '0.5rem' }}>
            <button
              className="sim-toggle-btn"
              style={{ width: '100%', justifyContent: 'center', padding: '0.5rem' }}
              onClick={export7DayEEG}
            >
              <Download size={14} />
              <span>Save 7-Day EEG Telemetry to SSD (.CSV)</span>
            </button>
          </div>
        </div>
      </div>

      {/* Channel Density Selector Modal */}
      {showChannelModal && (
        <div className="modal-overlay" onClick={() => setShowChannelModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '520px' }}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Layers size={18} style={{ color: 'var(--cyan-accent)' }} />
                <span className="modal-title">Select Electrode Array Density</span>
              </div>
              <button
                onClick={() => setShowChannelModal(false)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
              >
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                Scale from commercial EEG to high-density neuroimaging and micro-electrode arrays (up to 1024+ channels):
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                {CHANNEL_PRESETS.map((preset) => (
                  <button
                    key={preset.channels}
                    className={`scenario-btn ${channelCount === preset.channels ? 'active' : ''}`}
                    onClick={() => {
                      onChangeChannels(preset.channels);
                      setShowChannelModal(false);
                      if (onTriggerNotification) {
                        onTriggerNotification(`Electrode density switched to ${preset.channels}-Channels (${preset.label}).`);
                      }
                    }}
                  >
                    <div style={{ fontWeight: 600, fontSize: '13px', color: channelCount === preset.channels ? 'var(--cyan-accent)' : '#fff' }}>
                      {preset.channels} Channels
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{preset.label}</div>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono' }}>
                      {preset.rate} • {preset.type}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 7-Day Detailed Wave Inspector Modal */}
      {selectedWaveDetail && (
        <div className="modal-overlay" onClick={() => setSelectedWaveDetail(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '600px' }}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Waves size={18} style={{ color: selectedWaveDetail.color }} />
                <span className="modal-title">{selectedWaveDetail.name} • 7-Day Retrospective</span>
              </div>
              <button
                onClick={() => setSelectedWaveDetail(null)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
              >
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                {selectedWaveDetail.desc} (Frequency: {selectedWaveDetail.freq})
              </div>

              {/* 7-Day Trend History Table */}
              <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '8px', padding: '12px', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--cyan-accent)', marginBottom: '8px' }}>
                  Daily Relative Spectral Power (%) over Past 7 Days:
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '6px', textAlign: 'center' }}>
                  {['Day -6', 'Day -5', 'Day -4', 'Day -3', 'Day -2', 'Yest.', 'Today'].map((day, idx) => {
                    const mockVal = Math.round(selectedWaveDetail.data.power + Math.sin(idx * 1.5) * 6);
                    return (
                      <div key={day} style={{ background: 'rgba(255,255,255,0.03)', padding: '8px 4px', borderRadius: '6px' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{day}</div>
                        <div style={{ fontSize: '14px', fontWeight: 700, color: selectedWaveDetail.color, marginTop: '4px' }}>
                          {mockVal}%
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
                <button
                  className="sim-toggle-btn"
                  onClick={() => {
                    export7DayEEG();
                    setSelectedWaveDetail(null);
                  }}
                >
                  <Download size={14} />
                  <span>Download Complete 7-Day Epoch (.CSV)</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
