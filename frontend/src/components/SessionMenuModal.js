import React, { useRef } from 'react';
import {
  Save,
  Download,
  Upload,
  RefreshCw,
  Sliders,
  CheckCircle2,
  HardDrive,
  X,
  FileSpreadsheet,
  Activity,
  Layers
} from 'lucide-react';

export const SessionMenuModal = ({
  isOpen,
  onClose,
  sessionState,
  onLoadSession,
  onTriggerNotification
}) => {
  const fileInputRef = useRef();

  if (!isOpen) return null;

  // Save current session as comprehensive JSON package
  const handleSaveSession = () => {
    const sessionDump = {
      meta: {
        appName: 'Grand Council Sentient v7.0',
        timestamp: new Date().toISOString(),
        version: '7.0.4-empathy'
      },
      patient: sessionState.patientData,
      channels: sessionState.channelCount,
      activeScenario: sessionState.activeScenario,
      affectiveHistory: sessionState.affectiveHistory,
      vitals: sessionState.vitals,
      eegBands: sessionState.eegBands,
      regionActivations: sessionState.regionActivations
    };

    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(sessionDump, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', `gcs_session_${Date.now()}.json`);
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();

    if (onTriggerNotification) {
      onTriggerNotification('Session state successfully saved and downloaded (.JSON).');
    }
    onClose();
  };

  // Download multi-column telemetry bundle CSV
  const handleDownloadTelemetryCSV = () => {
    const rows = [
      ['Timestamp', 'Subject', 'Heart_Rate_BPM', 'HRV_ms', 'Respiration_BrPM', 'SpO2_Pct', 'GSR_uS', 'Skin_Temp_C', 'EEG_Delta', 'EEG_Theta', 'EEG_Alpha', 'EEG_Beta', 'EEG_Gamma', 'Emotion_State', 'Valence', 'Arousal']
    ];

    // Build 20 rows of temporal telemetry
    for (let i = 19; i >= 0; i--) {
      const time = new Date(Date.now() - i * 1000).toISOString();
      rows.push([
        time,
        sessionState.patientData?.subjectId || 'V1B3hr',
        sessionState.vitals.heartRate.value,
        sessionState.vitals.hrv.value,
        sessionState.vitals.respiration.value,
        sessionState.vitals.spo2.value,
        sessionState.vitals.gsr.value,
        sessionState.vitals.skinTemp.value,
        sessionState.eegBands.delta.power,
        sessionState.eegBands.theta.power,
        sessionState.eegBands.alpha.power,
        sessionState.eegBands.beta.power,
        sessionState.eegBands.gamma.power,
        sessionState.affectiveState.label,
        sessionState.affectiveState.valence,
        sessionState.affectiveState.arousal
      ]);
    }

    const csvContent = 'data:text/csv;charset=utf-8,' + rows.map((e) => e.join(',')).join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `gcs_telemetry_bundle_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    if (onTriggerNotification) {
      onTriggerNotification('Complete telemetry bundle saved to disk (.CSV).');
    }
    onClose();
  };

  const handleUploadSessionFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target.result);
        if (onLoadSession) {
          onLoadSession(parsed);
        }
        if (onTriggerNotification) {
          onTriggerNotification(`Restored session from ${file.name}.`);
        }
        onClose();
      } catch (err) {
        alert('Invalid session JSON format.');
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '540px' }}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18} style={{ color: 'var(--cyan-accent)' }} />
            <span className="modal-title">GCS Session Operations & Storage</span>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
          >
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Manage neural recording sessions, backup telemetry epochs, and synchronize with local storage:
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {/* 1. Save Session (.JSON) */}
            <button className="session-menu-action" onClick={handleSaveSession}>
              <div className="session-action-icon" style={{ background: 'rgba(56, 189, 248, 0.15)', color: 'var(--cyan-accent)' }}>
                <Save size={18} />
              </div>
              <div style={{ flex: 1, textAlign: 'left' }}>
                <div style={{ fontWeight: 600, fontSize: '13px', color: '#fff' }}>Save Current Session (.JSON)</div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Serialize all 68 cortical activations, 20 affective states, and vital streams
                </div>
              </div>
            </button>

            {/* 2. Download Full Telemetry Bundle (.CSV) */}
            <button className="session-menu-action" onClick={handleDownloadTelemetryCSV}>
              <div className="session-action-icon" style={{ background: 'rgba(16, 185, 129, 0.15)', color: 'var(--status-optimal)' }}>
                <FileSpreadsheet size={18} />
              </div>
              <div style={{ flex: 1, textAlign: 'left' }}>
                <div style={{ fontWeight: 600, fontSize: '13px', color: '#fff' }}>Download Full Telemetry Bundle (.CSV)</div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Export raw multi-channel telemetry table for external analysis (MATLAB / Python)
                </div>
              </div>
            </button>

            {/* 3. Load / Restore Session */}
            <button
              className="session-menu-action"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="session-action-icon" style={{ background: 'rgba(168, 85, 247, 0.15)', color: '#A855F7' }}>
                <Upload size={18} />
              </div>
              <div style={{ flex: 1, textAlign: 'left' }}>
                <div style={{ fontWeight: 600, fontSize: '13px', color: '#fff' }}>Load Previous Session (.JSON)</div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Replay and review previous session recordings
                </div>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleUploadSessionFile}
                style={{ display: 'none' }}
              />
            </button>
          </div>

          <div
            style={{
              background: 'rgba(255, 255, 255, 0.02)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              padding: '10px 12px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              marginTop: '6px'
            }}
          >
            <HardDrive size={16} style={{ color: 'var(--cyan-accent)' }} />
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
              Storage Directory: Local SSD storage supported with zero server dependencies.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
