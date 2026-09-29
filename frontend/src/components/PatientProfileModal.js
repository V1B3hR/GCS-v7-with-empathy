import React, { useState } from 'react';
import { User, Upload, Download, Save, X, FileText, CheckCircle2 } from 'lucide-react';

export const PatientProfileModal = ({
  isOpen,
  onClose,
  patientData,
  onSavePatientData,
  onTriggerNotification
}) => {
  const [profile, setProfile] = useState({
    subjectId: patientData?.subjectId || 'V1B3hr',
    fullName: patientData?.fullName || 'Subject V1B3hr',
    age: patientData?.age || 32,
    sex: patientData?.sex || 'Male',
    handedness: patientData?.handedness || 'Right-Handed',
    electrodeCap: patientData?.electrodeCap || '64-ch OpenBCI + E4 Physio',
    restingAlpha: patientData?.restingAlpha || '10.2 Hz',
    restingHR: patientData?.restingHR || '68 BPM',
    notes: patientData?.notes || 'GCS Empathy Synchronization Protocol v7.0. Normal baseline homeostatic range.'
  });

  if (!isOpen) return null;

  const handleChange = (field, val) => {
    setProfile((prev) => ({ ...prev, [field]: val }));
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target.result);
        setProfile((prev) => ({
          ...prev,
          ...json
        }));
        if (onTriggerNotification) {
          onTriggerNotification(`Loaded patient profile from file: ${file.name}`);
        }
      } catch (err) {
        alert('Could not parse JSON profile file. Please ensure valid JSON.');
      }
    };
    reader.readAsText(file);
  };

  const handleSave = () => {
    onSavePatientData(profile);
    if (onTriggerNotification) {
      onTriggerNotification(`Updated details for patient ${profile.subjectId}.`);
    }
    onClose();
  };

  const handleExportProfile = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(profile, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `gcs_patient_${profile.subjectId}_profile.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();

    if (onTriggerNotification) {
      onTriggerNotification(`Exported patient profile for ${profile.subjectId}.`);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '580px' }}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <User size={18} style={{ color: 'var(--cyan-accent)' }} />
            <span className="modal-title">Subject & Patient Records</span>
          </div>
          <button
            onClick={onClose}
            style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
          >
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {/* File Upload / Import Strip (Requirement 9) */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              background: 'rgba(56, 189, 248, 0.08)',
              border: '1px dashed rgba(56, 189, 248, 0.35)',
              borderRadius: '8px',
              padding: '10px 14px'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Upload size={16} style={{ color: 'var(--cyan-accent)' }} />
              <div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: '#fff' }}>Upload Patient File (.JSON / .CSV)</div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Load historical calibrations or subject records</div>
              </div>
            </div>

            <label className="sim-toggle-btn" style={{ cursor: 'pointer', margin: 0, padding: '0.35rem 0.75rem' }}>
              <span>Choose File</span>
              <input type="file" accept=".json,.csv" onChange={handleFileUpload} style={{ display: 'none' }} />
            </label>
          </div>

          {/* Form Fields */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '6px' }}>
            <div className="input-group">
              <label className="input-label">Subject ID / Code</label>
              <input
                className="custom-input"
                value={profile.subjectId}
                onChange={(e) => handleChange('subjectId', e.target.value)}
              />
            </div>

            <div className="input-group">
              <label className="input-label">Full Clinical Name</label>
              <input
                className="custom-input"
                value={profile.fullName}
                onChange={(e) => handleChange('fullName', e.target.value)}
              />
            </div>

            <div className="input-group">
              <label className="input-label">Age</label>
              <input
                className="custom-input"
                type="number"
                value={profile.age}
                onChange={(e) => handleChange('age', e.target.value)}
              />
            </div>

            <div className="input-group">
              <label className="input-label">Biological Sex</label>
              <select
                className="custom-input"
                value={profile.sex}
                onChange={(e) => handleChange('sex', e.target.value)}
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div className="input-group">
              <label className="input-label">Handedness</label>
              <select
                className="custom-input"
                value={profile.handedness}
                onChange={(e) => handleChange('handedness', e.target.value)}
              >
                <option value="Right-Handed">Right-Handed</option>
                <option value="Left-Handed">Left-Handed</option>
                <option value="Ambidextrous">Ambidextrous</option>
              </select>
            </div>

            <div className="input-group">
              <label className="input-label">Electrode Cap / Array</label>
              <input
                className="custom-input"
                value={profile.electrodeCap}
                onChange={(e) => handleChange('electrodeCap', e.target.value)}
              />
            </div>
          </div>

          <div className="input-group" style={{ marginTop: '4px' }}>
            <label className="input-label">Clinical Baseline Notes</label>
            <textarea
              className="custom-input"
              rows={3}
              value={profile.notes}
              onChange={(e) => handleChange('notes', e.target.value)}
              style={{ resize: 'none' }}
            />
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px' }}>
            <button
              className="hud-btn"
              onClick={handleExportProfile}
              style={{ padding: '0.45rem 0.85rem' }}
            >
              <Download size={13} />
              <span>Export Record (.JSON)</span>
            </button>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                className="hud-btn"
                onClick={onClose}
                style={{ padding: '0.45rem 0.85rem' }}
              >
                Cancel
              </button>
              <button
                className="sim-toggle-btn"
                onClick={handleSave}
                style={{ padding: '0.45rem 1rem' }}
              >
                <Save size={14} />
                <span>Save Profile</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
