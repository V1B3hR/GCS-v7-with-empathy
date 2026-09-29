import React, { useState } from 'react';
import './App.css';
import { useBCIData, PRESET_SCENARIOS } from './hooks/useBCIData';
import { HeaderBar } from './components/HeaderBar';
import { PhysiologicalColumn } from './components/PhysiologicalColumn';
import { BrainCanvas } from './components/BrainCanvas';
import { VitruvianColumn } from './components/VitruvianColumn';
import { SimulationModal } from './components/SimulationModal';
import { PatientProfileModal } from './components/PatientProfileModal';
import { SessionMenuModal } from './components/SessionMenuModal';
import { Sparkles, CheckCircle2 } from 'lucide-react';

function App() {
  const [isSimModalOpen, setIsSimModalOpen] = useState(false);
  const [isPatientModalOpen, setIsPatientModalOpen] = useState(false);
  const [isSessionMenuOpen, setIsSessionMenuOpen] = useState(false);

  const {
    connectionStatus,
    networkPing,
    pushHz,
    activeScenario,
    selectScenario,
    channelCount,
    setChannelCount,
    activeBandFilter,
    setActiveBandFilter,
    patientData,
    setPatientData,
    notification,
    triggerNotification,
    affectiveState,
    affectiveHistory,
    vitals,
    eegBands,
    regionActivations,
    somaticHotspots,
    selectedRegion,
    setSelectedRegion,
    selectedHotspot,
    setSelectedHotspot,
    restoreSession
  } = useBCIData();

  const activeScenarioObj = PRESET_SCENARIOS[activeScenario];

  return (
    <div className="app-container">
      {/* Action Feedback Toast Notification */}
      {notification && (
        <div className="app-toast-notification">
          <CheckCircle2 size={16} style={{ color: 'var(--status-optimal)' }} />
          <span>{notification}</span>
        </div>
      )}

      {/* 1. Top Row User Info & System Status Header */}
      <HeaderBar
        connectionStatus={connectionStatus}
        networkPing={networkPing}
        pushHz={pushHz}
        channelCount={channelCount}
        patientData={patientData}
        activeScenarioName={activeScenarioObj?.name}
        onOpenSimModal={() => setIsSimModalOpen(true)}
        onOpenPatientModal={() => setIsPatientModalOpen(true)}
        onOpenSessionMenu={() => setIsSessionMenuOpen(true)}
      />

      {/* 2. Main 1440p 12-Column Responsive Dashboard Grid */}
      <main className="dashboard-grid">
        {/* Left Column: Physiological & Neural Readings (Requirement 3: No overlap, Requirement 4: Clickable channels & 7-day history) */}
        <PhysiologicalColumn
          vitals={vitals}
          eegBands={eegBands}
          channelCount={channelCount}
          onChangeChannels={(ch) => setChannelCount(ch)}
          activeBandFilter={activeBandFilter}
          onSelectBandFilter={setActiveBandFilter}
          onTriggerNotification={triggerNotification}
        />

        {/* Center Column: 3D Focal Brain Atlas (Requirement 1, 2, 5, 7: Smaller bubbles, dissolve, zoom focus, right-click actions) */}
        <BrainCanvas
          regionActivations={regionActivations}
          selectedRegion={selectedRegion}
          activeBandFilter={activeBandFilter}
          onSelectRegion={(reg) => {
            setSelectedRegion(reg);
            if (reg) {
              // Highlight corresponding wave in EEG panel
              setActiveBandFilter(reg.defaultBand);
            }
          }}
          onTriggerNotification={triggerNotification}
        />

        {/* Right Column: Somatic Biometric Map (2D/3D Organ View) & Empathy Engine (Requirement 6, 8: Detailed organs, right-click, 20-state history) */}
        <VitruvianColumn
          affectiveState={affectiveState}
          affectiveHistory={affectiveHistory}
          somaticHotspots={somaticHotspots}
          selectedHotspot={selectedHotspot}
          onSelectHotspot={setSelectedHotspot}
          onTriggerNotification={triggerNotification}
        />
      </main>

      {/* 3. Scenario & Emulation Controller Modal */}
      <SimulationModal
        isOpen={isSimModalOpen}
        onClose={() => setIsSimModalOpen(false)}
        activeScenario={activeScenario}
        onSelectScenario={selectScenario}
      />

      {/* 4. Patient Profile & Upload Records Modal (Requirement 9) */}
      <PatientProfileModal
        isOpen={isPatientModalOpen}
        onClose={() => setIsPatientModalOpen(false)}
        patientData={patientData}
        onSavePatientData={setPatientData}
        onTriggerNotification={triggerNotification}
      />

      {/* 5. System Session Menu Modal (Requirement 10) */}
      <SessionMenuModal
        isOpen={isSessionMenuOpen}
        onClose={() => setIsSessionMenuOpen(false)}
        sessionState={{
          patientData,
          channelCount,
          activeScenario,
          affectiveState,
          affectiveHistory,
          vitals,
          eegBands,
          regionActivations
        }}
        onLoadSession={restoreSession}
        onTriggerNotification={triggerNotification}
      />
    </div>
  );
}

export default App;
