import { useState, useEffect, useRef, useCallback } from 'react';
import { CORTICAL_REGIONS_68, getStatusColor } from '../constants/brainRegions';

// 28-category taxonomy from GCS Empathy Progression
export const EMOTIONS = [
  { label: 'joy', icon: '😊', defaultValence: 0.82, defaultArousal: 0.65 },
  { label: 'contentment', icon: '😌', defaultValence: 0.75, defaultArousal: 0.35 },
  { label: 'flow', icon: '🌊', defaultValence: 0.85, defaultArousal: 0.70 },
  { label: 'focused', icon: '🎯', defaultValence: 0.60, defaultArousal: 0.68 },
  { label: 'empathy', icon: '🤝', defaultValence: 0.78, defaultArousal: 0.50 },
  { label: 'anxiety', icon: '😟', defaultValence: -0.62, defaultArousal: 0.82 },
  { label: 'overload', icon: '⚡', defaultValence: -0.75, defaultArousal: 0.88 },
  { label: 'calm', icon: '🌿', defaultValence: 0.70, defaultArousal: 0.28 }
];

export const PRESET_SCENARIOS = {
  HOMEOSTASIS: {
    id: 'HOMEOSTASIS',
    name: 'Homeostatic Baseline',
    desc: 'Normal autonomic equilibrium and balanced cortical baseline',
    emotion: 'contentment',
    hr: 70,
    hrv: 62,
    respiration: 14,
    spo2: 99,
    gsr: 2.1,
    temp: 36.6,
    bands: { delta: 18, theta: 22, alpha: 38, beta: 16, gamma: 6 },
    regionBase: 48,
    crisis: false
  },
  FLOW: {
    id: 'FLOW',
    name: 'Deep Flow & Synchrony',
    desc: 'Enhanced Alpha-Theta coherence and high vagal tone',
    emotion: 'flow',
    hr: 64,
    hrv: 78,
    respiration: 11,
    spo2: 99,
    gsr: 1.8,
    temp: 36.7,
    bands: { delta: 12, theta: 28, alpha: 44, beta: 12, gamma: 4 },
    regionBase: 52,
    crisis: false
  },
  COGNITIVE: {
    id: 'COGNITIVE',
    name: 'High Cognitive Load',
    desc: 'Intense problem solving, DLPFC gamma sync, moderate arousal',
    emotion: 'focused',
    hr: 82,
    hrv: 48,
    respiration: 17,
    spo2: 98,
    gsr: 3.4,
    temp: 36.5,
    bands: { delta: 10, theta: 15, alpha: 22, beta: 35, gamma: 18 },
    regionBase: 72,
    crisis: false
  },
  ANXIETY: {
    id: 'ANXIETY',
    name: 'Affective Stress / Anxiety',
    desc: 'Sympathetic arousal, amygdala-insula hyperactivation',
    emotion: 'anxiety',
    hr: 102,
    hrv: 24,
    respiration: 22,
    spo2: 97,
    gsr: 5.8,
    temp: 36.3,
    bands: { delta: 14, theta: 20, alpha: 14, beta: 38, gamma: 14 },
    regionBase: 82,
    crisis: false
  },
  CRISIS: {
    id: 'CRISIS',
    name: 'Crisis State (Hyper-arousal)',
    desc: 'Stage 5 Issue Notification: Autonomous safety intervention triggered',
    emotion: 'overload',
    hr: 128,
    hrv: 16,
    respiration: 28,
    spo2: 95,
    gsr: 8.4,
    temp: 36.1,
    bands: { delta: 25, theta: 12, alpha: 8, beta: 35, gamma: 20 },
    regionBase: 92,
    crisis: true
  }
};

// Seed 20 historical affective states for initial audit log
const generateInitialAffectiveHistory = () => {
  const history = [];
  const baseTime = Date.now() - 20 * 12000;
  const sampleEmotions = [
    { label: 'calm', icon: '🌿', val: 0.68, ar: 0.30 },
    { label: 'contentment', icon: '😌', val: 0.72, ar: 0.35 },
    { label: 'focused', icon: '🎯', val: 0.62, ar: 0.58 },
    { label: 'flow', icon: '🌊', val: 0.82, ar: 0.65 },
    { label: 'joy', icon: '😊', val: 0.80, ar: 0.60 }
  ];

  for (let i = 0; i < 20; i++) {
    const item = sampleEmotions[i % sampleEmotions.length];
    const timestamp = new Date(baseTime + i * 12000).toLocaleTimeString();
    history.push({
      timestamp,
      label: item.label,
      icon: item.icon,
      valence: item.val,
      arousal: item.arousal,
      confidence: 0.92,
      strength: 50 + (i % 5) * 3,
      crisisDetected: false,
      empathicResponse: {
        content: `Co-adaptive state verified at ${timestamp}.`,
        intensity: 'balanced',
        type: 'validation'
      }
    });
  }
  return history;
};

export function useBCIData() {
  const [connectionStatus, setConnectionStatus] = useState('CONNECTING');
  const [activeScenario, setActiveScenario] = useState('HOMEOSTASIS');
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [selectedHotspot, setSelectedHotspot] = useState(null);
  const [activeBandFilter, setActiveBandFilter] = useState(null); // 'alpha', 'beta', etc.
  const [channelCount, setChannelCount] = useState(64);
  const [networkPing, setNetworkPing] = useState(14);
  const [pushHz, setPushHz] = useState(3.0);
  const [notification, setNotification] = useState(null);

  // Patient Profile Metadata (Requirement 9)
  const [patientData, setPatientData] = useState({
    subjectId: 'V1B3hr',
    fullName: 'Subject V1B3hr',
    age: 32,
    sex: 'Male',
    handedness: 'Right-Handed',
    electrodeCap: '64-ch OpenBCI + E4 Physio',
    restingAlpha: '10.2 Hz',
    restingHR: '68 BPM',
    notes: 'GCS Empathy Synchronization Protocol v7.0. Normal baseline homeostatic range.'
  });

  // Affective & Empathy State
  const [affectiveState, setAffectiveState] = useState({
    label: 'contentment',
    icon: '😌',
    strength: 52,
    valence: 0.72,
    arousal: 0.38,
    confidence: 0.91,
    crisisDetected: false,
    empathicResponse: {
      content: 'Neural and physiological metrics are in homeostatic balance. Synchrony stable.',
      intensity: 'balanced',
      type: 'validation'
    }
  });

  // 20-State Affective History Log (Requirement 8)
  const [affectiveHistory, setAffectiveHistory] = useState(generateInitialAffectiveHistory);

  // Vitals State
  const [vitals, setVitals] = useState({
    heartRate: { value: 72, unit: 'BPM', min: 60, max: 90, status: 'Normal', history: [70, 71, 72, 73, 72, 71, 72, 74, 73, 72] },
    hrv: { value: 64, unit: 'ms', min: 45, max: 100, status: 'Normal', history: [62, 63, 65, 64, 66, 64, 63, 65, 64, 64] },
    respiration: { value: 14, unit: 'BrPM', min: 12, max: 20, status: 'Normal', history: [13, 14, 14, 15, 14, 13, 14, 14, 15, 14] },
    spo2: { value: 99, unit: '%', min: 95, max: 100, status: 'Normal', history: [99, 99, 98, 99, 99, 99, 98, 99, 99, 99] },
    gsr: { value: 2.2, unit: 'µS', min: 1.0, max: 4.5, status: 'Normal', history: [2.1, 2.2, 2.2, 2.3, 2.2, 2.1, 2.2, 2.2, 2.3, 2.2] },
    skinTemp: { value: 36.6, unit: '°C', min: 36.0, max: 37.2, status: 'Normal', history: [36.5, 36.6, 36.6, 36.5, 36.6, 36.6, 36.7, 36.6, 36.6, 36.6] }
  });

  // EEG Spectral Bands
  const [eegBands, setEegBands] = useState({
    delta: { power: 18, freq: '0.5 - 4 Hz', history: [17, 18, 19, 18, 17, 18, 19, 18] },
    theta: { power: 22, freq: '4 - 8 Hz', history: [21, 22, 23, 22, 22, 21, 22, 22] },
    alpha: { power: 38, freq: '8 - 12 Hz', history: [37, 38, 39, 38, 38, 39, 38, 38] },
    beta: { power: 16, freq: '12 - 30 Hz', history: [15, 16, 16, 17, 16, 15, 16, 16] },
    gamma: { power: 6, freq: '30 - 100 Hz', history: [5, 6, 6, 7, 6, 5, 6, 6] },
    faa: { power: 0.14, label: 'L/R Balance', status: 'Optimal' }
  });

  // 68 Cortical Region Activations
  const [regionActivations, setRegionActivations] = useState(() => {
    const initial = {};
    CORTICAL_REGIONS_68.forEach((region) => {
      const strength = 45 + Math.floor(Math.random() * 12);
      initial[region.nodeId] = {
        strength,
        status: getStatusColor(strength)
      };
    });
    return initial;
  });

  // Somatic Vitruvian Hotspots
  const [somaticHotspots, setSomaticHotspots] = useState({
    head: { id: 'head', name: 'Cephalic / Cortical', strength: 50, connected: 'EEG Gamma & Prefrontal Sync' },
    chest: { id: 'chest', name: 'Cardiopulmonary / Heart', strength: 48, connected: 'HR & HRV Vagal Tone' },
    lungs: { id: 'lungs', name: 'Thoracic / Respiratory', strength: 45, connected: 'Respiration & SpO2' },
    arms: { id: 'arms', name: 'Peripheral Electrodermal', strength: 42, connected: 'GSR / Skin Conductance' },
    core: { id: 'core', name: 'Enteric / Autonomic Core', strength: 49, connected: 'Sympathetic / Parasympathetic' },
    legs: { id: 'legs', name: 'Somatic Motor / Peripheral', strength: 40, connected: 'Motor Striatum & Reflex' }
  });

  const wsRef = useRef(null);
  const stepRef = useRef(0);
  const lastHistoryLoggedRef = useRef(0);

  const triggerNotification = useCallback((msg) => {
    setNotification(msg);
  }, []);

  // Dismiss notification after 3.5s
  useEffect(() => {
    if (notification) {
      const t = setTimeout(() => setNotification(null), 3500);
      return () => clearTimeout(t);
    }
  }, [notification]);

  // WebSocket Connection Handler
  useEffect(() => {
    let reconnectTimeout = null;
    let isSubscribed = true;

    const connect = () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws');
        wsRef.current = ws;

        ws.onopen = () => {
          if (!isSubscribed) return;
          setConnectionStatus('CONNECTED');
          setNetworkPing(Math.floor(10 + Math.random() * 8));
          triggerNotification('Hardware Sync Engaged: Connected to ws://localhost:8000/ws');
        };

        ws.onmessage = (event) => {
          if (!isSubscribed) return;
          try {
            const data = JSON.parse(event.data);
            if (data.affective) {
              const newAff = {
                label: data.affective.label || 'contentment',
                icon: data.affective.icon || '😌',
                strength: data.affective.strength || 50,
                valence: data.affective.valence ?? 0.7,
                arousal: data.affective.arousal ?? 0.35,
                confidence: data.affective.confidence ?? 0.9,
                crisisDetected: data.crisis_detected ?? false,
                empathicResponse: data.empathic_response || { content: 'Sensing state.', intensity: 'balanced' }
              };
              setAffectiveState(newAff);

              // Append to history buffer if changed or interval elapsed
              setAffectiveHistory((prev) => [
                ...prev.slice(-29), // keep up to 30 states
                {
                  ...newAff,
                  timestamp: new Date().toLocaleTimeString()
                }
              ]);
            }
          } catch (e) {
            console.error('WebSocket parse error:', e);
          }
        };

        ws.onclose = () => {
          if (!isSubscribed) return;
          setConnectionStatus('SIMULATED');
          reconnectTimeout = setTimeout(connect, 5000);
        };

        ws.onerror = () => {
          if (ws) ws.close();
        };
      } catch (e) {
        setConnectionStatus('SIMULATED');
        reconnectTimeout = setTimeout(connect, 5000);
      }
    };

    connect();

    return () => {
      isSubscribed = false;
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, [triggerNotification]);

  // Update physiological & neural state based on scenario or live stream
  useEffect(() => {
    const interval = setInterval(() => {
      stepRef.current += 1;
      const step = stepRef.current;
      const currentPreset = PRESET_SCENARIOS[activeScenario] || PRESET_SCENARIOS.HOMEOSTASIS;

      // Realistic biological oscillations
      const rsaWave = Math.sin(step * 0.4);
      const alphaWave = Math.cos(step * 0.25);
      const noise = (Math.random() - 0.5) * 2;

      // Instantaneous vital values
      const currentHR = Math.round(currentPreset.hr + rsaWave * 3.5 + noise * 1.5);
      const currentHRV = Math.max(10, Math.round(currentPreset.hrv - rsaWave * 4 + noise * 2));
      const currentResp = Math.round(currentPreset.respiration + Math.sin(step * 0.2) * 1.2);
      const currentSpO2 = Math.min(100, Math.max(94, Math.round(currentPreset.spo2 + (Math.random() > 0.8 ? -1 : 0))));
      const currentGSR = parseFloat((currentPreset.gsr + Math.sin(step * 0.15) * 0.3 + noise * 0.1).toFixed(2));
      const currentTemp = parseFloat((currentPreset.temp + Math.sin(step * 0.05) * 0.08).toFixed(1));

      const pushHistory = (arr, val) => [...arr.slice(-15), val];

      setVitals((prev) => ({
        heartRate: {
          ...prev.heartRate,
          value: currentHR,
          status: currentHR > 110 ? 'Danger' : currentHR > 85 ? 'Elevated' : 'Normal',
          history: pushHistory(prev.heartRate.history, currentHR)
        },
        hrv: {
          ...prev.hrv,
          value: currentHRV,
          status: currentHRV < 25 ? 'Danger' : currentHRV < 40 ? 'Attention' : 'Normal',
          history: pushHistory(prev.hrv.history, currentHRV)
        },
        respiration: {
          ...prev.respiration,
          value: currentResp,
          status: currentResp > 24 ? 'Danger' : currentResp > 18 ? 'Elevated' : 'Normal',
          history: pushHistory(prev.respiration.history, currentResp)
        },
        spo2: {
          ...prev.spo2,
          value: currentSpO2,
          status: currentSpO2 < 95 ? 'Danger' : currentSpO2 < 97 ? 'Attention' : 'Normal',
          history: pushHistory(prev.spo2.history, currentSpO2)
        },
        gsr: {
          ...prev.gsr,
          value: currentGSR,
          status: currentGSR > 6.0 ? 'Danger' : currentGSR > 4.0 ? 'Elevated' : 'Normal',
          history: pushHistory(prev.gsr.history, currentGSR)
        },
        skinTemp: {
          ...prev.skinTemp,
          value: currentTemp,
          status: currentTemp > 37.5 || currentTemp < 35.8 ? 'Attention' : 'Normal',
          history: pushHistory(prev.skinTemp.history, currentTemp)
        }
      }));

      // Update EEG power bands
      const b = currentPreset.bands;
      const dVal = Math.round(b.delta + Math.sin(step * 0.1) * 2);
      const tVal = Math.round(b.theta + Math.cos(step * 0.12) * 2);
      const aVal = Math.round(b.alpha + alphaWave * 3);
      const bVal = Math.round(b.beta + Math.sin(step * 0.3) * 2);
      const gVal = Math.round(b.gamma + Math.cos(step * 0.4) * 1.5);
      const faaVal = parseFloat((0.15 + Math.sin(step * 0.2) * 0.08).toFixed(2));

      setEegBands((prev) => ({
        delta: { ...prev.delta, power: dVal, history: pushHistory(prev.delta.history, dVal) },
        theta: { ...prev.theta, power: tVal, history: pushHistory(prev.theta.history, tVal) },
        alpha: { ...prev.alpha, power: aVal, history: pushHistory(prev.alpha.history, aVal) },
        beta: { ...prev.beta, power: bVal, history: pushHistory(prev.beta.history, bVal) },
        gamma: { ...prev.gamma, power: gVal, history: pushHistory(prev.gamma.history, gVal) },
        faa: { ...prev.faa, power: faaVal }
      }));

      // Update 68 Cortical Regions
      setRegionActivations((prev) => {
        const next = { ...prev };
        CORTICAL_REGIONS_68.forEach((region) => {
          let base = currentPreset.regionBase;
          if (activeScenario === 'COGNITIVE' && region.lobe === 'Frontal Lobe') {
            base += 12;
          } else if (activeScenario === 'FLOW' && (region.key.includes('cingulate') || region.key.includes('precuneus'))) {
            base += 8;
          } else if (activeScenario === 'ANXIETY' && (region.key.includes('insula') || region.key.includes('orbitofrontal'))) {
            base += 18;
          } else if (activeScenario === 'CRISIS') {
            base += Math.random() > 0.4 ? 14 : 6;
          }

          const localMod = Math.sin(step * 0.2 + region.nodeId * 0.3) * 6;
          const strength = Math.min(100, Math.max(20, Math.round(base + localMod + noise)));
          next[region.nodeId] = {
            strength,
            status: getStatusColor(strength)
          };
        });
        return next;
      });

      // Update Vitruvian hotspots
      setSomaticHotspots((prev) => ({
        head: { ...prev.head, strength: Math.min(100, currentPreset.regionBase + (activeScenario === 'COGNITIVE' ? 12 : 0)) },
        chest: { ...prev.chest, strength: Math.min(100, Math.round((currentHR / 120) * 85)) },
        lungs: { ...prev.lungs, strength: Math.min(100, Math.round((currentResp / 26) * 75)) },
        arms: { ...prev.arms, strength: Math.min(100, Math.round((currentGSR / 8.0) * 90)) },
        core: { ...prev.core, strength: Math.min(100, Math.round((100 - currentHRV) * 0.85)) },
        legs: { ...prev.legs, strength: Math.min(100, Math.round(35 + (currentHR > 90 ? 25 : 5))) }
      }));

      // When in simulation, update affective engine
      if (connectionStatus !== 'CONNECTED') {
        const emot = EMOTIONS.find((e) => e.label === currentPreset.emotion) || EMOTIONS[0];
        let empathicMsg = 'Neural metrics stable. Cognitive synchrony verified.';
        if (activeScenario === 'FLOW') {
          empathicMsg = 'Sustained theta-alpha coherence detected. Deep contemplative flow engaged.';
        } else if (activeScenario === 'COGNITIVE') {
          empathicMsg = 'High prefrontal gamma activity detected. Complex problem-solving state identified.';
        } else if (activeScenario === 'ANXIETY') {
          empathicMsg = 'Heightened autonomic arousal and insula activation. Recommending paced diaphragm breathing.';
        } else if (activeScenario === 'CRISIS') {
          empathicMsg = 'CRITICAL ALERT: Hyper-arousal threshold exceeded. Engaging GCS Stage 5 autonomous stabilization.';
        }

        const newAffectiveState = {
          label: emot.label,
          icon: emot.icon,
          strength: currentPreset.regionBase,
          valence: parseFloat((emot.defaultValence + Math.sin(step * 0.1) * 0.05).toFixed(2)),
          arousal: parseFloat((emot.defaultArousal + Math.cos(step * 0.1) * 0.05).toFixed(2)),
          confidence: 0.91,
          crisisDetected: currentPreset.crisis,
          empathicResponse: {
            content: empathicMsg,
            intensity: currentPreset.crisis ? 'critical' : activeScenario === 'ANXIETY' ? 'high' : 'balanced',
            type: currentPreset.crisis ? 'crisis_intervention' : 'validation'
          }
        };

        setAffectiveState(newAffectiveState);

        // Every ~15 steps, record to historical log
        if (step - lastHistoryLoggedRef.current >= 12) {
          lastHistoryLoggedRef.current = step;
          setAffectiveHistory((prev) => [
            ...prev.slice(-24), // retain at least 25 states
            {
              ...newAffectiveState,
              timestamp: new Date().toLocaleTimeString()
            }
          ]);
        }
      }
    }, 1000 / pushHz);

    return () => clearInterval(interval);
  }, [activeScenario, connectionStatus, pushHz]);

  const selectScenario = useCallback((scenarioKey) => {
    if (PRESET_SCENARIOS[scenarioKey]) {
      setActiveScenario(scenarioKey);
      triggerNotification(`Engaged BCI Scenario: ${PRESET_SCENARIOS[scenarioKey].name}`);
    }
  }, [triggerNotification]);

  const restoreSession = useCallback((sessionDump) => {
    if (sessionDump.patient) setPatientData(sessionDump.patient);
    if (sessionDump.channels) setChannelCount(sessionDump.channels);
    if (sessionDump.activeScenario) setActiveScenario(sessionDump.activeScenario);
    if (sessionDump.affectiveHistory) setAffectiveHistory(sessionDump.affectiveHistory);
    if (sessionDump.vitals) setVitals(sessionDump.vitals);
    if (sessionDump.eegBands) setEegBands(sessionDump.eegBands);
    if (sessionDump.regionActivations) setRegionActivations(sessionDump.regionActivations);
    triggerNotification('Session snapshot successfully restored.');
  }, [triggerNotification]);

  return {
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
  };
}
