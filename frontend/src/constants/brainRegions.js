/**
 * Desikan-Killiany Cortical Atlas Definition (68 Cortical Regions)
 * 34 regions per hemisphere matching GCS v7.0 cortical_nodes: 68
 * Enhanced with Region-Specific Neuromodulation Actions & Clinical Protocols
 */

export const LOBES = {
  FRONTAL: 'Frontal Lobe',
  PARIETAL: 'Parietal Lobe',
  TEMPORAL: 'Temporal Lobe',
  OCCIPITAL: 'Occipital Lobe',
  CINGULATE: 'Cingulate Cortex',
  INSULAR: 'Insular Cortex',
  SUBCORTICAL: 'Subcortical / Limbic',
  CEREBELLUM: 'Cerebellum',
  BRAINSTEM: 'Brainstem'
};

export const DESIKAN_REGIONS_RAW = [
  {
    key: 'superiorfrontal',
    name: 'Superior Frontal Gyrus',
    lobe: LOBES.FRONTAL,
    brodmann: 'BA 6, 8, 9',
    function: 'Executive control, higher cognitive processing, working memory',
    basePos: [0.65, 1.25, 0.45],
    defaultBand: 'Beta',
    actions: [
      { id: 'focus_boost', label: 'Optimize Executive Focus', desc: 'Deliver 20 Hz tACS to frontoparietal network', type: 'stimulation' },
      { id: 'admin_nootropic', label: 'Administrate Nootropic Agent', desc: 'Infuse cognitive enhancer (Citicoline / Modafinil protocol)', type: 'medicine' },
      { id: 'fatigue_relief', label: 'Alleviate Mental Fatigue', desc: 'Induce theta-alpha rest rhythm', type: 'stress' }
    ]
  },
  {
    key: 'rostralmiddlefrontal',
    name: 'Rostral Middle Frontal Gyrus (DLPFC)',
    lobe: LOBES.FRONTAL,
    brodmann: 'BA 9, 10, 46',
    function: 'Dorsolateral prefrontal cortex, emotional regulation, deliberate reasoning, decision making',
    basePos: [1.15, 0.95, 0.95],
    defaultBand: 'Gamma',
    actions: [
      { id: 'dlpfc_tms', label: 'High-Frequency rTMS (10 Hz)', desc: 'Prefrontal stimulation for decision clarity & mood lift', type: 'stimulation' },
      { id: 'admin_dopamine', label: 'Administer Dopaminergic Agonist', desc: 'Enhance working memory capacity', type: 'medicine' },
      { id: 'lower_stress', label: 'Lower Prefrontal Stress Load', desc: 'Dampen hyperactive executive loops', type: 'stress' }
    ]
  },
  {
    key: 'caudalmiddlefrontal',
    name: 'Caudal Middle Frontal Gyrus',
    lobe: LOBES.FRONTAL,
    brodmann: 'BA 6, 8',
    function: 'Frontal eye fields, premotor coordination, selective visual attention',
    basePos: [1.35, 1.05, 0.25],
    defaultBand: 'Beta',
    actions: [
      { id: 'vis_attention', label: 'Enhance Saccadic Attention', desc: 'Frontal eye field sync protocol', type: 'stimulation' },
      { id: 'admin_relax', label: 'Administer Mild Sedative', desc: 'Reduce oculomotor tension', type: 'medicine' }
    ]
  },
  {
    key: 'parsopercularis',
    name: 'Pars Opercularis (Broca Area)',
    lobe: LOBES.FRONTAL,
    brodmann: 'BA 44',
    function: 'Speech production, syntactic processing, mirror neuron system',
    basePos: [1.55, 0.45, 0.65],
    defaultBand: 'Beta',
    actions: [
      { id: 'speech_sync', label: 'Broca Speech Facilitation', desc: 'Anodal tDCS to enhance verbal articulation', type: 'stimulation' },
      { id: 'admin_neuroprotect', label: 'Administer Neuroprotective Agent', desc: 'Perfusion support for language networks', type: 'medicine' }
    ]
  },
  {
    key: 'parstriangularis',
    name: 'Pars Triangularis',
    lobe: LOBES.FRONTAL,
    brodmann: 'BA 45',
    function: 'Verbal communication, verbal fluency, syntactic processing',
    basePos: [1.60, 0.40, 1.05],
    defaultBand: 'Alpha',
    actions: [
      { id: 'verbal_fluency', label: 'Stimulate Semantic Retrieval', desc: 'Gamma burst entrainment', type: 'stimulation' }
    ]
  },
  {
    key: 'parsorbitalis',
    name: 'Pars Orbitalis',
    lobe: LOBES.FRONTAL,
    brodmann: 'BA 47',
    function: 'Language syntax, affective valuation, response inhibition',
    basePos: [1.45, -0.15, 1.25],
    defaultBand: 'Theta',
    actions: [
      { id: 'inhibition_control', label: 'Reinforce Impulse Inhibition', desc: 'Orbitofrontal stability protocol', type: 'stimulation' },
      { id: 'admin_serotonin', label: 'Administer Serotonergic Modulator', desc: 'Calibrate social valuation', type: 'medicine' }
    ]
  },
  {
    key: 'lateralorbitofrontal',
    name: 'Lateral Orbitofrontal Cortex',
    lobe: LOBES.FRONTAL,
    brodmann: 'BA 11, 47',
    function: 'Reward appraisal, punishment expectation, empathetic decision making',
    basePos: [1.05, -0.35, 1.15],
    defaultBand: 'Theta',
    actions: [
      { id: 'reward_rebalance', label: 'Reward Expectation Rebalance', desc: 'Targeted ultrasound neuromodulation (TUS)', type: 'stimulation' },
      { id: 'lower_anxiety', label: 'Lower Anticipatory Anxiety', desc: 'Vagal afferent sync to OFC', type: 'stress' }
    ]
  },
  {
    key: 'medialorbitofrontal',
    name: 'Medial Orbitofrontal Cortex',
    lobe: LOBES.FRONTAL,
    brodmann: 'BA 10, 11',
    function: 'Pleasure processing, positive valence, social empathy',
    basePos: [0.35, -0.35, 1.05],
    defaultBand: 'Theta',
    actions: [
      { id: 'empathy_burst', label: 'Engage Empathic Coherence', desc: '40 Hz entrainment for prosocial alignment', type: 'stimulation' },
      { id: 'admin_oxytocin', label: 'Administer Intranasal Oxytocin', desc: 'Prosocial neurochemical modulation', type: 'medicine' }
    ]
  },
  {
    key: 'precentral',
    name: 'Precentral Gyrus (Primary Motor)',
    lobe: LOBES.FRONTAL,
    brodmann: 'BA 4',
    function: 'Primary somatic motor cortex, voluntary muscle execution',
    basePos: [1.35, 0.95, -0.25],
    defaultBand: 'Beta',
    actions: [
      { id: 'motor_rehab', label: 'Motor Imagery Biofeedback', desc: 'Mu-rhythm desynchronization training', type: 'stimulation' },
      { id: 'admin_spasmolytic', label: 'Administer Muscle Relaxant', desc: 'Relieve somatic muscular spasticity', type: 'medicine' },
      { id: 'lower_tremor', label: 'Suppress Motor Overload', desc: 'Dampen corticospinal hyperactivity', type: 'stress' }
    ]
  },
  {
    key: 'paracentral',
    name: 'Paracentral Lobule',
    lobe: LOBES.FRONTAL,
    brodmann: 'BA 4, 5',
    function: 'Motor and sensory innervation of lower extremities and pelvic organs',
    basePos: [0.35, 1.15, -0.35],
    defaultBand: 'Beta',
    actions: [
      { id: 'lower_limb_sync', label: 'Lower Limb Motor Calibration', desc: 'Somatosensory cortical pacing', type: 'stimulation' }
    ]
  },
  {
    key: 'frontalpole',
    name: 'Frontal Pole (Frontopolar)',
    lobe: LOBES.FRONTAL,
    brodmann: 'BA 10',
    function: 'Metacognition, branching mental tasks, future planning, self-awareness',
    basePos: [0.65, 0.45, 1.75],
    defaultBand: 'Gamma',
    actions: [
      { id: 'metacognitive_probe', label: 'Metacognitive Depth Probe', desc: 'High-frequency gamma binding probe', type: 'stimulation' },
      { id: 'admin_adaptogen', label: 'Administer Adaptogenic Compound', desc: 'Support executive cognitive resilience', type: 'medicine' }
    ]
  },
  {
    key: 'postcentral',
    name: 'Postcentral Gyrus (Primary Somatosensory)',
    lobe: LOBES.PARIETAL,
    brodmann: 'BA 1, 2, 3',
    function: 'Tactile perception, proprioception, visceral bodily sensations',
    basePos: [1.35, 0.85, -0.65],
    defaultBand: 'Alpha',
    actions: [
      { id: 'sensory_gate', label: 'Somatosensory Gating Protocol', desc: 'Reduce tactile hypersensitivity', type: 'stimulation' },
      { id: 'admin_analgesic', label: 'Administer Endogenous Analgesic', desc: 'Targeted central pain modulation', type: 'medicine' }
    ]
  },
  {
    key: 'superiorparietal',
    name: 'Superior Parietal Lobule',
    lobe: LOBES.PARIETAL,
    brodmann: 'BA 7',
    function: 'Visuospatial attention, sensory integration, body schema orientation',
    basePos: [0.85, 1.05, -1.05],
    defaultBand: 'Alpha',
    actions: [
      { id: 'spatial_orient', label: 'Realign Body Schema Coordinates', desc: 'Parietal visual integration stimulus', type: 'stimulation' }
    ]
  },
  {
    key: 'inferiorparietal',
    name: 'Inferior Parietal Cortex (TPJ)',
    lobe: LOBES.PARIETAL,
    brodmann: 'BA 39, 40',
    function: 'Theory of mind, empathetic perspective taking, multimodal synthesis',
    basePos: [1.55, 0.55, -1.15],
    defaultBand: 'Alpha',
    actions: [
      { id: 'tom_alignment', label: 'Enhance Theory-of-Mind Network', desc: 'TPJ empathy resonance protocol', type: 'stimulation' },
      { id: 'lower_egocentric', label: 'Reduce Egocentric Bias', desc: 'Facilitate compassionate connection', type: 'stress' }
    ]
  },
  {
    key: 'supramarginal',
    name: 'Supramarginal Gyrus',
    lobe: LOBES.PARIETAL,
    brodmann: 'BA 40',
    function: 'Phonological processing, empathy for others pain, tactile interpretation',
    basePos: [1.60, 0.65, -0.75],
    defaultBand: 'Alpha',
    actions: [
      { id: 'pain_empathy_mod', label: 'Compassionate Pain Attenuation', desc: 'Modulate affective distress response', type: 'stress' }
    ]
  },
  {
    key: 'precuneus',
    name: 'Precuneus (Default Mode Network Hub)',
    lobe: LOBES.PARIETAL,
    brodmann: 'BA 7, 31',
    function: 'Self-referential thought, autobiographical memory, consciousness state',
    basePos: [0.35, 0.85, -1.35],
    defaultBand: 'Theta',
    actions: [
      { id: 'dmn_quieting', label: 'DMN Quieting (Flow Induction)', desc: 'Inhibit rumination and promote presence', type: 'stimulation' },
      { id: 'admin_mindful', label: 'Administer Mindfulness Enhancer', desc: 'Promote centered ego-dissolution', type: 'medicine' },
      { id: 'lower_rumination', label: 'Halt Negative Thought Loops', desc: 'Anti-rumination alpha burst', type: 'stress' }
    ]
  },
  {
    key: 'superiortemporal',
    name: 'Superior Temporal Gyrus',
    lobe: LOBES.TEMPORAL,
    brodmann: 'BA 22, 41, 42',
    function: 'Auditory cortex, language comprehension, voice prosody recognition',
    basePos: [1.65, 0.15, -0.25],
    defaultBand: 'Beta',
    actions: [
      { id: 'auditory_prosody', label: 'Tone Prosody Decoupling', desc: 'Enhance emotional voice sensitivity', type: 'stimulation' },
      { id: 'admin_tinnitus_relief', label: 'Administer Auditory Calmer', desc: 'Reduce auditory hyperactivity', type: 'medicine' }
    ]
  },
  {
    key: 'middletemporal',
    name: 'Middle Temporal Gyrus',
    lobe: LOBES.TEMPORAL,
    brodmann: 'BA 21',
    function: 'Semantic memory, recognizing facial expressions, contemplation',
    basePos: [1.65, -0.25, -0.45],
    defaultBand: 'Theta',
    actions: [
      { id: 'semantic_probe', label: 'Activate Semantic Recall', desc: 'Temporal theta oscillation drive', type: 'stimulation' }
    ]
  },
  {
    key: 'inferiortemporal',
    name: 'Inferior Temporal Gyrus',
    lobe: LOBES.TEMPORAL,
    brodmann: 'BA 20',
    function: 'Complex visual processing, object identification, visual memory',
    basePos: [1.55, -0.65, -0.35],
    defaultBand: 'Beta',
    actions: [
      { id: 'object_recognition', label: 'Object Feature Enhancement', desc: 'Ventral stream excitation', type: 'stimulation' }
    ]
  },
  {
    key: 'transversetemporal',
    name: 'Transverse Temporal (Heschl Gyrus)',
    lobe: LOBES.TEMPORAL,
    brodmann: 'BA 41, 42',
    function: 'Primary auditory perception, pitch, frequency discrimination',
    basePos: [1.35, 0.15, -0.45],
    defaultBand: 'Gamma',
    actions: [
      { id: 'binaural_40hz', label: 'Binaural 40 Hz Gamma Entrainment', desc: 'Synchronize primary auditory cortex', type: 'stimulation' }
    ]
  },
  {
    key: 'fusiform',
    name: 'Fusiform Gyrus',
    lobe: LOBES.TEMPORAL,
    brodmann: 'BA 37',
    function: 'Facial recognition, emotion extraction from faces, categorization',
    basePos: [1.15, -0.75, -0.85],
    defaultBand: 'Gamma',
    actions: [
      { id: 'face_affect_sync', label: 'Facial Affect Decoding Boost', desc: 'N170 visual ERP priming', type: 'stimulation' }
    ]
  },
  {
    key: 'parahippocampal',
    name: 'Parahippocampal Gyrus',
    lobe: LOBES.TEMPORAL,
    brodmann: 'BA 36',
    function: 'Memory encoding, contextual retrieval, environmental scene recognition',
    basePos: [0.75, -0.65, -0.75],
    defaultBand: 'Theta',
    actions: [
      { id: 'memory_consolidation', label: 'Memory Consolidation Protocol', desc: 'Slow wave sleep phase coupling', type: 'stimulation' }
    ]
  },
  {
    key: 'entorhinal',
    name: 'Entorhinal Cortex',
    lobe: LOBES.TEMPORAL,
    brodmann: 'BA 28, 34',
    function: 'Gateway to hippocampus, spatial grid cells, memory consolidation',
    basePos: [0.65, -0.65, 0.15],
    defaultBand: 'Theta',
    actions: [
      { id: 'hippocampal_pacemaker', label: 'Entorhinal Theta Pacemaker', desc: 'Drive 6 Hz hippocampal rhythm', type: 'stimulation' },
      { id: 'admin_neurotrophin', label: 'Administer BDNF Expression Inducer', desc: 'Promote synaptic neuroplasticity', type: 'medicine' }
    ]
  },
  {
    key: 'temporalpole',
    name: 'Temporal Pole',
    lobe: LOBES.TEMPORAL,
    brodmann: 'BA 38',
    function: 'Socio-emotional processing, narrative comprehension, personal empathy',
    basePos: [1.35, -0.45, 0.85],
    defaultBand: 'Alpha',
    actions: [
      { id: 'narrative_sync', label: 'Narrative Identity Harmonization', desc: 'Socio-emotional binding', type: 'stimulation' }
    ]
  },
  {
    key: 'banksssts',
    name: 'Banks of Superior Temporal Sulcus',
    lobe: LOBES.TEMPORAL,
    brodmann: 'BA 21, 22',
    function: 'Biological motion detection, social gaze tracking, intention inferencing',
    basePos: [1.60, 0.35, -0.75],
    defaultBand: 'Beta',
    actions: [
      { id: 'social_perception', label: 'Social Intention Tracking Boost', desc: 'STS biological motion tuning', type: 'stimulation' }
    ]
  },
  {
    key: 'lateraloccipital',
    name: 'Lateral Occipital Cortex',
    lobe: LOBES.OCCIPITAL,
    brodmann: 'BA 18, 19',
    function: 'Higher-order visual processing, visual feature synthesis',
    basePos: [1.25, 0.15, -1.85],
    defaultBand: 'Alpha',
    actions: [
      { id: 'visual_synthesis', label: 'Visual Synthesis Clarification', desc: 'Alpha desynchronization training', type: 'stimulation' }
    ]
  },
  {
    key: 'cuneus',
    name: 'Cuneus',
    lobe: LOBES.OCCIPITAL,
    brodmann: 'BA 17, 18',
    function: 'Visual orientation, contrast processing, visual imagery during dreaming',
    basePos: [0.45, 0.45, -1.75],
    defaultBand: 'Alpha',
    actions: [
      { id: 'photobiomod', label: 'Near-Infrared Photobiomodulation (810nm)', desc: 'Stimulate mitochondrial ATP in visual cortex', type: 'stimulation' }
    ]
  },
  {
    key: 'lingual',
    name: 'Lingual Gyrus',
    lobe: LOBES.OCCIPITAL,
    brodmann: 'BA 18, 19',
    function: 'Vision-language link, word recognition, dreaming visual generation',
    basePos: [0.55, -0.35, -1.65],
    defaultBand: 'Alpha',
    actions: [
      { id: 'word_form_sync', label: 'Visual Word Form Accelerator', desc: 'Facilitate reading speed', type: 'stimulation' }
    ]
  },
  {
    key: 'pericalcarine',
    name: 'Pericalcarine Cortex (Primary Visual V1)',
    lobe: LOBES.OCCIPITAL,
    brodmann: 'BA 17',
    function: 'Primary retinotopic mapping, visual stimulus entry point',
    basePos: [0.35, 0.05, -1.95],
    defaultBand: 'Alpha',
    actions: [
      { id: 'vep_diagnostic', label: 'Trigger Visual Evoked Potential (VEP)', desc: 'Measure retinogeniculate transmission latency', type: 'stimulation' },
      { id: 'alpha_calm', label: 'Induce Occipital Alpha Rest Wave', desc: 'Reduce ocular and visual sensory strain', type: 'stress' }
    ]
  },
  {
    key: 'rostralanteriorcingulate',
    name: 'Rostral Anterior Cingulate (rACC)',
    lobe: LOBES.CINGULATE,
    brodmann: 'BA 24, 32',
    function: 'Affective monitoring, empathy for pain, conflict resolution',
    basePos: [0.25, 0.45, 0.95],
    defaultBand: 'Theta',
    actions: [
      { id: 'empathy_pain_gate', label: 'Compassionate Pain Gating', desc: 'Downregulate distressing affective amplification', type: 'stress' },
      { id: 'admin_endorphin', label: 'Administer Endogenous Opioid Stimulus', desc: 'Central affective relief', type: 'medicine' }
    ]
  },
  {
    key: 'caudalanteriorcingulate',
    name: 'Caudal Anterior Cingulate (dACC)',
    lobe: LOBES.CINGULATE,
    brodmann: 'BA 24',
    function: 'Error detection, autonomic arousal modulation, decision effort',
    basePos: [0.25, 0.75, 0.35],
    defaultBand: 'Theta',
    actions: [
      { id: 'error_suppress', label: 'Dampen Error-Related Negativity (ERN)', desc: 'Prevent catastrophic spiral in performance anxiety', type: 'stress' },
      { id: 'autonomic_pacing', label: 'Cingulate Autonomic Pacing', desc: 'Re-align heart rate variability with respiratory rhythm', type: 'stimulation' }
    ]
  },
  {
    key: 'posteriorcingulate',
    name: 'Posterior Cingulate Cortex (PCC)',
    lobe: LOBES.CINGULATE,
    brodmann: 'BA 23, 31',
    function: 'Core of Default Mode Network, introspective consciousness, rumination',
    basePos: [0.25, 0.65, -0.65],
    defaultBand: 'Alpha',
    actions: [
      { id: 'pcc_decouple', label: 'PCC Meditation Decoupling', desc: 'Induce egoless deep meditative state', type: 'stimulation' },
      { id: 'lower_self_rumination', label: 'Quench Self-Critical Rumination', desc: 'Gamma suppression burst', type: 'stress' }
    ]
  },
  {
    key: 'isthmuscingulate',
    name: 'Isthmus of Cingulate Cortex',
    lobe: LOBES.CINGULATE,
    brodmann: 'BA 29, 30',
    function: 'Retrosplenial navigation, episodic memory recall, affective transition',
    basePos: [0.25, 0.25, -1.05],
    defaultBand: 'Theta',
    actions: [
      { id: 'episodic_recall', label: 'Reinforce Episodic Navigation', desc: 'Retrosplenial theta coupling', type: 'stimulation' }
    ]
  },
  {
    key: 'insula',
    name: 'Insular Cortex (Interoceptive Hub)',
    lobe: LOBES.INSULAR,
    brodmann: 'BA 13, 14',
    function: 'Visceral bodily feeling, autonomic regulation, compassion, gut-brain link',
    basePos: [1.10, 0.15, 0.15],
    defaultBand: 'Gamma',
    actions: [
      { id: 'vagus_sync', label: 'Transcutaneous Vagus Nerve Stimulation (tVNS)', desc: 'Deliver 25 Hz auricular vagus pulse to lower heart rate & systemic stress', type: 'stress' },
      { id: 'admin_gaba', label: 'Administer GABA-A Receptor Agonist', desc: 'Direct interoceptive calm (L-Theanine / GABAergic protocol)', type: 'medicine' },
      { id: 'interoceptive_calm', label: 'Somatic Panic Dampening Protocol', desc: 'Halt sympathetic cascade and reset viscero-sensory threshold', type: 'stress' }
    ]
  }
];

// Generate 68 bilateral regions (34 LH + 34 RH)
export const CORTICAL_REGIONS_68 = (() => {
  const regions = [];
  let index = 0;

  // Left Hemisphere
  DESIKAN_REGIONS_RAW.forEach((item) => {
    regions.push({
      nodeId: index,
      key: `lh_${item.key}`,
      hemisphere: 'LH',
      hemisphereLabel: 'Left Hemisphere',
      name: `Left ${item.name}`,
      shortName: `L-${item.name.replace('Gyrus', 'G.').replace('Cortex', 'Cx.')}`,
      lobe: item.lobe,
      brodmann: item.brodmann,
      function: item.function,
      defaultBand: item.defaultBand,
      position: [-Math.abs(item.basePos[0]), item.basePos[1], item.basePos[2]],
      connectedVital: getConnectedVital(item.lobe, item.key),
      actions: item.actions || []
    });
    index++;
  });

  // Right Hemisphere
  DESIKAN_REGIONS_RAW.forEach((item) => {
    regions.push({
      nodeId: index,
      key: `rh_${item.key}`,
      hemisphere: 'RH',
      hemisphereLabel: 'Right Hemisphere',
      name: `Right ${item.name}`,
      shortName: `R-${item.name.replace('Gyrus', 'G.').replace('Cortex', 'Cx.')}`,
      lobe: item.lobe,
      brodmann: item.brodmann,
      function: item.function,
      defaultBand: item.defaultBand,
      position: [Math.abs(item.basePos[0]), item.basePos[1], item.basePos[2]],
      connectedVital: getConnectedVital(item.lobe, item.key),
      actions: item.actions || []
    });
    index++;
  });

  return regions;
})();

function getConnectedVital(lobe, key) {
  if (key === 'insula' || key.includes('cingulate')) return 'HRV & Autonomic Vagal';
  if (lobe === LOBES.FRONTAL) return 'EEG Beta/Gamma & Cognitive Load';
  if (lobe === LOBES.TEMPORAL) return 'Voice Prosody & Emotion';
  if (lobe === LOBES.PARIETAL) return 'GSR Conductance & Somatosensory';
  if (lobe === LOBES.OCCIPITAL) return 'EEG Alpha Rhythm & Visual';
  return 'General Neuro-Somatic Sync';
}

/**
 * Maps an activation strength (0 - 100) to the 4-tier BCI status palette
 */
export function getStatusColor(strength) {
  if (strength < 60) {
    return {
      status: 'Normal',
      tier: 'optimal',
      hex: '#10B981', // Emerald Green
      rgb: [0.06, 0.72, 0.50],
      glow: 'rgba(16, 185, 129, 0.4)'
    };
  } else if (strength < 75) {
    return {
      status: 'Pending',
      tier: 'attention',
      hex: '#F59E0B', // Amber / Orange
      rgb: [0.96, 0.62, 0.04],
      glow: 'rgba(245, 158, 11, 0.4)'
    };
  } else if (strength < 85) {
    return {
      status: 'High Activity',
      tier: 'elevated',
      hex: '#EA580C', // Bloody Orange
      rgb: [0.91, 0.34, 0.05],
      glow: 'rgba(234, 88, 12, 0.5)'
    };
  } else {
    return {
      status: 'Danger',
      tier: 'danger',
      hex: '#EF4444', // Red
      rgb: [0.93, 0.27, 0.27],
      glow: 'rgba(239, 68, 68, 0.6)'
    };
  }
}
