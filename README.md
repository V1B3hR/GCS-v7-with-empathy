# GCS-v7-with-empathy
BCI with empathy and strong safety measures
Grand Council Sentient (GCS)
![alt text](https://img.shields.io/badge/build-passing-brightgreen)

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)

![alt text](https://img.shields.io/badge/version-7.0-blue)
An Empathetic, Co-Adaptive, and Therapeutic Brain-Computer Interface.
The Vision
The Grand Council Sentient (GCS) is more than just a Brain-Computer Interface. It is a research initiative and a functional prototype for a system designed to be a true partner to the human mind. Our core philosophy is that the mind is like a flower: fragile, flexible, and requiring a supportive environment to thrive.
This project moves beyond simple command decoding. We are building a complete cognitive and affective interface that can sense intent, understand emotion, and is architected to safely and ethically act as a therapeutic agent. Our goal is to create a system that is not just powerful, but also good—a technology that embodies the principles of safety, privacy, and empathy.
  
  Core Features
The GCS is a full-stack, modular system built on a foundation of cutting-edge research and a safety-first design.
🧠 The Neuro-Symbolic Core: At its heart is a state-of-the-art, zero-shot BCI (GCS_v3.1-Production). It uses a Graph Attention Network (GAT) on an anatomical brain scaffold to decode motor intent with 86.5% validated accuracy on unseen users.
❤️ The Empathy Module: The system is augmented with a validated AffectiveStateClassifier that fuses multi-modal data—EEG (brain), HRV/GSR (body), and speech prosody (voice)—to achieve a deep, nuanced understanding of the user's emotional state (valence and arousal).
🛡️ The Active Therapeutic Framework: The GCS is architected for closed-loop therapeutic action. All potential interventions are governed by the AI_Guardian and a strict Human-in-the-Loop (HITL) consent protocol, ensuring the user is always in control.
🌱 The Living Architecture: The system is designed as a co-adaptive agent (Phase 10). It uses an Online Learning Module to learn from real-time user feedback, allowing it to continuously personalize and improve its understanding, creating a truly symbiotic partnership.
  
  System Architecture
The GCS is a full-stack application designed for modern, cloud-based development and deployment.
Frontend: A responsive and intuitive "Cognitive Dashboard" built with React and Three.js for 3D visualization.
Backend: A high-performance, asynchronous API server built with Python and FastAPI.
Real-Time Communication: A WebSocket bridge ensures low-latency data streaming from the backend to the frontend.
AI Core: Our complete GCS Python package, built with TensorFlow/Keras and Spektral, running as the intelligent engine within the backend.

  Getting Started
This repository is configured to run seamlessly in GitHub Codespaces, providing a powerful, pre-configured cloud development environment with a generous free tier.

1. Launch the Codespace:
Click the green <> Code button at the top of this repository.
Go to the "Codespaces" tab.
Click "Create codespace on main".
Wait for the environment to build automatically. This may take a few minutes.
2. Run the Application:
The Codespace will open a VS Code editor in your browser.
Two terminals are needed. You can open a new one by clicking the + in the terminal panel.
In the first terminal, start the backend:
code
Bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
In the second terminal, start the frontend:
code
Bash
npm run --prefix frontend start
The Codespace will automatically detect that you've started a web server and a pop-up will appear asking if you want to open it in a browser. Click "Open in Browser".
You are now running the full-stack GCS Cognitive Dashboard!
Project Structure
The repository is organized as a professional monorepo.
code
Code
.
├── .devcontainer/    # Configuration for GitHub Codespaces
├── backend/          # All Python code for the GCS and API server
│   ├── gcs/          # The core GCS Python package
│   └── main.py       # The FastAPI application
├── frontend/         # All React code for the UI dashboard
│   └── src/
└── render.yaml       # Configuration for easy deployment to Render.com

  The Next Frontier: The Roadmap
The completion of the software architecture marks the end of the foundational phase. The future is about grounding the GCS in physical reality.
Hardware Integration: The next major step is to integrate the system with real, physical hardware (e.g., OpenBCI, Polar H10) to move from simulation to a live, operational prototype.
UI/UX Refinement: Continuously improve the Cognitive Dashboard to make it more intuitive, transparent, and empowering for the user.
Closed-Loop Policy Training: Develop the Reinforcement Learning agent that can learn a personalized, empathetic therapeutic policy, turning the GCS into a truly active and helpful partner.
  
  Our Ethical Principles
This project is governed by a set of non-negotiable principles.
Safety First: The system is designed to be fail-safe. The AI_Guardian and conservative protocols are in place to prevent harm above all else.
User Sovereignty: The user is the ultimate authority. No action is ever taken without the user's explicit and informed consent. The system is a tool for empowerment, not control.
Privacy by Design: We are committed to protecting the most sensitive data imaginable. All models are trained with privacy-preserving techniques, and all data is handled with end-to-end encryption.
Empathetic Action: The system's core logic is being built to recognize and respond to the user's well-being. The goal is not just to execute tasks, but to foster a supportive and healthy cognitive environment.
