import asyncio
import json
import logging
import random
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def gcs_data_stream(websocket: WebSocket):
    """Simulates the GCS live data stream."""
    while True:
        try:
            # --- SIMULATED DATA FOR DEMONSTRATION ---
            cognitive_data = {
                "intent": random.choice(["IDLE", "MOTOR_IMAGERY_LEFT", "FOCUS"]),
                "confidence": round(random.uniform(0.7, 0.99), 2),
                "activityDots": [] # Placeholder for 3D dot data
            }
            
            emotion_label = random.choice(["Calm", "Anxious", "Focused", "Joyful"])
            strength = 55 if emotion_label == "Calm" else random.randint(65, 90)
            icon = "😊" if emotion_label == "Calm" else "😟"
            
            affective_data = {
                "label": emotion_label,
                "icon": icon,
                "strength": strength
            }
            
            payload = json.dumps({
                "cognitive": cognitive_data,
                "affective": affective_data
            })
            await websocket.send_text(payload)
            
            await asyncio.sleep(2) # Stream data every 2 seconds

        except Exception as e:
            logging.error(f"Error in GCS stream: {e}")
            break

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("Frontend client connected.")
    try:
        await gcs_data_stream(websocket)
    except Exception as e:
        logging.error(f"WebSocket connection closed: {e}")
    finally:
        logging.info("Frontend client disconnected.")
