# ParkNav AI — Adaptive Parking Navigation System

AI-powered parking navigation for India using Q-learning reinforcement learning with a crowdsourced feedback loop.

## Features

- Full-screen interactive map (Leaflet.js + OpenStreetMap)
- Search any location in India (Nominatim geocoding)
- GPS current location detection
- Route drawing from source to destination (OSRM API)
- Route color changes near destination based on parking congestion (green/yellow/red)
- AI-predicted parking spots near destination with availability %, status, and confidence
- Crowdsourced feedback — users report parking availability, AI learns in real-time
- Community verified spots appear after 3+ reports at the same location
- Q-table persists between sessions (qtable.json)
- IST time display with peak hour warnings

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript, Leaflet.js
- **Backend**: Python Flask
- **AI**: Q-learning (Reinforcement Learning)
- **Routing**: OSRM API (free)
- **Geocoding**: Nominatim API (free)
- No paid APIs — everything is free

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open your browser: [http://localhost:5000](http://localhost:5000)

## Deployment State

Local development uses the JSON files in this folder for feedback, learned spots, and Q-table updates. In production, set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to store the same state in Supabase instead of writing to local files.

Create the Supabase table by running `supabase_schema.sql` in the Supabase SQL editor. Also set `GEMINI_API_KEY` in your deployment environment for the AI chat route.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/parking/nearby?lat=&lng=` | GET | 5 simulated parking spots near coordinates |
| `/api/predict?location_type=&hour=` | GET | RL prediction for location type + hour |
| `/api/feedback` | POST | Submit crowdsourced parking feedback |
| `/api/learned-spots` | GET | Community-verified parking locations |
| `/api/health` | GET | System health + Q-table stats |

## Folder Structure

```
ParkNav AI/
├── app.py                      # Flask backend
├── rl/q_agent.py               # Q-learning RL agent
├── templates/index.html        # Single-file frontend (HTML+CSS+JS)
├── qtable.json                 # Auto-created — persisted Q-table
├── parking_reports.json        # Auto-created — crowdsourced reports
├── known_parking_spots.json    # Auto-created — community verified spots
├── requirements.txt
└── README.md
```

## Test With

- Source: "IIT Delhi, New Delhi"
- Destination: "Select City Walk Mall, Saket"

## Academic Project

This is a final-year academic project demonstrating:
- Reinforcement Learning (Q-learning) for real-world prediction
- Crowdsourced feedback loops for continuous AI improvement
- Full-stack web development with Python + JavaScript
- Integration of free mapping APIs (OpenStreetMap, OSRM, Nominatim)
