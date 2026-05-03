# Product Requirement Document: ParkNav AI

## 1. Product Overview
ParkNav AI is an adaptive parking navigation system designed for urban environments in India. It leverages Reinforcement Learning (Q-learning) and XGBoost-based predictive modeling to provide real-time parking availability estimates. The system incorporates a crowdsourced feedback loop, allowing users to report observed conditions, which in turn improves the AI's accuracy over time.

## 2. Core Goals
- **Real-Time Discovery**: Enable users to find nearby parking locations on an interactive map.
- **Predictive Analytics**: Provide AI-powered availability percentages for parking spots based on time, day, and historical data.
- **Crowdsourced Intelligence**: Implement a feedback loop where user reports refine the AI's knowledge base.
- **Trust & Verification**: Identify "Community-Verified" spots that have consistent, reliable user reports.
- **Seamless Navigation**: Center the map on the user's GPS location and visualize optimized routes with congestion-aware coloring.

## 3. Key Features & User Flows

### 3.1 Interactive Map and Search
- **Description**: A full-screen Leaflet.js map with a search bar for geocoding locations in India.
- **User Flow**: 
    1. User navigates to the dashboard.
    2. User enters a location (e.g., "IIT Delhi") in the search bar.
    3. The map centers on the location and displays parking markers.
    4. User clicks a marker to see specific details (Address, Type, Base Availability).

### 3.2 AI Availability Prediction
- **Description**: Displays a "Live Availability" percentage for any selected destination or parking spot.
- **User Flow**:
    1. User selects a destination or clicks a parking marker.
    2. The system fetches a prediction from the AI model (XGBoost/Q-Agent).
    3. The UI displays a percentage (e.g., "85% Available") and a confidence level.

### 3.3 Crowdsourced Feedback Loop
- **Description**: Users can report "High", "Medium", or "Low" availability at their current location.
- **User Flow**:
    1. User clicks "Report Availability" for a selected spot.
    2. User selects the observed status.
    3. Upon submission, the AI's Q-table is updated in real-time.
    4. A confirmation toast appears, and the map marker may update its visual state.

### 3.4 Community-Verified Spots
- **Description**: Locations with 3+ consistent reports within a short timeframe are flagged as "Verified".
- **User Flow**:
    1. User views the map and notices a marker with a "Verified" badge.
    2. User clicks the marker to see the summary of recent reports that led to verification.

### 3.5 GPS & Smart Routing
- **Description**: Centers map on the user and draws routes that change color (Green -> Yellow -> Red) near the destination based on predicted parking congestion.
- **User Flow**:
    1. User clicks the "GPS" icon to center on their location.
    2. User sets a destination; the system draws the route via OSRM.
    3. The last 500m of the route changes color based on the destination's predicted availability.

## 4. Technical Stack
- **Frontend**: HTML5, Vanilla CSS, JavaScript, Leaflet.js
- **Backend**: Python Flask
- **AI/ML**: XGBoost (for trend prediction), Q-Learning (for real-time adaptation)
- **Data**: OpenStreetMap (via Nominatim and OSRM)

## 5. Success Criteria
- Map renders correctly with all interactive markers.
- Search successfully returns coordinates for valid Indian addresses.
- AI predictions update dynamically when the simulated "Time" or "Day" changes.
- User feedback submissions correctly update the `parking_reports.json` and Q-table.
