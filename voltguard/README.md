# VoltGuard

React + Vite + TypeScript + Tailwind + Recharts dashboard for an ESP32 electrical fault monitor (voltage, current, temperature) fed by Node-RED.

    npm install
    npm run dev      # http://localhost:5173, starts in Simulation mode
    npm run build

## Where things live
- `src/core.ts`: types, validation/normalization, threshold logic, simulator, Node-RED client (HTTP + WebSocket), prediction service, connection test
- `src/store.tsx`: state, mode switching, events. `src/ui.tsx`, `src/pages.tsx`, `src/App.tsx`: UI

## Connect Node-RED (when the ESP32 is ready)
Open **System settings** (or copy `.env.example` to `.env`) and change only:
1. **Data source** → Live (Node-RED)
2. **Node-RED base URL**, default `http://localhost:1880`
3. HTTP: **Latest-readings path**, default `/api/readings/latest`. WebSocket: **Transport** → WebSocket and set **WebSocket URL**, default `ws://localhost:1880/ws/readings`
4. Press **Test connection**, then **Save settings**

These defaults are placeholders. Nothing here has connected to your flow.

**HTTP flow:** `http in` (GET `/api/readings/latest`) → function node returning the latest reading from flow context → `http response`. Allow the dashboard origin in Node-RED `settings.js`: `httpNodeCors: {origin: "*"}`.
**WebSocket flow:** your MQTT/serial-in → `websocket out` node with path `/ws/readings`. Send one JSON object (or an array) per message.

## Expected JSON
    {"timestamp":"2026-10-16T08:30:00+05:30","voltage":230.4,"current":1.82,"temperature":42.6,
     "prediction":{"label":"normal","confidence":0.97}}
- `voltage`, `current`, `temperature` are required numbers; others are optional.
- A missing or invalid `timestamp` falls back to arrival time. Repeated timestamps are ignored.
- `prediction` is optional. Without it, live mode shows "Prediction unavailable". Any `status` field you send is ignored; status is derived from thresholds plus the prediction.
- Malformed messages are dropped, never displayed. If Node-RED goes offline the UI shows Offline; it never switches to simulated data on its own.

## Model endpoint
If Node-RED does not attach predictions, set **Prediction endpoint** in settings. The app POSTs `{voltage,current,temperature}` and reads the label and confidence from the response fields you name there.

## Honesty notes
- Simulated data is always tagged SIMULATION. Its prediction label is a fixed rule, not a model, and has no confidence value.
- Live history is only what this browser received this session. Node-RED persistence is not assumed.
- Threshold warnings (calculated here) are kept separate from model classifications (from your model).
