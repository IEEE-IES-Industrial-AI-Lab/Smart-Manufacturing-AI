"""
Digital Twin Synchronization Interface
-----------------------------------------
Bridges the TwinSimulator to real industrial hardware via MQTT or OPC-UA.

In "shadow mode" (default), the twin runs alongside real equipment and
its state is continuously updated from live sensor feeds. Divergence
between simulated and real sensor values triggers diagnostic alerts.

Supported protocols:
  - MQTT  (IoT edge gateways, most PLCs with MQTT bridge)
  - OPC-UA (industrial standard for SCADA/DCS systems)   [stub]
  - Modbus TCP                                            [stub]

When real hardware is not available, TwinSync can replay recorded CSV
data from the simulator for offline development and testing.

Configuration: see configs/digital_twin.yaml → sync section.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class SensorReading:
    """A single timestamped sensor measurement from one stage."""
    stage_id: int
    stage_name: str
    timestamp: float
    temperature: float
    vibration: float
    pressure: float
    status: str


@dataclass
class SyncState:
    """Current synchronization state of the digital twin."""
    connected: bool = False
    last_update_time: float = 0.0
    n_messages_received: int = 0
    n_divergence_alerts: int = 0
    latest_readings: Dict[int, SensorReading] = field(default_factory=dict)


# =============================================================================
# MQTT backend
# =============================================================================

class MQTTBackend:
    """
    MQTT subscriber for receiving industrial sensor data.

    Requires: pip install paho-mqtt

    Topic convention (configurable via topic_prefix):
      <topic_prefix>/stage/<stage_id>/<sensor_name>

    Or a JSON payload on:
      <topic_prefix>/stage/<stage_id>/telemetry
      with fields: temperature, vibration, pressure, status

    Parameters
    ----------
    broker_host : str
    broker_port : int
    topic_prefix : str
    on_message : callable
        Callback invoked with (stage_id, reading_dict) for each message.
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic_prefix: str = "smai/twin",
        on_message: Optional[Callable] = None,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_prefix = topic_prefix
        self.on_message_cb = on_message
        self._client = None
        self._connected = False

    def connect(self) -> bool:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.warning(
                "paho-mqtt not installed. Install with: pip install paho-mqtt. "
                "TwinSync will run in offline/replay mode."
            )
            return False

        self._client = mqtt.Client(client_id="smai-digital-twin")
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        try:
            self._client.connect(self.broker_host, self.broker_port, keepalive=60)
            self._client.loop_start()
            time.sleep(0.5)
            return self._connected
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            topic = f"{self.topic_prefix}/stage/+/telemetry"
            client.subscribe(topic)
            logger.info(f"MQTT connected. Subscribed to {topic}")
        else:
            logger.error(f"MQTT connection refused (rc={rc})")

    def _on_message(self, client, userdata, msg):
        try:
            topic_parts = msg.topic.split("/")
            stage_id = int(topic_parts[-2])
            payload = json.loads(msg.payload.decode())
            if self.on_message_cb:
                self.on_message_cb(stage_id, payload)
        except Exception as e:
            logger.warning(f"Failed to parse MQTT message: {e}")


# =============================================================================
# TwinSync
# =============================================================================

class TwinSync:
    """
    Digital twin synchronization manager.

    Keeps the TwinSimulator in sync with real hardware (via MQTT/OPC-UA)
    or replays recorded data for offline testing. Detects divergence between
    the simulated state and real sensor readings.

    Parameters
    ----------
    simulator : TwinSimulator
        The digital twin simulator instance.
    protocol : {"mqtt", "opcua", "replay"}
        Communication protocol.
    broker_host : str
        MQTT broker hostname.
    broker_port : int
    topic_prefix : str
    divergence_threshold : float
        Z-score threshold above which a sensor reading is flagged
        as diverging from the twin's prediction.
    replay_csv : str | Path, optional
        Path to a CSV file to replay when protocol="replay".
    """

    def __init__(
        self,
        simulator,
        protocol: str = "mqtt",
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic_prefix: str = "smai/twin",
        divergence_threshold: float = 3.0,
        replay_csv: Optional[str | Path] = None,
    ) -> None:
        self.simulator = simulator
        self.protocol = protocol
        self.divergence_threshold = divergence_threshold
        self.replay_csv = Path(replay_csv) if replay_csv else None

        self.state = SyncState()
        self._message_queue: Queue = Queue(maxsize=1000)
        self._backend: Optional[MQTTBackend] = None
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None

        if protocol == "mqtt":
            self._backend = MQTTBackend(
                broker_host=broker_host,
                broker_port=broker_port,
                topic_prefix=topic_prefix,
                on_message=self._enqueue_message,
            )
        elif protocol not in ("opcua", "replay"):
            raise ValueError(f"Unsupported protocol: {protocol}. Use 'mqtt', 'opcua', or 'replay'.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Start the synchronization loop.

        Returns True if successfully connected to real hardware,
        False if running in offline/replay mode.
        """
        if self.protocol == "mqtt" and self._backend is not None:
            connected = self._backend.connect()
            self.state.connected = connected
            if not connected:
                logger.info("Falling back to offline mode (no MQTT connection).")
        elif self.protocol == "replay":
            self.state.connected = False
            self._start_replay()

        self._running = True
        self._sync_thread = threading.Thread(
            target=self._sync_loop, daemon=True
        )
        self._sync_thread.start()
        logger.info(f"TwinSync started (protocol={self.protocol}, connected={self.state.connected})")
        return self.state.connected

    def stop(self) -> None:
        """Stop the synchronization loop."""
        self._running = False
        if self._backend is not None:
            self._backend.disconnect()
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=5)
        logger.info("TwinSync stopped.")

    # ------------------------------------------------------------------
    # Synchronization loop
    # ------------------------------------------------------------------

    def _sync_loop(self) -> None:
        while self._running:
            try:
                stage_id, payload = self._message_queue.get(timeout=0.1)
                self._process_reading(stage_id, payload)
            except Empty:
                pass

    def _enqueue_message(self, stage_id: int, payload: Dict) -> None:
        try:
            self._message_queue.put_nowait((stage_id, payload))
        except Exception:
            pass  # Queue full — drop oldest? For now skip

    def _process_reading(self, stage_id: int, payload: Dict) -> None:
        """
        Update twin state from an incoming sensor reading and
        check for divergence.
        """
        self.state.n_messages_received += 1
        self.state.last_update_time = time.time()

        if stage_id >= len(self.simulator._stages):
            return

        real = SensorReading(
            stage_id=stage_id,
            stage_name=self.simulator._stages[stage_id].name,
            timestamp=payload.get("timestamp", time.time()),
            temperature=float(payload.get("temperature", 0)),
            vibration=float(payload.get("vibration", 0)),
            pressure=float(payload.get("pressure", 0)),
            status=payload.get("status", "unknown"),
        )
        self.state.latest_readings[stage_id] = real

        # Compare against twin prediction
        twin_stage = self.simulator._stages[stage_id]
        divergences = self._check_divergence(real, twin_stage)
        if divergences:
            self.state.n_divergence_alerts += 1
            logger.warning(
                f"[Stage {stage_id} | {real.stage_name}] "
                f"Divergence detected: {divergences}"
            )

    def _check_divergence(self, real: SensorReading, twin_state) -> List[str]:
        """
        Compare real sensor values against twin predictions.
        Returns a list of diverging sensor names.
        """
        alerts = []
        checks = [
            ("temperature", real.temperature, twin_state.temperature),
            ("vibration", real.vibration, twin_state.vibration),
            ("pressure", real.pressure, twin_state.pressure),
        ]
        for name, real_val, twin_val in checks:
            # Normalized deviation — in practice this would use a running
            # baseline std; here we use a fixed scale for demonstration
            scale = max(abs(twin_val), 1e-6)
            z = abs(real_val - twin_val) / scale
            if z > self.divergence_threshold:
                alerts.append(f"{name}(real={real_val:.2f}, twin={twin_val:.2f}, z={z:.2f})")
        return alerts

    # ------------------------------------------------------------------
    # Replay mode
    # ------------------------------------------------------------------

    def _start_replay(self) -> None:
        """Replay recorded CSV data in a background thread."""
        if self.replay_csv is None or not self.replay_csv.exists():
            logger.warning(f"Replay CSV not found: {self.replay_csv}. No data will be replayed.")
            return

        def _replay_worker():
            import csv
            with open(self.replay_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not self._running:
                        break
                    # Extract stage 0 sensor data as example
                    payload = {
                        "temperature": float(row.get("stage_0_machining_temperature", 0)),
                        "vibration": float(row.get("stage_0_machining_vibration", 0)),
                        "pressure": float(row.get("stage_0_machining_pressure", 0)),
                        "status": row.get("stage_0_machining_status", "unknown"),
                        "timestamp": float(row.get("sim_time", 0)),
                    }
                    self._enqueue_message(0, payload)
                    time.sleep(0.01)  # 100 Hz replay

        t = threading.Thread(target=_replay_worker, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        return {
            "connected": self.state.connected,
            "protocol": self.protocol,
            "messages_received": self.state.n_messages_received,
            "divergence_alerts": self.state.n_divergence_alerts,
            "last_update_age_s": (
                round(time.time() - self.state.last_update_time, 2)
                if self.state.last_update_time > 0
                else None
            ),
            "active_stages": list(self.state.latest_readings.keys()),
        }

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"TwinSync(protocol={self.protocol}, "
            f"connected={s['connected']}, "
            f"messages={s['messages_received']}, "
            f"alerts={s['divergence_alerts']})"
        )
