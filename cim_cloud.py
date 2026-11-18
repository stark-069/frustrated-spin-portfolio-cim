# File: cim_cloud.py
# Licensed under the MIT License
import time
import requests
from typing import Dict, List, Optional, Any, Union

DEFAULT_BASE_URL = "https://quantum-ising-optimizer.shashankg554.workers.dev"

class CIMResult:
    """Represents the execution outcome of a problem solved on the optical CIM."""
    def __init__(self, data: Dict[str, Any]):
        self.raw = data
        self.job_id: str = data.get("job_id", "")
        self.status: str = data.get("status", "")
        self.problem_type: str = data.get("problem_type", "")
        self.num_variables: int = data.get("num_variables", 0)
        self.spins: Dict[str, int] = data.get("result_spins", {}) or {}
        self.energy: Optional[float] = data.get("result_energy")
        self.execution_time_ms: Optional[float] = data.get("execution_time_ms")
        self.hardware_mode: str = data.get("hardware_mode", "")
        self.error_message: Optional[str] = data.get("error_message")

class CIMJob:
    """Handle to an asynchronous or synchronous job dispatched to the CIM cloud."""
    def __init__(self, client: "CIMClient", job_data: Dict[str, Any]):
        self.client = client
        self.id: str = job_data.get("job_id", "")
        self.status: str = job_data.get("status", "QUEUED")
        self.queue_position: int = job_data.get("queue_position", 0)
        self.problem_type: str = job_data.get("problem_type", "")
        self.label: Optional[str] = job_data.get("label")
        self._cached_result: Optional[CIMResult] = None

    def refresh(self) -> str:
        data = self.client._get(f"/api/v1/jobs/{self.id}")
        self.status = data.get("status", self.status)
        self.queue_position = data.get("queue_position", 0)
        if self.status in ("COMPLETED", "FAILED", "CANCELLED"):
            self._cached_result = CIMResult(data)
        return self.status

    def wait_for_completion(self, timeout_seconds: float = 60.0, poll_interval: float = 0.5) -> CIMResult:
        start = time.time()
        while time.time() - start < timeout_seconds:
            self.refresh()
            if self.status == "COMPLETED":
                return self._cached_result
            elif self.status == "FAILED":
                err = self._cached_result.error_message if self._cached_result else "Unknown error"
                raise RuntimeError(f"Job failed on CIM hardware: {err}")
            elif self.status == "CANCELLED":
                raise RuntimeError("Job was cancelled.")
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {self.id} timed out.")

class CIMClient:
    """Client for interacting directly with the Quantum Ising Optimizer Cloud Gateway."""
    def __init__(self, api_key: Optional[str] = None, token: Optional[str] = None, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.token = token
        self.session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-CIM-API-Key"] = self.api_key
        elif self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        r = self.session.get(url, headers=self._headers(), timeout=15)
        if not r.ok:
            raise RuntimeError(f"API Error ({r.status_code}): {r.text}")
        return r.json()

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        r = self.session.post(url, json=payload, headers=self._headers(), timeout=20)
        if not r.ok:
            raise RuntimeError(f"API Error ({r.status_code}): {r.text}")
        return r.json()

    def submit_qubo(self, Q: Dict[Any, float], label: Optional[str] = None, shots: int = 20) -> CIMJob:
        serialized_qubo = {}
        for k, v in Q.items():
            if isinstance(k, tuple):
                serialized_qubo[f"{k[0]},{k[1]}"] = float(v)
            else:
                serialized_qubo[str(k)] = float(v)

        payload = {
            "label": label or "QUBO Problem",
            "problem_type": "QUBO",
            "qubo": serialized_qubo,
            "shots": shots
        }
        res = self._post("/api/v1/jobs/submit", payload)
        return CIMJob(self, res)
