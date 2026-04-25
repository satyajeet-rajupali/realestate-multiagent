import requests
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class A2AClient:
    """Call agents by task name, using the info from their agent cards."""

    def __init__(self, cards: Dict[str, Any]):
        # Build a lookup table: task → {agent, url, endpoint, method}
        self._capability_map = {}
        for name, card in cards.items():
            for cap in card.get("capabilities", []):
                self._capability_map[cap["task"]] = {
                    "agent": name,
                    "base_url": card["base_url"],
                    "endpoint": cap["endpoint"],
                    "method": cap.get("method", "POST"),
                }

    def call(self, task: str, params: Optional[Dict] = None, path_params: Optional[Dict] = None) -> Dict[str, Any]:
        if task not in self._capability_map:
            raise ValueError(f"Task '{task}' not found in any agent card.")
        cap = self._capability_map[task]
        url = cap["base_url"] + cap["endpoint"]

        # Inject path parameters if needed (e.g., /customer/{customer_id})
        if path_params:
            url = url.format(**path_params)

        method = cap["method"].upper()
        try:
            if method == "POST":
                resp = requests.post(url, json=params or {}, timeout=180)
            elif method == "GET":
                resp = requests.get(url, params=params or {}, timeout=180)
            else:
                raise ValueError(f"Unsupported method: {method}")
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"A2A call to {task} failed: {e}")
            raise