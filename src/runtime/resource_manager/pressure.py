"""Resource Pressure State Evaluator with Hysteresis.

Section 14:
- GREEN: Normal operation (< 70% RAM, normal queue)
- YELLOW: Reduce background work (70% - 80% RAM)
- ORANGE: Pause expensive background generation (80% - 88% RAM)
- RED: Unload non-critical models (88% - 94% RAM)
- CRITICAL: Preserve core state, pause generation, perform recovery (> 94% RAM)

Uses hysteresis thresholds to prevent rapid oscillation between states.
"""
from enum import Enum
from typing import Dict, Any


class PressureState(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"
    CRITICAL = "CRITICAL"


class PressureEvaluator:
    def __init__(self,
                 yellow_enter: float = 72.0, yellow_exit: float = 68.0,
                 orange_enter: float = 82.0, orange_exit: float = 78.0,
                 red_enter: float = 89.0, red_exit: float = 85.0,
                 critical_enter: float = 94.0, critical_exit: float = 91.0):
        self.ye_enter = yellow_enter
        self.ye_exit = yellow_exit
        self.or_enter = orange_enter
        self.or_exit = orange_exit
        self.re_enter = red_enter
        self.re_exit = red_exit
        self.cr_enter = critical_enter
        self.cr_exit = critical_exit
        self.current_state = PressureState.GREEN
        self.last_percent = 0.0

    def evaluate(self, ram_percent: float) -> PressureState:
        """Evaluates pressure state using hysteresis to avoid state flutter."""
        self.last_percent = float(ram_percent)
        
        if self.current_state == PressureState.GREEN:
            if ram_percent >= self.cr_enter:
                self.current_state = PressureState.CRITICAL
            elif ram_percent >= self.re_enter:
                self.current_state = PressureState.RED
            elif ram_percent >= self.or_enter:
                self.current_state = PressureState.ORANGE
            elif ram_percent >= self.ye_enter:
                self.current_state = PressureState.YELLOW
                
        elif self.current_state == PressureState.YELLOW:
            if ram_percent >= self.cr_enter:
                self.current_state = PressureState.CRITICAL
            elif ram_percent >= self.re_enter:
                self.current_state = PressureState.RED
            elif ram_percent >= self.or_enter:
                self.current_state = PressureState.ORANGE
            elif ram_percent <= self.ye_exit:
                self.current_state = PressureState.GREEN
                
        elif self.current_state == PressureState.ORANGE:
            if ram_percent >= self.cr_enter:
                self.current_state = PressureState.CRITICAL
            elif ram_percent >= self.re_enter:
                self.current_state = PressureState.RED
            elif ram_percent <= self.or_exit:
                self.current_state = PressureState.YELLOW
                
        elif self.current_state == PressureState.RED:
            if ram_percent >= self.cr_enter:
                self.current_state = PressureState.CRITICAL
            elif ram_percent <= self.re_exit:
                self.current_state = PressureState.ORANGE
                
        elif self.current_state == PressureState.CRITICAL:
            if ram_percent <= self.cr_exit:
                self.current_state = PressureState.RED

        return self.current_state

    def status(self) -> Dict[str, Any]:
        return {
            "state": self.current_state.value,
            "ram_percent": round(self.last_percent, 1),
            "allow_background_ai": self.current_state in (PressureState.GREEN, PressureState.YELLOW),
            "allow_speculative_work": self.current_state == PressureState.GREEN,
            "recovery_required": self.current_state == PressureState.CRITICAL
        }
