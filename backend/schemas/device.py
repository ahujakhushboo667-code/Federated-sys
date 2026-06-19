from pydantic import BaseModel
from typing import Optional, Dict, Any
from enum import Enum

class HardwareType(str, Enum):
    MI300X = "MI300X"
    RX_7900_XTX = "RX_7900_XTX"
    STEAM_DECK = "Steam_Deck"
    CPU_ONLY = "CPU_only"
    RYZEN_AI_LAPTOP = "Ryzen AI Laptop"
    RADEON_WORKSTATION = "Radeon Workstation"
    STEAM_DECK_MARKETING = "Steam Deck"
    EDGE_SERVER = "Edge Server"

class EdgeDevice(BaseModel):
    id: str
    name: str
    hardwareType: HardwareType
    status: str
    cpuUsage: float
    memoryUsage: float
    lastSync: str
    contributionScore: float
    region: str

    class Config:
        from_attributes = True

class RegionData(BaseModel):
    name: str
    deviceCount: int
    x: int
    y: int

class DeviceRegister(BaseModel):
    client_id: str
    hardware_type: HardwareType
    device_info: Dict[str, Any]
    contribution_weight: float = 1.0
    region: Optional[str] = "US-East"

class DeviceHeartbeat(BaseModel):
    status: str
    cpu_usage: float
    memory_usage: float
