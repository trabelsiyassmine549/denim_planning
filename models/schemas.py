"""
models/schemas.py — Pydantic request / response schemas for the FastAPI layer
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    token:        str
    commandeIds:  Optional[List[int]] = None   # None → all "En attente"
    maxMachinesPerOp: int = Field(default=1, ge=1, le=3)# 1 min → 1 machine par opération, 2 → 2 machines par opération, 3max → 3 machines par opération  

class GanttRow(BaseModel):
    numeroCommande:          str
    quantite:                int
    recetteId:               int
    urgence:                 int
    nomOperation:            str
    machineId:               int
    machineName:             str
    startPM:                 int
    endPM:                   int
    dureeMinutes:            int
    tempsChargementMinutes:  int
    tempsDecharementMinutes: int
    dureeTotale:             int
    lotSize:                 int
    quantiteLot:             int
    lotIdx:                  int
    nbLots:                  int
    dateStart:               str
    dateEnd:                 str
    dateExport:              str


class RunResponse(BaseModel):
    status:       str
    makespanDays: int
    makespanPM:   int
    startDate:    str
    rows:         List[GanttRow]
    warnings:     List[str]