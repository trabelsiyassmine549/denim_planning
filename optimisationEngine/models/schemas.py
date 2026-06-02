from typing import List, Optional
from pydantic import BaseModel, Field

class RunRequest(BaseModel):
    token:        str
    commandeIds:  Optional[List[int]] = None  
    maxMachinesPerOp: int = Field(default=1, ge=1, le=3)
    startDatetime: Optional[str] = None


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