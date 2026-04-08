from dataclasses import dataclass, field


@dataclass
class ProductionTask:
    NumeroCommande: str
    NomOperation: str
    MachineId: int
    DureeMinutes: int
    QuantiteLot: int
    EarliestStart: int
    LatestEnd: int

    # Filled by the solver
    Start: int = -1
    End: int = -1
    AssignedMachineId: int = -1