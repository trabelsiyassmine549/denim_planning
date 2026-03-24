from dataclasses import dataclass

@dataclass
class ProductionTask:
    NumeroCommande: str   # référence commande (NumeroCommande)
    NomOperation: str     # stonage / javel / javelisation / snow legs / rags /
                          # poudre / préparation poudre / ATOMS / blanchiment / rinçage
    MachineId: int        # FK → Machines.Id
    DureeMinutes: int     # durée réelle en minutes (depuis OperationRecette)
    QuantiteLot: int      # taille du lot traité dans ce task
    EarliestStart: int    # jour ouvré le plus tôt (offset en minutes)
    LatestEnd: int        # jour ouvré limite (offset en minutes)

    # Remplis par le solver
    Start: int = -1
    End: int = -1
    AssignedMachineId: int = -1