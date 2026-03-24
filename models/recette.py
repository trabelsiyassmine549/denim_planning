from dataclasses import dataclass

@dataclass
class Recette:
    Id: int
    NomRecette: str

    def __repr__(self):
        return f"Recette({self.Id} | {self.NomRecette})"


@dataclass
class OperationRecette:
    Id: int
    RecetteId: int
    Ordre: int
    NomOperation: str
    DureeMinutes: int                # machine running time (minutes)
    QuantiteLot: int                 # max pieces per batch
    TempsChargementMinutes: int      # loading time before cycle (pieces, chemicals, water, etc.)
    TempsDecharementMinutes: int     # unloading time after cycle (empty machine, drain, etc.)

    @property
    def DureeTotale(self) -> int:
        """Total machine occupation = loading + cycle + unloading"""
        return self.TempsChargementMinutes + self.DureeMinutes + self.TempsDecharementMinutes

    def __repr__(self):
        return (f"OperationRecette(recette={self.RecetteId} | ordre={self.Ordre} | "
                f"{self.NomOperation} | "
                f"{self.TempsChargementMinutes}+{self.DureeMinutes}+{self.TempsDecharementMinutes}min"
                f" | lot={self.QuantiteLot}pcs)")