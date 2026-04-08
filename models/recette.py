from dataclasses import dataclass


@dataclass
class Recette:
    Id: int
    NomRecette: str

    def __repr__(self) -> str:
        return f"Recette({self.Id} | {self.NomRecette})"


@dataclass
class OperationRecette:
    Id: int
    RecetteId: int
    Ordre: int
    NomOperation: str
    DureeMinutes: int
    QuantiteLot: int
    TempsChargementMinutes: int
    TempsDecharementMinutes: int

    @property
    def DureeTotale(self) -> int:
        """Total machine occupation = loading + cycle + unloading."""
        return self.TempsChargementMinutes + self.DureeMinutes + self.TempsDecharementMinutes

    def __repr__(self) -> str:
        return (
            f"OperationRecette(recette={self.RecetteId} | ordre={self.Ordre} | "
            f"{self.NomOperation} | "
            f"{self.TempsChargementMinutes}+{self.DureeMinutes}+{self.TempsDecharementMinutes}min"
            f" | lot={self.QuantiteLot}pcs)"
        )