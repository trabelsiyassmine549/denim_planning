from dataclasses import dataclass


@dataclass
class Commande:
    Id: int
    NumeroCommande: str
    DateExport: str
    Urgence: int          # 1=urgent  2=haute  3=normal  4=basse  5=flexible
    Quantite: int
    RecetteId: int
    Statut: str
    DateCreation: str
    DateModification: str

    def __repr__(self) -> str:
        return (
            f"Commande({self.NumeroCommande} | "
            f"qty={self.Quantite} | recetteId={self.RecetteId} | urgence={self.Urgence})"
        )