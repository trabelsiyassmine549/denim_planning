from dataclasses import dataclass

@dataclass
class Commande:
    Id: int                    # PK
    NumeroCommande: str        # ex: CMD-2026-001
    DateExport: str            # date d'export au format ISO (ex: 2026-03-25)
    Urgence: int               # 1=urgent, 2=normal, 3=flexible
    Quantite: int              # nombre de pièces total
    RecetteId: int             # FK → Recettes.Id
    Statut: str                # ex: "En attente", "En cours", "Terminé"
    DateCreation: str          # ISO datetime
    DateModification: str      # ISO datetime

    def __repr__(self):
        return (f"Commande({self.NumeroCommande} | "
                f"qty={self.Quantite} | recetteId={self.RecetteId} | urgence={self.Urgence})")