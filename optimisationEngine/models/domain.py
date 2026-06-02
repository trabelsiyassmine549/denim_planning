class Commande:
    def __init__(self, d: dict):
        self.Id             = d["id"]
        self.NumeroCommande = d["numeroCommande"]
        self.DateExport     = d["dateExport"][:10]
        self.Urgence        = d["urgence"]
        self.Quantite       = d["quantite"]
        self.RecetteId      = d["recetteId"]


class Machine:
    def __init__(self, d: dict):
        self.Id          = d["id"]
        self.NomMachine  = d["nomMachine"]
        self.CapaciteMax = d["capaciteMax"]
        self.Statut      = d["statut"]
        self.Operations  = d.get("operations") or ""

    def operations_list(self):
        return [op.strip() for op in self.Operations.split(",") if op.strip()]

    def is_available(self):
        return self.Statut.strip().lower() == "fonctionnel"

    def supports_operation(self, op: str):
        return op.lower() in [o.lower() for o in self.operations_list()]


class OperationRecette:
    def __init__(self, d: dict):
        self.Id                      = d["id"]
        self.RecetteId               = d.get("recetteId", 0)
        self.Ordre                   = d["ordre"]
        self.NomOperation            = d["nomOperation"]
        self.DureeMinutes            = d["dureeMinutes"]
        self.QuantiteLot             = d["quantiteLot"]
        self.TempsChargementMinutes  = d.get("tempsChargementMinutes", 5)
        self.TempsDecharementMinutes = d.get("tempsDecharementMinutes", 5)

    @property
    def DureeTotale(self):
        return (
            self.TempsChargementMinutes
            + self.DureeMinutes
            + self.TempsDecharementMinutes
        )