from dataclasses import dataclass, field
from typing import List

@dataclass
class Machine:
    Id: int                       # PK
    NomMachine: str               # ex: Brongo 2, Tupesa 1, Brongo 5
    CapaciteMax: int              # nombre max de pièces par lot
    Statut: str                   # "Fonctionnel" / "Non fonctionnel" (case-insensitive check)
    Operations: str               # comma-separated list of supported operations

    def operations_list(self) -> List[str]:
        return [op.strip() for op in self.Operations.split(",")]

    def is_available(self) -> bool:
        # Case-insensitive comparison: handles "fonctionnel", "Fonctionnel", etc.
        return self.Statut.strip().lower() == "fonctionnel"

    def supports_operation(self, operation: str) -> bool:
        # Case-insensitive operation match
        return operation.lower() in [op.lower() for op in self.operations_list()]

    def __repr__(self):
        return (f"Machine({self.Id} | {self.NomMachine} | "
                f"cap={self.CapaciteMax} | {self.Statut})")