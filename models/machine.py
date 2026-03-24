from dataclasses import dataclass, field
from typing import List

@dataclass
class Machine:
    Id: int                       # PK
    NomMachine: str               # ex: Brongo 1, Tupesa 3, OMI 2, Tonello 5
    CapaciteMax: int              # nombre max de pièces par lot
    Statut: str                   # "fonctionnel" ou "non fonctionnel"
    Operations: str               # comma-separated list of supported operations

    def operations_list(self) -> List[str]:
        return [op.strip() for op in self.Operations.split(",")]

    def is_available(self) -> bool:
        return self.Statut == "fonctionnel"

    def supports_operation(self, operation: str) -> bool:
        return operation in self.operations_list()

    def __repr__(self):
        return (f"Machine({self.Id} | {self.NomMachine} | "
                f"cap={self.CapaciteMax} | {self.Statut})")