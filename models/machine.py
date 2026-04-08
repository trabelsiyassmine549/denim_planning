from dataclasses import dataclass
from typing import List


@dataclass
class Machine:
    Id: int
    NomMachine: str
    CapaciteMax: int
    Statut: str
    Operations: str

    def operations_list(self) -> List[str]:
        return [op.strip() for op in self.Operations.split(",")]

    def is_available(self) -> bool:
        return self.Statut.strip().lower() == "fonctionnel"

    def supports_operation(self, operation: str) -> bool:
        return operation.lower() in [op.lower() for op in self.operations_list()]

    def __repr__(self) -> str:
        return (
            f"Machine({self.Id} | {self.NomMachine} | "
            f"cap={self.CapaciteMax} | {self.Statut})"
        )