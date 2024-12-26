from abc import ABC, abstractmethod

from fastapi import APIRouter


class ControllerAbstract(APIRouter, ABC):
    def __init__(self, prefix: str):
        super().__init__(prefix=prefix)
        self._setup_services()
        self._setup_routes()

    @abstractmethod
    def _setup_services(self) -> None:
        pass

    @abstractmethod
    def _setup_routes(self) -> None:
        pass
