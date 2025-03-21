from fastapi import FastAPI
from pydantic import BaseModel
import sys
import uvicorn


class Service(BaseModel):
    host: str
    port: int


class ServiceImpl:
    app = FastAPI()

    def __init__(self, service: Service) -> None:
        self.service = service

    @app.get("/")
    async def root(self) -> dict[str, str]:
        return {"message": f"Serving on {self.service.host}:{self.service.port}"}


if __name__ == "__main__":
    service = Service.model_validate_json(sys.argv[1])
    service_impl = ServiceImpl(service)
    uvicorn.run(service_impl.app, host=service.host, port=service.port)
