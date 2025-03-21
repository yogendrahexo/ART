import asyncio
import httpx
from fastapi import FastAPI
import os
from pydantic import BaseModel
import sys
import uvicorn


class Service(BaseModel):
    host: str
    port: int
    # process: asyncio.subprocess.Process | None = None

    class Config:
        arbitrary_types_allowed = True
        # exclude = {"process"}
        extra = "allow"

    async def start_openai_server(self) -> None: ...

    async def stop_openai_server(self) -> None: ...

    async def root(self) -> dict[str, str]:
        return {"message": f"Serving on {self.host}:{self.port}"}

    async def serve(self, serve: bool) -> "Service":
        if not serve:
            return self
        # Ensure logs directory exists
        os.makedirs("./logs", exist_ok=True)

        # Serialize the service object to JSON
        service_json = self.model_dump_json()

        # Open log file
        log_file = open("./logs/unsloth.log", "w")

        # Start the service as an asyncio subprocess
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            __file__,
            service_json,
            stdout=log_file,
            stderr=log_file,
        )

        # Create a client for the service
        client = httpx.AsyncClient(base_url=f"http://{self.host}:{self.port}")

        # Define the root endpoint
        async def root() -> dict[str, str]:
            response = await client.get("/")
            response.raise_for_status()
            return response.json()

        # Set the root endpoint
        self.root = root

        print(f"Started service on {self.host}:{self.port} (PID: {self.process.pid})")
        return self


if __name__ == "__main__":
    service = Service.model_validate_json(sys.argv[1])
    app = FastAPI()
    app.get("/")(service.root)
    uvicorn.run(app, host=service.host, port=service.port)
