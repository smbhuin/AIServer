import uvicorn

from settings import (
    ConfigFileSettings,
)

import api

def main():

    config = ConfigFileSettings()

    server_settings = config.server
    models_settings = config.models
    app = api.create_app(server_settings, models_settings)

    uvicorn.run(
        app,
        host=server_settings.host,
        port=int(server_settings.port),
        ssl_keyfile=server_settings.ssl_keyfile,
        ssl_certfile=server_settings.ssl_certfile,
        timeout_keep_alive=300
    )

if __name__ == "__main__":
    main()