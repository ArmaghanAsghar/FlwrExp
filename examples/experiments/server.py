from flwr.server.app import start_server
from flwr.server.app import ServerConfig


if __name__ == "__main__":
    start_server(config=ServerConfig(num_rounds=3))