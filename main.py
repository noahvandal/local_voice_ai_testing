from src.server import create_app
import uvicorn
import os

def main():
    reload_flag = os.getenv("RELOAD", "0") == "1"
    port = int(os.getenv("PORT", 8000))

    if reload_flag:
        uvicorn.run(
            "src.server:create_app",
            factory=True,
            host="0.0.0.0",
            port=port,
            reload=True,
        )
    else:
        # Instantiate app once and run without the reloader parent
        uvicorn.run(create_app(), host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
