import argparse, os, uvicorn

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=3000)
    p.add_argument("--model_path", type=str, default="default_model")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mpx", "mps"])
    p.add_argument("--reload", action="store_true", help="Reload the model or not")
    p.add_argument("--ssl-keyfile", type=str, default=None, help="Path to SSL key file")
    p.add_argument("--ssl-certfile", type=str, default=None, help="Path to SSL certificate file")
    args = p.parse_args()
    
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["MODEL_DEVICE"] = args.device

    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )

if __name__ == "__main__":
    main()
