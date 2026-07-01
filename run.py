"""Launch the web app.  Usage:  python run.py  ->  http://127.0.0.1:8000"""

import uvicorn

if __name__ == "__main__":
    # Load .env before importing app modules so config.py sees the vars at startup.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; rely on shell env vars

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
