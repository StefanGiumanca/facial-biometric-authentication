#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "dev.config.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the VisionAuth development stack.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the dev JSON config.")
    parser.add_argument(
        "--connection",
        choices=("wifi", "ngrok"),
        default=os.getenv("DEV_CONNECTION_MODE", "wifi"),
        help="Use wifi/LAN URLs or expose the backend through ngrok for mobile data.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    processes: list[subprocess.Popen[str]] = []

    env = os.environ.copy()
    env["DATABASE_URL"] = config["database_url"]
    env["ADMIN_KEY"] = config.get("admin_key", "dev-admin-key")
    env["EXPO_PUBLIC_CONNECTION_MODE"] = args.connection
    env["VITE_CONNECTION_MODE"] = args.connection

    try:
        start_postgres(config)
        ensure_database(config)

        backend_process = start_service(config, "backend", "backend", env)
        if backend_process is not None:
            processes.append(backend_process)

        ngrok_url = None
        if args.connection == "ngrok":
            ngrok_process, ngrok_url = start_ngrok_tunnel(8000)
            processes.append(ngrok_process)
            env["EXPO_PUBLIC_NGROK_API_BASE_URL"] = ngrok_url
            env["VITE_NGROK_API_BASE_URL"] = ngrok_url

        processes.extend(start_services(config, env, skip={"backend"}, connection=args.connection))
        print_urls(args.connection, ngrok_url)
        wait_until_interrupted(processes)
    except KeyboardInterrupt:
        print("\n[dev] Stopping VisionAuth stack...")
    finally:
        stop_processes(processes)

    return 0


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    with config_path.open() as file:
        return json.load(file)


def start_postgres(config: dict[str, Any]) -> None:
    postgres = config.get("postgres", {})
    if not postgres.get("enabled", False):
        return

    host = postgres.get("host", "localhost")
    port = int(postgres.get("port", 5432))
    if is_port_open(host, port):
        print(f"[postgres] Already reachable on {host}:{port}")
        return

    command = postgres.get("start_command")
    if not command:
        print("[postgres] Not reachable, and no start_command configured.")
        return

    print(f"[postgres] Starting with: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=False)

    deadline = time.time() + 20
    while time.time() < deadline:
        if is_port_open(host, port):
            print(f"[postgres] Ready on {host}:{port}")
            return
        time.sleep(0.5)

    print("[postgres] Still not reachable. Check your PostgreSQL service manually.")


def ensure_database(config: dict[str, Any]) -> None:
    postgres = config.get("postgres", {})
    if not postgres.get("enabled", False):
        return

    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    except ImportError:
        print("[postgres] psycopg2 not installed; skipping database existence check.")
        return

    host = postgres.get("host", "localhost")
    port = int(postgres.get("port", 5432))
    database = postgres.get("database", "visionauth")
    maintenance_database = postgres.get("maintenance_database", "postgres")
    user = postgres.get("user", "postgres")
    password = postgres.get("password", "postgres")

    if not is_port_open(host, port):
        return

    try:
        connection = psycopg2.connect(
            dbname=maintenance_database,
            user=user,
            password=password,
            host=host,
            port=port,
        )
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = connection.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        exists = cursor.fetchone() is not None
        if not exists:
            cursor.execute(f'CREATE DATABASE "{database}"')
            print(f"[postgres] Created database {database}")
        cursor.close()
        connection.close()
    except Exception as error:
        print(f"[postgres] Could not verify/create database: {error}")


def start_services(
    config: dict[str, Any],
    env: dict[str, str],
    *,
    skip: set[str] | None = None,
    connection: str = "wifi",
) -> list[subprocess.Popen[str]]:
    processes: list[subprocess.Popen[str]] = []
    for key, label in (("backend", "backend"), ("web_admin", "web-admin"), ("mobile", "expo")):
        if skip and key in skip:
            continue

        process = start_service(config, key, label, env, connection=connection)
        if process is not None:
            processes.append(process)
    return processes


def start_service(
    config: dict[str, Any],
    key: str,
    label: str,
    env: dict[str, str],
    *,
    connection: str = "wifi",
) -> subprocess.Popen[str] | None:
    service = config.get(key, {})
    if not service.get("enabled", False):
        return None

    command = build_service_command(key, service["command"], connection)
    cwd = ROOT / service.get("cwd", ".")
    print(f"[{label}] Starting with: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=stream_output, args=(label, process), daemon=True).start()
    time.sleep(0.5)
    return process


def build_service_command(key: str, command: list[str], connection: str) -> list[str]:
    if key != "mobile":
        return command

    expo_mode = "--tunnel" if connection == "ngrok" else "--lan"
    return [part for part in command if part not in {"--lan", "--tunnel", "--localhost"}] + [expo_mode]


def start_ngrok_tunnel(port: int) -> tuple[subprocess.Popen[str], str]:
    command = ["ngrok", "http", str(port), "--log=stdout"]
    print(f"[ngrok] Starting with: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=stream_output, args=("ngrok", process), daemon=True).start()

    try:
        public_url = wait_for_ngrok_url(port)
    except Exception:
        process.terminate()
        raise

    print(f"[ngrok] Backend tunnel: {public_url}")
    return process, public_url


def wait_for_ngrok_url(port: int) -> str:
    deadline = time.time() + 30
    last_error = "ngrok tunnel was not ready in time"

    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = str(error)
            time.sleep(0.5)
            continue

        for tunnel in payload.get("tunnels", []):
            public_url = tunnel.get("public_url", "")
            config_addr = tunnel.get("config", {}).get("addr", "")
            if public_url.startswith("https://") and str(port) in config_addr:
                return public_url

        time.sleep(0.5)

    raise RuntimeError(f"Could not read ngrok public URL: {last_error}")


def stream_output(label: str, process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return

    for line in process.stdout:
        print(f"[{label}] {line}", end="")


def print_urls(connection: str, ngrok_url: str | None = None) -> None:
    host_ip = get_lan_ip()
    print("\n[dev] VisionAuth is starting. Use these entry points:")
    print("[dev] Swagger / API docs:      http://127.0.0.1:8000/docs")
    print("[dev] Backend health/root:     http://127.0.0.1:8000")
    print("[dev] Web admin on this PC:    http://127.0.0.1:5173")
    if connection == "ngrok":
        print(f"[dev] Backend through ngrok:   {ngrok_url}")
        print("[dev] Expo Go: scan the tunnel QR code below; it works from mobile data.")
    else:
        print(f"[dev] Web admin on phone:      http://{host_ip}:5173")
        print(f"[dev] Mobile API base on phone should be: http://{host_ip}:8000")
        print("[dev] Expo Go: wait for the QR code below, then scan it with the Expo Go app.")
    print("[dev] PostgreSQL: runs in the background on localhost:5432; view it with pgAdmin/TablePlus/psql.")
    print("[dev] Stop everything started here with Ctrl+C.\n")


def wait_until_interrupted(processes: list[subprocess.Popen[str]]) -> None:
    while True:
        for process in processes:
            if process.poll() is not None:
                print(f"[dev] A service exited with code {process.returncode}.")
                return
        time.sleep(1)


def stop_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)

    deadline = time.time() + 8
    for process in processes:
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if process.poll() is None:
            process.terminate()


def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.7):
            return True
    except OSError:
        return False


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "YOUR_COMPUTER_IP"


if __name__ == "__main__":
    raise SystemExit(main())
