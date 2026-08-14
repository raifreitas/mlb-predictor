"""Dispara runner-mlb via workflow_dispatch (respaldo de cron-job.org).

Uso:
    python disparar_runner.py --token ghp_xxx
    python disparar_runner.py                          (usa env DISPATCH_PAT)
    python disparar_runner.py --horizonte 2000 --resguardo 1
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = ("https://api.github.com/repos/{repo}/actions/"
       "workflows/{workflow}/dispatches")


def main():
    parser = argparse.ArgumentParser(
        description="Dispara el workflow runner-mlb manualmente.")
    parser.add_argument("--token", default=os.environ.get("DISPATCH_PAT", ""),
                        help="PAT con permiso Actions:write (o env DISPATCH_PAT)")
    parser.add_argument("--repo", default="raifreitas/mlb-predictor")
    parser.add_argument("--workflow", default="runner-mlb.yml")
    parser.add_argument("--horizonte", default=None)
    parser.add_argument("--resguardo", default=None)
    args = parser.parse_args()

    if not args.token:
        print("Falta el token: --token o variable DISPATCH_PAT.", file=sys.stderr)
        sys.exit(2)

    body = {"ref": "main"}
    inputs = {}
    if args.horizonte:
        inputs["horizonte"] = args.horizonte
    if args.resguardo:
        inputs["resguardo"] = args.resguardo
    if inputs:
        body["inputs"] = inputs

    peticion = urllib.request.Request(
        API.format(repo=args.repo, workflow=args.workflow),
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {args.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(peticion) as respuesta:
            print(f"Dispatch OK (HTTP {respuesta.status}): "
                  f"{args.repo}/{args.workflow}")
    except urllib.error.HTTPError as e:
        print(f"Error HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
