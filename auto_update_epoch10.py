#!/usr/bin/env python3
"""
Subskrybuje ntfy SSE. Gdy trening wyśle wiadomość o epoce będącej
wielokrotnością 5 (5, 10, 15, …), aktualizuje manifest.json —
wyniki.html odczyta go automatycznie przy następnym otwarciu/odświeżeniu.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

NTFY_TOPIC   = "magisterka-akustyka-serce-2026"
MANIFEST     = Path("/Users/dawidnowak/Documents/Magisterka_hifigan/manifest.json")
SAMPLES_BASE = Path("/Users/dawidnowak/Documents/Magisterka_hifigan/output_hifigan/samples")


def parse_epoch_msg(msg: str):
    """Zwraca (epoch_num, losses_dict) lub (None, None) jeśli nie pasuje."""
    m = re.search(
        r'Epoka\s+(\d+)/\d+\s+'
        r'loss_D=\+?([\d.]+).*?loss_G=\+?([\d.]+).*?'
        r'adv=\+?([\d.]+).*?fm=\+?([\d.]+).*?sd=\+?([\d.]+)'
        r'.*?\((\d+)s\)',
        msg
    )
    if not m:
        return None, None
    epoch = int(m.group(1))
    losses = {
        'loss_D': m.group(2),
        'loss_G': m.group(3),
        'adv':    m.group(4),
        'fm':     m.group(5),
        'sd':     m.group(6),
        'time_s': int(m.group(7)),
    }
    return epoch, losses


def samples_exist(epoch: int) -> bool:
    d = SAMPLES_BASE / f"epoch_{epoch:04d}"
    return d.is_dir() and any(d.glob("*.wav"))


def update_manifest(epoch: int, losses: dict):
    data = json.loads(MANIFEST.read_text())
    # Zawsze aktualizuj latest_epoch
    if epoch >= data.get('latest_epoch', 0):
        data['latest_epoch']  = epoch
        data['latest_losses'] = losses
    # Co 5 epok z próbkami — dodaj do milestones
    if epoch % 5 == 0 and samples_exist(epoch):
        data['epochs'][str(epoch)] = losses
        print(f"  milestone E{epoch} dodany do epochs.", flush=True)
    MANIFEST.write_text(json.dumps(data, indent=2))
    print(f"manifest.json zaktualizowany — latest_epoch={epoch}.", flush=True)


def main():
    print(f"Subskrybuję ntfy: {NTFY_TOPIC} …", flush=True)
    url = f"https://ntfy.sh/{NTFY_TOPIC}/sse"

    with urllib.request.urlopen(url, timeout=None) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            try:
                data = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue

            msg = data.get("message", "")
            if not msg:
                continue

            epoch, losses = parse_epoch_msg(msg)
            if epoch is None:
                continue

            print(f"  ntfy → Epoka {epoch}: {losses}", flush=True)

            update_manifest(epoch, losses)


if __name__ == "__main__":
    main()
