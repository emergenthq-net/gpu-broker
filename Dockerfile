# gpu-broker API. GPU work happens in the model servers; this image only needs nvidia-smi
# (injected by the NVIDIA Container Toolkit with `capabilities: [utility]`), the docker CLI
# for the docker driver, ssh for the proxmox driver, and git + hf for downloads.
FROM python:3.12-slim-trixie

RUN apt-get update \
 && apt-get install -y --no-install-recommends git openssh-client curl ca-certificates docker-cli \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY gpu_broker ./gpu_broker
RUN pip install --no-cache-dir '.[download]'

RUN useradd --system --uid 10001 --create-home broker \
 && mkdir -p /etc/gpu-broker /var/lib/gpu-broker /var/log/gpu-broker /models \
 && chown -R broker:broker /etc/gpu-broker /var/lib/gpu-broker /var/log/gpu-broker /models
USER broker

# Config: /etc/gpu-broker/config.yaml if mounted (else built-in defaults); BROKER_TOKEN from the env.
EXPOSE 8095
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8095/health || exit 1
ENTRYPOINT ["gpu-broker"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8095"]
