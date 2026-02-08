FROM python:3.12 AS builder

RUN pip install uv

WORKDIR /app

COPY gymnasium/requirements.txt .

RUN uv venv --python 3.12 && \
    .venv/bin/pip install -r requirements.txt

FROM python:3.12

WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY gymnasium/ .

EXPOSE 5555

CMD [".venv/bin/python", "env_server.py"]