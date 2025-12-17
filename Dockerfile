# ─ Use official slim Python runtime ────────────────────────────────────────
FROM python:3.12-slim
# # ─ Use the official Oryx Python runtime (includes Kudu & App Service wiring) ─
# FROM mcr.microsoft.com/oryx/python:3.12

# ─  Noninteractive APT + timezone ───────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Chicago \
    WEBSITES_ENABLE_APP_SERVICE_STORAGE=true\
    WEBSITE_SKIP_RUNNING_KUDUAGENT=false

# ─ Install system deps + MS ODBC for SQL Server ────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tzdata \
    unixodbc \
    unixodbc-dev \
    libpq-dev \
    gcc \
    gnupg2 \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-archive-keyring.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ─ App dependencies ─────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ─ Copy all of your app code ───────
COPY . .

# ─ Expose & launch FastAPI app ────────────────────────────────────────
EXPOSE 8000
CMD ["ddtrace-run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
