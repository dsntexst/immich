#!/usr/bin/env python3
"""
Deploy Immich to Azure Container Apps with SSL/TLS.

Zero external dependencies — uses only the Python standard library to call
Azure Resource Manager REST APIs directly, so it can run from Claude Code
or any environment with Python 3.10+.

Usage:
    python3 deploy_immich.py --azure --force \
        --tenant-id <TENANT_ID> \
        --subscription-id <SUBSCRIPTION_ID> \
        --client-id <CLIENT_ID> \
        --client-secret <CLIENT_SECRET> \
        --dockerhub-username <USERNAME> \
        --dockerhub-token <TOKEN>
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import string
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("deploy_immich")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_REGION = "eastus"
DEFAULT_RESOURCE_GROUP = "immich-rg"
DEFAULT_ENVIRONMENT = "immich-env"
DEFAULT_STORAGE_ACCOUNT = "immichstorage"
DEFAULT_IMMICH_VERSION = "release"

DEFAULT_SERVER_IMAGE = "ghcr.io/immich-app/immich-server"
DEFAULT_ML_IMAGE = "ghcr.io/immich-app/immich-machine-learning"
DEFAULT_REDIS_IMAGE = "docker.io/valkey/valkey:9"
DEFAULT_POSTGRES_IMAGE = (
    "ghcr.io/immich-app/postgres:14-vectorchord0.4.3-pgvectors0.2.0"
)

FILESHARE_UPLOAD = "immich-upload"
FILESHARE_DB = "immich-db"
FILESHARE_ML_CACHE = "immich-ml-cache"

ARM_BASE = "https://management.azure.com"
LOGIN_BASE = "https://login.microsoftonline.com"

# Azure REST API versions
API_RESOURCE_GROUPS = "2024-03-01"
API_STORAGE = "2023-05-01"
API_LOG_ANALYTICS = "2023-09-01"
API_CONTAINER_APPS_ENV = "2024-03-01"
API_CONTAINER_APPS = "2024-03-01"


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------
class AzureError(Exception):
    """Raised when an Azure API call fails."""

    def __init__(self, status: int, body: str, url: str = ""):
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status} for {url}: {body[:500]}")


def _http(
    method: str,
    url: str,
    *,
    body: dict | str | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    backoff: float = 2.0,
) -> dict | str:
    """Perform an HTTP request with retries. Returns parsed JSON or raw text."""
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)

    data: bytes | None = None
    if body is not None:
        data = (json.dumps(body) if isinstance(body, dict) else body).encode()

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        if attempt > 0:
            wait = backoff * (2 ** (attempt - 1))
            log.warning("  Retry %d/%d in %.0fs ...", attempt, retries, wait)
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode()
                if raw and resp.headers.get("Content-Type", "").startswith("application/json"):
                    return json.loads(raw)
                return raw
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode() if exc.fp else ""
            # 404 = not found (often expected), 409 = conflict — no retry
            if exc.code in (404, 409):
                raise AzureError(exc.code, err_body, url) from exc
            last_exc = AzureError(exc.code, err_body, url)
            log.warning("  HTTP %d: %s", exc.code, err_body[:200])
        except urllib.error.URLError as exc:
            last_exc = exc
            log.warning("  Network error: %s", exc.reason)

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Azure authentication (OAuth2 client-credentials flow)
# ---------------------------------------------------------------------------
class AzureClient:
    """Thin wrapper around Azure ARM REST API using stdlib only."""

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        subscription_id: str,
    ) -> None:
        self.subscription_id = subscription_id
        self._token = self._acquire_token(tenant_id, client_id, client_secret)

    # ---- auth ----
    @staticmethod
    def _acquire_token(tenant_id: str, client_id: str, client_secret: str) -> str:
        log.info("Authenticating to Azure via service principal ...")
        url = f"{LOGIN_BASE}/{tenant_id}/oauth2/v2.0/token"
        payload = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://management.azure.com/.default",
        }).encode()
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                log.info("Authentication successful.")
                return data["access_token"]
        except urllib.error.HTTPError as exc:
            body = exc.read().decode() if exc.fp else ""
            log.error("Authentication failed (HTTP %d): %s", exc.code, body[:500])
            sys.exit(1)

    # ---- low-level ARM helpers ----
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _arm_url(self, path: str, api_version: str) -> str:
        base = f"{ARM_BASE}/subscriptions/{self.subscription_id}"
        sep = "&" if "?" in path else "?"
        return f"{base}{path}{sep}api-version={api_version}"

    def arm_get(self, path: str, api_version: str) -> dict | str:
        return _http("GET", self._arm_url(path, api_version),
                      headers=self._auth_headers())

    def arm_put(self, path: str, api_version: str, body: dict) -> dict | str:
        return _http("PUT", self._arm_url(path, api_version),
                      body=body, headers=self._auth_headers())

    def arm_patch(self, path: str, api_version: str, body: dict) -> dict | str:
        return _http("PATCH", self._arm_url(path, api_version),
                      body=body, headers=self._auth_headers())

    def arm_delete(self, path: str, api_version: str) -> dict | str:
        return _http("DELETE", self._arm_url(path, api_version),
                      headers=self._auth_headers())

    def arm_post(self, path: str, api_version: str,
                 body: dict | None = None) -> dict | str:
        return _http("POST", self._arm_url(path, api_version),
                      body=body or {}, headers=self._auth_headers())

    def resource_exists(self, path: str, api_version: str) -> bool:
        try:
            self.arm_get(path, api_version)
            return True
        except AzureError as exc:
            if exc.status == 404:
                return False
            raise

    # ---- resource group ----
    def ensure_resource_group(self, name: str, location: str) -> None:
        log.info("Ensuring resource group '%s' in '%s' ...", name, location)
        self.arm_put(
            f"/resourceGroups/{name}", API_RESOURCE_GROUPS,
            {"location": location},
        )
        log.info("Resource group '%s' ready.", name)

    # ---- storage ----
    def ensure_storage_account(self, rg: str, name: str, location: str) -> None:
        log.info("Ensuring storage account '%s' ...", name)
        path = f"/resourceGroups/{rg}/providers/Microsoft.Storage/storageAccounts/{name}"
        self.arm_put(path, API_STORAGE, {
            "location": location,
            "sku": {"name": "Standard_LRS"},
            "kind": "StorageV2",
            "properties": {},
        })
        # Wait for provisioning
        for _ in range(30):
            resp = self.arm_get(path, API_STORAGE)
            state = resp.get("properties", {}).get("provisioningState", "") if isinstance(resp, dict) else ""
            if state == "Succeeded":
                log.info("Storage account '%s' ready.", name)
                return
            time.sleep(5)
        log.warning("Storage account may still be provisioning.")

    def get_storage_keys(self, rg: str, name: str) -> str:
        path = (f"/resourceGroups/{rg}/providers/Microsoft.Storage"
                f"/storageAccounts/{name}/listKeys")
        resp = self.arm_post(path, API_STORAGE)
        return resp["keys"][0]["value"]

    def ensure_fileshare(self, rg: str, account: str, share: str) -> None:
        log.info("Ensuring file share '%s' ...", share)
        path = (f"/resourceGroups/{rg}/providers/Microsoft.Storage"
                f"/storageAccounts/{account}/fileServices/default/shares/{share}")
        self.arm_put(path, API_STORAGE, {
            "properties": {"shareQuota": 100},
        })
        log.info("File share '%s' ready.", share)

    # ---- Log Analytics workspace ----
    def ensure_log_analytics(self, rg: str, location: str,
                             name: str = "immich-logs") -> tuple[str, str]:
        log.info("Ensuring Log Analytics workspace '%s' ...", name)
        path = (f"/resourceGroups/{rg}/providers"
                f"/Microsoft.OperationalInsights/workspaces/{name}")
        self.arm_put(path, API_LOG_ANALYTICS, {
            "location": location,
            "properties": {"sku": {"name": "PerGB2018"}, "retentionInDays": 30},
        })
        # Wait for provisioning
        for _ in range(30):
            ws = self.arm_get(path, API_LOG_ANALYTICS)
            state = ws.get("properties", {}).get("provisioningState", "") if isinstance(ws, dict) else ""
            if state == "Succeeded":
                break
            time.sleep(5)

        ws = self.arm_get(path, API_LOG_ANALYTICS)
        customer_id = ws["properties"]["customerId"]

        keys_resp = self.arm_post(f"{path}/sharedKeys", API_LOG_ANALYTICS)
        primary_key = keys_resp["primarySharedKey"]
        log.info("Log Analytics workspace '%s' ready.", name)
        return customer_id, primary_key

    # ---- Container Apps environment ----
    def ensure_container_apps_env(
        self, rg: str, location: str, env_name: str,
        workspace_id: str, workspace_key: str,
    ) -> None:
        log.info("Ensuring Container Apps environment '%s' ...", env_name)
        path = (f"/resourceGroups/{rg}/providers"
                f"/Microsoft.App/managedEnvironments/{env_name}")
        self.arm_put(path, API_CONTAINER_APPS_ENV, {
            "location": location,
            "properties": {
                "appLogsConfiguration": {
                    "destination": "log-analytics",
                    "logAnalyticsConfiguration": {
                        "customerId": workspace_id,
                        "sharedKey": workspace_key,
                    },
                },
            },
        })
        # Wait — environment creation can take a few minutes
        log.info("Waiting for environment provisioning (this may take 2-3 min) ...")
        for i in range(60):
            env = self.arm_get(path, API_CONTAINER_APPS_ENV)
            state = env.get("properties", {}).get("provisioningState", "") if isinstance(env, dict) else ""
            if state == "Succeeded":
                log.info("Container Apps environment '%s' ready.", env_name)
                return
            if state == "Failed":
                log.error("Environment provisioning failed: %s", env)
                sys.exit(1)
            time.sleep(5)
        log.warning("Environment may still be provisioning — continuing anyway.")

    def bind_storage_to_env(
        self, rg: str, env_name: str, storage_name: str,
        account_name: str, account_key: str, share_name: str,
    ) -> None:
        log.info("Binding storage '%s' -> share '%s' to environment ...",
                 storage_name, share_name)
        path = (f"/resourceGroups/{rg}/providers/Microsoft.App"
                f"/managedEnvironments/{env_name}/storages/{storage_name}")
        self.arm_put(path, API_CONTAINER_APPS_ENV, {
            "properties": {
                "azureFile": {
                    "accountName": account_name,
                    "accountKey": account_key,
                    "shareName": share_name,
                    "accessMode": "ReadWrite",
                },
            },
        })
        log.info("Storage '%s' bound.", storage_name)

    # ---- Container Apps ----
    def _env_id(self, rg: str, env_name: str) -> str:
        return (f"/subscriptions/{self.subscription_id}/resourceGroups/{rg}"
                f"/providers/Microsoft.App/managedEnvironments/{env_name}")

    def deploy_container_app(
        self,
        rg: str,
        env_name: str,
        app_name: str,
        image: str,
        *,
        target_port: int | None = None,
        ingress_type: str | None = None,  # "external" | "internal"
        env_vars: list[dict] | None = None,
        cpu: float = 1.0,
        memory: str = "2Gi",
        min_replicas: int = 1,
        max_replicas: int = 1,
        volumes: list[dict] | None = None,
        volume_mounts: list[dict] | None = None,
        registry_server: str | None = None,
        registry_username: str | None = None,
        registry_password: str | None = None,
        force: bool = False,
    ) -> dict | str | None:
        log.info("Deploying container app '%s' (image=%s) ...", app_name, image)
        path = (f"/resourceGroups/{rg}/providers"
                f"/Microsoft.App/containerApps/{app_name}")

        if self.resource_exists(path, API_CONTAINER_APPS):
            if not force:
                log.info("  '%s' already exists (use --force to recreate).", app_name)
                return None
            log.info("  Deleting existing '%s' ...", app_name)
            self.arm_delete(path, API_CONTAINER_APPS)
            # Wait for deletion
            for _ in range(30):
                if not self.resource_exists(path, API_CONTAINER_APPS):
                    break
                time.sleep(5)

        # Build container definition
        container: dict = {
            "name": app_name,
            "image": image,
            "resources": {
                "cpu": cpu,
                "memory": memory,
            },
        }
        if env_vars:
            container["env"] = env_vars
        if volume_mounts:
            container["volumeMounts"] = volume_mounts

        # Build ingress config
        ingress_cfg = None
        if ingress_type and target_port:
            ingress_cfg = {
                "external": ingress_type == "external",
                "targetPort": target_port,
                "transport": "http",
                "allowInsecure": False,
            }

        # Build template
        template: dict = {
            "containers": [container],
            "scale": {
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
            },
        }
        if volumes:
            template["volumes"] = volumes

        # Build configuration
        configuration: dict = {}
        if ingress_cfg:
            configuration["ingress"] = ingress_cfg

        # Registry credentials
        if registry_server and registry_username and registry_password:
            configuration["registries"] = [{
                "server": registry_server,
                "username": registry_username,
                "passwordSecretRef": "registry-password",
            }]
            configuration["secrets"] = [{
                "name": "registry-password",
                "value": registry_password,
            }]

        body = {
            "location": "",  # filled by caller via env lookup
            "properties": {
                "managedEnvironmentId": self._env_id(rg, env_name),
                "configuration": configuration,
                "template": template,
            },
        }

        # Resolve the environment location
        env_path = (f"/resourceGroups/{rg}/providers"
                    f"/Microsoft.App/managedEnvironments/{env_name}")
        env_info = self.arm_get(env_path, API_CONTAINER_APPS_ENV)
        body["location"] = env_info["location"] if isinstance(env_info, dict) else "eastus"

        result = self.arm_put(path, API_CONTAINER_APPS, body)

        # Wait for provisioning
        log.info("  Waiting for '%s' to provision ...", app_name)
        for _ in range(60):
            app = self.arm_get(path, API_CONTAINER_APPS)
            state = app.get("properties", {}).get("provisioningState", "") if isinstance(app, dict) else ""
            if state == "Succeeded":
                log.info("  Container app '%s' ready.", app_name)
                return app
            if state == "Failed":
                log.error("  Container app '%s' FAILED: %s", app_name, app)
                sys.exit(1)
            time.sleep(5)

        log.warning("  '%s' may still be provisioning.", app_name)
        return result

    def get_app_fqdn(self, rg: str, app_name: str) -> str | None:
        path = (f"/resourceGroups/{rg}/providers"
                f"/Microsoft.App/containerApps/{app_name}")
        try:
            app = self.arm_get(path, API_CONTAINER_APPS)
            if isinstance(app, dict):
                return (app.get("properties", {})
                        .get("configuration", {})
                        .get("ingress", {})
                        .get("fqdn"))
        except AzureError:
            pass
        return None

    def add_custom_domain(self, rg: str, app_name: str, hostname: str,
                          env_name: str, bind_cert: bool) -> None:
        log.info("Adding custom domain '%s' to '%s' ...", hostname, app_name)
        path = (f"/resourceGroups/{rg}/providers"
                f"/Microsoft.App/containerApps/{app_name}")
        app = self.arm_get(path, API_CONTAINER_APPS)
        if not isinstance(app, dict):
            log.error("Could not read app config for custom domain.")
            return

        ingress = app.get("properties", {}).get("configuration", {}).get("ingress", {})
        custom_domains = ingress.get("customDomains") or []
        domain_entry: dict = {"name": hostname, "bindingType": "Disabled"}
        if bind_cert:
            domain_entry["bindingType"] = "SniEnabled"
            # Managed certificate — Azure provisions automatically
            domain_entry["certificateId"] = (
                f"{self._env_id(rg, env_name)}"
                f"/managedCertificates/{hostname.replace('.', '-')}"
            )
        custom_domains.append(domain_entry)
        ingress["customDomains"] = custom_domains
        app["properties"]["configuration"]["ingress"] = ingress

        self.arm_put(path, API_CONTAINER_APPS, app)
        log.info("Custom domain '%s' configured.", hostname)

    def delete_resource_group(self, name: str) -> None:
        log.info("Deleting resource group '%s' (async) ...", name)
        url = (f"{ARM_BASE}/subscriptions/{self.subscription_id}"
               f"/resourceGroups/{name}?api-version={API_RESOURCE_GROUPS}")
        _http("DELETE", url, headers=self._auth_headers())
        log.info("Deletion initiated. Check Azure portal for status.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_password(length: int = 32) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def env_var(name: str, value: str) -> dict:
    """Build a Container Apps environment variable entry."""
    return {"name": name, "value": value}


def secret_env_var(name: str, secret_name: str) -> dict:
    """Build a Container Apps env var that references a secret."""
    return {"name": name, "secretRef": secret_name}


# ---------------------------------------------------------------------------
# Deployment orchestration
# ---------------------------------------------------------------------------
def deploy(args: argparse.Namespace) -> None:
    region = args.region
    rg = args.resource_group
    env_name = args.environment
    storage_account = args.storage_account
    immich_version = args.immich_version
    force = args.force

    db_password = args.db_password or generate_password()

    client = AzureClient(
        args.tenant_id, args.client_id, args.client_secret, args.subscription_id,
    )

    # Registry credentials for Docker Hub images (e.g. valkey/redis)
    reg_server = reg_user = reg_pass = None
    if args.dockerhub_username and args.dockerhub_token:
        reg_server = "docker.io"
        reg_user = args.dockerhub_username
        reg_pass = args.dockerhub_token

    # ---- 1. Resource group ----
    client.ensure_resource_group(rg, region)

    # ---- 2. Log Analytics ----
    workspace_id, workspace_key = client.ensure_log_analytics(rg, region)

    # ---- 3. Container Apps environment ----
    client.ensure_container_apps_env(rg, region, env_name,
                                     workspace_id, workspace_key)

    # ---- 4. Storage (Azure Files) ----
    client.ensure_storage_account(rg, storage_account, region)
    account_key = client.get_storage_keys(rg, storage_account)
    for share in (FILESHARE_UPLOAD, FILESHARE_DB, FILESHARE_ML_CACHE):
        client.ensure_fileshare(rg, storage_account, share)

    # Bind storage mounts to environment
    client.bind_storage_to_env(rg, env_name, "uploadstorage",
                               storage_account, account_key, FILESHARE_UPLOAD)
    client.bind_storage_to_env(rg, env_name, "dbstorage",
                               storage_account, account_key, FILESHARE_DB)
    client.bind_storage_to_env(rg, env_name, "mlcachestorage",
                               storage_account, account_key, FILESHARE_ML_CACHE)

    # ---- 5. PostgreSQL (internal) ----
    client.deploy_container_app(
        rg, env_name, "immich-database",
        DEFAULT_POSTGRES_IMAGE,
        target_port=5432,
        ingress_type="internal",
        cpu=1.0, memory="2Gi",
        env_vars=[
            env_var("POSTGRES_PASSWORD", db_password),
            env_var("POSTGRES_USER", "postgres"),
            env_var("POSTGRES_DB", "immich"),
            env_var("POSTGRES_INITDB_ARGS", "--data-checksums"),
        ],
        volumes=[{
            "name": "db-vol",
            "storageName": "dbstorage",
            "storageType": "AzureFile",
        }],
        volume_mounts=[{
            "volumeName": "db-vol",
            "mountPath": "/var/lib/postgresql/data",
        }],
        force=force,
    )

    # ---- 6. Redis (internal) ----
    client.deploy_container_app(
        rg, env_name, "immich-redis",
        DEFAULT_REDIS_IMAGE,
        target_port=6379,
        ingress_type="internal",
        cpu=0.5, memory="1Gi",
        registry_server=reg_server,
        registry_username=reg_user,
        registry_password=reg_pass,
        force=force,
    )

    # ---- 7. Machine Learning (internal) ----
    client.deploy_container_app(
        rg, env_name, "immich-ml",
        f"{DEFAULT_ML_IMAGE}:{immich_version}",
        target_port=3003,
        ingress_type="internal",
        cpu=2.0, memory="4Gi",
        env_vars=[
            env_var("IMMICH_PORT", "3003"),
            env_var("MACHINE_LEARNING_CACHE_FOLDER", "/cache"),
            env_var("MACHINE_LEARNING_WORKERS", "1"),
            env_var("MACHINE_LEARNING_WORKER_TIMEOUT", "120"),
        ],
        volumes=[{
            "name": "mlcache-vol",
            "storageName": "mlcachestorage",
            "storageType": "AzureFile",
        }],
        volume_mounts=[{
            "volumeName": "mlcache-vol",
            "mountPath": "/cache",
        }],
        force=force,
    )

    # ---- 8. Immich Server (external — HTTPS/SSL) ----
    client.deploy_container_app(
        rg, env_name, "immich-server",
        f"{DEFAULT_SERVER_IMAGE}:{immich_version}",
        target_port=2283,
        ingress_type="external",
        cpu=2.0, memory="4Gi",
        env_vars=[
            env_var("IMMICH_PORT", "2283"),
            env_var("DB_HOSTNAME", "immich-database"),
            env_var("DB_PORT", "5432"),
            env_var("DB_USERNAME", "postgres"),
            env_var("DB_PASSWORD", db_password),
            env_var("DB_DATABASE_NAME", "immich"),
            env_var("REDIS_HOSTNAME", "immich-redis"),
            env_var("REDIS_PORT", "6379"),
            env_var("IMMICH_MACHINE_LEARNING_URL", "http://immich-ml:3003"),
        ],
        volumes=[{
            "name": "upload-vol",
            "storageName": "uploadstorage",
            "storageType": "AzureFile",
        }],
        volume_mounts=[{
            "volumeName": "upload-vol",
            "mountPath": "/data",
        }],
        force=force,
    )

    # ---- 9. Print endpoint ----
    fqdn = client.get_app_fqdn(rg, "immich-server")
    log.info("=" * 60)
    if fqdn:
        log.info("Immich is deployed and available at:")
        log.info("")
        log.info("  https://%s", fqdn)
        log.info("")
        log.info("SSL/TLS is automatically managed by Azure Container Apps.")
    else:
        log.info("Deployment submitted. Check the Azure portal for the URL.")
    log.info("=" * 60)

    # ---- 10. Custom domain (optional) ----
    if args.custom_domain:
        client.add_custom_domain(
            rg, "immich-server", args.custom_domain,
            env_name, args.enable_managed_cert,
        )
        if fqdn:
            log.info("Point a CNAME for '%s' -> '%s'", args.custom_domain, fqdn)

    log.info("Deployment complete.")


def teardown(args: argparse.Namespace) -> None:
    client = AzureClient(
        args.tenant_id, args.client_id, args.client_secret, args.subscription_id,
    )
    client.delete_resource_group(args.resource_group)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deploy Immich photo management to Azure with SSL/TLS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full deployment (run directly from Claude Code — no az CLI needed)
  python3 deploy_immich.py --azure --force \\
    --tenant-id <TENANT_ID> \\
    --subscription-id <SUBSCRIPTION_ID> \\
    --client-id <CLIENT_ID> \\
    --client-secret <CLIENT_SECRET> \\
    --dockerhub-username <USERNAME> \\
    --dockerhub-token <TOKEN>

  # Deploy with custom domain + managed SSL certificate
  python3 deploy_immich.py --azure --force \\
    --tenant-id <TENANT_ID> \\
    --subscription-id <SUBSCRIPTION_ID> \\
    --client-id <CLIENT_ID> \\
    --client-secret <CLIENT_SECRET> \\
    --custom-domain photos.example.com \\
    --enable-managed-cert

  # Tear down all resources
  python3 deploy_immich.py --azure --teardown \\
    --tenant-id <TENANT_ID> \\
    --subscription-id <SUBSCRIPTION_ID> \\
    --client-id <CLIENT_ID> \\
    --client-secret <CLIENT_SECRET>
""",
    )

    p.add_argument("--azure", action="store_true", required=True,
                   help="Deploy to Azure (required flag)")
    p.add_argument("--force", action="store_true", default=False,
                   help="Force re-creation of existing container apps")
    p.add_argument("--teardown", action="store_true", default=False,
                   help="Remove all deployed resources")

    auth = p.add_argument_group("Azure Authentication")
    auth.add_argument("--tenant-id", required=True)
    auth.add_argument("--subscription-id", required=True)
    auth.add_argument("--client-id", required=True)
    auth.add_argument("--client-secret", required=True)

    docker = p.add_argument_group("Docker Hub Credentials (optional)")
    docker.add_argument("--dockerhub-username", default=None,
                        help="Docker Hub username (for rate-limited images like valkey)")
    docker.add_argument("--dockerhub-token", default=None,
                        help="Docker Hub access token")

    opts = p.add_argument_group("Deployment Options")
    opts.add_argument("--region", default=DEFAULT_REGION,
                      help=f"Azure region (default: {DEFAULT_REGION})")
    opts.add_argument("--resource-group", default=DEFAULT_RESOURCE_GROUP,
                      help=f"Resource group (default: {DEFAULT_RESOURCE_GROUP})")
    opts.add_argument("--environment", default=DEFAULT_ENVIRONMENT,
                      help=f"Container Apps env (default: {DEFAULT_ENVIRONMENT})")
    opts.add_argument("--storage-account", default=DEFAULT_STORAGE_ACCOUNT,
                      help=f"Storage account (default: {DEFAULT_STORAGE_ACCOUNT})")
    opts.add_argument("--immich-version", default=DEFAULT_IMMICH_VERSION,
                      help=f"Immich version tag (default: {DEFAULT_IMMICH_VERSION})")
    opts.add_argument("--db-password", default=None,
                      help="PostgreSQL password (auto-generated if omitted)")

    ssl = p.add_argument_group("SSL / Custom Domain")
    ssl.add_argument("--custom-domain", default=None,
                     help="Custom domain (e.g. photos.example.com)")
    ssl.add_argument("--enable-managed-cert", action="store_true", default=False,
                     help="Provision Azure-managed SSL cert for custom domain")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.teardown:
        teardown(args)
    else:
        deploy(args)


if __name__ == "__main__":
    main()
