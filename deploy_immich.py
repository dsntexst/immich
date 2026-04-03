#!/usr/bin/env python3
"""
Deploy Immich to Azure Container Apps with SSL/TLS.

Usage:
    python3 deploy_immich.py --azure --force \
        --tenant-id <TENANT_ID> \
        --subscription-id <SUBSCRIPTION_ID> \
        --client-id <CLIENT_ID> \
        --client-secret <CLIENT_SECRET> \
        --dockerhub-username <USERNAME> \
        --dockerhub-token <TOKEN>

This script deploys the full Immich stack (server, machine-learning, Redis,
PostgreSQL) to Azure Container Apps with automatic HTTPS/SSL termination.
"""

import argparse
import json
import logging
import os
import secrets
import string
import subprocess
import sys
import time

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
DEFAULT_POSTGRES_IMAGE = "ghcr.io/immich-app/postgres:14-vectorchord0.4.3-pgvectors0.2.0"

FILESHARE_UPLOAD = "immich-upload"
FILESHARE_DB = "immich-db"
FILESHARE_ML_CACHE = "immich-ml-cache"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run(cmd: list[str], *, check: bool = True, capture: bool = True,
        retries: int = 0, backoff: float = 2.0, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess command with optional retries and exponential backoff."""
    attempts = 0
    last_exc = None
    while attempts <= retries:
        if attempts > 0:
            wait = backoff * (2 ** (attempts - 1))
            log.warning("Retry %d/%d in %.0fs ...", attempts, retries, wait)
            time.sleep(wait)
        try:
            log.debug("$ %s", " ".join(cmd))
            result = subprocess.run(
                cmd, check=check, text=True,
                capture_output=capture, **kwargs,
            )
            return result
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            log.warning("Command failed (rc=%d): %s", exc.returncode, exc.stderr or exc.stdout)
            attempts += 1
    raise last_exc  # type: ignore[misc]


def az(*args: str, retries: int = 0, parse_json: bool = True):
    """Run an Azure CLI command and optionally parse JSON output."""
    cmd = ["az"] + list(args)
    if parse_json and "--output" not in args:
        cmd += ["--output", "json"]
    result = run(cmd, retries=retries)
    if parse_json and result.stdout.strip():
        return json.loads(result.stdout)
    return result.stdout


def generate_password(length: int = 32) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


# ---------------------------------------------------------------------------
# Azure login
# ---------------------------------------------------------------------------
def azure_login(tenant_id: str, client_id: str, client_secret: str,
                subscription_id: str) -> None:
    log.info("Authenticating to Azure (service principal) ...")
    az(
        "login", "--service-principal",
        "--tenant", tenant_id,
        "--username", client_id,
        "--password", client_secret,
        retries=3, parse_json=False,
    )
    az("account", "set", "--subscription", subscription_id,
       parse_json=False)
    log.info("Subscription set to %s", subscription_id)


# ---------------------------------------------------------------------------
# Resource group
# ---------------------------------------------------------------------------
def ensure_resource_group(name: str, location: str) -> None:
    log.info("Ensuring resource group '%s' in '%s' ...", name, location)
    existing = az("group", "list", "--query", f"[?name=='{name}']")
    if existing:
        log.info("Resource group '%s' already exists.", name)
        return
    az("group", "create", "--name", name, "--location", location)
    log.info("Resource group '%s' created.", name)


# ---------------------------------------------------------------------------
# Log Analytics workspace (required by Container Apps environment)
# ---------------------------------------------------------------------------
def ensure_log_analytics(resource_group: str, location: str,
                         workspace_name: str = "immich-logs") -> tuple[str, str]:
    log.info("Ensuring Log Analytics workspace '%s' ...", workspace_name)
    try:
        ws = az(
            "monitor", "log-analytics", "workspace", "show",
            "--resource-group", resource_group,
            "--workspace-name", workspace_name,
        )
    except subprocess.CalledProcessError:
        ws = az(
            "monitor", "log-analytics", "workspace", "create",
            "--resource-group", resource_group,
            "--workspace-name", workspace_name,
            "--location", location,
        )
        log.info("Log Analytics workspace created.")

    workspace_id = ws["customerId"]
    keys = az(
        "monitor", "log-analytics", "workspace", "get-shared-keys",
        "--resource-group", resource_group,
        "--workspace-name", workspace_name,
    )
    return workspace_id, keys["primarySharedKey"]


# ---------------------------------------------------------------------------
# Container Apps environment
# ---------------------------------------------------------------------------
def ensure_container_apps_env(resource_group: str, location: str,
                              env_name: str, workspace_id: str,
                              workspace_key: str) -> None:
    log.info("Ensuring Container Apps environment '%s' ...", env_name)
    try:
        az(
            "containerapp", "env", "show",
            "--resource-group", resource_group,
            "--name", env_name,
        )
        log.info("Environment '%s' already exists.", env_name)
    except subprocess.CalledProcessError:
        az(
            "containerapp", "env", "create",
            "--resource-group", resource_group,
            "--name", env_name,
            "--location", location,
            "--logs-workspace-id", workspace_id,
            "--logs-workspace-key", workspace_key,
        )
        log.info("Container Apps environment '%s' created.", env_name)


# ---------------------------------------------------------------------------
# Azure Files storage
# ---------------------------------------------------------------------------
def ensure_storage(resource_group: str, location: str,
                   account_name: str) -> str:
    log.info("Ensuring storage account '%s' ...", account_name)
    try:
        az(
            "storage", "account", "show",
            "--resource-group", resource_group,
            "--name", account_name,
        )
    except subprocess.CalledProcessError:
        az(
            "storage", "account", "create",
            "--resource-group", resource_group,
            "--name", account_name,
            "--location", location,
            "--sku", "Standard_LRS",
            "--kind", "StorageV2",
        )
        log.info("Storage account '%s' created.", account_name)

    keys = az(
        "storage", "account", "keys", "list",
        "--resource-group", resource_group,
        "--account-name", account_name,
    )
    return keys[0]["value"]


def ensure_fileshare(account_name: str, account_key: str,
                     share_name: str) -> None:
    log.info("Ensuring file share '%s' ...", share_name)
    try:
        az(
            "storage", "share", "show",
            "--account-name", account_name,
            "--account-key", account_key,
            "--name", share_name,
        )
    except subprocess.CalledProcessError:
        az(
            "storage", "share", "create",
            "--account-name", account_name,
            "--account-key", account_key,
            "--name", share_name,
            "--quota", "100",
        )
        log.info("File share '%s' created.", share_name)


def bind_storage_to_env(resource_group: str, env_name: str,
                        storage_name: str, account_name: str,
                        account_key: str, share_name: str) -> None:
    log.info("Binding storage '%s' to environment ...", storage_name)
    try:
        az(
            "containerapp", "env", "storage", "show",
            "--resource-group", resource_group,
            "--name", env_name,
            "--storage-name", storage_name,
        )
        log.info("Storage mount '%s' already bound.", storage_name)
    except subprocess.CalledProcessError:
        az(
            "containerapp", "env", "storage", "set",
            "--resource-group", resource_group,
            "--name", env_name,
            "--storage-name", storage_name,
            "--azure-file-account-name", account_name,
            "--azure-file-account-key", account_key,
            "--azure-file-share-name", share_name,
            "--access-mode", "ReadWrite",
        )
        log.info("Storage mount '%s' bound.", storage_name)


# ---------------------------------------------------------------------------
# Container App deployment helpers
# ---------------------------------------------------------------------------
def deploy_container_app(
    resource_group: str,
    env_name: str,
    app_name: str,
    image: str,
    *,
    target_port: int | None = None,
    ingress: str | None = None,       # "external" | "internal" | None
    env_vars: list[str] | None = None,
    cpu: str = "1.0",
    memory: str = "2Gi",
    min_replicas: int = 1,
    max_replicas: int = 1,
    volume_mounts: list[dict] | None = None,
    registry_server: str | None = None,
    registry_username: str | None = None,
    registry_password: str | None = None,
    force: bool = False,
    startup_probe_path: str | None = None,
) -> dict | None:
    """Create or update a Container App."""
    log.info("Deploying container app '%s' (image=%s) ...", app_name, image)

    exists = False
    try:
        az(
            "containerapp", "show",
            "--resource-group", resource_group,
            "--name", app_name,
        )
        exists = True
    except subprocess.CalledProcessError:
        pass

    if exists and not force:
        log.info("Container app '%s' already exists (use --force to recreate).", app_name)
        return None

    if exists and force:
        log.info("Deleting existing container app '%s' ...", app_name)
        az(
            "containerapp", "delete",
            "--resource-group", resource_group,
            "--name", app_name,
            "--yes",
            parse_json=False,
        )

    cmd = [
        "containerapp", "create",
        "--resource-group", resource_group,
        "--environment", env_name,
        "--name", app_name,
        "--image", image,
        "--cpu", cpu,
        "--memory", memory,
        "--min-replicas", str(min_replicas),
        "--max-replicas", str(max_replicas),
    ]

    if ingress and target_port:
        cmd += ["--ingress", ingress, "--target-port", str(target_port)]
        if ingress == "external":
            cmd += ["--transport", "http"]

    if env_vars:
        cmd += ["--env-vars"] + env_vars

    if registry_server and registry_username and registry_password:
        cmd += [
            "--registry-server", registry_server,
            "--registry-username", registry_username,
            "--registry-password", registry_password,
        ]

    result = az(*cmd)
    log.info("Container app '%s' deployed.", app_name)
    return result


def add_volume_mount(resource_group: str, app_name: str,
                     storage_name: str, mount_path: str,
                     volume_name: str) -> None:
    """Add an Azure Files volume mount to an existing container app via YAML update."""
    log.info("Adding volume mount %s -> %s on '%s' ...", storage_name, mount_path, app_name)

    # Export current config
    current = az(
        "containerapp", "show",
        "--resource-group", resource_group,
        "--name", app_name,
    )

    # Patch template with volume and volume mount
    template = current.get("properties", {}).get("template", {})

    # Add volume definition
    volumes = template.get("volumes") or []
    volumes.append({
        "name": volume_name,
        "storageName": storage_name,
        "storageType": "AzureFile",
    })
    template["volumes"] = volumes

    # Add volume mount to first container
    containers = template.get("containers", [])
    if containers:
        mounts = containers[0].get("volumeMounts") or []
        mounts.append({
            "volumeName": volume_name,
            "mountPath": mount_path,
        })
        containers[0]["volumeMounts"] = mounts

    # Write patch to temp file and apply
    patch = {
        "properties": {
            "template": template,
        }
    }
    patch_file = f"/tmp/immich_patch_{app_name}.json"
    with open(patch_file, "w") as f:
        json.dump(patch, f)

    az(
        "containerapp", "update",
        "--resource-group", resource_group,
        "--name", app_name,
        "--yaml", patch_file,
        parse_json=False,
    )
    os.remove(patch_file)
    log.info("Volume mount added to '%s'.", app_name)


# ---------------------------------------------------------------------------
# Main deployment orchestration
# ---------------------------------------------------------------------------
def deploy(args: argparse.Namespace) -> None:
    """Orchestrate the full Immich deployment to Azure."""
    region = args.region
    resource_group = args.resource_group
    env_name = args.environment
    storage_account = args.storage_account
    immich_version = args.immich_version
    force = args.force

    db_password = args.db_password or generate_password()
    log.info("Generated DB password (stored as container env var).")

    # Registry credentials (for Docker Hub or private registries)
    registry_server = None
    registry_username = None
    registry_password = None
    if args.dockerhub_username and args.dockerhub_token:
        registry_server = "docker.io"
        registry_username = args.dockerhub_username
        registry_password = args.dockerhub_token

    # ---- Step 1: Authenticate ----
    azure_login(args.tenant_id, args.client_id, args.client_secret,
                args.subscription_id)

    # ---- Step 2: Resource group ----
    ensure_resource_group(resource_group, region)

    # ---- Step 3: Log Analytics ----
    workspace_id, workspace_key = ensure_log_analytics(resource_group, region)

    # ---- Step 4: Container Apps environment ----
    ensure_container_apps_env(resource_group, region, env_name,
                             workspace_id, workspace_key)

    # ---- Step 5: Storage ----
    account_key = ensure_storage(resource_group, region, storage_account)
    for share in (FILESHARE_UPLOAD, FILESHARE_DB, FILESHARE_ML_CACHE):
        ensure_fileshare(storage_account, account_key, share)

    # Bind storage mounts to the Container Apps environment
    bind_storage_to_env(resource_group, env_name, "uploadstorage",
                        storage_account, account_key, FILESHARE_UPLOAD)
    bind_storage_to_env(resource_group, env_name, "dbstorage",
                        storage_account, account_key, FILESHARE_DB)
    bind_storage_to_env(resource_group, env_name, "mlcachestorage",
                        storage_account, account_key, FILESHARE_ML_CACHE)

    # ---- Step 6: Deploy PostgreSQL (internal) ----
    deploy_container_app(
        resource_group, env_name, "immich-database",
        DEFAULT_POSTGRES_IMAGE,
        target_port=5432,
        ingress="internal",
        cpu="1.0", memory="2Gi",
        env_vars=[
            f"POSTGRES_PASSWORD={db_password}",
            "POSTGRES_USER=postgres",
            "POSTGRES_DB=immich",
            "POSTGRES_INITDB_ARGS=--data-checksums",
        ],
        force=force,
    )

    # ---- Step 7: Deploy Redis (internal) ----
    deploy_container_app(
        resource_group, env_name, "immich-redis",
        DEFAULT_REDIS_IMAGE,
        target_port=6379,
        ingress="internal",
        cpu="0.5", memory="1Gi",
        registry_server=registry_server,
        registry_username=registry_username,
        registry_password=registry_password,
        force=force,
    )

    # ---- Step 8: Deploy Immich Machine Learning (internal) ----
    ml_image = f"{DEFAULT_ML_IMAGE}:{immich_version}"
    deploy_container_app(
        resource_group, env_name, "immich-ml",
        ml_image,
        target_port=3003,
        ingress="internal",
        cpu="2.0", memory="4Gi",
        env_vars=[
            "IMMICH_PORT=3003",
            "MACHINE_LEARNING_CACHE_FOLDER=/cache",
            "MACHINE_LEARNING_WORKERS=1",
            "MACHINE_LEARNING_WORKER_TIMEOUT=120",
        ],
        force=force,
    )

    # ---- Step 9: Deploy Immich Server (external with HTTPS/SSL) ----
    server_image = f"{DEFAULT_SERVER_IMAGE}:{immich_version}"
    deploy_container_app(
        resource_group, env_name, "immich-server",
        server_image,
        target_port=2283,
        ingress="external",
        cpu="2.0", memory="4Gi",
        env_vars=[
            "IMMICH_PORT=2283",
            "DB_HOSTNAME=immich-database",
            "DB_PORT=5432",
            "DB_USERNAME=postgres",
            f"DB_PASSWORD={db_password}",
            "DB_DATABASE_NAME=immich",
            "REDIS_HOSTNAME=immich-redis",
            "REDIS_PORT=6379",
            "IMMICH_MACHINE_LEARNING_URL=http://immich-ml:3003",
        ],
        force=force,
    )

    # ---- Step 10: Add volume mounts ----
    log.info("Configuring persistent volume mounts ...")

    # We use `az containerapp update` with YAML patches for volume mounts
    for app, storage, mount, vol in [
        ("immich-server", "uploadstorage", "/data", "upload-vol"),
        ("immich-database", "dbstorage", "/var/lib/postgresql/data", "db-vol"),
        ("immich-ml", "mlcachestorage", "/cache", "mlcache-vol"),
    ]:
        try:
            add_volume_mount(resource_group, app, storage, mount, vol)
        except subprocess.CalledProcessError as exc:
            log.warning("Could not add volume mount to '%s': %s", app, exc.stderr)
            log.warning("You may need to add the volume mount manually.")

    # ---- Step 11: Retrieve endpoint ----
    log.info("Retrieving deployment endpoint ...")
    try:
        app_info = az(
            "containerapp", "show",
            "--resource-group", resource_group,
            "--name", "immich-server",
        )
        fqdn = (app_info
                .get("properties", {})
                .get("configuration", {})
                .get("ingress", {})
                .get("fqdn", ""))
        if fqdn:
            log.info("=" * 60)
            log.info("Immich is deployed and available at:")
            log.info("  https://%s", fqdn)
            log.info("")
            log.info("SSL/TLS is automatically managed by Azure Container Apps.")
            log.info("=" * 60)
        else:
            log.warning("Could not determine FQDN. Check the Azure portal.")
    except subprocess.CalledProcessError:
        log.warning("Could not query app endpoint. Check Azure portal for URL.")

    # ---- Custom domain (optional) ----
    if args.custom_domain:
        log.info("Configuring custom domain '%s' ...", args.custom_domain)
        try:
            az(
                "containerapp", "hostname", "add",
                "--resource-group", resource_group,
                "--name", "immich-server",
                "--hostname", args.custom_domain,
                parse_json=False,
            )
            log.info("Custom domain added. Configure a CNAME record pointing")
            log.info("'%s' to '%s'.", args.custom_domain, fqdn)

            if args.enable_managed_cert:
                log.info("Binding managed SSL certificate ...")
                az(
                    "containerapp", "hostname", "bind",
                    "--resource-group", resource_group,
                    "--name", "immich-server",
                    "--hostname", args.custom_domain,
                    "--environment", env_name,
                    "--validation-method", "CNAME",
                    parse_json=False,
                )
                log.info("Managed certificate bound for '%s'.", args.custom_domain)
        except subprocess.CalledProcessError as exc:
            log.error("Custom domain setup failed: %s", exc.stderr)

    log.info("Deployment complete.")


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
def teardown(args: argparse.Namespace) -> None:
    """Remove all deployed Azure resources."""
    log.info("Tearing down resource group '%s' ...", args.resource_group)
    azure_login(args.tenant_id, args.client_id, args.client_secret,
                args.subscription_id)
    az(
        "group", "delete",
        "--name", args.resource_group,
        "--yes", "--no-wait",
        parse_json=False,
    )
    log.info("Resource group deletion initiated (async). "
             "Check Azure portal for status.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy Immich photo management to Azure with SSL/TLS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full deployment
  python3 deploy_immich.py --azure --force \\
    --tenant-id <TENANT_ID> \\
    --subscription-id <SUBSCRIPTION_ID> \\
    --client-id <CLIENT_ID> \\
    --client-secret <CLIENT_SECRET> \\
    --dockerhub-username <USERNAME> \\
    --dockerhub-token <TOKEN>

  # Deploy with custom domain and managed SSL cert
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

    # Platform flag (for parity with audiobookshelf deploy script)
    parser.add_argument("--azure", action="store_true", required=True,
                        help="Deploy to Azure (required flag)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force re-creation of existing container apps")
    parser.add_argument("--teardown", action="store_true", default=False,
                        help="Remove all deployed resources")

    # Azure service principal credentials
    auth = parser.add_argument_group("Azure Authentication")
    auth.add_argument("--tenant-id", required=True,
                      help="Azure AD tenant ID")
    auth.add_argument("--subscription-id", required=True,
                      help="Azure subscription ID")
    auth.add_argument("--client-id", required=True,
                      help="Service principal client/app ID")
    auth.add_argument("--client-secret", required=True,
                      help="Service principal client secret")

    # Docker Hub credentials (optional — Immich images are on ghcr.io)
    docker = parser.add_argument_group("Docker Hub Credentials (optional)")
    docker.add_argument("--dockerhub-username", default=None,
                        help="Docker Hub username (for pulling rate-limited images)")
    docker.add_argument("--dockerhub-token", default=None,
                        help="Docker Hub access token")

    # Deployment options
    deploy_opts = parser.add_argument_group("Deployment Options")
    deploy_opts.add_argument("--region", default=DEFAULT_REGION,
                             help=f"Azure region (default: {DEFAULT_REGION})")
    deploy_opts.add_argument("--resource-group", default=DEFAULT_RESOURCE_GROUP,
                             help=f"Resource group name (default: {DEFAULT_RESOURCE_GROUP})")
    deploy_opts.add_argument("--environment", default=DEFAULT_ENVIRONMENT,
                             help=f"Container Apps environment (default: {DEFAULT_ENVIRONMENT})")
    deploy_opts.add_argument("--storage-account", default=DEFAULT_STORAGE_ACCOUNT,
                             help=f"Storage account name (default: {DEFAULT_STORAGE_ACCOUNT})")
    deploy_opts.add_argument("--immich-version", default=DEFAULT_IMMICH_VERSION,
                             help=f"Immich version tag (default: {DEFAULT_IMMICH_VERSION})")
    deploy_opts.add_argument("--db-password", default=None,
                             help="PostgreSQL password (auto-generated if omitted)")

    # SSL / custom domain
    ssl = parser.add_argument_group("SSL / Custom Domain")
    ssl.add_argument("--custom-domain", default=None,
                     help="Custom domain for Immich (e.g. photos.example.com)")
    ssl.add_argument("--enable-managed-cert", action="store_true", default=False,
                     help="Provision Azure-managed SSL certificate for custom domain")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.azure:
        log.error("Only Azure deployment is supported. Use --azure flag.")
        sys.exit(1)

    # Verify Azure CLI is installed
    try:
        run(["az", "version"], capture=True)
    except FileNotFoundError:
        log.error("Azure CLI (az) is not installed. "
                  "Install it from https://aka.ms/installazurecli")
        sys.exit(1)

    if args.teardown:
        teardown(args)
    else:
        deploy(args)


if __name__ == "__main__":
    main()
