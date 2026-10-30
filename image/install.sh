#!/bin/sh
# Shared build step of both TransitBench images (docs/IMAGE.md, docs/ENV_DESIGN.md section 2.6).
#   install.sh <tag>    run as root from the directory holding requirements.txt
# Installs the jail tools, uv, the pinned libraries and the per-slot users, and writes <tag> to
# /etc/transitbench-image. It installs no TransitBench code and no data: setup writes the code.
set -eu

TAG="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"

# uv on PATH, so verifiers' harness setup skips its own install. Pinned release, checked against
# its published sha256 before it is unpacked.
UV_VERSION=0.12.18
case "$(dpkg --print-architecture)" in
    amd64) UV_ARCH=x86_64; UV_SHA256=89eadd7c76fc063887959510d5ba0ab1264dfd5f1143b925ddb73021a40acf16 ;;
    arm64) UV_ARCH=aarch64; UV_SHA256=afb6291f3f0a6b4521fc67b947822506c41dde5b60d2189dd8f3695b2ac8c9e7 ;;
    *) echo "install.sh: unsupported architecture $(dpkg --print-architecture)" >&2; exit 1 ;;
esac

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends bubblewrap util-linux procps ca-certificates curl
rm -rf /var/lib/apt/lists/*

UV_DIR="uv-${UV_ARCH}-unknown-linux-gnu"
curl -LsSf "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${UV_DIR}.tar.gz" -o /tmp/uv.tar.gz
echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum -c -
tar -xzf /tmp/uv.tar.gz -C /tmp
install -m 0755 "/tmp/${UV_DIR}/uv" "/tmp/${UV_DIR}/uvx" /usr/local/bin/
rm -rf /tmp/uv.tar.gz "/tmp/${UV_DIR}"

# The libraries, wheels only, every file checked against requirements.txt (the uv.lock versions).
python3 -m pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore \
    --require-hashes --no-deps --only-binary=:all: -r "${HERE}/requirements.txt"
python3 -m pip check
python3 -c "import numpy, scipy, astropy, sklearn"

# One unprivileged user per runner slot: tb00..tb15, uid = gid = 2000..2015.
i=0
while [ "$i" -lt 16 ]; do
    name="$(printf 'tb%02d' "$i")"
    id=$((2000 + i))
    groupadd --gid "$id" "$name"
    useradd --uid "$id" --gid "$id" --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin --no-log-init "$name"
    i=$((i + 1))
done

install -d -m 0700 -o root -g root /opt/tb-grade
install -d -m 0755 -o root -g root /app
printf '%s\n' "$TAG" > /etc/transitbench-image
chmod 0644 /etc/transitbench-image
