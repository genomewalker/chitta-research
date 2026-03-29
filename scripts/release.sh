#!/bin/bash
# Release script for chitta-research
#
# Usage:
#   ./scripts/release.sh patch       # Bug fixes (1.0.0 → 1.0.1)
#   ./scripts/release.sh minor       # New features (1.0.0 → 1.1.0)
#   ./scripts/release.sh major       # Breaking changes (1.0.0 → 2.0.0)
#   ./scripts/release.sh 1.2.0       # Explicit version
#   ./scripts/release.sh minor -y    # Skip confirmation
#
# SemVer Guidelines:
#   MAJOR: Breaking changes to the agent protocol, graph format, or chittad RPC
#   MINOR: New agents, new agenda fields, new plan step prefixes
#   PATCH: Bug fixes, prompt improvements, no API changes
#
# Compatibility:
#   chitta-research 1.x requires cc-soul >= 5.0.1 (chittad socket protocol v5)
#   Update [workspace.metadata.compatibility].min_cc_soul_version when the
#   minimum cc-soul version changes.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

get_current_version() {
    grep '^version' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/'
}

bump_version() {
    local current="$1"
    local type="$2"
    IFS='.' read -r major minor patch <<< "$current"
    case "$type" in
        major) echo "$((major + 1)).0.0" ;;
        minor) echo "$major.$((minor + 1)).0" ;;
        patch) echo "$major.$minor.$((patch + 1))" ;;
        *)     echo "$type" ;;
    esac
}

validate_version() {
    [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

sedi() {
    if [[ "$OSTYPE" == "darwin"* ]]; then sed -i '' "$@"; else sed -i "$@"; fi
}

BUMP_TYPE=""
AUTO_CONFIRM=false

for arg in "$@"; do
    case "$arg" in
        -y|--yes) AUTO_CONFIRM=true ;;
        *) [[ -z "$BUMP_TYPE" ]] && BUMP_TYPE="$arg" ;;
    esac
done

if [[ -z "$BUMP_TYPE" ]]; then
    echo "Usage: $0 <patch|minor|major|X.Y.Z> [-y|--yes]"
    echo ""
    echo "SemVer:"
    echo "  patch  Bug fixes, prompt tuning (1.0.0 → 1.0.1)"
    echo "  minor  New agents, new step prefixes (1.0.0 → 1.1.0)"
    echo "  major  Breaking protocol/graph changes (1.0.0 → 2.0.0)"
    exit 1
fi

CURRENT_VERSION=$(get_current_version)
NEW_VERSION=$(bump_version "$CURRENT_VERSION" "$BUMP_TYPE")

if ! validate_version "$NEW_VERSION"; then
    echo "Error: Invalid version: $NEW_VERSION"
    exit 1
fi

MIN_SOUL=$(grep 'min_cc_soul_version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')

echo "=== chitta-research: $CURRENT_VERSION → $NEW_VERSION ==="
echo "    requires cc-soul >= $MIN_SOUL"
echo ""

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: Uncommitted changes. Commit or stash first."
    exit 1
fi

if [[ "$AUTO_CONFIRM" != "true" ]]; then
    read -p "Proceed with release v$NEW_VERSION? [y/N] " confirm
    [[ "$confirm" != "y" && "$confirm" != "Y" ]] && { echo "Aborted."; exit 0; }
fi

# Bump [workspace.package] version in root Cargo.toml
echo "Updating Cargo.toml..."
sedi "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" Cargo.toml
grep -q "\"$NEW_VERSION\"" Cargo.toml || { echo "Cargo.toml update failed"; exit 1; }

# Commit, tag, push
echo "Committing version bump..."
git add Cargo.toml
git commit -m "chore: bump version to $NEW_VERSION"

echo "Creating tag v$NEW_VERSION..."
git tag "v$NEW_VERSION"

echo "Pushing..."
git push origin main
git push origin "v$NEW_VERSION"

echo ""
echo "=== Release v$NEW_VERSION complete ==="
echo ""
echo "GitHub: https://github.com/genomewalker/chitta-research/releases/tag/v$NEW_VERSION"
echo ""
echo "Compatibility: requires cc-soul >= $MIN_SOUL"
