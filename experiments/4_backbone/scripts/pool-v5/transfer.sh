#!/bin/bash
# abdata-v5 transfer Coulson -> this Mac -> Isambard (scp relay), sha256 of every file on all three sides.
#   bash transfer.sh SRC_ON_COULSON DEST_ON_ISAMBARD     (directories; DEST must not exist yet; SRC/shards (build scratch) is skipped)
set -euo pipefail
SRC=$1; DEST=$2; C=Coulson-Chemistry; I=u6xq.aip2.isambard
L=/private/tmp/claude-501/-Users-alexi-isambard-ssh/ff273e5c-e06f-4d00-b3bb-87e52de7a000/scratchpad/relay/$(echo "$DEST" | tr / _)
ssh -o BatchMode=yes $I "test ! -e '$DEST'" || { echo "$DEST exists on Isambard" >&2; exit 3; }
mkdir -p "$L"
ssh -o BatchMode=yes $C "cd '$SRC' && find . -type f -not -path './shards/*' | LC_ALL=C sort | xargs sha256sum" > "$L.coulson.sha256"
rsync -a --exclude /shards -e "ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=20" "$C:$SRC/" "$L/"
(cd "$L" && find . -type f | LC_ALL=C sort | xargs shasum -a 256) > "$L.mac.sha256"
diff "$L.coulson.sha256" "$L.mac.sha256" > /dev/null || { echo "Coulson -> Mac checksum mismatch" >&2; exit 4; }
ssh -o BatchMode=yes $I "mkdir -p '$(dirname "$DEST")'"
rsync -a -e "ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=20" "$L/" "$I:$DEST/"
ssh -o BatchMode=yes $I "cd '$DEST' && find . -type f | LC_ALL=C sort | xargs sha256sum" > "$L.isambard.sha256"
diff "$L.coulson.sha256" "$L.isambard.sha256" > /dev/null || { echo "Coulson -> Isambard checksum mismatch" >&2; exit 5; }
echo "OK $(wc -l < "$L.coulson.sha256") files, sha256 equal on Coulson / Mac / Isambard: $SRC -> $DEST"
cp "$L.coulson.sha256" "$(dirname "$0")/sha256-$(basename "$DEST").txt"
rm -rf "$L"
