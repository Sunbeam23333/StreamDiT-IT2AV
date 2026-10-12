#!/usr/bin/env bash
# ============================================================================
# 拉取第三方框架到 utils/{audio,video,image}/ (固定 commit) 并打上本仓库的兼容补丁
#
# 用法:
#   bash third_party/fetch_third_party.sh                 # 默认: audio 组 (TTS / SFX / BGM)
#   bash third_party/fetch_third_party.sh video image     # 按组拉取: audio | video | image | optional | all
#   bash third_party/fetch_third_party.sh CosyVoice LatentSync   # 按名字拉取
#   bash third_party/fetch_third_party.sh --force LatentSync     # 删除已有目录后重新拉取
#   bash third_party/fetch_third_party.sh --no-patch LiveAvatar  # 只拉取不打补丁
#   bash third_party/fetch_third_party.sh --list                 # 查看清单
#
# 说明:
#   - SongGeneration 上游 GitHub 仓库已下线, 已直接 vendored 在 utils/audio/SongGeneration (含补丁)
#   - CosyVoice 的 third_party/Matcha-TTS 是 git submodule, 浅拉取后为空目录,
#     脚本会自动把它链接到 utils/audio/Matcha-TTS (这是我们踩过的坑之一)
#   - 需要代理时在仓库根目录 .env 中设置 IT2AV_PROXY_URL
# ============================================================================
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/script/common/env.sh"
MANIFEST="${ROOT}/third_party/manifest.tsv"
PATCH_DIR="${ROOT}/third_party/patches"
MARKER=".it2av_fetched"

FORCE=0
APPLY_PATCH=1
SELECT=()
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --no-patch) APPLY_PATCH=0 ;;
        --list)
            printf "%-20s %-9s %-34s %s\n" NAME GROUP DEST PATCH
            grep -v '^#' "${MANIFEST}" | awk -F'\t' '{printf "%-20s %-9s %-34s %s\n", $1, $2, $3, $6}'
            exit 0 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) SELECT+=("$arg") ;;
    esac
done
[[ ${#SELECT[@]} -eq 0 ]] && SELECT=(audio)

selected() {
    local name=$1 group=$2 s
    for s in "${SELECT[@]}"; do
        [[ "$s" == "all" && "$group" != "optional" ]] && return 0
        [[ "$s" == "$group" || "$s" == "$name" ]] && return 0
    done
    return 1
}

fetch_one() {
    local name=$1 dest=$2 repo=$3 commit=$4 patch=$5
    local dir="${ROOT}/${dest}"

    if [[ -d "$dir" && -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
        if [[ $FORCE -eq 1 ]]; then
            echo "  ↻ --force: 删除 ${dest}"
            rm -rf "$dir"
        elif [[ -f "$dir/$MARKER" && "$(cat "$dir/$MARKER")" == "$commit" ]]; then
            echo "  ✓ ${name} 已就绪 (${commit:0:8})"
            return 0
        else
            echo "  ⚠️  ${dest} 已存在但不是本脚本拉取的版本, 跳过 (用 --force 覆盖)"
            return 0
        fi
    fi

    echo "  ⬇️  ${name}: github.com/${repo} @ ${commit:0:8}"
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" remote add origin "https://github.com/${repo}.git"
    if ! git -C "$dir" fetch -q --depth 1 origin "$commit"; then
        echo "     浅拉取失败, 改为完整 clone"
        git -C "$dir" fetch -q origin
    fi
    git -C "$dir" -c advice.detachedHead=false checkout -q "$commit"

    if [[ "$patch" != "-" && $APPLY_PATCH -eq 1 ]]; then
        git -C "$dir" apply --whitespace=nowarn "${PATCH_DIR}/${patch}"
        echo "     🩹 已应用 third_party/patches/${patch}"
    fi

    if [[ "$name" == "CosyVoice" ]]; then
        local matcha="${dir}/third_party/Matcha-TTS"
        if [[ -d "$matcha" && -z "$(ls -A "$matcha" 2>/dev/null)" ]]; then
            rmdir "$matcha"
            ln -s ../../Matcha-TTS "$matcha"
            echo "     🔗 third_party/Matcha-TTS -> utils/audio/Matcha-TTS"
        fi
    fi
    echo "$commit" > "$dir/$MARKER"
}

echo "============================================================"
echo "StreamDiT-IT2AV · fetch third-party frameworks (${SELECT[*]})"
echo "============================================================"
while IFS=$'\t' read -r name group dest repo commit patch license; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    selected "$name" "$group" || continue
    fetch_one "$name" "$dest" "$repo" "$commit" "$patch"
done < "${MANIFEST}"

if [[ " ${SELECT[*]} " =~ " audio " || " ${SELECT[*]} " =~ " CosyVoice " ]] && [[ ! -d "${ROOT}/utils/audio/Matcha-TTS/matcha" ]]; then
    echo "  ℹ️  CosyVoice 依赖 Matcha-TTS, 请确保同时拉取: bash third_party/fetch_third_party.sh Matcha-TTS"
fi
echo "✅ done"
