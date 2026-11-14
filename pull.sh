#!/bin/bash
# =============================================================================
# 一键拉取脚本
#   1. 拉取主仓库最新代码：主仓库允许存在未提交的改动，按 git 配置的 pull
#      合并方式执行 git pull；仅在 git pull 失败时终止
#   2. 把子模组同步到最新代码所记录的版本（默认）
#      也可以用 --remote 让子模组跟随各自远端分支的最新提交
# 用法: ./pull.sh [选项]
#   -b, --branch <分支>  指定要拉取的分支（默认当前分支）
#       --remote         子模组跟随各自远端分支的最新提交（忽略主仓库记录的版本）
#       --no-submodule   只拉主仓库，不动子模组
#   -h, --help           显示帮助
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

BRANCH=""
SUBMODULE_MODE="pinned"   # pinned | remote
SKIP_SUBMODULE=0
REMOTE_NAME="origin"

usage() {
    cat <<'EOF'
一键拉取脚本: 拉取主仓库最新代码，并把子模组同步到最新代码所记录的版本

用法: ./pull.sh [选项]
  -b, --branch <分支>  指定要拉取的分支（默认当前分支）
      --remote         子模组跟随各自远端分支的最新提交（忽略主仓库记录的版本）
      --no-submodule   只拉主仓库，不动子模组
  -h, --help           显示帮助

主仓库存在未提交的改动时脚本不会中止，git pull 会按 git 配置的合并方式执行；
只有当 git pull 失败（例如本地改动与远端改动冲突）时脚本才会终止。
EOF
}

log()  { echo "[pull] $*"; }
die()  { echo "[pull] 错误: $*" >&2; exit 1; }

# ----------------------------- 参数解析 --------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--branch)
            [[ $# -ge 2 ]] || die "$1 需要一个分支名"
            BRANCH="$2"
            shift 2
            ;;
        --remote)      SUBMODULE_MODE="remote"; shift ;;
        --no-submodule) SKIP_SUBMODULE=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "未知参数: $1"; echo; usage; exit 1 ;;
    esac
done

# ----------------------------- 环境检查 --------------------------------------
command -v git >/dev/null 2>&1 || die "未找到 git，请先安装 git"
git rev-parse --git-dir >/dev/null 2>&1 || die "$PROJECT_DIR 不是 git 仓库"
git remote get-url "$REMOTE_NAME" >/dev/null 2>&1 || die "找不到远端 '$REMOTE_NAME'"

if [[ -z "$BRANCH" ]]; then
    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
fi
[[ "$BRANCH" != "HEAD" ]] || die "当前处于 detached HEAD，请用 -b <分支> 指定要拉取的分支"

echo "========================================"
echo "Project:  $PROJECT_DIR"
echo "Remote:   $REMOTE_NAME"
echo "Branch:   $BRANCH"
echo "Submodule: $([[ $SKIP_SUBMODULE -eq 1 ]] && echo "跳过" || echo "$SUBMODULE_MODE")"
echo "========================================"

# ----------------------------- 拉取主仓库 ------------------------------------
echo ""
echo "[1/3] 拉取 $REMOTE_NAME/$BRANCH ..."
git fetch "$REMOTE_NAME" --prune --tags

# 切换到目标分支（本地不存在则基于远端创建）
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    git show-ref --verify --quiet "refs/heads/$BRANCH" \
        && git checkout "$BRANCH" \
        || git checkout -b "$BRANCH" --track "$REMOTE_NAME/$BRANCH"
fi

OLD_HEAD="$(git rev-parse HEAD)"
# 不指定 --ff-only/--merge/--rebase，由 git 配置的 pull 合并方式决定；
# 主仓库有未提交的改动时 git pull 会在不冲突的前提下保留这些改动
if ! git pull "$REMOTE_NAME" "$BRANCH"; then
    die "git pull 失败，请按 git 的提示处理本地未提交的改动（提交或暂存）或冲突后重试"
fi
NEW_HEAD="$(git rev-parse HEAD)"

echo ""
if [[ "$OLD_HEAD" == "$NEW_HEAD" ]]; then
    log "主仓库已是最新版本: $(git rev-parse --short HEAD)"
else
    log "主仓库更新: $(git rev-parse --short "$OLD_HEAD") -> $(git rev-parse --short "$NEW_HEAD")"
    echo "----------------------------------------"
    git --no-pager log --oneline --no-decorate "$OLD_HEAD..$NEW_HEAD"
    echo "----------------------------------------"
fi

# ----------------------------- 同步子模组 ------------------------------------
echo ""
if [[ $SKIP_SUBMODULE -eq 1 ]]; then
    echo "[2/3] 跳过子模组同步"
else
    echo "[2/3] 同步子模组（$SUBMODULE_MODE）..."
    git submodule sync --recursive >/dev/null
    if [[ "$SUBMODULE_MODE" == "remote" ]]; then
        # 跟随 .gitmodules 中配置的 branch（分支=main）拉取各自最新提交
        git submodule update --init --recursive --remote
    else
        # 检出当前主仓库提交所记录的固定版本
        git submodule update --init --recursive
    fi
    log "子模组同步完成"
fi

# ----------------------------- 结果汇总 --------------------------------------
echo ""
echo "[3/3] 当前版本:"
echo "  主仓库 $(git rev-parse --short HEAD) $(git log -1 --pretty=%s)"
if [[ $SKIP_SUBMODULE -eq 0 ]]; then
    git submodule status --recursive | sed 's/^/  /'
    if [[ "$SUBMODULE_MODE" == "remote" ]]; then
        log "提示: --remote 模式下子模组指向各自远端最新提交，主仓库会出现子模组指针改动"
    fi
fi

echo ""
echo "========================================"
echo "Pull succeeded!"
echo "========================================"
