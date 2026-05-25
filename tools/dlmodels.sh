#!/bin/sh

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"

printf "working dir is %s\n" "$ROOT_DIR"
echo "downloading requirement aria2 check."

if command -v aria2c > /dev/null 2>&1
then
    echo "aria2 command found"
else
    echo "failed. please install aria2"
    exit 1
fi

echo "dir check start."

check_dir() {
    [ -d "$1" ] && printf "dir %s checked\n" "$1" || \
    printf "failed. generating dir %s\n" "$1" && mkdir -p "$1"
}

check_dir "$ROOT_DIR/pretrain/pretrained"
check_dir "$ROOT_DIR/pretrain/pretrained_v2"
check_dir "$ROOT_DIR/pretrain/hubert"
check_dir "$ROOT_DIR/pretrain/rmvpe"

echo "dir check finished."

echo "required files check start."

check_file_pretrained() {
  printf "checking %s\n" "$2"
  if [ -f "$ROOT_DIR/pretrain/$1/$2" ]; then
      printf "%s in ./pretrain/%s checked.\n" "$2" "$1"
  else
      echo failed. starting download from huggingface.
      aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/$1/$2" -d "$ROOT_DIR/pretrain/$1" -o "$2"
      [ -f "$ROOT_DIR/pretrain/$1/$2" ] && echo "download successful." || { echo "please try again!" && exit 1; }
  fi
}

check_file_special() {
  printf "checking %s\n" "$2"
  if [ -f "$ROOT_DIR/pretrain/$1/$2" ]; then
      printf "%s in ./pretrain/%s checked.\n" "$2" "$1"
  else
      echo failed. starting download from huggingface.
      aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/$2" -d "$ROOT_DIR/pretrain/$1" -o "$2"
      [ -f "$ROOT_DIR/pretrain/$1/$2" ] && echo "download successful." || { echo "please try again!" && exit 1; }
  fi
}

check_file_pretrained pretrained D32k.pth
check_file_pretrained pretrained D40k.pth
check_file_pretrained pretrained D48k.pth
check_file_pretrained pretrained G32k.pth
check_file_pretrained pretrained G40k.pth
check_file_pretrained pretrained G48k.pth
check_file_pretrained pretrained_v2 f0D40k.pth
check_file_pretrained pretrained_v2 f0G40k.pth
check_file_pretrained pretrained_v2 D40k.pth
check_file_pretrained pretrained_v2 G40k.pth
check_file_special rmvpe rmvpe.pt
check_file_special hubert hubert_base.pt

echo "required files check finished."
