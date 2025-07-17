#!/bin/bash

# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
if ! command -v conda &>/dev/null; then
    echo "[INFO] miniconda 설치"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    export PATH="$HOME/miniconda/bin:$PATH"
else
    echo "[INFO] conda가 설치되어 있습니다."
fi

# Conda 환셩 생성 및 활성화
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "[INFO] 가상환경 생성"
conda create -n myenv python=3.9 -y
echo "[INFO] 가상환경 활성화"
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
echo "[INFO] mypy 패키지 설치"
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    base=$(basename "$file" .py)
    in="../input/${base}_input"
    out="../output/${base}_output"

    echo "[INFO] '$file' 실행"
    if python "$file" < "$in" > "$out"; then
        echo "[INFO] 성공"
    else
        echo "[INFO] 실패"
    fi
done

# mypy 테스트 실행 및 mypy_log.txt 저장
echo "[INFO] mypy 테스트"
mypy . > ../mypy_log.txt 2>&1

# conda.yml 파일 생성
echo "[INFO] conda.yml 파일 생성"
conda env export > ../conda.yaml

# 가상환경 비활성화
## TODO
cd ..
echo "[INFO] 가상환경 비활성화"
conda deactivate
echo "[INFO] 스크립트 종료"