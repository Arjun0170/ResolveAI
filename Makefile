.PHONY: setup data inspect test train baseline neural index evaluate benchmark benchmark-native cpp demo serve all

PYTHON := .venv/bin/python
CLI := .venv/bin/resolveai

setup:
	./setup.sh

data:
	$(CLI) download-data

inspect:
	$(CLI) inspect-data

test:
	$(PYTHON) -m unittest discover -s tests -v

baseline:
	$(CLI) train-baseline

neural:
	$(CLI) train-neural

index:
	$(CLI) build-index

evaluate:
	$(CLI) evaluate-rag

benchmark:
	$(CLI) benchmark-service

benchmark-native:
	$(CLI) benchmark

cpp:
	mkdir -p build
	g++ -O3 -std=c++17 -Wall -Wextra -Wpedantic -fPIC -shared cpp/topk.cpp -o build/libresolve_topk.so

demo:
	$(CLI) demo --text "please help me find the phone i lost"

serve:
	$(CLI) serve --host 127.0.0.1 --port 8000

all:
	$(CLI) train-all
