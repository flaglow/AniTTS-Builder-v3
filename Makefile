SHELL := /bin/bash
PROJECT ?= mydata

.PHONY: help setup setup-cuml setup-no-cuml setup-no-cache start start-gpu start-cpu start-mydata

# Allow positional project arg:
#   make start data_w
#   make start-gpu data_w
#   make start-cpu data_w
ifneq ($(filter start start-gpu start-cpu,$(firstword $(MAKECMDGOALS))),)
EXTRA_GOALS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
ifneq ($(strip $(EXTRA_GOALS)),)
PROJECT := $(firstword $(EXTRA_GOALS))
$(EXTRA_GOALS):
	@:
endif
endif

help:
	@echo "make setup          # auto-detect GPU and choose INSTALL_CUML automatically"
	@echo "make setup-cuml     # INSTALL_CUML=1 ./setup.sh"
	@echo "make setup-no-cuml  # INSTALL_CUML=0 ./setup.sh (recommended if cuML causes torch import errors)"
	@echo "make setup-no-cache # NO_CACHE=1 ./setup.sh"
	@echo "make start [project]     # USE_GPU=1 ./start.sh <project> (default: $(PROJECT))"
	@echo "make start-gpu [project] # USE_GPU=1 ./start.sh <project> (default: $(PROJECT))"
	@echo "make start-cpu [project] # USE_GPU=0 ./start.sh <project> (default: $(PROJECT))"
	@echo "make start-mydata   # USE_GPU=1 ./start.sh mydata"

setup:
	./setup.sh

setup-cuml:
	INSTALL_CUML=1 ./setup.sh

setup-no-cuml:
	INSTALL_CUML=0 ./setup.sh

setup-no-cache:
	NO_CACHE=1 ./setup.sh

start:
	USE_GPU=1 ./start.sh $(PROJECT)

start-gpu:
	USE_GPU=1 ./start.sh $(PROJECT)

start-cpu:
	USE_GPU=0 ./start.sh $(PROJECT)

start-mydata:
	USE_GPU=1 ./start.sh mydata
