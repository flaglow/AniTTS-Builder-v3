SHELL := /bin/bash
PROJECT ?= mydata

.PHONY: help setup setup-cuml setup-no-cuml setup-no-cache start start-gpu start-cpu start-mydata

help:
	@echo "make setup          # auto-detect GPU and choose INSTALL_CUML automatically"
	@echo "make setup-cuml     # INSTALL_CUML=1 ./setup.sh"
	@echo "make setup-no-cuml  # INSTALL_CUML=0 ./setup.sh (recommended if cuML causes torch import errors)"
	@echo "make setup-no-cache # NO_CACHE=1 ./setup.sh"
	@echo "make start          # USE_GPU=1 ./start.sh $(PROJECT)"
	@echo "make start-gpu      # USE_GPU=1 ./start.sh $(PROJECT)"
	@echo "make start-cpu      # USE_GPU=0 ./start.sh $(PROJECT)"
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
