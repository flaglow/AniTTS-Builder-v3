SHELL := /bin/bash
PROJECT ?= mydata

.PHONY: help setup setup-no-cache start start-mydata

help:
	@echo "make setup          # ./setup.sh"
	@echo "make setup-no-cache # NO_CACHE=1 ./setup.sh"
	@echo "make start          # ./start.sh $(PROJECT)"
	@echo "make start-mydata   # ./start.sh mydata"

setup:
	./setup.sh

setup-no-cache:
	NO_CACHE=1 ./setup.sh

start:
	./start.sh $(PROJECT)

start-mydata:
	./start.sh mydata
