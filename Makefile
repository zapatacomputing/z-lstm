.PHONY: all install test lint

all: install lint test

install:
	pip install -r requirements.txt

lint:
	pylint */*py

test:
	pytest

