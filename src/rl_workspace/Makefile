.PHONY: build up down up-%

build:
	docker compose -f docker/docker-compose.yml build

up:
	docker compose -f docker/docker-compose.yml up

down:
	docker compose -f docker/docker-compose.yml down --remove-orphans

# also make target for an override
up-%:
	docker compose \
		-f docker/docker-compose.yml \
		-f docker/docker-compose.$*.yml \
		up

config-%:
	docker compose \
		-f docker/docker-compose.yml \
		-f docker/docker-compose.$*.yml \
		config

report:
	docker compose -f docker/docker-compose.pandoc.yml build
	docker compose -f docker/docker-compose.pandoc.yml run --rm pandoc
