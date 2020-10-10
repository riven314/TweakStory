build: 
	docker-compose build

start:
	docker-compose up -d

run: build start

stop:
	docker-compose stop
