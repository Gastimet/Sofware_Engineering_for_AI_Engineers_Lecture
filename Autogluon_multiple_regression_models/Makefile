.PHONY: up down logs train seed
up:
docker compose up -d --build


down:
docker compose down -v


logs:
docker compose logs -f


train:
docker compose exec api python -m src.pipeline.train --task churn \
--data-csv-a example_data/source_a.csv \
--data-csv-b example_data/source_b.csv \
--target churned --promote