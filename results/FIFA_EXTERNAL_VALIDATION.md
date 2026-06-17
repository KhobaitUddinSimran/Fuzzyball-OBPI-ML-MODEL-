# FIFA External Validation

- status: complete
- benchmark_source: FIFA 23 complete player dataset
- benchmark_source_url: https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset
- fifa_version: 23
- fifa_update: 9
- fifa_update_date: 2023-01-13
- obpi_players: 252
- matched_players: 230
- unmatched_players: 22
- match_rate: 0.913
- match_type_counts: {'exact_name': 196, 'token_containment': 31, 'fuzzy_name': 3}

## Benchmark Validation

- fifa_shooting: rho=0.1024, p=0.1214, n=230
- fifa_pace: rho=0.0925, p=0.162, n=230
- fifa_potential: rho=0.0719, p=0.2778, n=230
- fifa_physic: rho=0.0510, p=0.4417, n=230
- fifa_reactions: rho=0.0129, p=0.8452, n=230
- fifa_overall: rho=0.0005, p=0.9943, n=230
- fifa_dribbling: rho=-0.0116, p=0.8616, n=230
- fifa_vision: rho=-0.1076, p=0.1036, n=230
- fifa_passing: rho=-0.1249, p=0.05861, n=230
- fifa_defending: rho=-0.1456, p=0.0273, n=230

## Interpretation

FIFA ratings are an external commercial player-quality benchmark, not an expert panel and not event-level ground truth. Spearman rho measures convergent validity between aggregate OBPI and FIFA attributes on the matched subset.

## Outputs

- ratings_output: data/external/fifa_ratings.csv
- match_audit_output: data/external/fifa_ratings_match_audit.csv

## Unmatched Sample

- André Ayew Pelé
- Daichi Kamada
- Georgiy Sudakov
- Ghilane Chalali
- Gustavo Adolfo Puerta Molano
- Hattan Babhir
- Jean-Mattéo Bahoya Négoce
- Kobbie Mainoo
- Maksim Mukhin
- Mert Kömür
- Miralem Pjanić
- Mohamed Ali Ben Romdhane
- Mohamed Zeki Amdouni
- Moriba Kourouma Kourouma
- Pep Biel Mas Jaume
- Peter Stroud
- Qazim Laçi
- Romario Andrés Ibarra Mina
- Vladimír Darida
- Warren Zaire Emery
- Youssef Msakni
- Éverton Augusto de Barros Ribeiro
