# python3 scripts/migrate_and_truncate.py

python3 detector.py --config configs/default.yaml label --set dataset.name=coqa --set llm.alias=llama2_7b
python3 detector.py --config configs/default.yaml label --set dataset.name=coqa --set llm.alias=llama3.1_8b
python3 detector.py --config configs/default.yaml label --set dataset.name=coqa --set llm.alias=opt_6.7b
python3 detector.py --config configs/default.yaml label --set dataset.name=coqa --set llm.alias=qwen2.5_7b

python3 detector.py --config configs/default.yaml label --set dataset.name=truthfulqa --set llm.alias=llama2_7b
python3 detector.py --config configs/default.yaml label --set dataset.name=truthfulqa --set llm.alias=llama3.1_8b
python3 detector.py --config configs/default.yaml label --set dataset.name=truthfulqa --set llm.alias=opt_6.7b
python3 detector.py --config configs/default.yaml label --set dataset.name=truthfulqa --set llm.alias=qwen2.5_7b

python3 detector.py --config configs/default.yaml label --set dataset.name=triviaqa --set llm.alias=llama2_7b
python3 detector.py --config configs/default.yaml label --set dataset.name=triviaqa --set llm.alias=llama3.1_8b
python3 detector.py --config configs/default.yaml label --set dataset.name=triviaqa --set llm.alias=opt_6.7b
python3 detector.py --config configs/default.yaml label --set dataset.name=triviaqa --set llm.alias=qwen2.5_7b
