from evaluator import EvaluationConfig, Evaluator

# Create an EvaluationConfig instance to evaluate your model, for example:
# config = EvaluationConfig(
#     model_name_or_path='/home/molly/workspace/models/rwkv-6-world-1b6-MLC',
#     model_args={'model_lib': '/home/molly/workspace/models/rwkv-6-world-1b6-MLC/libs/rwkv-6-world-1b6-MLC-q4f16-cuda.so'},
#     tokenizer_name='rwkv_vocab_v20230424',
#     model_type='rwkv_mlc',
#     data=[
#         'data/github_cpp_20240701to20240714.json',
#         'data/github_python_20240701to20240714.json',
#         'data/ao3_chinese_20240701to20240713.json',
#         'data/ao3_english_20240701to20240714.json',
#         'data/bbc_news_20240701to20240714.json',
#         'data/wikipedia_english_20240701to20240714.json',
#         'data/arxiv_computer_science_20240701to20240714.json',
#         'data/arxiv_physics_20240701to20240714.json',
#     ]
# )

# config = EvaluationConfig(
#     model_name_or_path='/home/molly/workspace/models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
#     tokenizer_name='rwkv_vocab_v20230424',
#     model_type='rwkv',
#     model_args={'strategy': 'cuda fp16i8'},

#     data=[
#         'data/github_cpp_20240701to20240714.json',
#         'data/github_python_20240701to20240714.json',
#         'data/ao3_chinese_20240701to20240713.json',
#         'data/ao3_english_20240701to20240714.json',
#         'data/bbc_news_20240701to20240714.json',
#         'data/wikipedia_english_20240701to20240714.json',
#         'data/arxiv_computer_science_20240701to20240714.json',
#         'data/arxiv_physics_20240701to20240714.json',
#     ]
# )

config = EvaluationConfig(
    model_name_or_path='/home/molly/workspace/models/rwkv-6-world-1b6/rwkv-6-world-1b6-Q4_1.gguf',
    tokenizer_name='rwkv_vocab_v20230424',
    model_type='llama_cpp',

    data=[
        'data/github_cpp_20240701to20240714.json',
        'data/github_python_20240701to20240714.json',
        'data/ao3_chinese_20240701to20240713.json',
        'data/ao3_english_20240701to20240714.json',
        'data/bbc_news_20240701to20240714.json',
        'data/wikipedia_english_20240701to20240714.json',
        'data/arxiv_computer_science_20240701to20240714.json',
        'data/arxiv_physics_20240701to20240714.json',
    ]
)


if __name__ == '__main__':
    try:
        evaluator = Evaluator()
        evaluator.evaluate(config)
    except Exception as e:
        print(f"Error: {e}")
