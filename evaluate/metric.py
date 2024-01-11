import nltk
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# 文本数据
def caculate_ppl(text="This is an example sentence for calculating perplexity."):
    # 将文本拆分为句子列表
    sentences = nltk.sent_tokenize(text)

    # 将句子转换为N-gram格式
    n = 3  # 设置N-gram的大小
    train_data, padded_sents = padded_everygram_pipeline(n, sentences)

    # 创建和训练N-gram语言模型
    model = MLE(n)
    model.fit(train_data, padded_sents)

    # 计算困惑度
    test_text = "This is another example sentence for perplexity calculation."
    test_tokens = nltk.word_tokenize(test_text)
    test_ngrams = list(model.vocab) + test_tokens  # 将测试文本转换为tokens
    test_ngrams = list(nltk.ngrams(test_ngrams, n))

    test_perplexity = model.perplexity(test_ngrams)
    print(f"Perplexity of the test text: {test_perplexity}")

def caculate_blue(reference_sentence="This is an example sentence for calculating perplexity.",candidate_sentence="This is an example sentence for calculating perplexity."):
    reference_tokens = list(reference_sentence)  # 假设这里以单个字符为单位分词
    candidate_tokens = list(candidate_sentence)  
    # 计算BLEU分数
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
    print(f"BLEU Score: {bleu_score}")

def caculate_rougr(reference_sentence="This is an example sentence for calculating perplexity.",candidate_sentence="This is an example sentence for calculating perplexity."):
    rouge = Rouge()
    scores = rouge.get_scores(reference_sentence, candidate_sentence)
    print(scores)