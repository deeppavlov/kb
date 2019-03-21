import json
import random

count_0 = 0
count_total = 0

C_LEN = 280

replace_vocab = {}

def replace_html(init_string, number_of_unused_token, replace_vocab):
    for i in range(33):
        html_string = "[{}={}]".format(init_string, str(i))
        new_string = "[unused{}]".format(str(3*i+number_of_unused_token))
        replace_vocab[html_string] = new_string
    return replace_vocab

def replace_html_no_tokens_left(init_string, number_of_unused_token, replace_vocab):
    for i in range(33, 100):
        html_string = "[{}={}]".format(init_string, str(i))
        new_string = "[unused{}]".format(str(number_of_unused_token))
        replace_vocab[html_string] = new_string
    return replace_vocab

replace_vocab = replace_html("Paragraph", 1, replace_vocab)
replace_vocab = replace_html("List", 2, replace_vocab)
replace_vocab = replace_html("Table", 3, replace_vocab)
replace_vocab = replace_html_no_tokens_left("Paragraph", 97, replace_vocab)
replace_vocab = replace_html_no_tokens_left("List", 98, replace_vocab)
replace_vocab = replace_html_no_tokens_left("Table", 99, replace_vocab)


dataset_ans = []
dataset_noans = []


def make_bert_sample(nq_sample):
    short_answers = nq_sample["short_answers"]
    long_answer = nq_sample["long_answer"]
    document_tokens = nq_sample["document_tokens"].split(' ')
    for l, tok in enumerate(document_tokens):
        if tok in replace_vocab:
            document_tokens[l] = replace_vocab[tok]
    document_len = len(document_tokens)
    sample = {}
    what_kind_of_sample = None
    sample["question"] = nq_sample["question_text"]
    answers = []
    if len(short_answers) == 0 and long_answer["start_token"] == -1:
        if document_len <= 300:
            start_span = 0
            end_span = document_len

        if document_len > 300:
            document_len = len(document_tokens) - 300
            start_span = random.randint(0, document_len)
            end_span = start_span + 280

        context_tokens = document_tokens[start_span:end_span]
        context = ' '.join(context_tokens)
        sample["context"] = context
        answer = {}
        answer["answer_start"] = -1
        answer["text"] = ""
        answers.append(answer)
        sample["answers"] = answers
        what_kind_of_sample = "noans"
        
    if len(short_answers) > 0:
        start_span = short_answers[0]["start_token"]
        end_span = short_answers[-1]["end_token"]
        answer_length = end_span - start_span

        if answer_length < 130:
            num_context_tokens = C_LEN - answer_length
            if document_len <= 280:
                num_context_tokens = document_len - answer_length
                left_side = start_span
                right_side = num_context_tokens - left_side
        
            if document_len > 280:
                left_side = random.randint(50, (num_context_tokens-50))
                if left_side > start_span:
                    left_side = start_span
                right_side = num_context_tokens - left_side

            context_tokens = document_tokens[(start_span - left_side):(end_span + right_side)]
            context = ' '.join(context_tokens)

            for n, answer in enumerate(short_answers):
                start_span = answer["start_token"]
                end_span = answer["end_token"]
                answer_text = ' '.join(document_tokens[start_span:end_span])
                
                start_span_char = 0
                for k in range((start_span - left_side), start_span):
                    start_span_char += len(document_tokens[k])
                    start_span_char += 1

                answer = {}
                answer["text"] = answer_text
                answer["answer_start"] = start_span_char
                answers.append(answer)

            sample["context"] = context
            sample["answers"] = answers
            
            what_kind_of_sample = "ans"

    return sample, what_kind_of_sample

for i in range(50):
    print(i)
    flname = "dataset_clean/dataset_clean_"+str(i)+".json"
    fl = open(flname).read()
    data = json.loads(fl)
    for j in range(len(data)):
        sample, what_kind_of_sample = make_bert_sample(data[j])
        if what_kind_of_sample == "noans":
            dataset_noans.append(sample)
        if what_kind_of_sample == "ans":
            dataset_ans.append(sample)


len_dataset_ans = len(dataset_ans)
print(len_dataset_ans)
dataset_noans = dataset_noans[:len_dataset_ans]

dataset = dataset_noans + dataset_ans
random.shuffle(dataset)

out = open("natural_questions_train.json", 'w')
json.dump(dataset, out)
out.close()

